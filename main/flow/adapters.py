"""Native pipeline helpers provide frozen conditioning; forwards stay differentiable."""

import hashlib
import re
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .data import map_tensors


MODEL_IDS = {
    'flux_klein': 'black-forest-labs/FLUX.2-klein-base-4B',
    'qwen_image21': 'Qwen/Qwen-Image-2.1',
}


def resolve_revision(model_id, revision=None):
    if Path(model_id).is_dir():
        return None
    # An immutable commit is already resolved. Loading its components will verify
    # availability; do not make a metadata request for every teacher/student/fake.
    if revision and re.fullmatch(r'[0-9a-fA-F]{40}', revision):
        return revision.lower()
    from huggingface_hub import model_info
    return model_info(model_id, revision=revision).sha


class FlowAdapter:
    def __init__(self, family, model_id=None, revision=None, device='cuda', dtype=torch.bfloat16,
                 teacher_guidance=None, scheduler_config=None):
        import diffusers
        from diffusers import Flux2KleinPipeline, QwenImage21Pipeline
        self.family = family
        self.model_id = model_id or MODEL_IDS[family]
        self.revision = resolve_revision(self.model_id, revision)
        self.device = torch.device(device)
        self.dtype = dtype
        self.teacher_guidance = teacher_guidance if teacher_guidance is not None else (4.0 if family == 'flux_klein' else 1.0)
        self.saved_schedules = {}
        self.keep_encoders_on_device = False
        cls = Flux2KleinPipeline if family == 'flux_klein' else QwenImage21Pipeline
        self.pipe = cls.from_pretrained(self.model_id, revision=self.revision, torch_dtype=dtype, transformer=None)
        self.transformer_class = diffusers.Flux2Transformer2DModel if family == 'flux_klein' else diffusers.QwenImage21Transformer2DModel
        self.transformer_config = self.transformer_class.load_config(self.model_id, subfolder='transformer', revision=self.revision)
        if scheduler_config is not None:
            self.pipe.scheduler = self.pipe.scheduler.__class__.from_config(scheduler_config)
        for component in (self.pipe.text_encoder, self.pipe.vae):
            component.requires_grad_(False).eval()
        self.identity = dict(format_version=1, family=family, model_id=self.model_id,
                             revision=self.revision, diffusers_version=diffusers.__version__,
                             teacher_guidance=self.teacher_guidance, dtype=str(dtype))
        if Path(self.model_id).is_dir():
            digest = hashlib.sha256()
            for file in sorted(Path(self.model_id).rglob('*')):
                if file.is_file():
                    digest.update(str(file.relative_to(self.model_id)).encode())
                    digest.update(str((file.stat().st_size, file.stat().st_mtime_ns)).encode())
                    if file.suffix == '.json':
                        digest.update(file.read_bytes())
            self.identity['local_fingerprint'] = digest.hexdigest()

    def load_transformer(self):
        return self.transformer_class.from_pretrained(
            self.model_id, subfolder='transformer', revision=self.revision, torch_dtype=self.dtype
        ).to(self.device)

    def offload_encoders(self):
        self.pipe.text_encoder.cpu()
        self.pipe.vae.cpu()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def metadata(self):
        return dict(self.identity, scheduler_config=dict(self.pipe.scheduler.config),
                    schedule_policy='native-linear-sigmas-v1')

    @torch.no_grad()
    def prepare(self, record):
        h, w = record['height'], record['width']
        multiple = self.pipe.vae_scale_factor * 2
        if h % multiple or w % multiple:
            raise ValueError(f'{self.family} output dimensions must be divisible by {multiple}')
        if self.family == 'qwen_image21' and len(record['reference_images']) > 10:
            raise ValueError('Qwen Image 2.1 supports at most 10 reference images')
        images = []
        for path in record['reference_images']:
            with Image.open(path) as image:
                images.append(image.convert('RGBA' if self.family == 'qwen_image21' else 'RGB'))
        self.pipe.text_encoder.to(self.device)
        try:
            if self.family == 'flux_klein':
                embeds, text_ids = self.pipe.encode_prompt(record['prompt'], device=self.device)
                negative = self.pipe.encode_prompt('', device=self.device) if self.teacher_guidance > 1 else (None, None)
                condition = dict(prompt_embeds=embeds, text_ids=text_ids,
                                 negative_prompt_embeds=negative[0], negative_text_ids=negative[1])
                processed = []
                for image in images:
                    self.pipe.image_processor.check_image_input(image)
                    if image.width * image.height > 1024 ** 2:
                        image = self.pipe.image_processor._resize_to_target_area(image, 1024 ** 2)
                    rw, rh = image.width // multiple * multiple, image.height // multiple * multiple
                    if min(rw, rh) == 0:
                        raise ValueError('Reference image is smaller than the native VAE alignment')
                    processed.append(self.pipe.image_processor.preprocess(image, height=rh, width=rw, resize_mode='crop'))
            else:
                from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
                input_images, processed, sizes = [], [], []
                for image in images:
                    rw, rh, _ = calculate_dimensions(h * w, image.width / image.height)
                    sizes.append((rw, rh))
                    input_images.append(self.pipe.image_processor.resize(image, width=rw, height=rh))
                    processed.append(self.pipe.image_processor.preprocess(image, width=rw, height=rh).unsqueeze(2))
                encoded = self.pipe.encode_prompt(record['prompt'], image=input_images or None, device=self.device)
                negative = self.pipe.encode_prompt('', image=input_images or None, device=self.device) if self.teacher_guidance > 1 else (None, None, None)
                condition = dict(prompt_embeds=encoded[0], prompt_mask=encoded[1], image_mask=encoded[2],
                                 negative_prompt_embeds=negative[0], negative_prompt_mask=negative[1], negative_image_mask=negative[2])
        finally:
            if not getattr(self, 'keep_encoders_on_device', False):
                self.pipe.text_encoder.cpu()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        self.pipe.vae.to(self.device)
        try:
            generator = torch.Generator(device=self.device).manual_seed(0)
            if self.family == 'flux_klein':
                latents, ids = self.pipe.prepare_latents(1, self.transformer_config['in_channels'] // 4,
                                                       h, w, self.dtype, self.device, generator)
                refs, ref_ids = self.pipe.prepare_image_latents(processed, 1, generator, self.device, self.dtype) if images else (None, None)
                condition.update(latent_ids=ids, reference_latents=refs, reference_ids=ref_ids)
            else:
                latents, refs = self.pipe.prepare_latents(processed or None, 1, self.transformer_config['in_channels'],
                                                        h, w, self.dtype, self.device, generator)
                condition['reference_latents'] = refs
                condition['img_shapes'] = [[*( (1, rh // self.pipe.vae_scale_factor, rw // self.pipe.vae_scale_factor) for rw, rh in sizes ),
                                            (1, h // self.pipe.vae_scale_factor, w // self.pipe.vae_scale_factor)]]
                for key in ('image_mask', 'negative_image_mask'):
                    mask = condition[key]
                    if mask is not None:
                        condition[key] = torch.cat([mask, mask.new_ones(1, latents.shape[1] // 4)], dim=1)
        finally:
            if not getattr(self, 'keep_encoders_on_device', False):
                self.pipe.vae.cpu()
        return dict(shape=list(latents.shape), condition=condition, height=h, width=w, task=record['task'])

    def schedule(self, sample, num_steps):
        key = f'{sample["height"]}x{sample["width"]}:{num_steps}'
        if key in getattr(self, 'saved_schedules', {}):
            return torch.tensor(self.saved_schedules[key], device=self.device, dtype=torch.float32)
        if num_steps == 1:
            # Terminal stretching (Qwen's shift_terminal=0.02) divides by zero
            # for a singleton [1] schedule. A one-step student goes directly
            # from pure noise to clean, so its endpoints are exactly [1, 0].
            return torch.tensor([1.0, 0.0], device=self.device)
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
        scheduler = self.pipe.scheduler.__class__.from_config(self.pipe.scheduler.config)
        length = sample['shape'][1]
        if self.family == 'flux_klein':
            mu = compute_empirical_mu(length, num_steps)
        else:
            config = scheduler.config
            mu = calculate_shift(length, config.get('base_image_seq_len', 256), config.get('max_image_seq_len', 4096),
                                 config.get('base_shift', 0.5), config.get('max_shift', 1.15))
        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        if scheduler.config.get('use_flow_sigmas', False):
            sigmas = None
        scheduler.set_timesteps(num_steps, device=self.device, sigmas=sigmas, mu=mu)
        if len(scheduler.sigmas) != num_steps + 1 or scheduler.sigmas[-1] != 0 or scheduler.sigmas[0] != 1:
            raise ValueError('Distillation requires a native schedule starting at pure noise and ending at sigma=0')
        if not torch.isfinite(scheduler.sigmas).all() or not (scheduler.sigmas[:-1] > scheduler.sigmas[1:]).all():
            raise ValueError('Native sigma schedule must be finite and strictly decreasing')
        return scheduler.sigmas.to(self.device).float()

    def predict(self, transformer, latents, sigma, condition, guidance_scale=1.0):
        # Accept Accelerate/DDP-wrapped models without accessing wrapper dtype.
        parameter = next(transformer.parameters())
        dtype, device = parameter.dtype, parameter.device
        condition = map_tensors(condition, lambda x: x.to(device=device, dtype=dtype if x.is_floating_point() else x.dtype))
        state = latents.to(device=device, dtype=dtype)
        sigma = torch.as_tensor(sigma, device=device, dtype=dtype).reshape(-1).expand(state.shape[0])
        refs = condition.get('reference_latents')
        if self.family == 'flux_klein':
            hidden = torch.cat([state, refs], dim=1) if refs is not None else state
            ids = torch.cat([condition['latent_ids'], condition['reference_ids']], dim=1) if refs is not None else condition['latent_ids']
            base = dict(hidden_states=hidden, timestep=sigma, guidance=None, img_ids=ids, return_dict=False)
            def forward(negative=False):
                return transformer(**base,
                    encoder_hidden_states=condition['negative_prompt_embeds' if negative else 'prompt_embeds'],
                    txt_ids=condition['negative_text_ids' if negative else 'text_ids'])[0][:, :state.shape[1]]
        else:
            hidden = torch.cat([refs, state], dim=1) if refs is not None else state
            base = dict(hidden_states=hidden, timestep=sigma, img_shapes=condition['img_shapes'],
                        kv_cache=None, kv_cache_mode=None, return_dict=False)
            def forward(negative=False):
                return transformer(**base,
                    encoder_hidden_states=condition['negative_prompt_embeds' if negative else 'prompt_embeds'],
                    encoder_hidden_states_mask=condition['negative_prompt_mask' if negative else 'prompt_mask'],
                    img_mask=condition['negative_image_mask' if negative else 'image_mask'])[0][:, -state.shape[1]:]
        predicted = forward()
        if guidance_scale > 1:
            if condition.get('negative_prompt_embeds') is None:
                raise ValueError('Guided teacher predictions require cached negative conditioning')
            uncond = forward(True)
            predicted = uncond + guidance_scale * (predicted - uncond)
        return predicted

    @torch.no_grad()
    def decode(self, latents, sample):
        vae = self.pipe.vae.to(self.device)
        h, w = sample['height'], sample['width']
        if self.family == 'flux_klein':
            factor = self.pipe.vae_scale_factor * 2
            ids = sample['condition']['latent_ids'].to(self.device)
            latents = self.pipe._unpack_latents_with_ids(latents, ids, h // factor, w // factor).to(vae.dtype)
            mean = vae.bn.running_mean.view(1, -1, 1, 1)
            std = (vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).sqrt()
            latents = self.pipe._unpatchify_latents(latents * std + mean)
            image = vae.decode(latents, return_dict=False)[0]
        else:
            latents = self.pipe._unpack_latents(latents, h, w, self.pipe.vae_scale_factor).to(vae.dtype)
            mean = torch.tensor(vae.config.latents_mean, device=self.device, dtype=vae.dtype).view(1, -1, 1, 1, 1)
            std = torch.tensor(vae.config.latents_std, device=self.device, dtype=vae.dtype).view(1, -1, 1, 1, 1)
            image = vae.decode(latents * std + mean, return_dict=False)[0][:, :, 0]
        return self.pipe.image_processor.postprocess(image, output_type='pil')[0]
