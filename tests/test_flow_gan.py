"""DMD2 diffusion-GAN port, checked against the authors' SD/SDXL implementation (main/sd_guidance.py)."""

import copy
import importlib
import io
import pickle
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from diffusers import DDIMScheduler, QwenImage21Transformer2DModel

from main.flow.checkpoints import configure_trainable
from main.flow.gan import (DiscriminatorHead, RealLatents, ddpm_to_flow_sigma, discriminator_gan_loss,
                           feature_tokens, frozen, generator_gan_loss, generator_gradient_diagnostics,
                           realism_logits, sample_gan_sigma, target_grid)
from main.flow.losses import interpolate
from main.prepare_flow_real_latents import NumpyOnlyUnpickler, load_shard
from main.train_flow import parse_args, resume_config_matches
from tests.test_flow import synthetic_adapter

SDXL_SCHEDULER = dict(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear')


@pytest.fixture
def author_guidance(monkeypatch):
    """The authors' SDGuidance built by its own __init__, with the SD/SDXL UNets replaced by a stub."""
    for name in ('matplotlib', 'matplotlib.backends', 'matplotlib.backends.backend_agg', 'matplotlib.pyplot',
                 'imageio', 'imageio.v2'):
        module = types.ModuleType(name)
        module.FigureCanvasAgg = object
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, 'main.utils', raising=False)
    monkeypatch.delitem(sys.modules, 'main.sd_guidance', raising=False)
    sd_guidance = importlib.import_module('main.sd_guidance')
    monkeypatch.setattr(sd_guidance.UNet2DConditionModel, 'from_pretrained', staticmethod(lambda *a, **k: nn.Linear(1, 1)))
    monkeypatch.setattr(sd_guidance.DDIMScheduler, 'from_pretrained', staticmethod(lambda *a, **k: DDIMScheduler(**SDXL_SCHEDULER)))
    args = SimpleNamespace(model_id='stub', gan_alone=False, use_fp16=False, num_train_timesteps=1000,
                           min_step_percent=0.02, max_step_percent=0.98, real_guidance_scale=8, fake_guidance_scale=1,
                           cls_on_clean_image=True, gen_cls_loss=True, sdxl=True, gradient_checkpointing=False,
                           diffusion_gan=True, diffusion_gan_max_timestep=1000)
    return sd_guidance.SDGuidance(args, accelerator=SimpleNamespace(is_local_main_process=False))


def test_gan_losses_match_authors(author_guidance):
    torch.manual_seed(0)
    logits = lambda image, **kwargs: image.mean(dim=[1, 2, 3]).unsqueeze(1) * 3
    author_guidance.compute_cls_logits = logits
    real, fake = torch.randn(4, 4, 8, 8), torch.randn(4, 4, 8, 8)
    author_gen = author_guidance.compute_generator_clean_cls_loss(fake, text_embedding=None)['gen_cls_loss']
    author_disc = author_guidance.compute_guidance_clean_cls_loss(real, fake, None, None)[0]['guidance_cls_loss']
    assert torch.allclose(generator_gan_loss(logits(fake)), author_gen)
    assert torch.allclose(discriminator_gan_loss(logits(real), logits(fake)), author_disc)


def test_head_tail_is_the_authors_sdxl_branch(author_guidance):
    author = author_guidance.cls_pred_branch
    head = DiscriminatorHead(in_channels=1280, grid=64)
    tail = head.cls_pred_branch[3:]
    assert [repr(m) for m in tail] == [repr(m) for m in author]
    for ours, theirs in zip(tail, author):
        ours.load_state_dict(theirs.state_dict())
    features = torch.randn(2, 1280, 32, 32)
    assert torch.allclose(tail(features), author(features), atol=1e-5)
    assert head(torch.randn(2, 64 * 64, 1280)).shape == (2, 1)


def test_diffusion_gan_noise_has_the_authors_snr(author_guidance):
    scheduler = author_guidance.scheduler
    timesteps = torch.tensor([0, 1, 250, 500, 750, 999])
    alpha = scheduler.alphas_cumprod[timesteps].double()
    sigma = ddpm_to_flow_sigma(timesteps).double()
    assert torch.allclose(((1 - sigma) / sigma) ** 2, alpha / (1 - alpha), rtol=1e-4)
    clean, noise = torch.randn(6, 4, 8, 8), torch.randn(6, 4, 8, 8)
    ddpm = scheduler.add_noise(clean, noise, timesteps)
    flow = interpolate(clean, noise, sigma.float())
    # A flow state is the authors' DDPM state up to a scalar, so both carry the same signal-to-noise ratio.
    scale = (alpha.sqrt() + (1 - alpha).sqrt()).float().view(-1, 1, 1, 1)
    assert torch.allclose(flow * scale, ddpm, atol=1e-5)
    sampled = sample_gan_sigma(4096, 'cpu', torch.Generator().manual_seed(0), 500)
    assert sampled.min() >= 0 and sampled.max() <= ddpm_to_flow_sigma(torch.tensor([499])) + 1e-6


def gan_model():
    return QwenImage21Transformer2DModel(in_channels=4, out_channels=4, num_layers=2, num_attention_heads=2,
                                         attention_head_dim=16, context_in_dim=32, axes_dims_rope=(4, 6, 6))


def gan_sample(grid=8):
    embeds = torch.randn(1, 3, 32)
    mask = torch.tensor([[False] * 3 + [True] * (grid * grid // 4)])
    condition = dict(prompt_embeds=embeds, prompt_mask=None, image_mask=mask, negative_prompt_embeds=embeds,
                     negative_prompt_mask=None, negative_image_mask=mask, img_shapes=[[(1, grid, grid)]],
                     reference_latents=None)
    return dict(shape=[1, grid * grid, 4], height=8 * grid, width=8 * grid, task='t2i', condition=condition)


def test_feature_tokens_are_the_mid_network_target_states():
    torch.manual_seed(0)
    model = gan_model()
    adapter = synthetic_adapter('qwen_image21', model)
    sample = gan_sample()
    latents = torch.randn(sample['shape'], requires_grad=True)
    captured = {}
    handle = model.transformer_blocks[0].register_forward_hook(lambda m, i, o: captured.__setitem__('h', o))
    adapter.predict(model, latents, .5, sample['condition'])
    handle.remove()
    tokens = feature_tokens(adapter, model, latents, .5, sample['condition'], 1)
    assert len(model.transformer_blocks) == 2
    assert torch.allclose(tokens, captured['h'][:, -64:], atol=1e-6)
    tokens.sum().backward()
    assert latents.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        feature_tokens(adapter, model, latents, .5, sample['condition'], 3)


def test_generator_turn_trains_student_only():
    torch.manual_seed(0)
    base = gan_model()
    adapter = synthetic_adapter('qwen_image21', base)
    fake = configure_trainable(copy.deepcopy(base), 'lora', rank=2, alpha=2, gradient_checkpointing=True)
    head = DiscriminatorHead(32, target_grid(gan_sample()["condition"]), width=64)
    clean = torch.randn(gan_sample()['shape'], requires_grad=True)
    with frozen(fake, head):
        gan_loss = generator_gan_loss(realism_logits(adapter, fake, head, clean, gan_sample()['condition'], 1))
        dm_loss = clean.square().mean()
        expected = torch.autograd.grad(dm_loss + .0005 * gan_loss, clean, retain_graph=True)[0]
        rng = torch.get_rng_state().clone()
        diagnostics = generator_gradient_diagnostics(dm_loss, gan_loss, clean, .0005)
        assert clean.grad is None and torch.equal(rng, torch.get_rng_state())
        assert diagnostics['dm_latent_gradient_norm'] > 0 and diagnostics['gan_latent_gradient_norm'] > 0
        assert diagnostics['weighted_gan_latent_gradient_norm'] == pytest.approx(.0005 * diagnostics['gan_latent_gradient_norm'])
        (dm_loss + .0005 * gan_loss).backward()
        assert torch.equal(clean.grad, expected)
    assert clean.grad.abs().sum() > 0
    trainable = [p for p in fake.parameters() if p.requires_grad] + list(head.parameters())
    assert trainable and all(p.grad is None for p in trainable)
    discriminator_gan_loss(realism_logits(adapter, fake, head, torch.randn(1, 64, 4), gan_sample()['condition'], 1),
                           realism_logits(adapter, fake, head, clean.detach(), gan_sample()['condition'], 1)).backward()
    assert all(p.grad is not None for p in head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fake.parameters() if p.requires_grad)


def test_gan_flags_validate_and_guard_resume(tmp_path):
    base = ['--model_family', 'qwen_image21', '--manifest', 'm.jsonl', '--output_dir', str(tmp_path), '--task', 't2i']
    gan = base + ['--gan_real_latents', 'r.pt', '--gen_cls_loss_weight', '5e-3', '--guidance_cls_loss_weight', '1e-2']
    args = parse_args(gan)
    assert args.use_gan and args.real_cache_dir.endswith('conditioning_cache-real')
    for bad in (base + ['--gen_cls_loss_weight', '5e-3'], gan + ['--gen_cls_loss_weight', '-1'],
                base + ['--gan_gradient_diagnostics_every', '100'], gan + ['--gan_gradient_diagnostics_every', '-1'],
                base + ['--gan_real_latents', 'r.pt'], gan + ['--diffusion_gan_max_timestep', '0'],
                [*gan[:-6], '--task', 'mixed', *gan[-6:]]):
        with pytest.raises(SystemExit):
            parse_args(bad)
    legacy = {key: value for key, value in vars(parse_args(base)).items() if 'gan' not in key and 'cls_loss' not in key}
    resume_config_matches(legacy, vars(parse_args(base)))
    with pytest.raises(ValueError, match='gen_cls_loss_weight'):
        resume_config_matches(legacy, vars(args))


def test_real_latents_and_safe_shard_loading(tmp_path):
    for shard, count in ((0, 3), (1, 2)):
        torch.save(dict(latents=torch.randn(count, 64, 4).half(), prompts=[f'p{shard}-{i}' for i in range(count)],
                        height=64, width=64), tmp_path / f'flow_latents_{shard:03d}.pt')
    real = RealLatents([tmp_path / 'flow_latents_000.pt', tmp_path / 'flow_latents_001.pt'])
    assert len(real) == 5 and real.latent(4).shape == (1, 64, 4) and real.prompts[3] == 'p1-0'
    assert real.records()[3] == dict(task='t2i', prompt='p1-0', reference_images=[], height=64, width=64)
    authors = dict(latents=np.random.randn(2, 4, 8, 8).astype(np.float16), prompts=np.array(['a', 'b']))
    torch.save(authors, tmp_path / 'vae_latents_000.pt', pickle_protocol=5)
    latents, prompts = load_shard(tmp_path / 'vae_latents_000.pt')
    assert np.array_equal(latents, authors['latents']) and prompts == ['a', 'b']

    class Exploit:
        def __reduce__(self):
            return (print, ('owned',))
    with pytest.raises(pickle.UnpicklingError):
        NumpyOnlyUnpickler(io.BytesIO(pickle.dumps(Exploit(), protocol=5))).load()
