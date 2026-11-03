"""DMD2's diffusion-GAN loss (Yin et al., 2024) on the flow fake-score network.

Ported from the authors' SD/SDXL implementation (main/sd_guidance.py, main/train_sd.py):

* The discriminator shares the fake-score backbone. Like ``classify_forward``, which stops the UNet after its
  mid block, the transformer runs only its first ``feature_block`` blocks; a convolutional head (the SDXL
  ``cls_pred_branch``) maps the target-image tokens to one realism logit.
* ``diffusion_gan``: real and generated latents are noised before classification, at a uniform DDPM timestep in
  [0, max_timestep). Flow models have no DDPM schedule, so the timestep is mapped to the flow sigma with the same
  signal-to-noise ratio under the SD/SDXL ``scaled_linear`` betas the authors used.
* Generator: ``softplus(-D(fake))``. Discriminator: ``softplus(D(fake)) + softplus(-D(real))``.
"""

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import interpolate

# SD v1.5 / SDXL DDPM noise schedule (scheduler_config: scaled_linear, 0.00085 -> 0.012, 1000 steps).
SD_BETA_START, SD_BETA_END, SD_TRAIN_TIMESTEPS = 0.00085, 0.012, 1000


def sd_alphas_cumprod(num_train_timesteps=SD_TRAIN_TIMESTEPS):
    betas = torch.linspace(SD_BETA_START ** 0.5, SD_BETA_END ** 0.5, num_train_timesteps, dtype=torch.float64) ** 2
    return torch.cumprod(1 - betas, dim=0)


def ddpm_to_flow_sigma(timesteps):
    """Flow sigma whose SNR ((1-s)/s)^2 equals the DDPM SNR a/(1-a) at the given SD timesteps."""
    alpha = sd_alphas_cumprod().to(timesteps.device)[timesteps.long()]
    signal, noise = alpha.sqrt(), (1 - alpha).sqrt()
    return (noise / (signal + noise)).float()


def sample_gan_sigma(batch_size, device, generator=None, max_timestep=SD_TRAIN_TIMESTEPS):
    timesteps = torch.randint(0, max_timestep, (batch_size,), device=device, generator=generator)
    return ddpm_to_flow_sigma(timesteps)


def head_layers(in_channels, grid, width=1280, groups=32):
    """First conv halves the token grid and maps the transformer width to the authors' 1280 channels; the
    remaining stride-2 convs, the stride-4 conv to 1x1 and the 1x1 logit conv follow the SDXL cls_pred_branch."""
    if grid < 8 or grid & (grid - 1):
        raise ValueError('The discriminator head needs a power-of-two token grid of at least 8')
    layers, channels = [], in_channels
    while grid > 4:
        layers += [nn.Conv2d(channels, width, kernel_size=4, stride=2, padding=1), nn.GroupNorm(groups, width), nn.SiLU()]
        channels, grid = width, grid // 2
    layers += [nn.Conv2d(width, width, kernel_size=4, stride=4, padding=0), nn.GroupNorm(groups, width), nn.SiLU(),
               nn.Conv2d(width, 1, kernel_size=1, stride=1, padding=0)]
    return layers


class DiscriminatorHead(nn.Module):
    def __init__(self, in_channels, grid, width=1280, groups=32):
        super().__init__()
        self.grid = grid
        self.cls_pred_branch = nn.Sequential(*head_layers(in_channels, grid, width, groups))

    def forward(self, tokens):
        batch, count, channels = tokens.shape
        if count != self.grid * self.grid:
            raise ValueError(f'Expected {self.grid}x{self.grid} target tokens, got {count}')
        features = tokens.float().transpose(1, 2).reshape(batch, channels, self.grid, self.grid)
        # The authors run cls_pred_branch outside their bf16 autocast context, on float32 features.
        with torch.autocast(features.device.type, enabled=False):
            return self.cls_pred_branch(features).squeeze(dim=3).squeeze(dim=2)


def target_grid(condition):
    _, height, width = condition['img_shapes'][0][-1]
    if height != width:
        raise ValueError('The discriminator head currently expects square images')
    return height


def unwrap(transformer):
    return getattr(transformer, 'module', transformer)


def feature_tokens(adapter, transformer, latents, sigma, condition, feature_block):
    """Target-image hidden states after ``feature_block`` transformer blocks; later blocks are skipped."""
    model = unwrap(transformer)
    blocks = model.transformer_blocks
    if not 1 <= feature_block <= len(blocks):
        raise ValueError(f'feature_block must be in [1, {len(blocks)}]')
    captured = {}
    handle = blocks[feature_block - 1].register_forward_hook(lambda module, inputs, output: captured.__setitem__('h', output))
    model.transformer_blocks = nn.ModuleList(list(blocks)[:feature_block])
    try:
        adapter.predict(transformer, latents, sigma, condition)
    finally:
        model.transformer_blocks = blocks
        handle.remove()
    return captured['h'][:, -latents.shape[1]:]


def realism_logits(adapter, fake, head, latents, condition, feature_block, generator=None, max_timestep=SD_TRAIN_TIMESTEPS):
    """compute_cls_logits with diffusion_gan: noise, run the shared fake-score backbone, classify."""
    sigma = sample_gan_sigma(latents.shape[0], latents.device, generator, max_timestep)
    noise = torch.randn(latents.shape, device=latents.device, generator=generator)
    noisy = interpolate(latents, noise, sigma)
    return head(feature_tokens(adapter, fake, noisy, sigma, condition, feature_block))


def generator_gan_loss(fake_logits):
    return F.softplus(-fake_logits).mean()


def generator_gradient_diagnostics(dm_loss, gan_loss, clean, gan_weight):
    """Compare objective gradients at the shared student output, without accumulating parameter grads.

    These are latent gradients, not parameter gradients or an average over a training batch.
    Retaining both graphs lets the ordinary combined backward run unchanged afterwards.
    """
    dm = torch.autograd.grad(dm_loss, clean, retain_graph=True)[0].detach().float()
    gan = torch.autograd.grad(gan_loss, clean, retain_graph=True)[0].detach().float()
    dm_norm, gan_norm = dm.norm(), gan.norm()
    denominator = float(dm_norm * gan_norm)
    return dict(dm_latent_gradient_norm=float(dm_norm), gan_latent_gradient_norm=float(gan_norm),
        weighted_gan_latent_gradient_norm=float(gan_weight * gan_norm),
        weighted_gan_to_dm_ratio=float(gan_weight * gan_norm / dm_norm) if dm_norm > 0 else None,
        gradient_cosine=float((dm * gan).sum()) / denominator if denominator > 0 else None)


def discriminator_gan_loss(real_logits, fake_logits):
    return F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()


def restore_real_sampler(generator, checkpoint, examples, draws):
    """Restore the real-photo stream, replaying scalar draws for older checkpoints."""
    from pathlib import Path
    path = Path(checkpoint) / 'real_sampler_rng.pt'
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['examples'] != examples or state['draws'] != draws:
            raise ValueError('Real-photo sampler metadata differs from the resumed run')
        generator.set_state(state['state'])
        return 'saved_state'
    # The original single-GPU GAN loop drew once per record on every critic turn.
    for _ in range(draws):
        torch.randint(examples, (), generator=generator)
    return 'legacy_replay'


@contextmanager
def frozen(*modules):
    """Generator turn: gradients reach the student through D, never D itself (the authors zero them instead)."""
    parameters = [p for module in modules for p in module.parameters() if p.requires_grad]
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter in parameters:
            parameter.requires_grad_(True)


class RealLatents:
    """Converted real photos (main/prepare_flow_real_latents.py) with the captions they were published with."""
    def __init__(self, paths):
        self.latents, self.prompts = [], []
        for path in paths:
            data = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
            if len(data['latents']) != len(data['prompts']):
                raise ValueError(f'Latents and prompts differ in length in {path}')
            self.latents.append(data['latents'])
            self.prompts.extend(data['prompts'])
            self.height, self.width = int(data['height']), int(data['width'])
        self.offsets = torch.tensor([0] + [len(x) for x in self.latents]).cumsum(0)
        if not self.prompts:
            raise ValueError('No real latents were loaded')

    def __len__(self):
        return len(self.prompts)

    def latent(self, index):
        shard = int(torch.searchsorted(self.offsets, index, right=True)) - 1
        return self.latents[shard][index - int(self.offsets[shard])].float().unsqueeze(0)

    def records(self):
        return [dict(task='t2i', prompt=prompt, reference_images=[], height=self.height, width=self.width)
                for prompt in self.prompts]
