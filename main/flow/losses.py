"""Losses use clean=0/noise=1 sigma and velocity=noise-clean."""

import torch
import torch.nn.functional as F


def sample_score_sigma(adapter, batch_size, device, generator=None):
    """FastVideo samples integer timesteps, then shifts them toward high noise.

    Keep the original continuous, bounded sampler for existing recipes. The
    shifted policy matches FastVideo's modular Wan recipe (no extra .02/.98
    clamp), independently of the student's inference schedule.
    """
    if getattr(adapter, 'score_noise_sampling', 'uniform') == 'uniform':
        return torch.rand(batch_size, device=device, generator=generator) * .96 + .02
    num_timesteps = int(adapter.pipe.scheduler.config.num_train_timesteps)
    u = torch.randint(num_timesteps, (batch_size,), device=device, generator=generator).float() / num_timesteps
    shift = adapter.score_noise_shift
    return shift * u / (1 + (shift - 1) * u)


def expand_sigma(sigma, sample):
    return torch.as_tensor(sigma, device=sample.device, dtype=torch.float32).reshape(
        -1, *([1] * (sample.ndim - 1))
    )


def interpolate(clean, noise, sigma):
    sigma = expand_sigma(sigma, clean)
    return (1 - sigma) * clean.float() + sigma * noise.float()


def reconstruct_clean(noisy, velocity, sigma):
    return noisy.float() - expand_sigma(sigma, noisy) * velocity.float()


def distribution_matching_loss(student_clean, teacher_clean, fake_clean, epsilon=1e-6):
    """DMD's surrogate gradient; critic estimates never receive gradients here."""
    with torch.no_grad():
        dims = tuple(range(1, student_clean.ndim))
        scale = (student_clean.float() - teacher_clean.float()).abs().mean(dims, keepdim=True)
        gradient = (fake_clean.float() - teacher_clean.float()) / scale.clamp_min(epsilon)
        gradient = torch.nan_to_num(gradient)
        target = student_clean.float() - gradient
    return 0.5 * F.mse_loss(student_clean.float(), target), gradient


def fake_score_loss(predicted_velocity, student_clean, noise):
    return F.mse_loss(predicted_velocity.float(), noise.float() - student_clean.detach().float())


def rollout_prediction(adapter, student, state, sigma, condition):
    if getattr(adapter, 'sampling_precision', 'fp32') == 'native':
        # Match FLUX's pinned native pipeline: round t=1000*sigma to the carry
        # dtype before dividing by 1000 for the transformer timestep input.
        sigma = (sigma * 1000).to(state.dtype) / 1000
    return adapter.predict(student, state, sigma, condition)


def rollout_update(adapter, state, velocity, sigma, next_sigma, generator=None):
    if getattr(adapter, 'rollout_mode', 'euler') == 'stochastic':
        clean = reconstruct_clean(state, velocity, sigma)
        if float(next_sigma) == 0:
            return clean.to(velocity.dtype) if getattr(adapter, 'sampling_precision', 'fp32') == 'native' else clean
        noise = torch.randn(clean.shape, device=clean.device, generator=generator)
        result = interpolate(clean, noise, next_sigma)
        return result.to(velocity.dtype) if getattr(adapter, 'sampling_precision', 'fp32') == 'native' else result
    if getattr(adapter, 'sampling_precision', 'fp32') == 'native':
        return (state.float() + (next_sigma - sigma) * velocity).to(velocity.dtype)
    return state + (next_sigma - sigma) * velocity.float()


def rollout_noise(adapter, student, shape, generator):
    state = torch.randn(shape, device=next(student.parameters()).device, generator=generator)
    if getattr(adapter, 'sampling_precision', 'fp32') == 'native':
        state = state.to(next(student.parameters()).dtype)
    return state


def student_clean_sample(adapter, student, sample, num_steps, generator=None, selected_step=None):
    """Train a random step on the student's own preceding inference states."""
    sigmas = adapter.schedule(sample, num_steps)
    device = next(student.parameters()).device
    state = rollout_noise(adapter, student, sample['shape'], generator)
    selected = int(torch.randint(num_steps, (), device=device, generator=generator)) if selected_step is None else int(selected_step)
    if not 0 <= selected < num_steps:
        raise ValueError('Selected rollout step is outside the trained schedule')
    with torch.no_grad():
        for index in range(selected):
            velocity = rollout_prediction(adapter, student, state, sigmas[index], sample['condition'])
            state = rollout_update(adapter, state, velocity, sigmas[index], sigmas[index + 1], generator)
    velocity = rollout_prediction(adapter, student, state.detach(), sigmas[selected], sample['condition'])
    return reconstruct_clean(state.detach(), velocity, sigmas[selected])


@torch.no_grad()
def sample_student(adapter, student, sample, num_steps, generator=None):
    sigmas = adapter.schedule(sample, num_steps)
    state = rollout_noise(adapter, student, sample['shape'], generator)
    for index in range(num_steps):
        velocity = rollout_prediction(adapter, student, state, sigmas[index], sample['condition'])
        state = rollout_update(adapter, state, velocity, sigmas[index], sigmas[index + 1], generator)
    return state


def normalized_direction(student_clean, target_clean, direction, epsilon=1e-6):
    """Scale a clean-space direction by DMD's per-sample |student - target| normalizer."""
    dims = tuple(range(1, student_clean.ndim))
    scale = (student_clean.float() - target_clean.float()).abs().mean(dims, keepdim=True)
    return torch.nan_to_num(direction.float() / scale.clamp_min(epsilon))


def cfg_term_sigma(adapter, batch_size, device, generator=None):
    """Independent CFG-augmentation noise level, mapped into [cfg_noise_min, cfg_noise_max]."""
    low, high = getattr(adapter, 'cfg_noise_min', 0.0), getattr(adapter, 'cfg_noise_max', 1.0)
    return low + (high - low) * sample_score_sigma(adapter, batch_size, device, generator)


def decoupled_gradient(adapter, teacher, fake, clean, sample, guidance_scale, sigma, noise, generator=None):
    """Decoupled DMD (Liu et al., 2025): the CFG target splits into a conditional
    distribution-matching term and a (w-1)(cond-uncond) CFG-augmentation term.

    The CFG term is scored at its own independently drawn noise level, so its
    weight and sigma range can be tuned separately from distribution matching.
    """
    noisy = interpolate(clean, noise, sigma)
    cond_clean = reconstruct_clean(noisy, adapter.predict(teacher, noisy, sigma, sample['condition']), sigma)
    fake_clean = reconstruct_clean(noisy, adapter.predict(fake, noisy, sigma, sample['condition']), sigma)
    gradient = normalized_direction(clean, cond_clean, fake_clean - cond_clean)
    weight = getattr(adapter, 'cfg_term_weight', 1.0)
    if guidance_scale > 1 and weight > 0:
        cfg_sigma = cfg_term_sigma(adapter, clean.shape[0], clean.device, generator)
        cfg_noisy = interpolate(clean, torch.randn(clean.shape, device=clean.device, generator=generator), cfg_sigma)
        cond_v, uncond_v = adapter.predict(teacher, cfg_noisy, cfg_sigma, sample['condition'], return_branches=True)
        cfg_cond = reconstruct_clean(cfg_noisy, cond_v, cfg_sigma)
        cfg_uncond = reconstruct_clean(cfg_noisy, uncond_v, cfg_sigma)
        guided = cfg_uncond + guidance_scale * (cfg_cond - cfg_uncond)
        augmentation = -(guidance_scale - 1) * (cfg_cond - cfg_uncond)
        gradient = gradient + weight * normalized_direction(clean, guided, augmentation)
    return gradient


def distillation_loss(adapter, student, teacher, fake, sample, num_steps, guidance_scale, generator=None,
                      scoring_generator=None, selected_step=None, keep_graph=False):
    """keep_graph returns the differentiable student sample, for losses added on the same generator turn."""
    clean = student_clean_sample(adapter, student, sample, num_steps, generator, selected_step)
    scoring_generator = generator if scoring_generator is None else scoring_generator
    sigma = sample_score_sigma(adapter, clean.shape[0], clean.device, scoring_generator)
    noise = torch.randn(clean.shape, device=clean.device, generator=scoring_generator)
    if getattr(adapter, 'dmd_guidance_mode', 'combined') == 'decoupled':
        with torch.no_grad():
            gradient = decoupled_gradient(adapter, teacher, fake, clean.detach().float(), sample, guidance_scale,
                                          sigma, noise, scoring_generator)
            target = clean.detach().float() - gradient
        return 0.5 * F.mse_loss(clean.float(), target), clean if keep_graph else clean.detach(), gradient
    with torch.no_grad():
        noisy = interpolate(clean.detach(), noise, sigma)
        teacher_v = adapter.predict(teacher, noisy, sigma, sample['condition'], guidance_scale)
        fake_v = adapter.predict(fake, noisy, sigma, sample['condition'])
        teacher_clean = reconstruct_clean(noisy, teacher_v, sigma)
        fake_clean = reconstruct_clean(noisy, fake_v, sigma)
    loss, gradient = distribution_matching_loss(clean, teacher_clean, fake_clean)
    return loss, clean if keep_graph else clean.detach(), gradient
