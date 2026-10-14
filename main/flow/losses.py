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


def distillation_loss(adapter, student, teacher, fake, sample, num_steps, guidance_scale, generator=None,
                      scoring_generator=None, selected_step=None):
    clean = student_clean_sample(adapter, student, sample, num_steps, generator, selected_step)
    scoring_generator = generator if scoring_generator is None else scoring_generator
    sigma = sample_score_sigma(adapter, clean.shape[0], clean.device, scoring_generator)
    noise = torch.randn(clean.shape, device=clean.device, generator=scoring_generator)
    with torch.no_grad():
        noisy = interpolate(clean.detach(), noise, sigma)
        teacher_v = adapter.predict(teacher, noisy, sigma, sample['condition'], guidance_scale)
        fake_v = adapter.predict(fake, noisy, sigma, sample['condition'])
        teacher_clean = reconstruct_clean(noisy, teacher_v, sigma)
        fake_clean = reconstruct_clean(noisy, fake_v, sigma)
    loss, gradient = distribution_matching_loss(clean, teacher_clean, fake_clean)
    return loss, clean.detach(), gradient
