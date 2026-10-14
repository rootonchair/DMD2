"""Bounded FLUX diagnostic using the raw noisy-marginal flow-KL gradient.

This is an explicit experimental entry point, not a change to the production
normalized-DMD default. It uses the same trainer, data, critic and artifact gates.
"""

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from main.flow.losses import expand_sigma, interpolate, reconstruct_clean, student_clean_sample


def flow_kl_gradient(teacher_clean, fake_clean, sigma):
    sigma = expand_sigma(sigma, fake_clean)
    if not ((sigma > 0) & (sigma < 1)).all():
        raise ValueError('Raw flow-KL scoring requires sigma strictly between zero and one')
    # z = a*x + sigma*epsilon, a=1-sigma. The score difference is
    # a/sigma^2 * (fake_clean-teacher_clean); differentiating z adds another a.
    return (1-sigma).square() / sigma.square() * (fake_clean.float()-teacher_clean.float())


def flow_kl_loss(student_clean, teacher_clean, fake_clean, sigma):
    with torch.no_grad():
        gradient = flow_kl_gradient(teacher_clean, fake_clean, sigma)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite raw flow-KL gradient')
        target = student_clean.float()-gradient
    return .5*F.mse_loss(student_clean.float(), target), gradient


def distillation_loss(adapter, student, teacher, fake, sample, num_steps, guidance_scale, generator=None):
    clean = student_clean_sample(adapter, student, sample, num_steps, generator)
    sigma = torch.rand(clean.shape[0], device=clean.device, generator=generator)*.96+.02
    noise = torch.randn(clean.shape, device=clean.device, generator=generator)
    with torch.no_grad():
        noisy = interpolate(clean.detach(), noise, sigma)
        teacher_v = adapter.predict(teacher, noisy, sigma, sample['condition'], guidance_scale)
        fake_v = adapter.predict(fake, noisy, sigma, sample['condition'])
        teacher_clean = reconstruct_clean(noisy, teacher_v, sigma)
        fake_clean = reconstruct_clean(noisy, fake_v, sigma)
    loss, gradient = flow_kl_loss(clean, teacher_clean, fake_clean, sigma)
    return loss, clean.detach(), gradient


def main(argv=None):
    from main import train_flow
    original_parse = train_flow.parse_args
    original_loss = train_flow.distillation_loss
    def parse(arguments=None):
        args = original_parse(arguments)
        if args.model_family != 'flux_klein' or args.task != 'i2i' or args.num_steps != 4 or not args.validation_manifest or args.resume or args.train_iters > 1000:
            raise ValueError('Raw flow-KL diagnostics require a fresh FLUX quality pilot of at most 1000 fake updates')
        args.loss_weighting = 'raw_flow_kl'
        args.objective_code_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return args
    try:
        train_flow.parse_args = parse
        train_flow.distillation_loss = distillation_loss
        print('EXPERIMENTAL_OBJECTIVE raw_flow_kl; normalized production default is unchanged', flush=True)
        train_flow.main(argv)
    finally:
        train_flow.parse_args = original_parse
        train_flow.distillation_loss = original_loss


if __name__ == '__main__':
    main()
