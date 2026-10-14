"""An oracle counterexample for adaptive normalization and raw flow-KL checks."""

import json
from pathlib import Path

import torch

from main.flow.losses import distribution_matching_loss
from main.flow_kl_diagnostic import flow_kl_gradient, flow_kl_loss
from main.validate_flow import gaussian_posterior


def validate_weighting(samples=262144):
    sigma, mean_q, mean_p, variance_q, variance_p = .8, 1.3, -.8, 2.0, .4
    a = 1-sigma
    qvar, pvar = a*a*variance_q+sigma*sigma, a*a*variance_p+sigma*sigma
    analytic = a*a*variance_q*(1/pvar-1/qvar)
    rows = []
    for seed in [0, 1, 2]:
        generator = torch.Generator().manual_seed(seed)
        xi = torch.randn(samples, 1, 1, generator=generator)
        noise = torch.randn(xi.shape, generator=generator)
        clean = mean_q+variance_q**.5*xi
        z = a*clean+sigma*noise
        q = gaussian_posterior(z, sigma, mean_q, variance_q)
        p = gaussian_posterior(z, sigma, mean_p, variance_p)
        _, normalized = distribution_matching_loss(clean, p, q)
        raw = flow_kl_gradient(p, q, sigma)
        normalized_variance = float((normalized*variance_q**.5*xi).mean())
        raw_variance = float((raw*variance_q**.5*xi).mean())
        assert normalized_variance < 0 and raw_variance > 0
        assert abs(raw_variance-analytic) < .002
        rows.append(dict(seed=seed, normalized_log_std_gradient=normalized_variance,
                         raw_log_std_gradient=raw_variance, analytic_log_std_gradient=analytic))
    clean = torch.ones(1, 2, 3, requires_grad=True)
    teacher = torch.zeros_like(clean, requires_grad=True)
    fake = torch.ones_like(clean, requires_grad=True)
    loss, gradient = flow_kl_loss(clean, teacher, fake, .5)
    loss.backward()
    assert (clean.grad > 0).all() and teacher.grad is None and fake.grad is None
    assert torch.equal(gradient, torch.ones_like(gradient))
    return dict(completed=True, raw_gradient_checks_passed=True,
        normalized_general_kl_descent_guarantee=False, oracle_counterexample=rows,
        gaussian=dict(mean_q=mean_q, mean_p=mean_p, variance_q=variance_q, variance_p=variance_p,
                      sigma=sigma, samples_per_seed=samples, student_family='Fixed mean; trainable log standard deviation'),
        finding='Adaptive per-sample normalization can reverse the expected variance update relative to unweighted noisy-marginal KL, even with oracle critics.',
        limits='A scalar constrained Gaussian counterexample, not proof of the FLUX failure. Original DMD2 uses the same normalization heuristic. Raw weighting still relies on learned approximate scores and needs image-quality validation.')


def main():
    result = validate_weighting()
    output = Path('outputs/flux-validation/weighting.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
