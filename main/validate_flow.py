"""Independent correctness gates for the experimental flow distillation backend.

Reports are evidence about particular gates, not a claim of editing convergence.
Run math without pretrained weights; native uses the pinned real FLUX teacher.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from main.flow.losses import distribution_matching_loss, fake_score_loss, reconstruct_clean


def tensor_error(actual, expected):
    actual, expected = actual.detach().double(), expected.detach().double()
    difference = actual - expected
    return dict(max_abs=float(difference.abs().max()), rms=float(difference.square().mean().sqrt()),
                relative_rms=float(difference.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-12)),
                finite=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()))


def gaussian_posterior(z, sigma, mean, variance):
    """Condition a Gaussian clean distribution on a linear noisy observation."""
    a = 1 - sigma
    marginal_variance = a * a * variance + sigma * sigma
    return mean + a * variance / marginal_variance * (z - a * mean)


def validate_math():
    # Independent analytic marginal KL, rather than a second implementation of the surrogate.
    # p_sigma and q_sigma have equal variance, making the mean KL derivative exact.
    variance, real_mean, fake_mean = 0.7, -0.8, 1.3
    rows = []
    for sigma in (0.02, 0.1, 0.3, 0.7, 0.98):
        a = 1 - sigma
        marginal_variance = a * a * variance + sigma * sigma
        z = torch.linspace(-3, 3, 40, dtype=torch.float64).reshape(5, 2, 4)
        teacher_clean = gaussian_posterior(z, sigma, real_mean, variance)
        fake_clean = gaussian_posterior(z, sigma, fake_mean, variance)
        teacher_score = -(z - a * real_mean) / marginal_variance
        fake_score = -(z - a * fake_mean) / marginal_variance
        # score_q - score_p = (1-sigma)/sigma^2 * (E_q[x0|z] - E_p[x0|z]).
        score_error = tensor_error(a / sigma**2 * (fake_clean - teacher_clean), fake_score - teacher_score)
        true_gradient = a * a * (fake_mean - real_mean) / marginal_variance
        reconstructed_gradient = a * a / sigma**2 * (fake_clean - teacher_clean)
        kl = lambda mu: 0.5 * a * a * (mu - real_mean)**2 / marginal_variance
        step = 1e-5
        finite_difference = (kl(fake_mean + step) - kl(fake_mean - step)) / (2 * step)
        assert score_error['max_abs'] < 1e-10
        assert torch.allclose(reconstructed_gradient, torch.full_like(z, true_gradient), atol=1e-10)
        assert abs(finite_difference - true_gradient) < 1e-9
        # The code intentionally uses a normalized heuristic, not the raw KL derivative.
        student = torch.full_like(z, fake_mean, requires_grad=True)
        teacher = teacher_clean.clone().requires_grad_(True)
        fake = fake_clean.clone().requires_grad_(True)
        loss, gradient = distribution_matching_loss(student, teacher, fake)
        loss.backward()
        assert (student.grad > 0).all(), 'Gradient descent must move the mean toward the teacher'
        assert teacher.grad is None and fake.grad is None
        expected_scale = (student.detach().float() - teacher.detach().float()).abs().mean((1, 2), keepdim=True)
        expected = (fake.detach().float() - teacher.detach().float()) / expected_scale
        assert torch.allclose(gradient, expected, atol=1e-6)
        # Perfect conditional flow velocity reconstructs the independently derived posterior.
        velocity = (z - teacher_clean) / sigma
        reconstruction_error = tensor_error(reconstruct_clean(z, velocity, sigma), teacher_clean)
        assert reconstruction_error['max_abs'] < 3e-6
        rows.append(dict(sigma=sigma, analytic_kl_gradient=true_gradient, finite_difference=finite_difference,
                         score_identity_error=score_error, reconstruction_error=reconstruction_error))
    x = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(5), requires_grad=True)
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(6))
    prediction = (noise - x.detach()).clone().requires_grad_(True)
    loss = fake_score_loss(prediction, x, noise)
    loss.backward()
    assert float(loss.detach()) == 0 and x.grad is None
    prediction = (noise - x.detach() + 0.2).requires_grad_(True)
    loss = fake_score_loss(prediction, x, noise)
    loss.backward()
    assert (prediction.grad > 0).all() and x.grad is None
    same = torch.randn(2, 3, 4)
    zero, gradient = distribution_matching_loss(same, same, same)
    assert float(zero) == 0 and not gradient.any()
    # Check a real learning trajectory with oracle critics on a fixed Gaussian family.
    mean = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([mean], lr=0.025)
    history = []
    for _ in range(160):
        sigma = 0.5
        clean = mean.expand(32, 1, 1)
        z = (1 - sigma) * clean.detach()
        q = gaussian_posterior(z, sigma, mean.detach(), variance)
        p = gaussian_posterior(z, sigma, real_mean, variance)
        objective, _ = distribution_matching_loss(clean, p, q)
        objective.backward()
        optimizer.step()
        optimizer.zero_grad()
        history.append(float(mean.detach()))
    assert abs(history[-1] - real_mean) < 0.05, history[-1]
    # Validate variance as well as mean, integrating both clean/noise Gaussian
    # variables with quadrature so Monte Carlo error cannot hide a wrong factor.
    nodes, weights = np.polynomial.hermite.hermgauss(12)
    nodes = torch.tensor(nodes * np.sqrt(2), dtype=torch.float64)
    weights = torch.tensor(weights / np.sqrt(np.pi), dtype=torch.float64)
    clean_nodes, noise_nodes = torch.meshgrid(nodes, nodes, indexing='ij')
    joint_weights = weights[:, None] * weights[None, :]
    covariance_cases = []
    for fake_variance, real_variance in [(2.0, .4), (.2, 1.4)]:
        for sigma in (.02, .3, .8, .98):
            a = 1 - sigma
            qvar = a*a*fake_variance + sigma*sigma
            pvar = a*a*real_variance + sigma*sigma
            clean = fake_mean + fake_variance**.5 * clean_nodes
            z = a*clean + sigma*noise_nodes
            q = gaussian_posterior(z, sigma, fake_mean, fake_variance)
            p = gaussian_posterior(z, sigma, real_mean, real_variance)
            raw_gradient = a*a/sigma**2*(q-p)
            mean_gradient = float((joint_weights*raw_gradient).sum())
            log_std_gradient = float((joint_weights*raw_gradient*fake_variance**.5*clean_nodes).sum())
            expected_mean = a*a*(fake_mean-real_mean)/pvar
            expected_log_std = a*a*fake_variance*(1/pvar-1/qvar)
            assert abs(mean_gradient-expected_mean) < 1e-9
            assert abs(log_std_gradient-expected_log_std) < 1e-9
            covariance_cases.append(dict(sigma=sigma, fake_variance=fake_variance,
                real_variance=real_variance, mean_gradient=mean_gradient,
                log_std_gradient=log_std_gradient, expected_log_std_gradient=expected_log_std))
    return dict(passed=True, cases=rows, covariance_cases=covariance_cases, oracle_student_initial_mean=2.0,
                oracle_student_final_mean=history[-1], oracle_teacher_mean=real_mean,
                scope='Score conversion, raw Gaussian KL gradient, normalized surrogate direction, detachment and oracle learning; not pretrained convergence.')


@torch.no_grad()
def native_case(adapter, model, record, output, label, guidance, steps=4, seed=42):
    """Capture the actual native call, replay wrapper forwards, then compare rollouts."""
    adapter.keep_encoders_on_device = True
    sample = adapter.prepare(record)
    h, w = record['height'], record['width']
    factor = adapter.pipe.vae_scale_factor * 2
    noise_grid = torch.randn((1, model.config.in_channels, h // factor, w // factor),
                            device=adapter.device, dtype=adapter.dtype,
                            generator=torch.Generator(device=adapter.device).manual_seed(seed))
    packed_noise = adapter.pipe._pack_latents(noise_grid)
    native_calls, native_states = [], []
    replaying = False
    def capture(module, args, kwargs, result):
        nonlocal replaying
        if replaying:
            return
        position = len(native_calls)
        negative = guidance > 1 and position % 2 == 1
        condition = dict(sample['condition'])
        if negative:
            condition['prompt_embeds'] = condition['negative_prompt_embeds']
            condition['text_ids'] = condition['negative_text_ids']
        state = kwargs['hidden_states'][:, :packed_noise.shape[1]]
        replaying = True
        try:
            replay = adapter.predict(model, state, kwargs['timestep'], condition)
        finally:
            replaying = False
        expected = result[0][:, :state.shape[1]]
        reference = sample['condition']['reference_latents']
        native_calls.append(dict(negative=negative, sigma=float(kwargs['timestep'][0]),
            velocity_error=tensor_error(replay, expected),
            prompt_error=tensor_error(condition['prompt_embeds'], kwargs['encoder_hidden_states']),
            text_ids_equal=torch.equal(condition['text_ids'], kwargs['txt_ids']),
            reference_error=tensor_error(reference, kwargs['hidden_states'][:, state.shape[1]:]) if reference is not None else None,
            velocity=expected.detach().clone(), state=state.detach().clone()))
    hook = model.register_forward_hook(capture, with_kwargs=True)
    adapter.pipe.transformer = model
    adapter.pipe.to(adapter.device)
    adapter.pipe.set_progress_bar_config(disable=True)
    references = []
    for path in record['reference_images']:
        with Image.open(path) as image:
            references.append(image.convert('RGB'))
    def callback(pipe, index, timestep, values):
        native_states.append(values['latents'].detach().clone())
        return values
    try:
        native_image = adapter.pipe(prompt=record['prompt'], image=references or None,
            height=h, width=w, guidance_scale=guidance, num_inference_steps=steps,
            latents=noise_grid.clone(), generator=torch.Generator(device=adapter.device).manual_seed(seed),
            callback_on_step_end=callback).images[0]
    finally:
        hook.remove()
    sigmas = adapter.schedule(sample, steps)
    native_sigmas = adapter.pipe.scheduler.sigmas.to(adapter.device).float()
    schedule_error = tensor_error(sigmas, native_sigmas)
    # Replay Euler using actual native predictions and timestep inputs to separate numeric
    # carry precision from prompt/packing/forward correctness.
    stride = 2 if guidance > 1 else 1
    transition_errors = []
    for index, native_state in enumerate(native_states):
        call = native_calls[index * stride]
        velocity = call['velocity']
        if stride == 2:
            uncond = native_calls[index * stride + 1]['velocity']
            velocity = uncond + guidance * (velocity - uncond)
        expected = call['state'].float() + (native_sigmas[index + 1] - native_sigmas[index]) * velocity
        transition_errors.append(tensor_error(expected.to(velocity.dtype), native_state))
    native_decoded = adapter.decode(native_states[-1], sample)
    decode_difference = np.asarray(native_decoded).astype(float) - np.asarray(native_image).astype(float)
    # An independent full custom rollout uses exact same initial tensor, not just a seed.
    state = packed_noise.float()
    for index in range(steps):
        velocity = adapter.predict(model, state, sigmas[index], sample['condition'], guidance)
        state = state + (sigmas[index + 1] - sigmas[index]) * velocity.float()
    custom_image = adapter.decode(state, sample)
    native_image.save(output / f'{label}-native.png')
    custom_image.save(output / f'{label}-custom.png')
    rollout_error = tensor_error(state, native_states[-1])
    checks = [call['velocity_error']['relative_rms'] < 1e-5 and call['prompt_error']['max_abs'] == 0
              and call['text_ids_equal'] and (call['reference_error'] is None or call['reference_error']['max_abs'] == 0)
              for call in native_calls]
    strict_rollout = adapter.dtype == torch.float32
    passed = all(checks) and schedule_error['max_abs'] < 1e-6 and all(x['max_abs'] < 1e-5 for x in transition_errors)
    passed = passed and float(np.abs(decode_difference).max()) <= 1
    if strict_rollout:
        passed = passed and rollout_error['relative_rms'] < 1e-4
    # Repeat with the native carry/timestep policy. This must agree even in BF16;
    # otherwise a loose FP32-only gate would conceal a model-integration defect.
    state_native_policy = packed_noise.clone()
    for index in range(steps):
        native_timestep = adapter.pipe.scheduler.timesteps[index].expand(1).to(state_native_policy.dtype) / 1000
        velocity = adapter.predict(model, state_native_policy, native_timestep, sample['condition'], guidance)
        state_native_policy = (state_native_policy.float() + (sigmas[index + 1] - sigmas[index]) * velocity).to(velocity.dtype)
    native_policy_error = tensor_error(state_native_policy, native_states[-1])
    passed = passed and native_policy_error['relative_rms'] < 1e-4
    for call in native_calls:
        del call['velocity'], call['state']
    return dict(label=label, passed=passed, dtype=str(adapter.dtype), task=record['task'],
                references=len(references), guidance=guidance, steps=steps, schedule_error=schedule_error,
                forward_calls=native_calls, transition_errors=transition_errors,
                decode_max_pixel_error=float(np.abs(decode_difference).max()), rollout_error=rollout_error,
                native_policy_error=native_policy_error,
                strict_rollout_gate=strict_rollout,
                note='BF16 full-rollout discrepancy reported separately: native rounds carry state and timestep; custom keeps Euler states FP32.')


def validate_native(args, output):
    from main.flow.adapters import FlowAdapter
    from main.flow.data import read_manifest
    records = read_manifest(args.manifest, task='i2i', height=512, width=512)
    results = []
    for dtype in (torch.float32, torch.bfloat16):
        adapter = FlowAdapter('flux_klein', args.model_id, args.revision, 'cuda', dtype=dtype, teacher_guidance=4)
        model = adapter.load_transformer().requires_grad_(False).eval()
        cases = [(dict(records[0]), 'i2i', 1), (dict(records[0]), 'i2i-cfg', 4)]
        if dtype == torch.float32:
            t2i = dict(records[0], task='t2i', prompt='A red bicycle parked next to a brick wall', reference_images=[])
            multi = dict(records[0], reference_images=[records[0]['reference_images'][0], records[1]['reference_images'][0]])
            cases += [(t2i, 't2i', 1), (t2i, 't2i-cfg', 4), (multi, 'two-references', 4)]
        for record, label, guidance in cases:
            label = ('fp32-' if dtype == torch.float32 else 'bf16-') + label
            result = native_case(adapter, model, record, output, label, guidance)
            results.append(result)
            (output / 'native.json').write_text(json.dumps(dict(passed=all(x['passed'] for x in results), complete=False, cases=results), indent=2) + '\n')
            print(label, 'PASS' if result['passed'] else 'FAIL', 'rollout relative RMS', result['rollout_error']['relative_rms'], flush=True)
        adapter.pipe.transformer = None
        del model, adapter
        gc.collect()
        torch.cuda.empty_cache()
    return dict(passed=all(x['passed'] for x in results), complete=True, cases=results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['math', 'native'], required=True)
    parser.add_argument('--output_dir', default='outputs/flux-validation')
    parser.add_argument('--manifest', default='data/flux-klein-i2i-hqedit/validation/manifest.jsonl')
    parser.add_argument('--model_id', default='black-forest-labs/FLUX.2-klein-base-4B')
    parser.add_argument('--revision', default='a3b4f4849157f664bdbc776fd7453c2783562f4d')
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = validate_math() if args.stage == 'math' else validate_native(args, output)
    (output / f'{args.stage}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(args.stage, 'PASS' if result['passed'] else 'FAIL', flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
