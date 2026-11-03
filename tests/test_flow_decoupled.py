"""Decoupled DMD guidance, saturation diagnostics and launcher ablation overrides."""

import copy
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from main.flow.checkpoints import configure_trainable
from main.flow.losses import (cfg_term_sigma, distillation_loss, interpolate, reconstruct_clean,
                              sample_score_sigma, student_clean_sample)
from main.flow.validation import color_drift, image_statistics
from main.train_flow import parse_args, resume_config_matches
from scripts.run_qwen_image21_t2i import apply_overrides, training_command
from tests.test_flow import synthetic_adapter, synthetic_sample, tiny_model


def models(family):
    base = tiny_model(family)
    adapter = synthetic_adapter(family, base)
    teacher = copy.deepcopy(base).requires_grad_(False)
    student = configure_trainable(copy.deepcopy(base), 'lora', rank=2, alpha=2)
    fake = configure_trainable(copy.deepcopy(base), 'lora', rank=2, alpha=2)
    with torch.no_grad():
        for parameter in fake.parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter) * 0.01)
    return adapter, teacher, student, fake


def generators(seed):
    return torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed + 1)


@pytest.mark.parametrize('family', ['flux_klein', 'qwen_image21'])
def test_predict_branches_compose_to_guided_prediction(family):
    model = tiny_model(family)
    adapter = synthetic_adapter(family, model)
    sample = synthetic_sample(family)
    state = torch.randn(sample['shape'])
    cond, uncond = adapter.predict(model, state, .6, sample['condition'], return_branches=True)
    assert torch.allclose(cond, adapter.predict(model, state, .6, sample['condition']))
    guided = adapter.predict(model, state, .6, sample['condition'], 2.5)
    assert torch.allclose(uncond + 2.5 * (cond - uncond), guided, atol=1e-6)


@pytest.mark.parametrize('family', ['flux_klein', 'qwen_image21'])
def test_unguided_decoupled_mode_equals_combined(family):
    torch.manual_seed(0)
    adapter, teacher, student, fake = models(family)
    sample = synthetic_sample(family)
    results = {}
    for mode in ('combined', 'decoupled'):
        adapter.dmd_guidance_mode = mode
        rollout, scoring = generators(7)
        results[mode] = distillation_loss(adapter, student, teacher, fake, sample, 4, 1.0, rollout, scoring)
    assert torch.allclose(results['combined'][2], results['decoupled'][2], atol=1e-6)
    assert torch.allclose(results['combined'][0], results['decoupled'][0], atol=1e-6)


@pytest.mark.parametrize('family', ['flux_klein', 'qwen_image21'])
def test_decoupled_gradient_matches_manual_decomposition(family):
    torch.manual_seed(0)
    adapter, teacher, student, fake = models(family)
    adapter.dmd_guidance_mode, adapter.cfg_term_weight = 'decoupled', .7
    adapter.cfg_noise_min, adapter.cfg_noise_max = .5, .9
    sample, w = synthetic_sample(family), 2.5
    condition = sample['condition']
    rollout, scoring = generators(11)
    loss, clean, gradient = distillation_loss(adapter, student, teacher, fake, sample, 4, w, rollout, scoring, 2)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters() if p.requires_grad)
    assert all(p.grad is None for p in fake.parameters()) and all(p.grad is None for p in teacher.parameters())

    rollout, scoring = generators(11)
    with torch.no_grad():
        expected_clean = student_clean_sample(adapter, student, sample, 4, rollout, 2)
        sigma = sample_score_sigma(adapter, 1, 'cpu', scoring)
        noisy = interpolate(expected_clean, torch.randn(expected_clean.shape, generator=scoring), sigma)
        cond = reconstruct_clean(noisy, adapter.predict(teacher, noisy, sigma, condition), sigma)
        fake_clean = reconstruct_clean(noisy, adapter.predict(fake, noisy, sigma, condition), sigma)
        cfg_sigma = cfg_term_sigma(adapter, 1, 'cpu', scoring)
        assert .5 <= float(cfg_sigma) <= .9
        cfg_noisy = interpolate(expected_clean, torch.randn(expected_clean.shape, generator=scoring), cfg_sigma)
        cfg_cond = reconstruct_clean(cfg_noisy, adapter.predict(teacher, cfg_noisy, cfg_sigma, condition), cfg_sigma)
        guided = reconstruct_clean(cfg_noisy, adapter.predict(teacher, cfg_noisy, cfg_sigma, condition, w), cfg_sigma)
        cfg_uncond = (w * cfg_cond - guided) / (w - 1)
    dm = (fake_clean - cond) / (expected_clean - cond).abs().mean()
    ca = -(w - 1) * (cfg_cond - cfg_uncond) / (expected_clean - guided).abs().mean()
    assert torch.allclose(clean, expected_clean)
    assert torch.allclose(gradient, dm + .7 * ca, atol=1e-4)

    adapter.cfg_term_weight = 0
    rollout, scoring = generators(11)
    _, _, matching_only = distillation_loss(adapter, student, teacher, fake, sample, 4, w, rollout, scoring, 2)
    assert torch.allclose(matching_only, dm, atol=1e-5)


def test_cfg_sigma_range_maps_score_sampler():
    adapter = type('Adapter', (), dict(score_noise_sampling='uniform', cfg_noise_min=.6, cfg_noise_max=.8))()
    sigma = cfg_term_sigma(adapter, 4096, 'cpu', torch.Generator().manual_seed(0))
    assert sigma.min() >= .6 and sigma.max() <= .8 and sigma.std() > .03


def test_decoupled_flags_validate_and_guard_resume(tmp_path):
    base = ['--model_family', 'qwen_image21', '--manifest', 'm.jsonl', '--output_dir', str(tmp_path)]
    args = parse_args(base + ['--dmd_guidance_mode', 'decoupled', '--cfg_term_weight', '.5',
                              '--cfg_noise_min', '.3', '--cfg_noise_max', '.9'])
    assert (args.cfg_term_weight, args.cfg_noise_min, args.cfg_noise_max) == (.5, .3, .9)
    for bad in (['--cfg_term_weight', '.5'], ['--dmd_guidance_mode', 'decoupled', '--cfg_noise_min', '.9', '--cfg_noise_max', '.3'],
                ['--dmd_guidance_mode', 'decoupled', '--cfg_term_weight', '-1']):
        with pytest.raises(SystemExit):
            parse_args(base + bad)
    legacy = {key: value for key, value in vars(parse_args(base)).items()
              if key not in ('dmd_guidance_mode', 'cfg_term_weight', 'cfg_noise_min', 'cfg_noise_max')}
    resume_config_matches(legacy, vars(parse_args(base)))
    with pytest.raises(ValueError, match='dmd_guidance_mode'):
        resume_config_matches(legacy, vars(args))


def test_saturation_statistics_and_teacher_drift():
    gray = image_statistics(Image.new('RGB', (32, 32), (128, 128, 128)))
    red = image_statistics(Image.new('RGB', (32, 32), (255, 0, 0)))
    assert gray['saturation_p95'] == 0 and gray['clipped_fraction'] == 0
    assert red['saturation_p95'] == pytest.approx(1) and red['clipped_fraction'] == 1
    rows = [dict(statistics=red, teacher_statistics=gray), dict(statistics=gray, teacher_statistics=None)]
    drift = color_drift(rows)
    assert drift['saturation_p95'] == pytest.approx(1) and drift['clipped_fraction'] == 1
    assert color_drift([dict(statistics=red, teacher_statistics=None)]) is None


def test_launcher_overrides_replace_or_append_flags(tmp_path):
    default = training_command(Path('/repo'), tmp_path, Path('/repo/data/repo-t2i'))
    command = training_command(Path('/repo'), tmp_path, Path('/repo/data/repo-t2i'),
        ['--fake_lora_alpha=64', '--train_iters=2000', '--dmd_guidance_mode=decoupled'])
    value = lambda flag: command[command.index(flag) + 1]
    assert value('--fake_lora_alpha') == '64' and value('--train_iters') == '2000'
    assert value('--dmd_guidance_mode') == 'decoupled' and len(command) == len(default) + 2
    betas = apply_overrides(default, ['--adam_betas=0 0.999'])
    assert betas[-3:] == ['--adam_betas', '0', '0.999']
    for bad in (['--output_dir=/elsewhere'], ['fake_lr=1'], ['--fake_lr']):
        with pytest.raises(ValueError):
            apply_overrides(default, bad)


def test_adam_betas_flag_validates_and_guards_resume(tmp_path):
    base = ['--model_family', 'qwen_image21', '--manifest', 'm.jsonl', '--output_dir', str(tmp_path)]
    default, zero = vars(parse_args(base)), vars(parse_args(base + ['--adam_betas', '0', '0.999']))
    assert default['adam_betas'] == [0.9, 0.999] and zero['adam_betas'] == [0.0, 0.999]
    with pytest.raises(SystemExit):
        parse_args(base + ['--adam_betas', '1', '0.999'])
    legacy = {key: value for key, value in default.items() if key != 'adam_betas'}
    resume_config_matches(legacy, default)
    with pytest.raises(ValueError, match='adam_betas'):
        resume_config_matches(legacy, zero)
