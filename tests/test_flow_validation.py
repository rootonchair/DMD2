import numpy as np
from PIL import Image
import torch

from main.flow.validation import artifact_reasons, evaluate_student, image_statistics
from main.validate_flow import validate_math
from main.train_flow import parse_args, resume_config_matches
from main.flow.losses import rollout_prediction, rollout_update


def test_independent_gaussian_score_and_learning_gate():
    result = validate_math()
    assert result['passed'] and len(result['cases']) == 5
    assert abs(result['oracle_student_final_mean'] - result['oracle_teacher_mean']) < .05


def test_artifact_gate_rejects_new_collapse_but_preserves_baseline_colors():
    generator = np.random.default_rng(42)
    baseline = image_statistics(Image.fromarray(generator.integers(40, 230, (64, 64, 3), dtype=np.uint8)))
    cyan = image_statistics(Image.new('RGB', (64, 64), (0, 255, 255)))
    black = image_statistics(Image.new('RGB', (64, 64), (0, 0, 0)))
    green = image_statistics(Image.new('RGB', (64, 64), (0, 255, 100)))
    assert 'large_new_cyan_region' in artifact_reasons(cyan, baseline)
    assert 'large_luminance_collapse' in artifact_reasons(black, baseline)
    assert 'large_new_flat_color_region' in artifact_reasons(green, baseline)
    assert artifact_reasons(cyan, cyan) == []
    assert artifact_reasons(baseline, baseline) == []


def test_validation_writes_comparisons_and_restores_model_mode(tmp_path, monkeypatch):
    source = tmp_path / 'source.png'
    Image.new('RGB', (64, 64), (80, 90, 100)).save(source)
    generator = np.random.default_rng(42)
    base = Image.fromarray(generator.integers(40, 230, (512, 512, 3), dtype=np.uint8))
    class Adapter:
        device = torch.device('cpu')
        output = base
        offloaded = False
        def decode(self, latent, sample):
            return self.output
        def offload_encoders(self):
            self.offloaded = True
    adapter = Adapter()
    model = torch.nn.Linear(1, 1).train()
    record = dict(prompt='Edit the scene', reference_images=[str(source)])
    monkeypatch.setattr('main.flow.losses.sample_student', lambda *a, **kw: torch.zeros(1, 4, 8))
    initial = evaluate_student(adapter, model, [{}], [record], tmp_path / 'base', [42, 43])
    assert initial['artifact_gate_passed'] and initial['samples'] == 2
    assert model.training and adapter.offloaded
    adapter.output = Image.new('RGB', (512, 512), (0, 255, 255))
    later = evaluate_student(adapter, model, [{}], [record], tmp_path / 'later', [42, 43], initial['results'])
    assert not later['artifact_gate_passed']
    assert (tmp_path / 'later/000-seed42-comparison.png').is_file()
    with Image.open(tmp_path / 'later/000-seed42-comparison.png') as image:
        assert image.width == 1536 and image.height > 544
        assert image.info['Prompt'] == record['prompt']
    assert model.training


def test_legacy_resume_defaults_to_zero_fake_warmup():
    args = vars(parse_args(['--model_family', 'flux_klein', '--manifest', 'inputs.jsonl', '--output_dir', 'run']))
    old = dict(args)
    del old['fake_warmup_iters']
    del old['sampling_precision']
    del old['fake_lora_rank'], old['fake_lora_alpha']
    del old['rollout_mode'], old['pilot_fixed_noise']
    del old['warmup_step']
    del old['score_noise_sampling'], old['score_noise_shift'], old['student_ema_decay']
    resume_config_matches(old, args)
    changed = dict(args, fake_warmup_iters=100)
    import pytest
    with pytest.raises(ValueError, match='fake_warmup_iters'):
        resume_config_matches(old, changed)
    with pytest.raises(ValueError, match='warmup_step'):
        resume_config_matches(old, dict(args, warmup_step=500))
    for field, value in [('score_noise_sampling', 'shifted'), ('score_noise_shift', 5),
                         ('student_ema_decay', .98)]:
        with pytest.raises(ValueError, match=field):
            resume_config_matches(old, dict(args, **{field: value}))


def test_fastvideo_score_sampling_matches_shifted_integer_timesteps():
    from types import SimpleNamespace
    from main.flow.losses import sample_score_sigma
    adapter = SimpleNamespace(score_noise_sampling='shifted', score_noise_shift=8,
        pipe=SimpleNamespace(scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))))
    actual = sample_score_sigma(adapter, 100000, 'cpu', torch.Generator().manual_seed(9))
    u = torch.randint(1000, (100000,), generator=torch.Generator().manual_seed(9)).float() / 1000
    assert torch.equal(actual, 8*u/(1+7*u))
    assert actual.min() == 0 and actual.max() < 1
    assert .29 < (actual > .95).float().mean() < .31
    legacy = sample_score_sigma(SimpleNamespace(), 100000, 'cpu', torch.Generator().manual_seed(9))
    assert torch.equal(legacy, torch.rand(100000, generator=torch.Generator().manual_seed(9))*.96+.02)


def test_student_ema_updates_on_critic_only_iterations_and_restores_on_exception():
    import pytest
    from main.flow.ema import StudentEMA
    model = torch.nn.Linear(1, 1)
    model.bias.requires_grad_(False)
    ema = StudentEMA(model, .98)
    with torch.no_grad():
        model.weight.fill_(2)
    ema.update(model)  # FastVideo initializes its active shadow at the first step end.
    with torch.no_grad():
        model.weight.fill_(10)
    ema.update(model)
    ema.update(model)  # Critic-only iteration: live student did not change.
    expected = torch.full_like(model.weight, 2*.98*.98+10*(1-.98*.98))
    assert torch.allclose(ema.shadow['weight'], expected) and ema.num_updates == 3
    assert 'bias' not in ema.shadow
    with pytest.raises(RuntimeError, match='restore'):
        with ema.apply(model):
            assert torch.allclose(model.weight, expected)
            raise RuntimeError('restore')
    assert model.weight.item() == 10
    reloaded = StudentEMA(model, .98)
    reloaded.load_state_dict(ema.state_dict())
    assert reloaded.num_updates == 3 and torch.equal(reloaded.shadow['weight'], ema.shadow['weight'])


def test_native_rollout_matches_native_bf16_timestep_and_carry_policy():
    class Adapter:
        sampling_precision = 'native'
        def predict(self, student, state, sigma, condition):
            self.sigma = sigma
            return torch.full_like(state, .3)
    adapter = Adapter()
    state = torch.ones(1, 2, 3, dtype=torch.bfloat16)
    sigma, next_sigma = torch.tensor(.958085358), torch.tensor(.88398188)
    velocity = rollout_prediction(adapter, None, state, sigma, {})
    assert adapter.sigma == (sigma * 1000).to(torch.bfloat16) / 1000
    actual = rollout_update(adapter, state, velocity, sigma, next_sigma)
    expected = (state.float() + (next_sigma - sigma) * velocity).to(torch.bfloat16)
    assert actual.dtype == torch.bfloat16 and torch.equal(actual, expected)


def test_raw_flow_kl_weighting_and_normalization_counterexample():
    from main.validate_flow_weighting import validate_weighting
    result = validate_weighting()
    assert result['raw_gradient_checks_passed']
    assert not result['normalized_general_kl_descent_guarantee']


def test_raw_flow_kl_rejects_undefined_endpoints():
    from main.flow_kl_diagnostic import flow_kl_gradient
    import pytest
    for sigma in [0, 1, -1]:
        with pytest.raises(ValueError, match='strictly between'):
            flow_kl_gradient(torch.zeros(1, 2, 3), torch.ones(1, 2, 3), sigma)


def test_stochastic_update_uses_fresh_forward_noise():
    from types import SimpleNamespace
    from main.flow.losses import interpolate
    adapter = SimpleNamespace(rollout_mode='stochastic', sampling_precision='fp32')
    clean = torch.randn(1, 20, 8, generator=torch.Generator().manual_seed(10))
    original_noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(11))
    noisy = interpolate(clean, original_noise, .8)
    velocity = original_noise-clean
    generator = torch.Generator().manual_seed(12)
    updated = rollout_update(adapter, noisy, velocity, torch.tensor(.8), torch.tensor(.4), generator)
    fresh = torch.randn(clean.shape, generator=torch.Generator().manual_seed(12))
    assert torch.allclose(updated, .6*clean+.4*fresh, atol=1e-6)
    assert not torch.allclose(updated, .6*clean+.4*original_noise)
    assert torch.allclose(rollout_update(adapter, noisy, velocity, torch.tensor(.8), torch.tensor(0.)), clean, atol=1e-6)


def test_stochastic_training_prefix_matches_inference_with_explicit_selection():
    from main.flow.losses import sample_student, student_clean_sample, reconstruct_clean
    class Adapter:
        rollout_mode = 'stochastic'
        sampling_precision = 'fp32'
        states = []
        velocities = []
        def schedule(self, sample, steps):
            return torch.tensor([1, .75, .5, .25, 0])
        def predict(self, student, state, sigma, condition):
            self.states.append(state.clone())
            velocity = state*.1+sigma*.2
            self.velocities.append(velocity.clone())
            return velocity
    model = torch.nn.Linear(1, 1)
    sample = dict(shape=[1, 4, 8], condition={})
    adapter = Adapter()
    sample_student(adapter, model, sample, 4, torch.Generator().manual_seed(73))
    states, velocities = list(adapter.states), list(adapter.velocities)
    adapter.states, adapter.velocities = [], []
    clean = student_clean_sample(adapter, model, sample, 4, torch.Generator().manual_seed(73), selected_step=2)
    assert all(torch.equal(actual, expected) for actual, expected in zip(adapter.states, states))
    assert torch.equal(clean, reconstruct_clean(states[2], velocities[2], .5))


def test_controlled_scoring_noise_is_independent_of_extra_rollout_draws():
    from main.train_flow import pilot_noise_generators
    a = pilot_noise_generators(44, 12, 3, 1, 'cpu')
    b = pilot_noise_generators(44, 12, 3, 1, 'cpu')
    torch.randn(1000, generator=a['rollout'])
    assert torch.equal(torch.randn(10, generator=a['dm']), torch.randn(10, generator=b['dm']))
    assert torch.equal(torch.randn(10, generator=a['fake']), torch.randn(10, generator=b['fake']))
    assert torch.equal(torch.randint(4, (10,), generator=a['step']), torch.randint(4, (10,), generator=b['step']))


def test_immutable_revision_uses_no_metadata_network_request(monkeypatch):
    from main.flow.adapters import resolve_revision
    def forbidden(*args, **kwargs):
        raise AssertionError('Immutable revision must not query model metadata')
    monkeypatch.setattr('huggingface_hub.model_info', forbidden)
    assert resolve_revision('black-forest-labs/FLUX.2-klein-base-4B', 'A'*40) == 'a'*40
