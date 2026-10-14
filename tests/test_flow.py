"""Small native-transformer checks; no pretrained weights or network needed."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel, QwenImage21Transformer2DModel
from PIL import Image

from main.flow.adapters import FlowAdapter
from main.flow.checkpoints import (configure_trainable, export_model, load_export, read_export_config,
                                  prune_checkpoints, register_compact_lora_hooks)
from main.flow.data import cache_key, prepare_cache, read_manifest
from main.flow.losses import (distillation_loss, distribution_matching_loss, fake_score_loss,
                              interpolate, reconstruct_clean, sample_student)
from main.train_flow import batch_indices, main as train_main, parse_args, resume_config_matches


def tiny_model(family):
    if family == 'flux_klein':
        return Flux2Transformer2DModel(in_channels=8, num_layers=1, num_single_layers=1,
            num_attention_heads=2, attention_head_dim=16, joint_attention_dim=32,
            timestep_guidance_channels=16, axes_dims_rope=(4, 4, 4, 4), guidance_embeds=False)
    return QwenImage21Transformer2DModel(in_channels=4, out_channels=4, num_layers=1,
        num_attention_heads=2, attention_head_dim=16, context_in_dim=32, axes_dims_rope=(4, 6, 6))


def synthetic_sample(family, references=0):
    channels = 8 if family == 'flux_klein' else 4
    condition = dict(reference_latents=torch.randn(1, references * 4, channels) if references else None)
    if family == 'flux_klein':
        ids = torch.tensor([[[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0]]])
        condition.update(prompt_embeds=torch.randn(1, 3, 32), text_ids=torch.zeros(1, 3, 4, dtype=torch.long),
            negative_prompt_embeds=torch.randn(1, 3, 32), negative_text_ids=torch.zeros(1, 3, 4, dtype=torch.long),
            latent_ids=ids, reference_ids=torch.cat([ids + torch.tensor([10 * (i + 1), 0, 0, 0]) for i in range(references)], dim=1) if references else None)
    else:
        embeds = torch.randn(1, 3 + references, 32)
        mask = torch.tensor([[False] * 3 + [True] * references + [True]])
        condition.update(prompt_embeds=embeds, prompt_mask=None, image_mask=mask,
            negative_prompt_embeds=torch.randn_like(embeds), negative_prompt_mask=None, negative_image_mask=mask,
            img_shapes=[[(1, 2, 2)] * (references + 1)])
    return dict(shape=[1, 4, channels], height=32, width=32, task='i2i' if references else 't2i', condition=condition)


def synthetic_adapter(family, model, device='cpu'):
    adapter = FlowAdapter.__new__(FlowAdapter)
    adapter.family = family
    adapter.device = torch.device(device)
    adapter.dtype = torch.float32
    adapter.model_id = 'synthetic-base'
    adapter.revision = 'synthetic-revision'
    adapter.teacher_guidance = 1.0
    adapter.identity = dict(family=family, model_id=adapter.model_id, revision=adapter.revision, diffusers_version='synthetic')
    adapter.pipe = SimpleNamespace(scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True))
    adapter.transformer_class = type(model)
    adapter.load_transformer = lambda: copy.deepcopy(model).to(device)
    return adapter


def test_flow_reconstruction_and_gradient_sign():
    clean, noise = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
    for sigma in (0, 0.5, 1, torch.tensor([0.2, 0.9])):
        noisy = interpolate(clean, noise, sigma)
        assert torch.allclose(reconstruct_clean(noisy, noise - clean, sigma), clean, atol=1e-6)
    student = torch.zeros(1, 2, 3, requires_grad=True)
    teacher = torch.ones_like(student, requires_grad=True)
    fake = torch.zeros_like(student, requires_grad=True)
    loss, gradient = distribution_matching_loss(student, teacher, fake)
    loss.backward()
    assert (student.grad < 0).all()  # Gradient descent moves the student toward the teacher.
    assert teacher.grad is None and fake.grad is None
    assert torch.isfinite(gradient).all()
    loss, gradient = distribution_matching_loss(student.detach(), student.detach(), student.detach())
    assert loss == 0 and torch.isfinite(gradient).all()


def test_fake_target_detached():
    student = torch.randn(1, 4, 8, requires_grad=True)
    prediction = torch.randn_like(student, requires_grad=True)
    fake_score_loss(prediction, student, torch.randn_like(student)).backward()
    assert student.grad is None and prediction.grad is not None


@pytest.mark.parametrize('family', ['flux_klein', 'qwen_image21'])
@pytest.mark.parametrize('references', [0, 1, 2])
def test_native_forward_and_output_slicing(family, references):
    model = tiny_model(family)
    adapter = synthetic_adapter(family, model)
    sample = synthetic_sample(family, references)
    state = torch.randn(sample['shape'], requires_grad=True)
    predicted = adapter.predict(model, state, 0.5, sample['condition'])
    assert predicted.shape == state.shape and torch.isfinite(predicted).all()
    predicted.square().mean().backward()
    assert state.grad is not None and state.grad.abs().sum() > 0
    # Compare explicitly to the native transformer call, including reference ordering/masks.
    c = sample['condition']
    if family == 'flux_klein':
        hidden = torch.cat([state, c['reference_latents']], dim=1) if references else state
        ids = torch.cat([c['latent_ids'], c['reference_ids']], dim=1) if references else c['latent_ids']
        expected = model(hidden_states=hidden, timestep=torch.tensor([0.5]), guidance=None,
            encoder_hidden_states=c['prompt_embeds'], img_ids=ids, txt_ids=c['text_ids'], return_dict=False)[0][:, :4]
    else:
        hidden = torch.cat([c['reference_latents'], state], dim=1) if references else state
        expected = model(hidden_states=hidden, timestep=torch.tensor([0.5]), encoder_hidden_states=c['prompt_embeds'],
            encoder_hidden_states_mask=c['prompt_mask'], img_shapes=c['img_shapes'], img_mask=c['image_mask'],
            kv_cache=None, kv_cache_mode=None, return_dict=False)[0][:, -4:]
    assert torch.allclose(predicted, expected)
    guided = adapter.predict(model, state, 0.5, c, 4.0)
    assert torch.isfinite(guided).all()


@pytest.mark.parametrize('family', ['flux_klein', 'qwen_image21'])
@pytest.mark.parametrize('mode', ['lora', 'full'])
@pytest.mark.parametrize('steps', [1, 4])
def test_distillation_gradients_and_export(family, mode, steps, tmp_path):
    base = tiny_model(family)
    adapter = synthetic_adapter(family, base)
    teacher = copy.deepcopy(base).requires_grad_(False)
    original = {name: p.clone() for name, p in teacher.named_parameters()}
    student = configure_trainable(copy.deepcopy(base), mode, rank=2, alpha=2)
    fake = configure_trainable(copy.deepcopy(base), mode, rank=2, alpha=2)
    # Diverge critic estimates so the generator's initial matching gradient is nonzero.
    with torch.no_grad():
        for parameter in fake.parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter) * 0.01)
    sample = synthetic_sample(family, 2)
    loss, clean, gradient = distillation_loss(adapter, student, teacher, fake, sample, steps, 1.0)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters() if p.requires_grad)
    assert all(p.grad is None for p in fake.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    for name, parameter in teacher.named_parameters():
        assert torch.equal(parameter, original[name])
    noise = torch.randn_like(clean)
    fake_score_loss(adapter.predict(fake, interpolate(clean, noise, 0.5), 0.5, sample['condition']), clean, noise).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fake.parameters() if p.requires_grad)
    metadata = dict(adapter.metadata(), num_steps=steps, task='mixed', height=32, width=32)
    export_model(student, tmp_path / 'student', metadata, mode)
    config = read_export_config(tmp_path / 'student')
    restored = load_export(adapter, tmp_path / 'student', config)
    student.eval()
    generator = torch.Generator().manual_seed(42)
    output = sample_student(adapter, student, sample, steps, generator)
    restored_output = sample_student(adapter, restored, sample, steps, torch.Generator().manual_seed(42))
    assert torch.allclose(output, restored_output, atol=1e-5)
    assert config['num_steps'] == steps


def test_manifest_validation_cache_and_invalidation(tmp_path):
    reference = tmp_path / 'ref.png'
    reference.write_bytes(b'first image')
    manifest = tmp_path / 'input.jsonl'
    records = [dict(task='t2i', prompt='A mountain', reference_images=[]),
               dict(task='i2i', prompt='Change the sky', reference_images=['ref.png'])]
    manifest.write_text('\n'.join(map(json.dumps, records)))
    loaded = read_manifest(manifest)
    assert loaded[1]['reference_images'] == [str(reference)]
    assert len(read_manifest(manifest, task='t2i')) == 1
    key = cache_key(loaded[1], {'revision': 'a'})
    reference.write_bytes(b'second image')
    assert cache_key(loaded[1], {'revision': 'a'}) != key
    assert cache_key(loaded[1], {'revision': 'b'}) != key
    calls = []
    adapter = SimpleNamespace(identity={'revision': 'a'}, prepare=lambda record: calls.append(record) or {'shape': [1, 4, 8]})
    prepare_cache(adapter, loaded, tmp_path / 'cache')
    prepare_cache(adapter, loaded, tmp_path / 'cache')
    assert len(calls) == 2
    manifest.write_text(json.dumps(dict(task='t2i', prompt='bad', reference_images=['ref.png'])))
    with pytest.raises(ValueError, match='t2i requires no references'):
        read_manifest(manifest)


def test_batches_and_resume_configuration():
    whole = [batch_indices(13, iteration, micro, 2, 2, 2, rank, 42)
             for iteration in range(3) for micro in range(2) for rank in range(2)]
    flattened = sum(whole, [])
    assert len(set(flattened[:13])) == 13
    assert whole[8] == batch_indices(13, 2, 0, 2, 2, 2, 0, 42)
    args = vars(parse_args(['--model_family', 'flux_klein', '--manifest', 'unused.jsonl', '--output_dir', 'unused']))
    resume_config_matches(args, args)
    with pytest.raises(ValueError, match='num_steps'):
        resume_config_matches(args, dict(args, num_steps=1))


def native_preprocessing_adapter(family):
    """Native packing, VAE normalization, processors, and decoding with cheap encoders."""
    from diffusers import Flux2KleinPipeline, QwenImage21Pipeline
    from types import MethodType
    class TestVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()), requires_grad=False)
            self.config = SimpleNamespace(block_out_channels=[1] * 4, batch_norm_eps=1e-5,
                z_dim=4, latents_mean=[0.1] * 4, latents_std=[0.5] * 4)
            self.bn = torch.nn.BatchNorm2d(8).eval()
            self.seen_channels = []
        @property
        def dtype(self):
            return self.weight.dtype
        def encode(self, image):
            self.seen_channels.append(image.shape[1])
            if family == 'flux_klein':
                result = torch.nn.functional.avg_pool2d(image[:, :2], 8)
            else:
                result = torch.nn.functional.avg_pool2d(image[:, :, 0], 16).unsqueeze(2)
            return SimpleNamespace(latents=result)
        def decode(self, latent, return_dict=False):
            if family == 'flux_klein':
                latent = torch.cat([latent, latent[:, :1]], dim=1)
                result = torch.nn.functional.interpolate(latent, scale_factor=8)
            else:
                result = torch.nn.functional.interpolate(latent[:, :, 0], scale_factor=16).unsqueeze(2)
            return (result,)
    vae = TestVAE()
    encoder = torch.nn.Linear(1, 1).requires_grad_(False)
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    if family == 'flux_klein':
        pipe = Flux2KleinPipeline(scheduler=scheduler, vae=vae, text_encoder=encoder, tokenizer=None, transformer=None)
        def encode_prompt(self, prompt, device=None):
            embeds = torch.ones(1, 3, 32) * (0.1 if prompt else 0)
            return embeds, torch.zeros(1, 3, 4, dtype=torch.long)
    else:
        class TestProcessor:
            tokenizer = SimpleNamespace(encode=lambda *a: [3])
            def apply_chat_template(self, *args, **kwargs):
                return [[1, 2]]
        processor = TestProcessor()
        pipe = QwenImage21Pipeline(scheduler=scheduler, vae=vae, text_encoder=encoder, processor=processor, transformer=None)
        def encode_prompt(self, prompt, image=None, device=None):
            slots = sum((img.width // 16) * (img.height // 16) // 4 for img in image or [])
            embeds = torch.ones(1, 3 + slots, 32) * (0.1 if prompt else 0)
            mask = torch.tensor([[False] * 3 + [True] * slots])
            return embeds, None, mask
    pipe.encode_prompt = MethodType(encode_prompt, pipe)
    model = tiny_model(family)
    adapter = synthetic_adapter(family, model)
    adapter.pipe = pipe
    adapter.transformer_config = dict(model.config)
    adapter.teacher_guidance = 4.0
    return adapter, model


@pytest.mark.parametrize('family', ['flux_klein', 'qwen_image21'])
@pytest.mark.parametrize('references', [0, 1, 2])
def test_native_preprocessing_decode_and_inference_cli(family, references, tmp_path, monkeypatch):
    adapter, base = native_preprocessing_adapter(family)
    paths = []
    for index in range(references):
        path = tmp_path / f'ref{index}.png'
        Image.new('RGBA', (64, 64), (80, 100, 120, 96)).save(path)
        paths.append(str(path))
    record = dict(task='i2i' if references else 't2i', prompt='test', reference_images=paths, height=64, width=64)
    sample = adapter.prepare(record)
    assert sample['shape'] == [1, 16, 8 if family == 'flux_klein' else 4]
    assert adapter.pipe.vae.seen_channels == [3 if family == 'flux_klein' else 4] * references
    predicted = adapter.predict(base, torch.randn(sample['shape']), 0.5, sample['condition'], 4)
    assert predicted.shape == tuple(sample['shape'])
    image = adapter.decode(torch.zeros(sample['shape']), sample)
    assert image.size == (64, 64)
    assert image.mode == ('RGB' if family == 'flux_klein' else 'RGBA')
    metadata = dict(adapter.metadata(), num_steps=1, task='mixed', height=64, width=64)
    export_model(base, tmp_path / 'student', metadata, 'full')
    monkeypatch.setattr('demo.flow_image.FlowAdapter', lambda *a, **kw: adapter)
    from demo.flow_image import main as infer_main
    command = ['--checkpoint', str(tmp_path / 'student'), '--prompt', 'test', '--output', str(tmp_path / 'output.png'), '--device', 'cpu']
    if references:
        command += ['--reference_images', *paths]
    infer_main(command)
    with Image.open(tmp_path / 'output.png') as output:
        assert output.size == (64, 64) and output.mode == image.mode
    with pytest.raises(SystemExit):
        infer_main(command + ['--num_steps', '4'])


def test_saved_schedule_reused():
    adapter = synthetic_adapter('flux_klein', tiny_model('flux_klein'))
    sample = synthetic_sample('flux_klein')
    adapter.saved_schedules = {'32x32:4': [1.0, 0.9, 0.7, 0.3, 0.0]}
    assert adapter.schedule(sample, 4).tolist() == pytest.approx([1, 0.9, 0.7, 0.3, 0])


def test_hqedit_subset_is_bounded_and_disjoint(tmp_path):
    from main.prepare_hqedit import materialize_subset
    rows = [dict(edit='Change the sky', input_image=Image.new('RGB', (128, 128), (index, 20, 30))) for index in range(6)]
    rows.insert(1, rows[0])  # A duplicate image with the same instruction must be excluded.
    selection = materialize_subset(iter(rows), tmp_path, 3, 2, reference_max_side=64)
    train_hashes = {entry['source_sha256'] for entry in selection['train']}
    val_hashes = {entry['source_sha256'] for entry in selection['validation']}
    assert len(train_hashes) == 3 and len(val_hashes) == 2 and train_hashes.isdisjoint(val_hashes)
    assert all(entry['reference_size'] == [64, 64] for split in selection.values() for entry in split)
    assert len(list(tmp_path.rglob('*.png'))) == 5


def test_checkpoint_retention_only_prunes_completed_own_run(tmp_path):
    for index in (5, 500, 1000):
        directory = tmp_path / f'checkpoint-{index:06d}'
        directory.mkdir()
        (directory / 'checkpoint_complete.json').write_text(json.dumps(dict(run_id='own', completed_iterations=index)))
    foreign = tmp_path / 'checkpoint-000001'
    foreign.mkdir()
    (foreign / 'checkpoint_complete.json').write_text(json.dumps(dict(run_id='other', completed_iterations=1)))
    incomplete = tmp_path / 'checkpoint-000002'
    incomplete.mkdir()
    prune_checkpoints(tmp_path, 2, 'own')
    assert not (tmp_path / 'checkpoint-000005').exists()
    assert (tmp_path / 'checkpoint-000500').exists() and (tmp_path / 'checkpoint-001000').exists()
    assert foreign.exists() and incomplete.exists()


def test_compact_checkpoint_hooks_and_legacy_reload(tmp_path):
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState
    AcceleratorState._reset_state(reset_partial_state=True)
    try:
        accelerator = Accelerator(cpu=True)
        base = tiny_model('flux_klein')
        adapter = synthetic_adapter('flux_klein', base)
        models = [configure_trainable(copy.deepcopy(base), 'lora', rank=2, alpha=2) for _ in range(2)]
        student, fake = accelerator.prepare(*models)
        original = {name: parameter.detach().clone() for name, parameter in student.named_parameters()}
        legacy = tmp_path / 'legacy'
        accelerator.save_state(str(legacy))
        metadata = dict(adapter.metadata(), num_steps=4, task='mixed', height=32, width=32)
        register_compact_lora_hooks(accelerator, metadata)
        with torch.no_grad():
            for parameter in student.parameters():
                if parameter.requires_grad:
                    parameter.add_(1)
        accelerator.load_state(str(legacy))
        assert all(torch.equal(parameter, original[name]) for name, parameter in student.named_parameters())
        compact = tmp_path / 'compact'
        accelerator.save_state(str(compact))
        assert (compact / 'student/adapter.safetensors').exists()
        assert (compact / 'fake/adapter.safetensors').exists()
        assert not (compact / 'model.safetensors').exists()
        assert not (compact / 'model_1.safetensors').exists()
        with torch.no_grad():
            for parameter in student.parameters():
                if parameter.requires_grad:
                    parameter.add_(1)
        accelerator.load_state(str(compact))
        assert all(torch.equal(parameter, original[name]) for name, parameter in student.named_parameters())
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)


@pytest.mark.parametrize('family', ['flux_klein', 'qwen_image21'])
@pytest.mark.parametrize('steps', [1, 4])
def test_published_scheduler_settings(family, steps):
    adapter = synthetic_adapter(family, tiny_model(family))
    adapter.pipe.scheduler = FlowMatchEulerDiscreteScheduler(
        use_dynamic_shifting=True, shift_terminal=0.02 if family == 'qwen_image21' else None)
    sigmas = adapter.schedule(synthetic_sample(family), steps)
    assert len(sigmas) == steps + 1 and torch.isfinite(sigmas).all()
    assert sigmas[0] == 1 and sigmas[-1] == 0
    assert (sigmas[:-1] > sigmas[1:]).all()
    if family == 'qwen_image21' and steps == 4:
        assert float(sigmas[-2]) == pytest.approx(0.02, abs=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('family', ['flux_klein', 'qwen_image21'])
@pytest.mark.parametrize('steps', [1, 4])
@pytest.mark.parametrize('warmup', [0, 4])
@pytest.mark.parametrize('fastvideo', [False, True])
def test_cuda_training_accumulation_and_resume(family, steps, warmup, fastvideo, tmp_path, monkeypatch, capsys):
    class TestAdapter(FlowAdapter):
        def __init__(self, family, model_id, revision, device, teacher_guidance=None):
            base = tiny_model(family).to(torch.bfloat16)
            self.__dict__.update(synthetic_adapter(family, base, device).__dict__)
            self.dtype = torch.bfloat16
            # A guided teacher creates a measurable student gradient immediately;
            # three tiny BF16 steps with identical unguided critics can round it away.
            self.teacher_guidance = 4.0
            self.identity['diffusers_version'] = 'test'
            self.identity['teacher_guidance'] = 4.0
        def prepare(self, record):
            # Isolate preprocessing RNG, keeping cache misses from affecting training seeds.
            with torch.random.fork_rng():
                torch.manual_seed(123)
                return synthetic_sample(self.family, int(record['task'] == 'i2i'))
        def offload_encoders(self):
            pass
    monkeypatch.setattr('main.train_flow.FlowAdapter', TestAdapter)
    manifest = tmp_path / 'data.jsonl'
    reference = tmp_path / 'reference.png'
    reference.write_bytes(b'reference content')
    manifest.write_text('\n'.join([json.dumps(dict(task='t2i', prompt='mountain')),
        json.dumps(dict(task='i2i', prompt='edit', reference_images=['reference.png']))]))
    common = ['--model_family', family, '--manifest', str(manifest), '--model_id', 'synthetic-base',
        '--lora_rank', '2', '--lora_alpha', '2', '--num_steps', str(steps), '--gradient_accumulation_steps', '2',
        '--fake_gen_ratio', '2', '--save_every', '1', '--log_every', '1', '--cache_dir', str(tmp_path / 'cache'),
        '--warmup_step', str(warmup)]
    if fastvideo:
        common += ['--score_noise_sampling', 'shifted', '--score_noise_shift', '8',
                   '--rollout_mode', 'stochastic', '--student_ema_decay', '.98']
    continuous = tmp_path / 'continuous'
    split = tmp_path / 'split'
    train_main(common + ['--output_dir', str(continuous), '--train_iters', '3'])
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    expected = [0, .25, .5] if warmup else [1, 1, 1]
    for row, factor in zip(records, expected):
        assert row['student_lr'] == pytest.approx(1e-5 * factor)
        assert row['fake_lr'] == pytest.approx(1e-5 * factor)
    assert len(records) == 3
    train_main(common + ['--output_dir', str(split), '--train_iters', '2'])
    train_main(common + ['--output_dir', str(split), '--train_iters', '3', '--resume', str(split / 'checkpoint-000002')])
    from safetensors.torch import load_file
    a = load_file(str(continuous / 'student/adapter.safetensors'))
    b = load_file(str(split / 'student/adapter.safetensors'))
    assert all(torch.equal(a[key], b[key]) for key in a)
    from main.audit_flow_run import audit_checkpoint
    verified = audit_checkpoint(continuous / 'checkpoint-000003', 3)
    assert verified['completed_fake_updates'] == 3 and verified['completed_student_updates'] == 2
    if fastvideo:
        assert verified['student_ema']['updates'] == 3
        assert verified['student_ema']['export_matches_shadow']
        a = load_file(str(continuous / 'student-ema/adapter.safetensors'))
        b = load_file(str(split / 'student-ema/adapter.safetensors'))
        assert all(torch.equal(a[key], b[key]) for key in a)
        a = torch.load(continuous / 'checkpoint-000003/custom_checkpoint_0.pkl', weights_only=True)
        b = torch.load(split / 'checkpoint-000003/custom_checkpoint_0.pkl', weights_only=True)
        assert a['num_updates'] == b['num_updates'] == 3
        assert all(torch.equal(a['shadow'][key], b['shadow'][key]) for key in a['shadow'])
        config = read_export_config(continuous / 'student-ema')
        assert config['rollout_mode'] == 'stochastic' and config['weight_source'] == 'ema'
    if warmup:
        initial = audit_checkpoint(continuous / 'checkpoint-000001', 1)
        assert initial['roles']['student']['lora_b_squared_norm'] == 0
        assert initial['roles']['fake']['lora_b_squared_norm'] == 0
        assert verified['roles']['student']['lr_schedule_steps'] == 3
        assert verified['roles']['fake']['lr_schedule_steps'] == 3
        for name in ('scheduler.bin', 'scheduler_1.bin'):
            a = torch.load(continuous / 'checkpoint-000003' / name, weights_only=True)
            b = torch.load(split / 'checkpoint-000003' / name, weights_only=True)
            assert a == b and a['last_epoch'] == 3
    with pytest.raises(ValueError, match='iteration count'):
        audit_checkpoint(continuous / 'checkpoint-000003', 4)
    for name in ('optimizer.bin', 'optimizer_1.bin'):
        a = torch.load(continuous / 'checkpoint-000003' / name, weights_only=True)
        b = torch.load(split / 'checkpoint-000003' / name, weights_only=True)
        assert a['param_groups'] == b['param_groups']
        for key in a['state']:
            for field in a['state'][key]:
                assert torch.equal(a['state'][key][field], b['state'][key][field])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('objective', ['normalized', 'raw_flow_kl', 'stochastic', 'fastvideo'])
def test_cuda_pilot_warmup_and_artifact_warning_does_not_stop_training(objective, tmp_path, monkeypatch):
    class TestAdapter(FlowAdapter):
        def __init__(self, family, model_id, revision, device, teacher_guidance=None):
            base = tiny_model(family).to(torch.bfloat16)
            self.__dict__.update(synthetic_adapter(family, base, device).__dict__)
            self.dtype = torch.bfloat16
            self.teacher_guidance = 4.0
            self.identity['diffusers_version'] = 'test'
            self.identity['teacher_guidance'] = 4.0
        def prepare(self, record):
            with torch.random.fork_rng():
                torch.manual_seed(123)
                return synthetic_sample(self.family, 1)
        def offload_encoders(self):
            pass
    monkeypatch.setattr('main.train_flow.FlowAdapter', TestAdapter)
    evaluations = []
    def verdict(*args, **kwargs):
        if objective == 'fastvideo' and evaluations:
            assert kwargs['weight_source'] == 'ema'
        evaluations.append(args[-1] if len(args) == 8 else None)
        return dict(results=[{}], artifact_gate_passed=len(evaluations) == 1)
    monkeypatch.setattr('main.train_flow.evaluate_student', verdict)
    reference = tmp_path / 'reference.png'
    reference.write_bytes(b'reference')
    manifest = tmp_path / 'train.jsonl'
    manifest.write_text(json.dumps(dict(task='i2i', prompt='Edit', reference_images=['reference.png'])) + '\n')
    output = tmp_path / 'pilot'
    entry = train_main
    if objective == 'raw_flow_kl':
        from main.flow_kl_diagnostic import main as entry
    extra = ['--rollout_mode', 'stochastic', '--pilot_fixed_noise'] if objective == 'stochastic' else []
    if objective == 'fastvideo':
        extra = ['--rollout_mode', 'stochastic', '--score_noise_sampling', 'shifted',
                 '--student_ema_decay', '.98']
    entry(['--model_family', 'flux_klein', '--model_id', 'synthetic-base', '--task', 'i2i',
                '--manifest', str(manifest), '--output_dir', str(output), '--train_iters', '8',
                '--lora_rank', '2', '--lora_alpha', '2', '--gradient_accumulation_steps', '2',
                '--fake_warmup_iters', '3', '--fake_gen_ratio', '2', '--validation_manifest', str(manifest),
                '--validation_every', '2', '--validation_limit', '1', *extra])
    summary = json.loads((output / 'pilot_summary.json').read_text())
    assert not summary['stopped_for_artifacts'] and summary['reached_requested_updates']
    assert summary['validation_policy'] == 'report_only'
    assert summary['completed_iterations'] == 8 and summary['completed_student_updates'] == 3
    assert len(evaluations) == 3
    from main.audit_flow_run import audit_checkpoint
    assert audit_checkpoint(output / 'checkpoint-000008', 8)['completed_student_updates'] == 3
    if objective == 'raw_flow_kl':
        config = json.loads((output / 'checkpoint-000008/training_config.json').read_text())
        assert config['loss_weighting'] == 'raw_flow_kl' and len(config['objective_code_sha256']) == 64
    if objective == 'stochastic':
        export = json.loads((output / 'student/flow_config.json').read_text())
        assert export['rollout_mode'] == 'stochastic'
        config = json.loads((output / 'checkpoint-000008/training_config.json').read_text())
        assert config['pilot_fixed_noise']
