import json
from pathlib import Path

import pytest
import torch

from main.flow.gan import restore_real_sampler
from main.train_flow import resume_validation_baseline
from scripts.run_qwen_image21_t2i import inherit_validation


def test_legacy_and_saved_real_sampler_preserve_the_next_draws(tmp_path):
    seed, count, draws = 1045, 19, 64
    continuous = torch.Generator().manual_seed(seed)
    for _ in range(draws):
        torch.randint(count, (), generator=continuous)
    expected = continuous.get_state().clone()
    resumed = torch.Generator().manual_seed(seed)
    global_rng = torch.get_rng_state().clone()
    assert restore_real_sampler(resumed, tmp_path, count, draws) == 'legacy_replay'
    assert torch.equal(resumed.get_state(), expected)
    assert torch.equal(torch.get_rng_state(), global_rng)
    torch.save(dict(state=expected, examples=count, draws=draws), tmp_path / 'real_sampler_rng.pt')
    resumed.manual_seed(0)
    assert restore_real_sampler(resumed, tmp_path, count, draws) == 'saved_state'
    assert torch.equal(resumed.get_state(), expected)
    assert torch.equal(torch.randint(count, (20,), generator=continuous), torch.randint(count, (20,), generator=resumed))
    with pytest.raises(ValueError, match='sampler metadata'):
        restore_real_sampler(resumed, tmp_path, count + 1, draws)


def test_continuation_keeps_original_baseline_and_checks_protocol(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'continued'
    output.mkdir()
    baseline = source / 'validation-000000'
    baseline.mkdir(parents=True)
    (source / 'teacher-references').mkdir()
    image = baseline / '000-seed42.png'
    image.write_bytes(b'baseline')
    row = dict(index=0, seed=42, prompt='A dog', reference_images=[], image=str(image), statistics={})
    (baseline / 'summary.json').write_text(json.dumps(dict(completed=True, results=[row])))
    config = dict(validation_manifest='heldout.jsonl', validation_every=100, validation_limit=1, validation_seeds=[42])
    records = [dict(prompt='A dog', reference_images=[])]
    inherit_validation(source, output)
    assert (output / 'validation-000000').is_symlink()
    assert resume_validation_baseline(config, config, output, records) == [row]
    with pytest.raises(ValueError, match='validation_seeds'):
        resume_validation_baseline(config, dict(config, validation_seeds=[43]), output, records)
    with pytest.raises(ValueError, match='baseline inputs'):
        resume_validation_baseline(config, config, output, [dict(prompt='A cat', reference_images=[])])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('legacy', [False, True])
def test_gan_validation_resume_matches_uninterrupted_training(tmp_path, monkeypatch, legacy):
    from PIL import Image
    from safetensors.torch import load_file
    from main.flow.adapters import FlowAdapter
    from main.train_flow import main as train
    from tests.test_flow import synthetic_adapter
    from tests.test_flow_gan import gan_model, gan_sample

    class Adapter(FlowAdapter):
        def __init__(self, family, model_id, revision, device, teacher_guidance=None):
            self.__dict__.update(synthetic_adapter(family, gan_model().to(torch.bfloat16), device).__dict__)
            self.dtype = torch.bfloat16
            self.keep_encoders_on_device = False
            self.teacher_guidance = 2.5
            self.identity.update(diffusers_version='test', teacher_guidance=2.5)
        def prepare(self, record):
            with torch.random.fork_rng():
                torch.manual_seed(123 + len(record['prompt']))
                return gan_sample()
        def offload_encoders(self):
            pass
        def decode(self, latents, sample):
            return Image.new('RGB', (64, 64), (100, 120, 140))

    monkeypatch.setattr('main.train_flow.FlowAdapter', Adapter)
    monkeypatch.setattr('main.train_flow.prepare_teacher_references', lambda *a: [])
    manifest, heldout, real = (tmp_path / name for name in ('train.jsonl', 'heldout.jsonl', 'real.pt'))
    manifest.write_text(json.dumps(dict(task='t2i', prompt='Training scene')) + '\n')
    heldout.write_text(json.dumps(dict(task='t2i', prompt='Held-out scene')) + '\n')
    torch.save(dict(latents=torch.randn(3, 64, 4), prompts=['photo a', 'photo bb', 'photo ccc'], height=64, width=64), real)
    common = ['--model_family', 'qwen_image21', '--model_id', 'synthetic-base', '--task', 't2i',
        '--manifest', str(manifest), '--height', '64', '--width', '64', '--num_steps', '4',
        '--lora_rank', '2', '--lora_alpha', '2', '--gan_feature_block', '1', '--gan_head_width', '64',
        '--gan_real_latents', str(real), '--gen_cls_loss_weight', '.005', '--guidance_cls_loss_weight', '.01',
        '--gradient_accumulation_steps', '2', '--fake_gen_ratio', '2', '--warmup_step', '2',
        '--rollout_mode', 'stochastic', '--score_noise_sampling', 'shifted', '--dmd_guidance_mode', 'decoupled',
        '--student_ema_decay', '.98', '--save_every', '1', '--validation_manifest', str(heldout),
        '--validation_limit', '1', '--validation_seeds', '42', '--validation_every', '1',
        '--cache_dir', str(tmp_path / 'cache')]
    continuous, split, resumed = (tmp_path / name for name in ('continuous', 'split', 'resumed'))
    train(common + ['--output_dir', str(continuous), '--train_iters', '5'])
    train(common + ['--output_dir', str(split), '--train_iters', '2'])
    if legacy:
        (split / 'checkpoint-000002/real_sampler_rng.pt').unlink()
    resumed.mkdir()
    inherit_validation(split, resumed)
    before = (split / 'validation-000000/summary.json').read_bytes()
    train(common + ['--output_dir', str(resumed), '--train_iters', '5', '--resume', str(split / 'checkpoint-000002')])
    assert (split / 'validation-000000/summary.json').read_bytes() == before
    for role in ('student', 'fake', 'student-ema'):
        a = load_file(str(continuous / f'checkpoint-000005/{role}/adapter.safetensors'))
        b = load_file(str(resumed / f'checkpoint-000005/{role}/adapter.safetensors'))
        assert a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)

    def equal_tree(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                equal_tree(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                equal_tree(x, y)
        else:
            assert a == b
    for name in ('optimizer.bin', 'optimizer_1.bin', 'custom_checkpoint_0.pkl',
                 'custom_checkpoint_1.pkl', 'scheduler.bin', 'scheduler_1.bin', 'real_sampler_rng.pt'):
        equal_tree(torch.load(continuous / 'checkpoint-000005' / name, weights_only=True),
                   torch.load(resumed / 'checkpoint-000005' / name, weights_only=True))
    from main.audit_flow_run import audit_checkpoint
    audit = audit_checkpoint(resumed / 'checkpoint-000005', 5)
    assert audit['discriminator']['finite'] and audit['real_sampler']['draws'] == 10
