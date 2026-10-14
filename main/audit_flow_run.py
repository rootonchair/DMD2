"""Verify the completed real FLUX I2I run against checkpoints and image artifacts."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops
from safetensors.torch import load_file
import torch

from main.flow.data import cache_key, read_manifest


def audit_checkpoint(directory, expected_updates):
    directory = Path(directory)
    config = json.loads((directory / 'training_config.json').read_text())
    progress = json.loads((directory / 'progress.json').read_text())
    marker = json.loads((directory / 'checkpoint_complete.json').read_text())
    if progress['completed_iterations'] != expected_updates or marker['completed_iterations'] != expected_updates:
        raise ValueError('Checkpoint iteration count differs from the requested completed run')
    if marker['run_id'] != config['run_id']:
        raise ValueError('Checkpoint completion marker belongs to a different run')
    result = dict(completed_fake_updates=expected_updates,
                  completed_student_updates=(max(0, expected_updates - config.get('fake_warmup_iters', 0)) + config['fake_gen_ratio'] - 1) // config['fake_gen_ratio'],
                  roles={})
    for role, optimizer in [('student', 'optimizer.bin'), ('fake', 'optimizer_1.bin')]:
        expected_steps = result['completed_student_updates'] if role == 'student' else expected_updates
        export = json.loads((directory / role / 'flow_config.json').read_text())
        if export['train_mode'] != 'lora' or export['num_steps'] != config['num_steps'] or export['task'] != config['task']:
            raise ValueError(f'{role} export settings differ from training settings')
        weights = load_file(str(directory / role / 'adapter.safetensors'))
        if not weights or not all(torch.isfinite(value).all() for value in weights.values()):
            raise ValueError(f'{role} adapter weights are missing or nonfinite')
        b_norm = sum(float(value.square().sum()) for name, value in weights.items() if 'lora_B' in name)
        last_update_index = ((config.get('fake_warmup_iters', 0) + (expected_steps - 1) * config['fake_gen_ratio'])
                             if role == 'student' else expected_updates - 1)
        zero_lr_only = config.get('warmup_step', 0) > 0 and last_update_index == 0
        if b_norm == 0 and not zero_lr_only:
            raise ValueError(f'{role} has no learned LoRA B weights')
        state = torch.load(directory / optimizer, map_location='cpu', weights_only=True)
        steps = {int(value['step']) for value in state['state'].values()}
        if steps != {expected_steps}:
            raise ValueError(f'{role} optimizer steps {steps} differ from {expected_steps}')
        for value in state['state'].values():
            for name in ('exp_avg', 'exp_avg_sq'):
                if not torch.isfinite(value[name]).all():
                    raise ValueError(f'{role} optimizer {name} is nonfinite')
        result['roles'][role] = dict(tensors=len(weights), finite=True, lora_b_squared_norm=b_norm,
                                    optimizer_steps=expected_steps)
        if config.get('warmup_step', 0):
            scheduler_name = 'scheduler.bin' if role == 'student' else 'scheduler_1.bin'
            scheduler = torch.load(directory / scheduler_name, map_location='cpu', weights_only=True)
            # Accelerate's default scheduler advances once per process for each
            # outer iteration, exactly as in the original training loop.
            expected_epoch = expected_updates * config['world_size']
            if scheduler['last_epoch'] != expected_epoch:
                raise ValueError(f'{role} LR schedule count differs from completed iterations')
            base_lr = config['student_lr'] if role == 'student' else config['fake_lr']
            expected_lr = base_lr * min(expected_epoch / config['warmup_step'], 1.0)
            if any(abs(group['lr'] - expected_lr) > 1e-12 for group in state['param_groups']):
                raise ValueError(f'{role} optimizer LR differs from its warm-up schedule')
            result['roles'][role].update(lr_schedule_steps=expected_epoch, next_lr=expected_lr)
    if config.get('student_ema_decay', 0):
        ema = torch.load(directory / 'custom_checkpoint_0.pkl', map_location='cpu', weights_only=True)
        if ema['decay'] != config['student_ema_decay'] or ema['num_updates'] != expected_updates:
            raise ValueError('Student EMA decay or per-iteration update count differs from training')
        if not ema['shadow'] or not all(torch.isfinite(value).all() for value in ema['shadow'].values()):
            raise ValueError('Student EMA parameters are missing or nonfinite')
        ema_export = json.loads((directory / 'student-ema/flow_config.json').read_text())
        if ema_export.get('weight_source') != 'ema' or ema_export.get('ema_updates') != expected_updates:
            raise ValueError('EMA export does not identify the saved averaged student')
        ema_weights = load_file(str(directory / 'student-ema/adapter.safetensors'))
        expected_weights = {name.replace('.default', ''): value for name, value in ema['shadow'].items()}
        if ema_weights.keys() != expected_weights.keys() or any(
                not torch.equal(value, expected_weights[name]) for name, value in ema_weights.items()):
            raise ValueError('EMA export weights differ from the saved EMA state')
        result['student_ema'] = dict(decay=ema['decay'], updates=ema['num_updates'], finite=True,
                                    export_matches_shadow=True)
    return result


def audit_evaluation(directory, manifest, expected_examples=64, expected_teachers=8):
    directory = Path(directory)
    manifest = Path(manifest).resolve()
    summary = json.loads((directory / 'summary.json').read_text())
    results = json.loads((directory / 'results.json').read_text())
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not summary['completed'] or summary['examples'] != expected_examples or len(results) != expected_examples:
        raise ValueError('Held-out evaluation is incomplete')
    if summary['teacher_comparisons'] != expected_teachers:
        raise ValueError('Teacher comparison count differs from the requested run')
    for index, result in enumerate(results):
        if result['index'] != index or result['prompt'] != records[index]['prompt'] or result['student_steps'] != 4:
            raise ValueError(f'Validation record {index} does not match the held-out manifest')
        if result['seed'] != 42 or result['teacher_steps'] != (50 if index < expected_teachers else None):
            raise ValueError(f'Validation record {index} uses unexpected sampling settings')
        references = [str((manifest.parent / path).resolve()) for path in records[index]['reference_images']]
        if result['references'] != references:
            raise ValueError(f'Validation record {index} uses different reference images')
        with Image.open(references[0]) as reference, Image.open(directory / f'{index:03d}-source.png') as exported:
            reference, exported = reference.convert('RGB'), exported.convert('RGB')
            if reference.size != exported.size or ImageChops.difference(reference, exported).getbbox() is not None:
                raise ValueError(f'Validation source {index} differs from the actual held-out source')
        paths = [directory / f'{index:03d}-{kind}.png' for kind in ('source', 'student', 'comparison')]
        if index < expected_teachers:
            paths.append(directory / f'{index:03d}-teacher.png')
        for path in paths:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if 'student' in path.name or 'teacher' in path.name:
                    if image.size != (512, 512):
                        raise ValueError(f'{path} has an unexpected output resolution')
    return dict(examples=expected_examples, teacher_comparisons=expected_teachers, images_verified=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', default='outputs/flux-klein-i2i-4step-lora')
    parser.add_argument('--data_dir', default='data/flux-klein-i2i-hqedit')
    args = parser.parse_args(argv)
    output, data = Path(args.output_dir), Path(args.data_dir)
    state = json.loads((output / 'run_status.json').read_text())
    if state['stage'] != 'complete' or state.get('completed_iterations') != 10000:
        raise ValueError('The real training/evaluation supervisor has not completed')
    checkpoint = output / 'checkpoint-010000'
    config = json.loads((checkpoint / 'training_config.json').read_text())
    required = dict(model_family='flux_klein', model_id='black-forest-labs/FLUX.2-klein-base-4B', task='i2i',
                    revision='a3b4f4849157f664bdbc776fd7453c2783562f4d', student_lr=1e-5, fake_lr=1e-5,
                    num_steps=4, student_train_mode='lora', fake_train_mode='lora', lora_rank=32, lora_alpha=32,
                    teacher_guidance=4.0, height=512, width=512, batch_size=1,
                    gradient_accumulation_steps=4, fake_gen_ratio=5, seed=42, world_size=1)
    for key, value in required.items():
        if config[key] != value:
            raise ValueError(f'Training configuration differs for {key}')
    provenance = json.loads((data / 'provenance.json').read_text())
    if provenance['dataset'] != 'UCSC-VLAA/HQ-Edit' or provenance['revision'] != '8d0a51255d4ebc82152806e1f3891c85b8e4e037' or not provenance['streaming'] or provenance['projected_columns'] != ['input_image', 'edit']:
        raise ValueError('Dataset provenance differs from the requested streamed HQ-Edit subset')
    selections = {split: json.loads((data / split / 'selection.json').read_text()) for split in ('train', 'validation')}
    if len(selections['train']) != 1024 or len(selections['validation']) != 64:
        raise ValueError('Dataset subset sizes differ from the agreed run')
    train_hashes = {row['source_sha256'] for row in selections['train']}
    val_hashes = {row['source_sha256'] for row in selections['validation']}
    if len(train_hashes) != 1024 or len(val_hashes) != 64 or not train_hashes.isdisjoint(val_hashes):
        raise ValueError('Source images are duplicated or overlap across train and validation')
    checkpoint_result = audit_checkpoint(checkpoint, 10000)
    if not (checkpoint / 'random_states_0.pkl').is_file():
        raise ValueError('Final checkpoint lacks the saved RNG state')
    export_config = json.loads((checkpoint / 'student/flow_config.json').read_text())
    identity_keys = ('format_version', 'family', 'model_id', 'revision', 'diffusers_version', 'teacher_guidance', 'dtype')
    identity = {key: export_config[key] for key in identity_keys}
    records = read_manifest(data / 'train/manifest.jsonl', task='i2i', height=512, width=512)
    if len(records) != 1024 or [cache_key(record, identity) for record in records] != config['data_keys']:
        raise ValueError('Current training source files or instructions differ from the conditioning actually used')
    sigmas = torch.tensor(export_config['schedules']['512x512:4'])
    if len(sigmas) != 5 or sigmas[0] != 1 or sigmas[-1] != 0 or not torch.isfinite(sigmas).all() or not (sigmas[:-1] > sigmas[1:]).all():
        raise ValueError('Final export does not contain a valid trained four-step sigma schedule')
    final = load_file(str(checkpoint / 'student/adapter.safetensors'))
    latest = load_file(str(output / 'student/adapter.safetensors'))
    if final.keys() != latest.keys() or not all(torch.equal(final[key], latest[key]) for key in final):
        raise ValueError('Latest student export differs from the final checkpoint')
    evaluation = audit_evaluation(output / 'validation', data / 'validation/manifest.jsonl')
    result = dict(verified_complete=True, checkpoint=checkpoint_result, evaluation=evaluation,
                  dataset=provenance, output_dir=str(output.resolve()))
    (output / 'completion_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
