"""Run isolated FLUX pilots after correctness and fixed-critic gates pass."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', default='outputs/flux-validation')
    parser.add_argument('--train_manifest', default='data/flux-klein-i2i-hqedit/train/manifest.jsonl')
    parser.add_argument('--validation_manifest', default='data/flux-klein-i2i-hqedit/validation/manifest.jsonl')
    parser.add_argument('--updates', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--variants', nargs='+', choices=['baseline', 'native-precision', 'slow-student', 'critic-rank128', 'critic-warmup'],
                        default=['baseline', 'native-precision', 'slow-student', 'critic-rank128'])
    args = parser.parse_args()
    if args.updates < 100 or len(set(args.variants)) != len(args.variants):
        parser.error('Use at least 100 updates and distinct variant names')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'pilots_status.json'
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        try:
            os.kill(previous['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('A validation-pilot supervisor is already live')
    for name, path in [('math', output / 'math.json'), ('native', output / 'native.json'),
                       ('critic', output / 'critic-rank32/critic.json')]:
        result = json.loads(path.read_text())
        if not result['passed'] or (name == 'native' and not result['complete']):
            raise ValueError(f'The {name} gate has not passed')
    from main.flow.data import read_manifest
    records = {name: read_manifest(root / path, 'i2i', 512, 512)[:size]
               for name, path, size in [('train', args.train_manifest, 16), ('validation', args.validation_manifest, 8)]}
    hashes = {name: {hashlib.sha256(Path(r['reference_images'][0]).read_bytes()).hexdigest() for r in rows}
              for name, rows in records.items()}
    if len(hashes['train']) != 16 or len(hashes['validation']) != 8 or not hashes['train'].isdisjoint(hashes['validation']):
        raise ValueError('Pilots require 16 unique train and eight disjoint validation sources')
    manifests = {}
    for name, rows in records.items():
        manifests[name] = output / f'pilot-{name}.jsonl'
        manifests[name].write_text(''.join(json.dumps(row) + '\n' for row in rows))
    code = [root / 'main/train_flow.py', *sorted((root / 'main/flow').glob('*.py'))]
    snapshot = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code}
    (output / 'pilot_code_snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    env = dict(os.environ, HF_HOME=str(root / '.cache/huggingface'),
               HF_HUB_CACHE=str(root / '.cache/huggingface/hub'), OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    state = dict(pid=os.getpid(), started_at=time.time(), stage='starting', variants=args.variants, seed=args.seed,
                 requested_updates=args.updates, train_examples=16, validation_examples=8, results={})
    def update(stage, **fields):
        state.update(stage=stage, updated_at=time.time(), **fields)
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(status_path)
        print(stage, fields.get('variant', ''), flush=True)
    options = {'baseline': [], 'native-precision': ['--sampling_precision', 'native'],
               'slow-student': ['--student_lr', '1e-6'],
               'critic-rank128': ['--fake_lora_rank', '128', '--fake_lora_alpha', '128'],
               'critic-warmup': ['--fake_warmup_iters', '200']}
    try:
        for variant in args.variants:
            if any(hashlib.sha256((root / path).read_bytes()).hexdigest() != digest for path, digest in snapshot.items()):
                raise RuntimeError('Training code changed between controlled pilots')
            directory = output / f'pilot-{variant}-seed{args.seed}'
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / 'pilot_summary.json').exists():
                raise ValueError(f'{directory} already has results; choose a fresh output directory or seed')
            command = [sys.executable, '-u', '-m', 'main.train_flow', '--model_family', 'flux_klein',
                '--model_id', 'black-forest-labs/FLUX.2-klein-base-4B',
                '--revision', 'a3b4f4849157f664bdbc776fd7453c2783562f4d',
                '--manifest', str(manifests['train']), '--task', 'i2i', '--num_steps', '4',
                '--output_dir', str(directory), '--cache_dir', str(root / '.cache/flux-klein-i2i-conditioning'),
                '--train_iters', str(args.updates), '--student_lr', '1e-5', '--fake_lr', '1e-5',
                '--lora_rank', '32', '--lora_alpha', '32', '--teacher_guidance', '4',
                '--height', '512', '--width', '512', '--batch_size', '1', '--gradient_accumulation_steps', '4',
                '--fake_gen_ratio', '5', '--seed', str(args.seed), '--save_every', str(args.updates), '--keep_checkpoints', '1',
                '--validation_manifest', str(manifests['validation']), '--validation_limit', '8',
                '--validation_every', '20', '--validation_seeds', '42', '43', *options[variant]]
            (directory / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
            update('training_pilot', variant=variant)
            with (directory / 'run.log').open('w') as log:
                subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            summary = json.loads((directory / 'pilot_summary.json').read_text())
            state['results'][variant] = summary
            update('pilot_finished', variant=variant)
        update('complete', all_requested_pilots_completed=True,
               editing_quality_verified=False, note='Inspect comparison images; automatic artifact checks alone do not establish editing quality.')
    except Exception as error:
        update('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
