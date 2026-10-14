"""Repeat the baseline and test raw flow-KL weighting with a second training seed."""

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
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--raw_only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output_dir).resolve()
    previous = json.loads((output / 'pilots_status.json').read_text())
    if previous['stage'] != 'complete':
        raise ValueError('Finish the original controlled-pilot round before starting follow-up GPU work')
    if not json.loads((output / 'weighting.json').read_text())['raw_gradient_checks_passed']:
        raise ValueError('Raw flow-KL math checks must pass before testing that objective')
    status_path = output / f'weighting_status_seed{args.seed}.json'
    if status_path.exists():
        prior = json.loads(status_path.read_text())
        try:
            os.kill(prior['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('A weighting-pilot supervisor is already running')
    snapshot = json.loads((output / 'pilot_code_snapshot.json').read_text())
    if any(hashlib.sha256((root / path).read_bytes()).hexdigest() != digest for path, digest in snapshot.items()):
        raise ValueError('Production training code differs from the original controlled-pilot round')
    template = json.loads((output / 'pilot-baseline-seed42/command.json').read_text())
    env = dict(os.environ, HF_HOME=str(root / '.cache/huggingface'),
               HF_HUB_CACHE=str(root / '.cache/huggingface/hub'), OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    state = dict(pid=os.getpid(), seed=args.seed, started_at=time.time(), results={})
    def update(stage, **fields):
        state.update(stage=stage, updated_at=time.time(), **fields)
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(status_path)
        print(stage, fields.get('variant', ''), flush=True)
    variants = ['raw-flow-kl'] if args.raw_only else ['baseline', 'raw-flow-kl']
    try:
        for variant in variants:
            directory = output / f'pilot-{variant}-seed{args.seed}'
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / 'pilot_summary.json').exists():
                raise ValueError('Refusing to overwrite completed follow-up results')
            command = list(template)
            command[command.index('--seed')+1] = str(args.seed)
            command[command.index('--output_dir')+1] = str(directory)
            if variant == 'raw-flow-kl':
                command[command.index('main.train_flow')] = 'main.flow_kl_diagnostic'
            (directory / 'command.json').write_text(json.dumps(command, indent=2)+'\n')
            update('training_pilot', variant=variant)
            with (directory / 'run.log').open('w') as log:
                subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            state['results'][variant] = json.loads((directory / 'pilot_summary.json').read_text())
            update('pilot_finished', variant=variant)
        update('complete', editing_quality_verified=False)
    except Exception as error:
        update('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
