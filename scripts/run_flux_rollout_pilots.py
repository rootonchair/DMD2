"""Isolate stochastic re-noising versus Euler with independent fixed RNG streams."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'outputs/flux-validation'
    for name in ['pilots_status.json', 'weighting_status_seed43.json', 'tracking_status.json']:
        if json.loads((output / name).read_text())['stage'] != 'complete':
            raise ValueError('Finish the prior GPU jobs before starting rollout pilots')
    status_path = output / 'rollout_status_seed44.json'
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        try:
            os.kill(previous['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('A rollout-pilot supervisor is already live')
    template = json.loads((output / 'pilot-baseline-seed42/command.json').read_text())
    paths = [root / 'main/train_flow.py', *sorted((root / 'main/flow').glob('*.py'))]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    (output / 'rollout_code_snapshot.json').write_text(json.dumps(hashes, indent=2)+'\n')
    env = dict(os.environ, HF_HOME=str(root / '.cache/huggingface'),
               HF_HUB_CACHE=str(root / '.cache/huggingface/hub'), HF_HUB_OFFLINE='1', OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    state = dict(pid=os.getpid(), seed=44, started_at=time.time(), results={})
    def update(stage, **fields):
        state.update(stage=stage, updated_at=time.time(), **fields)
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(status_path)
        print(stage, fields.get('variant', ''), flush=True)
    try:
        for mode in ['euler', 'stochastic']:
            if any(hashlib.sha256((root / path).read_bytes()).hexdigest() != digest for path, digest in hashes.items()):
                raise ValueError('Training code changed between the controlled rollout pilots')
            name = f'{mode}-fixed'
            directory = output / f'pilot-{name}-seed44'
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / 'pilot_summary.json').exists():
                raise ValueError('Refusing to overwrite completed rollout results')
            command = list(template)
            command[command.index('--seed')+1] = '44'
            command[command.index('--output_dir')+1] = str(directory)
            command += ['--pilot_fixed_noise', '--rollout_mode', mode]
            (directory / 'command.json').write_text(json.dumps(command, indent=2)+'\n')
            update('training_pilot', variant=name)
            with (directory / 'run.log').open('w') as log:
                subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            state['results'][name] = json.loads((directory / 'pilot_summary.json').read_text())
            update('pilot_finished', variant=name)
        update('complete', editing_quality_verified=False)
    except Exception as error:
        update('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
