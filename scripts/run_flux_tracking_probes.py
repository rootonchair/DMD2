"""Read-only critic probes after all GPU training pilots have finished."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'outputs/flux-validation'
    for name in ['pilots_status.json', 'weighting_status_seed43.json']:
        state = json.loads((output / name).read_text())
        if state['stage'] != 'complete':
            raise ValueError('Finish GPU training pilots before running tracking probes')
    variants = [(name, 42) for name in ['baseline', 'native-precision', 'slow-student', 'critic-rank128']]
    variants += [('baseline', 43), ('raw-flow-kl', 43)]
    status_path = output / 'tracking_status.json'
    if status_path.exists():
        prior = json.loads(status_path.read_text())
        try:
            os.kill(prior['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('A tracking-probe supervisor is already running')
    state = dict(pid=os.getpid(), started_at=time.time(), completed=[])
    def update(stage, **fields):
        state.update(stage=stage, updated_at=time.time(), **fields)
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(status_path)
        print(stage, fields.get('variant', ''), flush=True)
    env = dict(os.environ, HF_HOME=str(root / '.cache/huggingface'),
               HF_HUB_CACHE=str(root / '.cache/huggingface/hub'), OMP_NUM_THREADS='4')
    try:
        for variant, seed in variants:
            directory = output / f'pilot-{variant}-seed{seed}'
            summary = json.loads((directory / 'pilot_summary.json').read_text())
            checkpoint = directory / f"checkpoint-{summary['completed_iterations']:06d}"
            update('probing', variant=variant, seed=seed)
            command = [sys.executable, '-u', '-m', 'main.probe_flow_tracking', '--checkpoint', str(checkpoint),
                       '--manifest', str(output / 'pilot-validation.jsonl'), '--output', str(directory / 'tracking-mixture.json')]
            if variant == 'baseline' and seed == 42:
                command += ['--teacher_examples', '3', '7']
            with (directory / 'tracking.log').open('w') as log:
                subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            state['completed'].append(dict(variant=variant, seed=seed))
            update('probe_finished', variant=variant, seed=seed)
        update('complete')
    except Exception as error:
        update('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
