"""Supervise smoke validation, the agreed 10k update run, and final comparisons."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', default='outputs/flux-klein-i2i-4step-lora')
    parser.add_argument('--data_dir', default='data/flux-klein-i2i-hqedit')
    parser.add_argument('--resume', action='store_true', help='Continue from the latest completed checkpoint')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output, data = root / args.output_dir, root / args.data_dir
    output.mkdir(parents=True, exist_ok=True)
    status_file = output / 'run_status.json'
    if status_file.exists():
        previous = json.loads(status_file.read_text())
        try:
            os.kill(previous['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('A supervisor for this output directory is already running')
    cache = root / '.cache'
    env = dict(os.environ, PYTHONUNBUFFERED='1', HF_HOME=str(cache / 'huggingface'),
        HF_HUB_CACHE=str(cache / 'huggingface/hub'), HF_XET_CACHE=str(cache / 'huggingface/xet'),
        HF_DATASETS_CACHE=str(cache / 'datasets'), TMPDIR=str(cache / 'tmp'), OMP_NUM_THREADS='4')
    for key in ('HF_HOME', 'HF_HUB_CACHE', 'HF_XET_CACHE', 'HF_DATASETS_CACHE', 'TMPDIR'):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    status = dict(pid=os.getpid(), started_at=time.time(), dataset='UCSC-VLAA/HQ-Edit', task='i2i',
                  model='black-forest-labs/FLUX.2-klein-base-4B', target_iterations=10000)
    def update(stage, **fields):
        status.update(stage=stage, updated_at=time.time(), **fields)
        (output / 'run_status.json').write_text(json.dumps(status, indent=2) + '\n')
        print(stage, flush=True)
    def run(stage, command, log):
        update(stage)
        with (output / log).open('a') as stream:
            subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    try:
        if shutil.disk_usage(root).free < 50 * 2 ** 30:
            raise RuntimeError('This run requires at least 50 GiB free workspace space before preprocessing')
        if not (data / 'provenance.json').exists():
            run('dataset_preparation', [sys.executable, '-u', '-m', 'main.prepare_hqedit',
                '--output_dir', str(data), '--train_size', '1024', '--validation_size', '64'], 'hqedit_dataset.log')
        provenance = json.loads((data / 'provenance.json').read_text())
        if provenance['dataset'] != 'UCSC-VLAA/HQ-Edit' or provenance['train_size'] != 1024 or provenance['validation_size'] != 64:
            raise ValueError('This supervised run requires the agreed HQ-Edit 1024/64 subset')
        base = [sys.executable, '-u', '-m', 'main.train_flow', '--model_family', 'flux_klein',
            '--model_id', 'black-forest-labs/FLUX.2-klein-base-4B',
            '--revision', 'a3b4f4849157f664bdbc776fd7453c2783562f4d',
            '--manifest', str(data / 'train/manifest.jsonl'), '--task', 'i2i', '--num_steps', '4',
            '--student_train_mode', 'lora', '--fake_train_mode', 'lora', '--lora_rank', '32', '--lora_alpha', '32',
            '--student_lr', '1e-5', '--fake_lr', '1e-5', '--teacher_guidance', '4', '--height', '512', '--width', '512',
            '--batch_size', '1', '--gradient_accumulation_steps', '4', '--fake_gen_ratio', '5', '--seed', '42',
            '--cache_dir', str(cache / 'flux-klein-i2i-conditioning'), '--output_dir', str(output), '--keep_checkpoints', '2']
        completed = 0
        checkpoint = None
        if args.resume:
            candidates = [(json.loads((directory / 'progress.json').read_text())['completed_iterations'], directory)
                for directory in output.glob('checkpoint-*')
                if (directory / 'checkpoint_complete.json').exists()]
            if not candidates:
                raise ValueError('No completed checkpoint available to resume')
            completed, checkpoint = max(candidates)
        else:
            run('smoke_training', base + ['--train_iters', '5', '--save_every', '5', '--log_every', '1'], 'smoke.log')
            completed, checkpoint = 5, output / 'checkpoint-000005'
        evaluation = [sys.executable, '-u', '-m', 'main.evaluate_flow_i2i', '--checkpoint', str(output / 'student'),
                      '--manifest', str(data / 'validation/manifest.jsonl'), '--seed', '42']
        if not (output / 'smoke_validation/summary.json').exists():
            run('smoke_inference', evaluation + ['--limit', '4', '--teacher_count', '0',
                '--output_dir', str(output / 'smoke_validation')], 'smoke_inference.log')
        if completed < 10000:
            run('training_10000', base + ['--resume', str(checkpoint),
                '--train_iters', '10000', '--save_every', '500', '--log_every', '10'], 'train.log')
        run('final_evaluation', evaluation + ['--limit', '64', '--teacher_count', '8',
            '--output_dir', str(output / 'validation')], 'evaluation.log')
        update('complete', completed_iterations=10000)
    except Exception as error:
        update('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
