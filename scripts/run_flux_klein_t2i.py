"""Supervise or queue a one/four-step FLUX Klein T2I run on repo LAION captions."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def process_start(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    return fields[19] if fields[0] != 'Z' else None


def wait_for_run(directory, update, poll_seconds=30):
    """Wait for successful completion and supervisor exit, including evaluation."""
    path = Path(directory).resolve() / 'run_status.json'
    initial = json.loads(path.read_text())
    identity = initial['pid'], initial['started_at']
    birth = process_start(initial['pid'])
    while True:
        status = json.loads(path.read_text())
        if (status['pid'], status['started_at']) != identity:
            raise RuntimeError('Predecessor run identity changed while this run was queued')
        update('queued', after_run=str(path.parent), predecessor_stage=status['stage'],
               predecessor_completed_iterations=status.get('completed_iterations', 0))
        if status['stage'] in {'failed', 'superseded', 'cancelled', 'dependency_failed'}:
            raise RuntimeError(f"Predecessor run ended with stage={status['stage']}")
        current_birth = process_start(status['pid'])
        running = birth is not None and current_birth == birth
        if status['stage'] == 'complete' and not running:
            return
        if status['stage'] != 'complete' and not running:
            raise RuntimeError('Predecessor supervisor exited before completing its run')
        time.sleep(poll_seconds)


def training_command(root, output, data, num_steps):
    cache = root / '.cache'
    return [sys.executable, '-u', '-m', 'main.train_flow', '--model_family', 'flux_klein',
        '--model_id', 'black-forest-labs/FLUX.2-klein-base-4B',
        '--revision', 'a3b4f4849157f664bdbc776fd7453c2783562f4d',
        '--manifest', str(data / 'train.jsonl'), '--task', 't2i', '--num_steps', str(num_steps),
        '--student_train_mode', 'lora', '--fake_train_mode', 'lora', '--lora_rank', '64', '--lora_alpha', '8',
        '--fake_lora_rank', '64', '--fake_lora_alpha', '8',
        '--student_lr', '5e-5', '--fake_lr', '5e-5', '--teacher_guidance', '4',
        '--max_grad_norm', '10', '--warmup_step', '500',
        '--score_noise_sampling', 'shifted', '--score_noise_shift', '8',
        '--rollout_mode', 'stochastic', '--student_ema_decay', '.98',
        '--height', '512', '--width', '512', '--batch_size', '1', '--gradient_accumulation_steps', '8',
        '--fake_gen_ratio', '5', '--seed', '42', '--train_iters', '10000',
        '--output_dir', str(output), '--cache_dir', str(cache / 'flux-klein-t2i-conditioning'),
        '--conditioning_cache_limit', '512', '--save_every', '500', '--keep_checkpoints', '2',
        '--validation_manifest', str(data / 'validation.jsonl'), '--validation_limit', '8',
        '--validation_every', '100', '--validation_seeds', '42', '43']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir')
    parser.add_argument('--data_dir', default='data/repo-t2i')
    parser.add_argument('--num_steps', type=int, choices=[1, 4], default=4)
    parser.add_argument('--after_run', help='Queue until this run completes training, audit and final evaluation')
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir or f'outputs/flux-klein-t2i-{args.num_steps}step-lora'
    root = Path(__file__).resolve().parents[1]
    output, data = (root / args.output_dir).resolve(), (root / args.data_dir).resolve()
    if args.after_run and (root / args.after_run).resolve() == output:
        parser.error('--after_run must refer to a different output directory')
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'run_status.json'
    if status_path.exists():
        raise ValueError('Output already has a run; choose a fresh directory to avoid duplicate training')
    cache = root / '.cache'
    env = dict(os.environ, PYTHONUNBUFFERED='1', HF_HOME=str(cache / 'huggingface'),
        HF_HUB_CACHE=str(cache / 'huggingface/hub'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='4', TMPDIR=str(cache / 'tmp'))
    status = dict(pid=os.getpid(), started_at=time.time(), target_iterations=10000,
        target_student_updates=2000, task='t2i', dataset='DMD2 original LAION captions',
        stage='starting', quality_verified=False, completed_iterations=0, num_steps=args.num_steps,
        initialization='pretrained_base', completed_student_updates=0)
    def update(stage, **fields):
        status.update(stage=stage, updated_at=time.time(), **fields)
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(status, indent=2) + '\n')
        temp.replace(status_path)
    def run(stage, command, log_name):
        (output / f'{stage}_command.json').write_text(json.dumps(command, indent=2) + '\n')
        update(stage)
        with (output / log_name).open('w') as log:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            update(stage, child_pid=child.pid)
            for line in child.stdout:
                log.write(line)
                log.flush()
                if line.startswith('{'):
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if 'iteration' in row:
                        update(stage, completed_iterations=row['iteration'],
                            completed_student_updates=row['completed_student_updates'], last_metrics=row)
            code = child.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
    try:
        command = training_command(root, output, data, args.num_steps)
        (output / 'planned_training_command.json').write_text(json.dumps(command, indent=2) + '\n')
        if args.after_run:
            wait_for_run(root / args.after_run, update)
        status['training_started_at'] = time.time()
        provenance = json.loads((data / 'provenance.json').read_text())
        if provenance.get('hqedit_used') is not False or provenance['train_size'] < 2_000_000:
            raise ValueError('Expected the full prepared original repo LAION corpus')
        update('prepared', train_examples=provenance['train_size'], validation_examples=provenance['validation_size'])
        sources = [root / 'main/train_flow.py', root / 'main/flow/data.py',
            root / 'main/audit_flow_validation.py', *sorted((root / 'main/flow').glob('*.py'))]
        hashes = {}
        for source in set(sources):
            relative = source.relative_to(root)
            hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
            archived = output / 'code' / relative
            archived.parent.mkdir(parents=True, exist_ok=True)
            archived.write_bytes(source.read_bytes())
        (output / 'code_snapshot.json').write_text(json.dumps(hashes, indent=2) + '\n')
        run('training', command, 'train.log')
        summary = json.loads((output / 'pilot_summary.json').read_text())
        update('auditing', completed_iterations=summary['completed_iterations'],
            completed_student_updates=summary['completed_student_updates'])
        sys.path.insert(0, str(root))
        from main.audit_flow_validation import audit_pilot
        audit = audit_pilot(output)
        (output / 'completion_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        run('final_evaluation', [sys.executable, '-u', '-m', 'main.evaluate_flow_t2i',
            '--checkpoint', str(output / 'student-ema'), '--manifest', str(data / 'validation.jsonl'),
            '--output_dir', str(output / 'validation'), '--limit', '64', '--teacher_count', '8', '--seed', '42'], 'evaluation.log')
        update('complete', completed_iterations=10000, completed_student_updates=2000,
            artifact_audit_passed=True, quality_verified=False,
            note='Training and artifact verification complete; inspect held-out teacher/student comparisons for generation quality')
    except Exception as error:
        update('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
