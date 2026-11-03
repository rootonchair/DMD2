"""Supervise or queue fresh and resumed Qwen Image 2.1 T2I distillation on repo captions."""

import argparse
import codecs
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_flux_klein_t2i import training_command as flux_training_command, wait_for_run
from main.flow.adapters import QWEN_DISTILL_SCHEDULE_POLICY


MODEL_ID = 'Qwen/Qwen-Image-2.1'
REVISION = 'd26bb61231c349cf6b7896fa83353113880e1ba3'


def apply_overrides(command, overrides):
    """Replace or append '--flag=value' ablation overrides; a bare '--flag' appends a switch and
    space-separated values append as multiple arguments (e.g. '--adam_betas=0 0.999')."""
    command = list(command)
    for override in overrides or []:
        flag, separator, value = override.partition('=')
        if not flag.startswith('--') or flag in ('--output_dir', '--manifest', '--validation_manifest'):
            raise ValueError(f'Unsupported override: {override}')
        if flag in command:
            if not separator:
                raise ValueError(f'Override for an existing flag needs a value: {override}')
            command[command.index(flag) + 1] = value
        else:
            command.extend([flag, *value.split()] if separator else [flag])
    return command


def command_value(command, flag, default=None):
    return command[command.index(flag) + 1] if flag in command else default


def inherit_validation(source, output):
    """Reference the original baseline/history without modifying the completed run."""
    source, output = Path(source).resolve(), Path(output).resolve()
    for path in [source / 'teacher-references', *sorted(source.glob('validation-*'))]:
        if path.is_dir():
            (output / path.name).symlink_to(path, target_is_directory=True)


def training_command(root, output, data, ablation_overrides=None):
    command = flux_training_command(root, output, data, 4)
    overrides = {
        '--model_family': 'qwen_image21', '--model_id': MODEL_ID, '--revision': REVISION,
        '--teacher_guidance': '2.5', '--height': '1024', '--width': '1024',
        '--cache_dir': str(root / '.cache/qwen-image21-t2i-conditioning'),
    }
    for flag, value in overrides.items():
        command[command.index(flag) + 1] = value
    command.extend(['--student_schedule_policy', QWEN_DISTILL_SCHEDULE_POLICY])
    return apply_overrides(command, ablation_overrides)


def supervise_child(command, log_path, update, stage, target_iterations=10000):
    """Keep recording heartbeats even while loading, rendering, or saving."""
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    pending = ''
    with Path(log_path).open('wb') as log, selectors.DefaultSelector() as selector:
        child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=os.environ.copy())
        selector.register(child.stdout, selectors.EVENT_READ)
        update(stage, child_pid=child.pid, child_started_at=time.time())
        while selector.get_map():
            events = selector.select(timeout=30)
            if not events:
                update(stage, heartbeat_at=time.time())
            for key, _ in events:
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                log.write(chunk)
                log.flush()
                pending += decoder.decode(chunk)
                while '\n' in pending:
                    line, pending = pending.split('\n', 1)
                    if line.startswith('TEACHER_REFERENCE '):
                        fields = line.split()
                        update(stage, training_phase='teacher_references',
                            teacher_reference_index=int(fields[1]), teacher_reference_seed=int(fields[2]))
                    elif line.startswith('PILOT_BASELINE_READY'):
                        update(stage, training_phase='optimizer_updates')
                    elif line.startswith('PILOT_VALIDATION'):
                        update(stage, last_validation=line)
                    elif line.startswith('{'):
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        if 'iteration' in row:
                            update(stage, training_phase='optimizer_updates',
                                completed_iterations=row['iteration'],
                                completed_student_updates=row['completed_student_updates'], last_metrics=row,
                                estimated_training_seconds_remaining=row['seconds_per_iteration'] * (target_iterations-row['iteration']))
        code = child.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', default='outputs/qwen-image21-t2i-4step-cfg2p5-1024-20261015')
    parser.add_argument('--data_dir', default='data/repo-t2i')
    parser.add_argument('--model_cache_dir', help='Existing Hugging Face hub cache; weights are reused in place')
    parser.add_argument('--resume_checkpoint', help='Continue a complete training checkpoint into a new output directory')
    parser.add_argument('--after_run', help='Queue until this run completes training, audit and evaluation')
    parser.add_argument('--override', action='append', default=[], metavar='--FLAG=VALUE',
                        help='Ablation override of one train_flow flag, e.g. --override=--fake_lora_alpha=64 (repeatable)')
    args = parser.parse_args(argv)
    output, data = (ROOT / args.output_dir).resolve(), (ROOT / args.data_dir).resolve()
    resume = (ROOT / args.resume_checkpoint).resolve() if args.resume_checkpoint else None
    after_run = (ROOT / args.after_run).resolve() if args.after_run else None
    if after_run == output or (resume and resume.parent == output):
        parser.error('Use a new output directory distinct from the predecessor and checkpoint source')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory for this fresh run')
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='4', TMPDIR=str(ROOT / '.cache/tmp'))
    if args.model_cache_dir:
        os.environ['HF_HUB_CACHE'] = str(Path(args.model_cache_dir).resolve())
    command = training_command(ROOT, output, data, args.override)
    if resume:
        command.extend(['--resume', str(resume)])
    target_iterations = int(command_value(command, '--train_iters'))
    target_student_updates = -(-target_iterations // int(command_value(command, '--fake_gen_ratio')))
    status = dict(pid=os.getpid(), started_at=time.time(), model_id=MODEL_ID, revision=REVISION,
        task='t2i', dataset='DMD2 original LAION captions', num_steps=4,
        teacher_guidance=float(command_value(command, '--teacher_guidance')),
        height=1024, width=1024, initialization='checkpoint_resume' if resume else 'pretrained_base', target_iterations=target_iterations,
        student_schedule_policy=QWEN_DISTILL_SCHEDULE_POLICY, overrides=args.override,
        target_student_updates=target_student_updates, completed_iterations=0, completed_student_updates=0,
        quality_verified=False, training_phase='loading', resume_checkpoint=str(resume) if resume else None)
    def update(stage, **fields):
        status.update(stage=stage, updated_at=time.time(), **fields)
        interval = int(command_value(command, '--save_every'))
        first_save = (status.get('resumed_from_iteration', 0) // interval + 1) * interval
        first_checkpoint = output / f'checkpoint-{first_save:06d}'
        if (first_checkpoint / 'checkpoint_complete.json').is_file() and not status.get('first_checkpoint_verified'):
            # Observe the first save without imposing a visual-quality stop.
            from main.audit_flow_run import audit_checkpoint
            try:
                audit = audit_checkpoint(first_checkpoint, first_save)
                (output / 'first_checkpoint_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
                status['first_checkpoint_verified'] = True
                status['first_checkpoint_iteration'] = first_save
                status.pop('first_checkpoint_audit_error', None)
            except Exception as error:
                status['first_checkpoint_audit_error'] = str(error)
        temporary = output / 'run_status.tmp'
        temporary.write_text(json.dumps(status, indent=2)+'\n')
        temporary.replace(output / 'run_status.json')
    def run(stage, command, log_name):
        (output / f'{stage}_command.json').write_text(json.dumps(command, indent=2)+'\n')
        supervise_child(command, output / log_name, update, stage, target_iterations)
    try:
        update('preparing')
        provenance = json.loads((data / 'provenance.json').read_text())
        if provenance.get('hqedit_used') is not False or provenance['train_size'] < 2_000_000:
            raise ValueError('Expected the full prepared original repo LAION corpus')
        sources = [ROOT / 'main/train_flow.py', ROOT / 'main/evaluate_flow_t2i.py',
            ROOT / 'main/audit_flow_validation.py', ROOT / 'main/audit_flow_run.py',
            Path(__file__).resolve(), ROOT / 'scripts/run_flux_klein_t2i.py',
            *sorted((ROOT / 'main/flow').glob('*.py'))]
        hashes = {}
        for source in sources:
            relative = source.relative_to(ROOT)
            content = source.read_bytes()
            hashes[str(relative)] = hashlib.sha256(content).hexdigest()
            archived = output / 'code' / relative
            archived.parent.mkdir(parents=True, exist_ok=True)
            archived.write_bytes(content)
        (output / 'code_snapshot.json').write_text(json.dumps(hashes, indent=2)+'\n')
        (output / 'planned_training_command.json').write_text(json.dumps(command, indent=2)+'\n')
        if after_run:
            wait_for_run(after_run, update)
        # A queued job must execute the trainer it archived when it was submitted.
        if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest for name, digest in hashes.items()):
            raise ValueError('Queued source files changed; review the queued job before training')
        if resume:
            from main.audit_flow_run import audit_checkpoint
            progress = json.loads((resume / 'progress.json').read_text())['completed_iterations']
            audit = audit_checkpoint(resume, progress)
            (output / 'resume_checkpoint_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
            inherit_validation(resume.parent, output)
            status.update(completed_iterations=progress, completed_student_updates=audit['completed_student_updates'],
                resumed_from_iteration=progress)
        update('prepared', train_examples=provenance['train_size'], validation_examples=provenance['validation_size'])
        status['training_started_at'] = time.time()
        run('training', command, 'train.log')
        update('auditing')
        from main.audit_flow_validation import audit_pilot
        audit = audit_pilot(output)
        (output / 'completion_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
        run('final_evaluation', [sys.executable, '-u', '-m', 'main.evaluate_flow_t2i',
            '--checkpoint', str(output / 'student-ema'), '--manifest', str(data / 'validation.jsonl'),
            '--output_dir', str(output / 'validation'), '--limit', '64', '--teacher_count', '8', '--seed', '42'],
            'evaluation.log')
        update('complete', completed_iterations=target_iterations, completed_student_updates=target_student_updates,
            artifact_audit_passed=True, quality_verified=False)
    except Exception as error:
        update('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
