"""Distribution-matching distillation for flow models, without a GAN loss."""

import argparse
from contextlib import nullcontext
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import time
import uuid

from accelerate import Accelerator
from accelerate.utils import DistributedType, set_seed
from diffusers.optimization import get_scheduler
import torch

from main.flow.adapters import FlowAdapter, MODEL_IDS
from main.flow.checkpoints import configure_trainable, export_model, prune_checkpoints, register_compact_lora_hooks
from main.flow.data import CachedConditioning, OnDemandConditioning, cache_key, prepare_cache, read_manifest
from main.flow.losses import distillation_loss, fake_score_loss, interpolate, sample_score_sigma, student_clean_sample
from main.flow.ema import StudentEMA
from main.flow.validation import evaluate_student
from main.flow.comparisons import prepare_flux_teacher_references


@lru_cache(maxsize=2)
def epoch_order(length, seed, epoch):
    return torch.randperm(length, generator=torch.Generator().manual_seed(seed + epoch)).tolist()


def batch_indices(length, iteration, microstep, batch_size, accumulation, world_size, rank, seed):
    start = ((iteration * accumulation + microstep) * world_size + rank) * batch_size
    return [epoch_order(length, seed, offset // length)[offset % length]
            for offset in range(start, start + batch_size)]


def pilot_noise_generators(seed, iteration, microstep, index, device):
    result = {}
    for name in ('rollout', 'step', 'dm', 'fake'):
        digest = hashlib.sha256(f'{seed}:{iteration}:{microstep}:{index}:{name}'.encode()).digest()
        value = int.from_bytes(digest[:8], 'little') % (2**63-1)
        result[name] = torch.Generator(device=device).manual_seed(value)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_family', choices=list(MODEL_IDS), required=True)
    parser.add_argument('--model_id')
    parser.add_argument('--revision')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--task', choices=['t2i', 'i2i', 'mixed'], default='mixed')
    parser.add_argument('--num_steps', type=int, choices=[1, 4], default=4)
    parser.add_argument('--student_train_mode', choices=['lora', 'full'], default='lora')
    parser.add_argument('--fake_train_mode', choices=['lora', 'full'], default='lora')
    parser.add_argument('--lora_rank', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--fake_lora_rank', type=int, help='Independent fake-score rank; defaults to --lora_rank')
    parser.add_argument('--fake_lora_alpha', type=int, help='Independent fake-score alpha; defaults to --lora_alpha')
    parser.add_argument('--student_lr', type=float)
    parser.add_argument('--fake_lr', type=float)
    parser.add_argument('--teacher_guidance', type=float)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--no_gradient_checkpointing', action='store_true')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--train_iters', type=int, default=10000, help='Number of fake-score optimizer updates')
    parser.add_argument('--fake_gen_ratio', type=int, default=5)
    parser.add_argument('--warmup_step', type=int, default=0,
                        help='Linear LR warm-up for both optimizers; original DMD2 uses 500 (0 preserves legacy constant LR)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--cache_dir')
    parser.add_argument('--conditioning_cache_limit', type=int,
                        help='Encode training inputs on demand, retaining at most N cache files (single GPU)')
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--keep_checkpoints', type=int, help='Retain this many completed checkpoints from this run')
    parser.add_argument('--resume', help='checkpoint-N directory containing Accelerate state')
    parser.add_argument('--validation_manifest', help='Fixed held-out inputs for short single-GPU quality pilots')
    parser.add_argument('--validation_every', type=int, default=20, help='Evaluate every N student updates during a validation pilot')
    parser.add_argument('--validation_limit', type=int, default=8)
    parser.add_argument('--validation_seeds', type=int, nargs='+', default=[42, 43])
    parser.add_argument('--fake_warmup_iters', type=int, default=0, help='Initial fake-only optimizer updates before joint training')
    parser.add_argument('--sampling_precision', choices=['fp32', 'native'], default='fp32',
                        help='Euler carry/timestep policy; native matches the pinned FLUX pipeline')
    parser.add_argument('--rollout_mode', choices=['euler', 'stochastic'], default='euler',
                        help='Euler updates or clean prediction followed by fresh independent noise')
    parser.add_argument('--score_noise_sampling', choices=['uniform', 'shifted'], default='uniform',
                        help='Legacy uniform sigma .02-.98 or FastVideo shifted integer timesteps over the full range')
    parser.add_argument('--score_noise_shift', type=float, default=8.0,
                        help='Rational timestep shift for shifted score-noise sampling (FastVideo Wan: 8)')
    parser.add_argument('--student_ema_decay', type=float, default=0.0,
                        help='Student EMA decay, updated every outer iteration from the start; 0 disables, FastVideo uses .98')
    parser.add_argument('--pilot_fixed_noise', action='store_true',
                        help='Separate reproducible rollout/selection/scoring RNG streams for controlled pilots')
    args = parser.parse_args(argv)
    if args.conditioning_cache_limit is not None and args.conditioning_cache_limit < 1:
        parser.error('--conditioning_cache_limit must be positive')
    for name in ('fake_lora_rank', 'fake_lora_alpha'):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f'--{name} must be positive')
    args.fake_lora_rank = args.fake_lora_rank or args.lora_rank
    args.fake_lora_alpha = args.fake_lora_alpha or args.lora_alpha
    for name in ('batch_size', 'gradient_accumulation_steps', 'train_iters', 'fake_gen_ratio', 'lora_rank',
                 'lora_alpha', 'save_every', 'log_every'):
        if getattr(args, name) < 1:
            parser.error(f'--{name} must be positive')
    if args.max_grad_norm <= 0:
        parser.error('--max_grad_norm must be positive')
    if args.warmup_step < 0:
        parser.error('--warmup_step must be nonnegative')
    if not 0 < args.score_noise_shift < float('inf'):
        parser.error('--score_noise_shift must be finite and positive')
    if not 0 <= args.student_ema_decay < 1:
        parser.error('--student_ema_decay must be in [0, 1)')
    if args.keep_checkpoints is not None and args.keep_checkpoints < 1:
        parser.error('--keep_checkpoints must be positive')
    if args.validation_every < 1 or args.validation_limit < 1 or args.fake_warmup_iters < 0:
        parser.error('Validation sizes/interval must be positive and fake warmup nonnegative')
    if args.validation_manifest and args.resume:
        parser.error('Quality pilots require fresh initialization so their baseline is the untrained base')
    if args.sampling_precision == 'native' and args.model_family != 'flux_klein':
        parser.error('Native carry/timestep policy is currently validated for FLUX Klein only')
    for name in ('student_lr', 'fake_lr'):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f'--{name} must be positive')
    if args.teacher_guidance is not None and args.teacher_guidance < 1:
        parser.error('--teacher_guidance must be >= 1')
    args.model_id = args.model_id or MODEL_IDS[args.model_family]
    args.cache_dir = args.cache_dir or str(Path(args.output_dir) / 'conditioning_cache')
    args.student_lr = args.student_lr or (1e-5 if args.student_train_mode == 'lora' else 5e-7)
    args.fake_lr = args.fake_lr or (1e-5 if args.fake_train_mode == 'lora' else 5e-7)
    return args


def resume_config_matches(saved, current):
    # These fields affect the next batch, optimizer, loss, or sampling schedule.
    fields = ('model_family', 'model_id', 'revision', 'task', 'num_steps', 'student_train_mode', 'fake_train_mode',
              'lora_rank', 'lora_alpha', 'student_lr', 'fake_lr', 'teacher_guidance', 'batch_size',
              'gradient_accumulation_steps', 'fake_gen_ratio', 'seed', 'height', 'width', 'max_grad_norm',
              'no_gradient_checkpointing')
    differences = [name for name in fields if saved.get(name) != current.get(name)]
    if saved.get('warmup_step', 0) != current.get('warmup_step', 0):
        differences.append('warmup_step')
    if saved.get('fake_warmup_iters', 0) != current.get('fake_warmup_iters', 0):
        differences.append('fake_warmup_iters')
    if saved.get('sampling_precision', 'fp32') != current.get('sampling_precision', 'fp32'):
        differences.append('sampling_precision')
    for name, default in [('rollout_mode', 'euler'), ('pilot_fixed_noise', False),
                          ('score_noise_sampling', 'uniform'), ('score_noise_shift', 8.0),
                          ('student_ema_decay', 0.0)]:
        if saved.get(name, default) != current.get(name, default):
            differences.append(name)
    for name, fallback in [('fake_lora_rank', 'lora_rank'), ('fake_lora_alpha', 'lora_alpha')]:
        if saved.get(name, saved.get(fallback)) != current.get(name, current.get(fallback)):
            differences.append(name)
    if differences:
        raise ValueError(f'Resume configuration differs for: {", ".join(differences)}')


def main(argv=None):
    args = parse_args(argv)
    if args.pilot_fixed_noise and (not args.validation_manifest or getattr(args, 'loss_weighting', 'normalized') != 'normalized'):
        raise ValueError('Fixed RNG streams currently require a fresh normalized-loss quality pilot')
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision='bf16')
    if accelerator.distributed_type not in (DistributedType.NO, DistributedType.MULTI_GPU):
        raise ValueError('Flow training currently supports single-GPU and CUDA DDP launches, not model sharding')
    if accelerator.device.type != 'cuda':
        raise RuntimeError('Real-checkpoint flow training requires a CUDA GPU with BF16 support')
    if args.validation_manifest and accelerator.num_processes != 1:
        raise ValueError('Quality pilots currently require a single CUDA process')
    if args.conditioning_cache_limit and accelerator.num_processes != 1:
        raise ValueError('On-demand conditioning currently requires a single CUDA process')
    set_seed(args.seed + accelerator.process_index)
    output = Path(args.output_dir)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'student').exists() and not args.resume:
            raise ValueError('Output already contains a student; use a new directory or --resume')
    accelerator.wait_for_everyone()
    records = read_manifest(args.manifest, args.task, args.height, args.width)
    # Pin the base checkpoint before constructing cache identities and models.
    adapter = FlowAdapter(args.model_family, args.model_id, args.revision, accelerator.device,
                          teacher_guidance=args.teacher_guidance)
    args.revision = adapter.revision
    args.teacher_guidance = adapter.teacher_guidance
    adapter.sampling_precision = args.sampling_precision
    adapter.rollout_mode = args.rollout_mode
    adapter.score_noise_sampling = args.score_noise_sampling
    adapter.score_noise_shift = args.score_noise_shift
    if accelerator.is_main_process and not args.conditioning_cache_limit:
        prepare_cache(adapter, records, args.cache_dir)
    accelerator.wait_for_everyone()
    paths = [Path(args.cache_dir) / (cache_key(record, adapter.identity) + '.pt') for record in records]
    dataset = (OnDemandConditioning(adapter, records, paths, args.conditioning_cache_limit)
               if args.conditioning_cache_limit else CachedConditioning(paths))
    validation_records, validation_samples = [], []
    if args.validation_manifest:
        validation_records = read_manifest(args.validation_manifest, args.task, args.height, args.width)[:args.validation_limit]
        validation_paths = prepare_cache(adapter, validation_records, args.cache_dir)
        validation_samples = [CachedConditioning(validation_paths)[i] for i in range(len(validation_paths))]
    adapter.offload_encoders()
    # Cache misses run frozen encoders and create dummy latent shapes. Their RNG
    # consumption must not change adapter initialization or the training stream.
    set_seed(args.seed + accelerator.process_index)
    teacher = adapter.load_transformer().requires_grad_(False).eval()
    student = configure_trainable(adapter.load_transformer(), args.student_train_mode, args.lora_rank,
                                  args.lora_alpha, not args.no_gradient_checkpointing)
    fake = configure_trainable(adapter.load_transformer(), args.fake_train_mode, args.fake_lora_rank,
                               args.fake_lora_alpha, not args.no_gradient_checkpointing)
    student_optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.student_lr)
    fake_optimizer = torch.optim.AdamW([p for p in fake.parameters() if p.requires_grad], lr=args.fake_lr)
    student_scheduler = fake_scheduler = None
    if args.warmup_step:
        student_scheduler = get_scheduler('constant_with_warmup', optimizer=student_optimizer,
            num_warmup_steps=args.warmup_step, num_training_steps=args.train_iters)
        fake_scheduler = get_scheduler('constant_with_warmup', optimizer=fake_optimizer,
            num_warmup_steps=args.warmup_step, num_training_steps=args.train_iters)
        student, fake, student_optimizer, fake_optimizer, student_scheduler, fake_scheduler = accelerator.prepare(
            student, fake, student_optimizer, fake_optimizer, student_scheduler, fake_scheduler)
    else:
        student, fake, student_optimizer, fake_optimizer = accelerator.prepare(student, fake, student_optimizer, fake_optimizer)
    student_ema = StudentEMA(accelerator.unwrap_model(student), args.student_ema_decay) if args.student_ema_decay else None
    if student_ema is not None:
        accelerator.register_for_checkpointing(student_ema)
    metadata = dict(adapter.metadata(), num_steps=args.num_steps, task=args.task, height=args.height, width=args.width,
                    teacher_guidance=args.teacher_guidance, sampling_precision=args.sampling_precision,
                    rollout_mode=args.rollout_mode, score_noise_sampling=args.score_noise_sampling,
                    score_noise_shift=args.score_noise_shift, student_ema_decay=args.student_ema_decay)
    schedules = {}
    for index, record in enumerate(records):
        key = f'{record["height"]}x{record["width"]}:{args.num_steps}'
        if key not in schedules:
            schedules[key] = adapter.schedule(dataset[index], args.num_steps).cpu().tolist()
    metadata['schedules'] = schedules
    adapter.saved_schedules = schedules
    run_config = dict(vars(args), world_size=accelerator.num_processes,
                      data_keys=[path.stem for path in paths], diffusers_version=adapter.identity['diffusers_version'])
    if args.student_train_mode == args.fake_train_mode == 'lora':
        register_compact_lora_hooks(accelerator, metadata)
    run_id = uuid.uuid4().hex
    start = 0
    if args.resume:
        saved = json.loads((Path(args.resume) / 'training_config.json').read_text())
        run_id = saved.get('run_id', run_id)
        resume_config_matches(saved, run_config)
        for key in ('world_size', 'data_keys', 'diffusers_version'):
            if saved[key] != run_config[key]:
                raise ValueError(f'Resume requires unchanged {key}')
        accelerator.load_state(args.resume)
        start = json.loads((Path(args.resume) / 'progress.json').read_text())['completed_iterations']
    if start >= args.train_iters:
        raise ValueError('--train_iters must exceed the number of completed iterations')
    run_config['run_id'] = run_id

    def save(completed):
        directory = output / f'checkpoint-{completed:06d}'
        accelerator.save_state(str(directory))
        if accelerator.is_main_process:
            (directory / 'training_config.json').write_text(json.dumps(run_config, indent=2) + '\n')
            (directory / 'progress.json').write_text(json.dumps({'completed_iterations': completed}) + '\n')
            if not (directory / 'compact_lora.json').exists():
                export_model(accelerator.unwrap_model(student), directory / 'student', metadata, args.student_train_mode)
            export_model(accelerator.unwrap_model(student), output / 'student', metadata, args.student_train_mode)
            if student_ema is not None:
                ema_metadata = dict(metadata, weight_source='ema', ema_updates=student_ema.num_updates)
                with student_ema.apply(accelerator.unwrap_model(student)):
                    export_model(accelerator.unwrap_model(student), directory / 'student-ema', ema_metadata, args.student_train_mode)
                    export_model(accelerator.unwrap_model(student), output / 'student-ema', ema_metadata, args.student_train_mode)
            (directory / 'checkpoint_complete.json').write_text(json.dumps(dict(run_id=run_id, completed_iterations=completed)) + '\n')
            prune_checkpoints(output, args.keep_checkpoints, run_id)
        accelerator.wait_for_everyone()

    training_started = time.monotonic()
    baseline = None
    comparison_options = {}
    if validation_samples and args.model_family == 'flux_klein' and args.task == 't2i':
        comparison_options['teacher_references'] = prepare_flux_teacher_references(adapter, teacher,
            validation_records, output / 'teacher-references', args.validation_seeds)
    if validation_samples:
        baseline_result = evaluate_student(adapter, accelerator.unwrap_model(student), validation_samples,
            validation_records, output / 'validation-000000', args.validation_seeds, **comparison_options)
        baseline = baseline_result['results']
        accelerator.print('PILOT_BASELINE_READY', len(baseline))
    student_updates = max(0, (start - args.fake_warmup_iters + args.fake_gen_ratio - 1) // args.fake_gen_ratio)
    for iteration in range(start, args.train_iters):
        student_turn = iteration >= args.fake_warmup_iters and (iteration - args.fake_warmup_iters) % args.fake_gen_ratio == 0
        student_lr, fake_lr = student_optimizer.param_groups[0]['lr'], fake_optimizer.param_groups[0]['lr']
        statistics = torch.zeros(3, device=accelerator.device)
        student_parameter_gradient_norm = None
        fake_parameter_gradient_norm = None
        for microstep in range(args.gradient_accumulation_steps):
            indices = batch_indices(len(dataset), iteration, microstep, args.batch_size,
                                    args.gradient_accumulation_steps, accelerator.num_processes,
                                    accelerator.process_index, args.seed)
            with accelerator.accumulate(student, fake):
                for index in indices:
                    sample = dataset[index]
                    generators = pilot_noise_generators(args.seed, iteration, microstep, index, accelerator.device) if args.pilot_fixed_noise else None
                    selected_step = int(torch.randint(args.num_steps, (), device=accelerator.device, generator=generators['step'])) if generators else None
                    with accelerator.autocast():
                        if student_turn:
                            extra = dict(generator=generators['rollout'], scoring_generator=generators['dm'], selected_step=selected_step) if generators else {}
                            dm_loss, clean, gradient = distillation_loss(adapter, student, teacher, fake, sample,
                                args.num_steps, args.teacher_guidance, **extra)
                            if not torch.isfinite(dm_loss):
                                raise FloatingPointError('Nonfinite student distribution-matching loss')
                            accelerator.backward(dm_loss / args.batch_size)
                            statistics[0] += dm_loss.detach()
                            statistics[2] += gradient.norm().detach()
                        else:
                            with torch.no_grad():
                                clean = student_clean_sample(adapter, student, sample, args.num_steps,
                                    generators['rollout'], selected_step) if generators else student_clean_sample(adapter, student, sample, args.num_steps)
                        noise = torch.randn(clean.shape, device=clean.device, generator=generators['fake']) if generators else torch.randn_like(clean)
                        sigma = sample_score_sigma(adapter, clean.shape[0], clean.device, generators['fake'] if generators else None)
                        noisy = interpolate(clean.detach(), noise, sigma)
                        fake_v = adapter.predict(fake, noisy, sigma, sample['condition'])
                        fake_loss = fake_score_loss(fake_v, clean, noise)
                        if not torch.isfinite(fake_loss):
                            raise FloatingPointError('Nonfinite fake-score loss')
                        accelerator.backward(fake_loss / args.batch_size)
                        statistics[1] += fake_loss.detach()
                if accelerator.sync_gradients:
                    if student_turn:
                        student_parameter_gradient_norm = accelerator.clip_grad_norm_(student.parameters(), args.max_grad_norm)
                    fake_parameter_gradient_norm = accelerator.clip_grad_norm_(fake.parameters(), args.max_grad_norm)
                    if not torch.isfinite(fake_parameter_gradient_norm) or (student_turn and not torch.isfinite(student_parameter_gradient_norm)):
                        raise FloatingPointError('Nonfinite optimizer parameter gradient')
                if student_turn:
                    student_optimizer.step()
                    student_optimizer.zero_grad()
                fake_optimizer.step()
                fake_optimizer.zero_grad()
        if student_ema is not None:
            student_ema.update(accelerator.unwrap_model(student))
        if student_scheduler is not None:
            # Original DMD2 advances BOTH schedules every outer iteration,
            # including iterations without a student optimizer update. Keep
            # these outside the gradient-accumulation microstep loop.
            student_scheduler.step()
            fake_scheduler.step()
        completed = iteration + 1
        student_updates += int(student_turn)
        if student_turn or completed % args.log_every == 0 or completed == 1:
            means = accelerator.reduce(statistics, reduction='mean') / (args.batch_size * args.gradient_accumulation_steps)
            accelerator.print(json.dumps(dict(iteration=completed, student_updated=student_turn,
                dm_loss=float(means[0]), fake_loss=float(means[1]), dm_gradient_norm=float(means[2]),
                student_parameter_gradient_norm=float(student_parameter_gradient_norm) if student_turn else None,
                fake_parameter_gradient_norm=float(fake_parameter_gradient_norm),
                completed_student_updates=student_updates,
                student_lr=student_lr, fake_lr=fake_lr,
                student_ema_updates=student_ema.num_updates if student_ema is not None else 0,
                seconds_per_iteration=(time.monotonic() - training_started) / (completed - start),
                peak_gpu_memory_gib=torch.cuda.max_memory_allocated(accelerator.device) / 2 ** 30)))
        if validation_samples and ((student_turn and student_updates % args.validation_every == 0) or completed == args.train_iters):
            with student_ema.apply(accelerator.unwrap_model(student)) if student_ema is not None else nullcontext():
                validation = evaluate_student(adapter, accelerator.unwrap_model(student), validation_samples,
                    validation_records, output / f'validation-{completed:06d}', args.validation_seeds, baseline,
                    weight_source='ema' if student_ema is not None else 'raw', **comparison_options)
            with (output / 'validation_checks.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(completed_iterations=completed, completed_student_updates=student_updates,
                    artifact_gate_passed=validation['artifact_gate_passed'],
                    directory=str((output / f'validation-{completed:06d}').resolve()))) + '\n')
            accelerator.print('PILOT_VALIDATION', completed,
                              'PASS' if validation['artifact_gate_passed'] else 'ARTIFACT_WARNING')
        if completed % args.save_every == 0 or completed == args.train_iters:
            save(completed)
    if validation_samples:
        (output / 'pilot_summary.json').write_text(json.dumps(dict(completed_iterations=completed,
            completed_student_updates=student_updates, stopped_for_artifacts=False,
            validation_policy='report_only',
            reached_requested_updates=completed == args.train_iters,
            quality_verified=False, note='Inspect held-out comparisons; passing the artifact gate alone does not establish quality.'), indent=2) + '\n')
    accelerator.end_training()


if __name__ == '__main__':
    main()
