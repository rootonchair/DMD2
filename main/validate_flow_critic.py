"""Freeze the real base student and test critic learning on fixed generated samples."""

import argparse
import json
from pathlib import Path
import time

import torch

from main.flow.adapters import FlowAdapter
from main.flow.checkpoints import configure_trainable, export_model
from main.flow.data import CachedConditioning, prepare_cache, read_manifest, map_tensors
from main.flow.losses import fake_score_loss, interpolate, student_clean_sample


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--train_manifest', default='data/flux-klein-i2i-hqedit/train/manifest.jsonl')
    parser.add_argument('--validation_manifest', default='data/flux-klein-i2i-hqedit/validation/manifest.jsonl')
    parser.add_argument('--cache_dir', default='.cache/flux-klein-i2i-conditioning')
    parser.add_argument('--updates', type=int, default=200)
    parser.add_argument('--train_size', type=int, default=16)
    parser.add_argument('--validation_size', type=int, default=8)
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--mode', choices=['lora', 'full'], default='lora')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if min(args.updates, args.train_size, args.validation_size, args.rank, args.accumulation) < 1 or args.lr <= 0:
        parser.error('sizes, rank, accumulation and learning rate must be positive')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'critic.json').exists():
        parser.error('Use a new output directory for each independent critic experiment')
    torch.manual_seed(args.seed)
    adapter = FlowAdapter('flux_klein', revision='a3b4f4849157f664bdbc776fd7453c2783562f4d', teacher_guidance=4)
    records = {name: read_manifest(path, task='i2i', height=512, width=512)[:size]
               for name, path, size in [('train', args.train_manifest, args.train_size),
                                        ('validation', args.validation_manifest, args.validation_size)]}
    if len(records['train']) != args.train_size or len(records['validation']) != args.validation_size:
        raise ValueError('Requested split sizes are not available')
    if {r['reference_images'][0] for r in records['train']} & {r['reference_images'][0] for r in records['validation']}:
        raise ValueError('Critic train and validation sources overlap')
    datasets = {name: CachedConditioning(prepare_cache(adapter, rows, args.cache_dir)) for name, rows in records.items()}
    adapter.offload_encoders()
    student = adapter.load_transformer().requires_grad_(False).eval()
    pool = {}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for name, dataset in datasets.items():
            pool[name] = []
            for i in range(len(dataset)):
                sample = dataset[i]
                generator = torch.Generator(device='cuda').manual_seed(args.seed + i)
                clean = student_clean_sample(adapter, student, sample, 4, generator)
                if not torch.isfinite(clean).all():
                    raise FloatingPointError('Frozen student generated nonfinite clean sample')
                pool[name].append((sample, clean.cpu()))
    # The student is never in the optimizer and is offloaded for all critic updates.
    student.cpu()
    versions = {name: p._version for name, p in student.named_parameters()}
    fake = configure_trainable(adapter.load_transformer(), args.mode, args.rank, args.rank, True)
    optimizer = torch.optim.AdamW([p for p in fake.parameters() if p.requires_grad], lr=args.lr)
    sigmas = [.05, .2, .5, .8, .95]

    @torch.no_grad()
    def evaluate():
        fake.eval()
        metrics = {}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for name, examples in pool.items():
                by_sigma = []
                for sigma_index, sigma in enumerate(sigmas):
                    errors = []
                    for i, (sample, cpu_clean) in enumerate(examples):
                        clean = cpu_clean.cuda()
                        noise = torch.randn(clean.shape, device='cuda', generator=torch.Generator(device='cuda').manual_seed(999 + i + sigma_index * 100))
                        prediction = adapter.predict(fake, interpolate(clean, noise, sigma), sigma, sample['condition'])
                        errors.append(float(fake_score_loss(prediction, clean, noise)))
                    by_sigma.append(dict(sigma=sigma, mse=sum(errors) / len(errors)))
                metrics[name] = dict(by_sigma=by_sigma, mean_mse=sum(x['mse'] for x in by_sigma) / len(by_sigma))
        fake.train()
        return metrics

    baseline = evaluate()
    history = [dict(update=0, metrics=baseline)]
    print('CRITIC_BASELINE', json.dumps(baseline), flush=True)
    started = time.monotonic()
    for update in range(1, args.updates + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0
        for microstep in range(args.accumulation):
            index = ((update - 1) * args.accumulation + microstep) % len(pool['train'])
            sample, cpu_clean = pool['train'][index]
            clean = cpu_clean.cuda()
            generator = torch.Generator(device='cuda').manual_seed(args.seed + update * args.accumulation + microstep)
            sigma = torch.rand(1, device='cuda', generator=generator) * .96 + .02
            noise = torch.randn(clean.shape, device='cuda', generator=generator)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = adapter.predict(fake, interpolate(clean, noise, sigma), sigma, sample['condition'])
                loss = fake_score_loss(prediction, clean, noise)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite critic loss')
            (loss / args.accumulation).backward()
            loss_sum += float(loss.detach()) / args.accumulation
        gradient_norm = torch.nn.utils.clip_grad_norm_(fake.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if update % 10 == 0:
            print(json.dumps(dict(update=update, loss=loss_sum, parameter_gradient_norm=float(gradient_norm))), flush=True)
        if update % 50 == 0 or update == args.updates:
            metrics = evaluate()
            history.append(dict(update=update, metrics=metrics))
            print('CRITIC_EVALUATION', update, json.dumps(metrics), flush=True)
    unchanged = all(p._version == versions[name] and p.grad is None and not p.requires_grad for name, p in student.named_parameters())
    final = history[-1]['metrics']
    ratio = final['validation']['mean_mse'] / baseline['validation']['mean_mse']
    train_ratio = final['train']['mean_mse'] / baseline['train']['mean_mse']
    bin_ratios = [new['mse'] / old['mse'] for new, old in zip(final['validation']['by_sigma'], baseline['validation']['by_sigma'])]
    # Predeclared diagnostic gate, not a scientific claim of adequate score accuracy.
    passed = unchanged and ratio < .95 and train_ratio < .95 and max(bin_ratios) < 1.2
    metadata = dict(adapter.metadata(), num_steps=4, task='i2i', height=512, width=512,
                    teacher_guidance=4, schedules={'512x512:4': adapter.schedule(pool['train'][0][0], 4).cpu().tolist()})
    export_model(fake, output / 'fake', metadata, args.mode)
    result = dict(passed=passed, completed=True, config=vars(args), frozen_student_unchanged=unchanged,
                  train_mse_ratio=train_ratio, validation_mse_ratio=ratio, validation_bin_ratios=bin_ratios,
                  history=history, elapsed_seconds=time.monotonic()-started,
                  gate='Mean train/held-out MSE improves at least 5%; no held-out sigma bin worsens more than 20%; frozen student unchanged.',
                  scope='Fixed clean samples from the frozen base student at random rollout steps; no claim about moving-distribution tracking.')
    (output / 'critic.json').write_text(json.dumps(result, indent=2) + '\n')
    print('CRITIC_RESULT', 'PASS' if passed else 'FAIL', 'validation ratio', ratio, flush=True)


if __name__ == '__main__':
    main()
