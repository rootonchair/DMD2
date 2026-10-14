"""Compare a trained critic with the base critic on the current student distribution.

Held-out conditional denoising risk diagnoses distribution tracking; it does not
provide oracle scores or establish image quality.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps

from main.flow.adapters import FlowAdapter
from main.flow.checkpoints import load_export, read_export_config
from main.flow.data import CachedConditioning, prepare_cache, read_manifest
from main.flow.losses import fake_score_loss, interpolate, reconstruct_clean, sample_student, student_clean_sample


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Completed pilot checkpoint containing student and fake exports')
    parser.add_argument('--manifest', default='data/flux-klein-i2i-hqedit/validation/manifest.jsonl')
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43])
    parser.add_argument('--output', required=True)
    parser.add_argument('--clean_mode', choices=['mixture', 'final'], default='mixture')
    parser.add_argument('--teacher_examples', type=int, nargs='*', default=[],
                        help='Held-out indices for teacher/fake clean-estimate images at each score sigma')
    args = parser.parse_args(argv)
    checkpoint = Path(args.checkpoint)
    config = read_export_config(checkpoint / 'student')
    if config['family'] != 'flux_klein' or config['task'] != 'i2i' or config['num_steps'] != 4:
        parser.error('This probe currently requires a four-step FLUX I2I checkpoint')
    if args.limit < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Use a positive limit and distinct sampling seeds')
    records = read_manifest(args.manifest, 'i2i', config['height'], config['width'])[:args.limit]
    adapter = FlowAdapter('flux_klein', config['model_id'], config['revision'], teacher_guidance=config['teacher_guidance'])
    adapter.saved_schedules = config['schedules']
    paths = prepare_cache(adapter, records, '.cache/flux-klein-i2i-conditioning')
    dataset = CachedConditioning(paths)
    adapter.offload_encoders()
    student = load_export(adapter, checkpoint / 'student', config)
    fake = load_export(adapter, checkpoint / 'fake')
    base = adapter.load_transformer().requires_grad_(False).eval()
    sigmas = [.05, .2, .5, .8, .95]
    by_sigma = {sigma: [] for sigma in sigmas}
    image_directory = Path(args.output).with_suffix('')
    if args.teacher_examples:
        image_directory.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for index in range(len(dataset)):
            sample = dataset[index]
            for seed in args.seeds:
                sampler = student_clean_sample if args.clean_mode == 'mixture' else sample_student
                clean = sampler(adapter, student, sample, 4,
                    torch.Generator(device='cuda').manual_seed(seed + index * 100))
                for sigma_index, sigma in enumerate(sigmas):
                    noise = torch.randn(clean.shape, device='cuda',
                        generator=torch.Generator(device='cuda').manual_seed(999 + seed + index * 100 + sigma_index * 10000))
                    noisy = interpolate(clean, noise, sigma)
                    trained_prediction = adapter.predict(fake, noisy, sigma, sample['condition'])
                    base_prediction = adapter.predict(base, noisy, sigma, sample['condition'])
                    trained_error = float(fake_score_loss(trained_prediction, clean, noise))
                    base_error = float(fake_score_loss(base_prediction, clean, noise))
                    if not torch.isfinite(torch.tensor([trained_error, base_error])).all():
                        raise FloatingPointError('Nonfinite distribution-tracking error')
                    by_sigma[sigma].append(dict(trained_mse=trained_error, base_mse=base_error))
                    if index in args.teacher_examples and seed == args.seeds[0]:
                        guided = adapter.predict(base, noisy, sigma, sample['condition'], config['teacher_guidance'])
                        student_image = adapter.decode(clean, sample)
                        teacher_image = adapter.decode(reconstruct_clean(noisy, guided, sigma), sample)
                        fake_image = adapter.decode(reconstruct_clean(noisy, trained_prediction, sigma), sample)
                        panels = [('Student clean: '+args.clean_mode, student_image),
                                  (f'Teacher clean estimate: sigma {sigma}', teacher_image),
                                  (f'Fake clean estimate: sigma {sigma}', fake_image)]
                        canvas = Image.new('RGB', (1536, 544), 'white')
                        draw = ImageDraw.Draw(canvas)
                        for column, (label, image) in enumerate(panels):
                            draw.text((column*512+8, 8), label, fill='black')
                            canvas.paste(ImageOps.pad(image.convert('RGB'), (512,512)), (column*512,32))
                        canvas.save(image_directory / f'{index:03d}-sigma{sigma:.2f}.png')
    rows = []
    for sigma, errors in by_sigma.items():
        trained = sum(row['trained_mse'] for row in errors) / len(errors)
        base_error = sum(row['base_mse'] for row in errors) / len(errors)
        rows.append(dict(sigma=sigma, trained_mse=trained, base_mse=base_error, ratio=trained/base_error))
    trained = sum(row['trained_mse'] for row in rows) / len(rows)
    baseline = sum(row['base_mse'] for row in rows) / len(rows)
    result = dict(completed=True, checkpoint=str(checkpoint.resolve()), examples=len(records), seeds=args.seeds, clean_mode=args.clean_mode,
        by_sigma=rows, mean_trained_mse=trained, mean_base_mse=baseline, trained_to_base_ratio=trained/baseline,
        scope='Held-out denoising risk on the selected current-student distribution; lower error does not establish correct oracle scores or editing quality. Clean-estimate images are posterior estimates at score sigmas, not native 50-step teacher renders.')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
