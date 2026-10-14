"""Export held-out source/student/teacher comparisons for a flow student."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw
import torch

from main.flow.adapters import FlowAdapter
from main.flow.checkpoints import load_export, read_export_config
from main.flow.data import read_manifest
from main.flow.losses import sample_student


def comparison_grid(source, student, teacher=None):
    images = [('Source', source), ('Teacher (50 steps)', teacher), ('Student (4 steps)', student)] if teacher else [('Source', source), ('Student (4 steps)', student)]
    canvas = Image.new('RGB', (512 * len(images), 544), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(images):
        canvas.paste(ImageOps.pad(image.convert('RGB'), (512, 512)), (index * 512, 32))
        draw.text((index * 512 + 8, 8), label, fill='black')
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--limit', type=int, default=64)
    parser.add_argument('--teacher_count', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.limit < 1 or args.teacher_count < 0:
        parser.error('limit must be positive and teacher_count nonnegative')
    config = read_export_config(args.checkpoint)
    if config['family'] != 'flux_klein':
        parser.error('This run evaluator currently supports FLUX Klein only')
    records = read_manifest(args.manifest, task='i2i', height=config['height'], width=config['width'])[:args.limit]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    adapter = FlowAdapter(config['family'], config['model_id'], config['revision'], 'cuda',
                          teacher_guidance=1, scheduler_config=config['scheduler_config'])
    adapter.saved_schedules = config.get('schedules', {})
    student = load_export(adapter, args.checkpoint, config)
    teacher = adapter.load_transformer().requires_grad_(False).eval() if args.teacher_count else None
    adapter.keep_encoders_on_device = True
    results = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for index, record in enumerate(records):
            sample = adapter.prepare(record)
            generator = torch.Generator(device='cuda').manual_seed(args.seed)
            latents = sample_student(adapter, student, sample, config['num_steps'], generator)
            if not torch.isfinite(latents).all():
                raise FloatingPointError(f'Nonfinite student output at validation record {index}')
            edited = adapter.decode(latents, sample)
            edited.save(output / f'{index:03d}-student.png')
            with Image.open(record['reference_images'][0]) as image:
                source = image.convert('RGB')
            source.save(output / f'{index:03d}-source.png')
            teacher_image = None
            if index < args.teacher_count:
                adapter.pipe.transformer = teacher
                adapter.pipe.to('cuda')
                adapter.pipe.set_progress_bar_config(disable=True)
                references = []
                for path in record['reference_images']:
                    with Image.open(path) as image:
                        references.append(image.convert('RGB'))
                teacher_image = adapter.pipe(prompt=record['prompt'], image=references,
                    height=record['height'], width=record['width'], guidance_scale=config['teacher_guidance'],
                    num_inference_steps=50, generator=torch.Generator(device='cuda').manual_seed(args.seed)).images[0]
                teacher_image.save(output / f'{index:03d}-teacher.png')
            comparison_grid(source, edited, teacher_image).save(output / f'{index:03d}-comparison.png')
            result = dict(index=index, prompt=record['prompt'], references=record['reference_images'], seed=args.seed,
                          student_steps=config['num_steps'], teacher_steps=50 if teacher_image else None)
            results.append(result)
            (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            print(f'Validation: {index + 1}/{len(records)}', flush=True)
    (output / 'summary.json').write_text(json.dumps(dict(checkpoint=str(Path(args.checkpoint).resolve()),
        examples=len(results), teacher_comparisons=min(len(results), args.teacher_count),
        seed=args.seed, completed=True), indent=2) + '\n')
    print('EVALUATION_READY', output, flush=True)


if __name__ == '__main__':
    main()
