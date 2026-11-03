"""Evaluate a flow text-to-image export against its native teacher."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
import torch

from main.flow.adapters import FlowAdapter
from main.flow.checkpoints import load_export, read_export_config
from main.flow.data import read_manifest
from main.flow.losses import sample_student
from main.flow.validation import image_statistics
from main.flow.comparisons import save_comparison, native_teacher_guidance, native_teacher_steps


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
    if config['task'] != 't2i':
        parser.error('Expected a text-to-image export')
    teacher_steps = native_teacher_steps(config['family'])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    adapter = FlowAdapter(config['family'], config['model_id'], config['revision'], 'cuda',
                          teacher_guidance=1, scheduler_config=config['scheduler_config'])
    adapter.saved_schedules = config.get('schedules', {})
    student = load_export(adapter, args.checkpoint, config)
    teacher = adapter.load_transformer().requires_grad_(False).eval() if args.teacher_count else None
    records = read_manifest(args.manifest, 't2i', config['height'], config['width'])[:args.limit]
    results = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for index, record in enumerate(records):
            sample = adapter.prepare(record)
            generator = torch.Generator(device='cuda').manual_seed(args.seed)
            latents = sample_student(adapter, student, sample, config['num_steps'], generator)
            if not torch.isfinite(latents).all():
                raise FloatingPointError('Nonfinite text-to-image output')
            image = adapter.decode(latents, sample)
            image.save(output / f'{index:03d}-student.png')
            panels = [(f'Student ({config["num_steps"]} steps)', image)]
            if index < args.teacher_count:
                original_transformer, original_scheduler = adapter.pipe.transformer, adapter.pipe.scheduler
                try:
                    adapter.pipe.transformer = teacher
                    adapter.pipe.scheduler = original_scheduler.__class__.from_config(original_scheduler.config)
                    adapter.pipe.to('cuda')
                    adapter.pipe.set_progress_bar_config(disable=True)
                    reference = adapter.pipe(prompt=record['prompt'], height=record['height'], width=record['width'],
                        **native_teacher_guidance(config['family'], config['teacher_guidance']),
                        num_inference_steps=teacher_steps,
                        generator=torch.Generator(device='cuda').manual_seed(args.seed)).images[0]
                finally:
                    adapter.pipe.transformer, adapter.pipe.scheduler = original_transformer, original_scheduler
                reference.save(output / f'{index:03d}-teacher.png')
                panels.insert(0, (f'Teacher ({teacher_steps} steps, CFG {config["teacher_guidance"]})', reference))
            save_comparison(output / f'{index:03d}-comparison.png', panels, record['prompt'], args.seed)
            results.append(dict(index=index, prompt=record['prompt'], references=[], seed=args.seed,
                student_steps=config['num_steps'], teacher_steps=teacher_steps if index < args.teacher_count else None,
                statistics=image_statistics(image)))
            (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            adapter.offload_encoders()
            print(f'Validation: {index+1}/{len(records)}', flush=True)
    (output / 'summary.json').write_text(json.dumps(dict(completed=True, examples=len(results),
        teacher_comparisons=min(len(results), args.teacher_count), seed=args.seed, task='t2i',
        quality_verified=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
