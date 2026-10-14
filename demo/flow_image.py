"""Generate or edit with a distilled flow student export."""

import argparse
from pathlib import Path

import torch

from main.flow.adapters import FlowAdapter
from main.flow.checkpoints import load_export, read_export_config
from main.flow.losses import sample_student


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--reference_images', nargs='+', default=[])
    parser.add_argument('--height', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--num_steps', type=int, choices=[1, 4], help='Must match the trained step count')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    config = read_export_config(args.checkpoint)
    if args.num_steps is not None and args.num_steps != config['num_steps']:
        parser.error('--num_steps must match the checkpoint trained step count')
    task = 'i2i' if args.reference_images else 't2i'
    if config['task'] != 'mixed' and config['task'] != task:
        parser.error(f'Checkpoint was trained for {config["task"]}; this input requests {task}')
    if config['family'] == 'qwen_image21' and Path(args.output).suffix.lower() not in ('.png', '.webp', '.tiff', '.tif'):
        parser.error('Qwen RGBA output requires PNG, WebP, or TIFF')
    if not args.prompt.strip():
        parser.error('--prompt must be nonempty')
    if any(value is not None and value < 32 for value in (args.height, args.width)):
        parser.error('--height and --width must be >= 32')
    adapter = FlowAdapter(config['family'], config['model_id'], config['revision'], args.device,
                          teacher_guidance=1.0, scheduler_config=config['scheduler_config'])
    adapter.saved_schedules = config.get('schedules', {})
    record = dict(task=task, prompt=args.prompt, reference_images=args.reference_images,
                  height=args.height or config['height'], width=args.width or config['width'])
    sample = adapter.prepare(record)
    adapter.offload_encoders()
    student = load_export(adapter, args.checkpoint, config)
    generator = torch.Generator(device=adapter.device).manual_seed(args.seed)
    with torch.inference_mode(), torch.autocast(adapter.device.type, dtype=torch.bfloat16,
                                               enabled=adapter.dtype == torch.bfloat16):
        latents = sample_student(adapter, student, sample, config['num_steps'], generator)
        image = adapter.decode(latents, sample)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f'Saved {task} image to {output}')


if __name__ == '__main__':
    main()
