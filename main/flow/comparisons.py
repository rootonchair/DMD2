"""Prompt-labelled comparisons and reusable native FLUX teacher references."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps, PngImagePlugin
import torch


def comparison_image(panels, prompt, seed, iteration=None):
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 18)
    except OSError:
        font = ImageFont.load_default()
    width = 512 * len(panels)
    lines, line = [], ''
    for word in ('Prompt: ' + prompt).split():
        candidate = (line + ' ' + word).strip()
        if font.getlength(candidate) <= width - 24:
            line = candidate
            continue
        if line:
            lines.append(line)
        line = ''
        for character in word:
            if font.getlength(line + character) > width - 24:
                lines.append(line)
                line = ''
            line += character
    if line:
        lines.append(line)
    image_y = 12 + 24 * len(lines) + 52
    canvas = Image.new('RGB', (width, image_y + 512), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, text in enumerate(lines):
        draw.text((12, 8 + index * 24), text, font=font, fill='black')
    detail = f'Seed: {seed}' + (f' | Critic updates: {iteration}' if iteration is not None else '')
    draw.text((12, image_y - 48), detail, font=font, fill='black')
    bounds = []
    for column, (label, panel) in enumerate(panels):
        x = column * 512
        draw.text((x + 12, image_y - 24), label, font=font, fill='black')
        canvas.paste(ImageOps.pad(panel.convert('RGB'), (512, 512)), (x, image_y))
        bounds.append(dict(label=label, box=[x, image_y, x + 512, image_y + 512]))
    return canvas, dict(prompt=prompt, seed=seed, iteration=iteration,
        prompt_lines=lines, image_y=image_y, panels=bounds)


def save_comparison(path, panels, prompt, seed, iteration=None):
    path = Path(path)
    canvas, layout = comparison_image(panels, prompt, seed, iteration)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text('Prompt', prompt)
    metadata.add_text('Comparison', json.dumps(layout))
    temporary = path.with_suffix('.tmp')
    canvas.save(temporary, format='PNG', pnginfo=metadata)
    temporary.replace(path)
    return layout


@torch.inference_mode()
def prepare_flux_teacher_references(adapter, teacher, records, directory, seeds):
    if adapter.family != 'flux_klein':
        raise ValueError('Native teacher comparisons currently require FLUX Klein')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(model_id=adapter.model_id, revision=adapter.revision,
        teacher_guidance=adapter.teacher_guidance, teacher_steps=50, records=records, seeds=list(seeds))
    summary_path = directory / 'summary.json'
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if previous['identity'] != identity:
            raise ValueError('Teacher reference inputs differ from their cached identity')
        if previous['completed'] and all(Path(row['image']).is_file() for row in previous['results']):
            return previous['results']
    original_transformer, original_scheduler = adapter.pipe.transformer, adapter.pipe.scheduler
    rows = []
    devices = [adapter.device] if adapter.device.type == 'cuda' else []
    try:
        # Reference rendering must not consume the optimizer training RNG stream.
        with torch.random.fork_rng(devices=devices), torch.autocast(adapter.device.type, dtype=adapter.dtype):
            adapter.pipe.transformer = teacher
            adapter.pipe.scheduler = original_scheduler.__class__.from_config(original_scheduler.config)
            adapter.pipe.to(adapter.device)
            adapter.pipe.set_progress_bar_config(disable=True)
            for index, record in enumerate(records):
                for seed in seeds:
                    references = []
                    for reference in record['reference_images']:
                        with Image.open(reference) as source:
                            references.append(source.convert('RGB'))
                    image = adapter.pipe(prompt=record['prompt'], image=references or None,
                        height=record['height'], width=record['width'], num_inference_steps=50,
                        guidance_scale=adapter.teacher_guidance,
                        generator=torch.Generator(device=adapter.device).manual_seed(seed)).images[0]
                    path = directory / f'{index:03d}-seed{seed}.png'
                    image.save(path)
                    rows.append(dict(index=index, seed=seed, prompt=record['prompt'],
                        reference_images=record['reference_images'], image=str(path.resolve()), teacher_steps=50))
                    temporary = summary_path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(dict(identity=identity, completed=False, results=rows), indent=2) + '\n')
                    temporary.replace(summary_path)
                    print('TEACHER_REFERENCE', index, seed, flush=True)
    finally:
        adapter.pipe.transformer = original_transformer
        adapter.pipe.scheduler = original_scheduler
        adapter.offload_encoders()
    summary_path.write_text(json.dumps(dict(identity=identity, completed=True, results=rows), indent=2) + '\n')
    return rows
