"""Cheap artifact diagnostics for fixed held-out editing cases.

These reject obvious collapse, but do not establish editing quality. Comparisons
must still be inspected; cyan can be a legitimate image color in some prompts.
"""

import numpy as np
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageOps
import torch

from .comparisons import save_comparison


def image_statistics(image):
    rgb = np.asarray(image.convert('RGB'))
    cyan = (rgb[:, :, 0] < 30) & (rgb[:, :, 1] > 200) & (rgb[:, :, 2] > 200)
    quantized = rgb.astype(np.int32) // 8
    codes = quantized[:, :, 0] * 1024 + quantized[:, :, 1] * 32 + quantized[:, :, 2]
    largest_bin = np.bincount(codes.reshape(-1), minlength=32768).max() / codes.size
    luminance = rgb[:, :, 0] * .2126 + rgb[:, :, 1] * .7152 + rgb[:, :, 2] * .0722
    return dict(cyan_fraction=float(cyan.mean()), largest_color_bin=float(largest_bin),
                mean_luminance=float(luminance.mean()), luminance_std=float(luminance.std()))


def artifact_reasons(current, baseline):
    reasons = []
    if current['cyan_fraction'] > .25 and current['cyan_fraction'] - baseline['cyan_fraction'] > .15:
        reasons.append('large_new_cyan_region')
    if current['largest_color_bin'] > .5 and current['largest_color_bin'] - baseline['largest_color_bin'] > .2:
        reasons.append('large_new_flat_color_region')
    if baseline['mean_luminance'] > 30 and current['mean_luminance'] < .2 * baseline['mean_luminance']:
        reasons.append('large_luminance_collapse')
    return reasons


@torch.no_grad()
def evaluate_student(adapter, student, samples, records, directory, seeds, baseline=None, teacher_references=None,
                     weight_source='raw'):
    from .losses import sample_student
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    previous_mode = student.training
    student.eval()
    results = []
    teacher_map = {(row['index'], row['seed']): row for row in (teacher_references or [])}
    try:
        with torch.autocast(adapter.device.type, dtype=torch.bfloat16):
            for index, (sample, record) in enumerate(zip(samples, records)):
                for seed in seeds:
                    generator = torch.Generator(device=adapter.device).manual_seed(seed)
                    latents = sample_student(adapter, student, sample, 4, generator)
                    if not torch.isfinite(latents).all():
                        raise FloatingPointError('Nonfinite held-out student latents')
                    image = adapter.decode(latents, sample)
                    filename = f'{index:03d}-seed{seed}.png'
                    image.save(directory / filename)
                    stats = image_statistics(image)
                    position = len(results)
                    original = baseline[position] if baseline is not None else None
                    reasons = artifact_reasons(stats, original['statistics']) if original else []
                    if record['reference_images']:
                        with Image.open(record['reference_images'][0]) as source:
                            panels = [('Source', source.convert('RGB'))]
                    else:
                        panels = []
                    teacher_reference = teacher_map.get((index, seed))
                    if teacher_reference is not None:
                        if teacher_reference['prompt'] != record['prompt'] or teacher_reference['teacher_steps'] != 50:
                            raise ValueError('Teacher comparison must match this prompt and use 50 steps')
                        with Image.open(teacher_reference['image']) as teacher_image:
                            panels.append(('Teacher, 50 steps', teacher_image.convert('RGB')))
                    elif original is not None:
                        with Image.open(original['image']) as base:
                            panels.append(('Initial student, 4 steps', base.convert('RGB')))
                    panels.append(('Student EMA, 4 steps' if weight_source == 'ema' else 'Student, 4 steps', image))
                    layout = save_comparison(directory / f'{index:03d}-seed{seed}-comparison.png',
                        panels, record['prompt'], seed, int(directory.name.split('-')[-1]) if directory.name.split('-')[-1].isdigit() else None)
                    if teacher_reference is not None:
                        layout.update(teacher_image=teacher_reference['image'], teacher_steps=50)
                    results.append(dict(index=index, seed=seed, prompt=record['prompt'], weight_source=weight_source,
                        reference_images=record['reference_images'], image=str((directory / filename).resolve()),
                        statistics=stats, artifact_reasons=reasons, comparison=layout,
                        latent_mean=float(latents.float().mean()), latent_std=float(latents.float().std()),
                        latent_absmax=float(latents.float().abs().max())))
    finally:
        student.train(previous_mode)
        adapter.offload_encoders()
    result = dict(completed=True, examples=len(records), samples=len(results),
                  artifact_gate_passed=not any(row['artifact_reasons'] for row in results), results=results,
                  scope='Automatic rejection of obvious artifacts relative to the untrained base; visual generation-quality review is still required.')
    (directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
