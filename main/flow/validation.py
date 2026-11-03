"""Cheap artifact diagnostics for fixed held-out editing cases.

These reject obvious collapse, but do not establish editing quality. Comparisons
must still be inspected; cyan can be a legitimate image color in some prompts.
"""

import numpy as np
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageOps
import torch

from .comparisons import save_comparison, native_teacher_steps


def image_statistics(image):
    rgb = np.asarray(image.convert('RGB'))
    cyan = (rgb[:, :, 0] < 30) & (rgb[:, :, 1] > 200) & (rgb[:, :, 2] > 200)
    quantized = rgb.astype(np.int32) // 8
    codes = quantized[:, :, 0] * 1024 + quantized[:, :, 1] * 32 + quantized[:, :, 2]
    largest_bin = np.bincount(codes.reshape(-1), minlength=32768).max() / codes.size
    luminance = rgb[:, :, 0] * .2126 + rgb[:, :, 1] * .7152 + rgb[:, :, 2] * .0722
    # HSV saturation; the tail and channel clipping expose DMD/CFG over-saturation
    # that leaves the mean nearly unchanged.
    high, low = rgb.max(-1).astype(np.float32), rgb.min(-1).astype(np.float32)
    saturation = np.where(high > 0, (high - low) / np.maximum(high, 1), 0)
    clipped = ((rgb <= 5) | (rgb >= 250)).any(-1)
    return dict(cyan_fraction=float(cyan.mean()), largest_color_bin=float(largest_bin),
                mean_luminance=float(luminance.mean()), luminance_std=float(luminance.std()),
                saturation_mean=float(saturation.mean()), saturation_p95=float(np.percentile(saturation, 95)),
                clipped_fraction=float(clipped.mean()))


COLOR_DRIFT_KEYS = ('saturation_mean', 'saturation_p95', 'clipped_fraction', 'luminance_std')


def color_drift(results):
    """Mean student-minus-teacher color statistics over samples with teacher references."""
    rows = [row for row in results if row.get('teacher_statistics')]
    if not rows:
        return None
    return {key: float(np.mean([row['statistics'][key] - row['teacher_statistics'][key] for row in rows]))
            for key in COLOR_DRIFT_KEYS}


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
                     weight_source='raw', num_steps=4):
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
                    latents = sample_student(adapter, student, sample, num_steps, generator)
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
                    teacher_stats = None
                    if teacher_reference is not None:
                        steps = native_teacher_steps(getattr(adapter, 'family', 'flux_klein'))
                        if teacher_reference['prompt'] != record['prompt'] or teacher_reference['teacher_steps'] != steps:
                            raise ValueError(f'Teacher comparison must match this prompt and use {steps} steps')
                        with Image.open(teacher_reference['image']) as teacher_image:
                            label = f'Teacher, {steps} steps'
                            if hasattr(adapter, 'teacher_guidance'):
                                label += f', CFG {adapter.teacher_guidance}'
                            panels.append((label, teacher_image.convert('RGB')))
                            teacher_stats = image_statistics(teacher_image)
                    elif original is not None:
                        with Image.open(original['image']) as base:
                            panels.append((f'Initial student, {num_steps} step' + ('s' if num_steps != 1 else ''), base.convert('RGB')))
                    label = 'Student EMA' if weight_source == 'ema' else 'Student'
                    panels.append((f'{label}, {num_steps} step' + ('s' if num_steps != 1 else ''), image))
                    layout = save_comparison(directory / f'{index:03d}-seed{seed}-comparison.png',
                        panels, record['prompt'], seed, int(directory.name.split('-')[-1]) if directory.name.split('-')[-1].isdigit() else None)
                    if teacher_reference is not None:
                        layout.update(teacher_image=teacher_reference['image'], teacher_steps=teacher_reference['teacher_steps'])
                        if hasattr(adapter, 'teacher_guidance'):
                            layout['teacher_guidance'] = adapter.teacher_guidance
                    results.append(dict(index=index, seed=seed, prompt=record['prompt'], weight_source=weight_source,
                        student_steps=num_steps,
                        reference_images=record['reference_images'], image=str((directory / filename).resolve()),
                        statistics=stats, teacher_statistics=teacher_stats, artifact_reasons=reasons, comparison=layout,
                        latent_mean=float(latents.float().mean()), latent_std=float(latents.float().std()),
                        latent_absmax=float(latents.float().abs().max())))
    finally:
        student.train(previous_mode)
        adapter.offload_encoders()
    result = dict(completed=True, examples=len(records), samples=len(results), student_steps=num_steps,
                  artifact_gate_passed=not any(row['artifact_reasons'] for row in results),
                  color_drift_vs_teacher=color_drift(results), results=results,
                  scope='Automatic rejection of obvious artifacts relative to the untrained base; visual generation-quality review is still required.')
    (directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
