"""Read-only conditioning, adapter and critic diagnostics for T2I prompt drift."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from main.flow.adapters import FlowAdapter
from main.flow.checkpoints import load_export, read_export_config
from main.flow.comparisons import prepare_flux_teacher_references, save_comparison
from main.flow.data import CachedConditioning, cache_key, read_manifest
from main.flow.losses import (fake_score_loss, interpolate, reconstruct_clean,
    rollout_noise, rollout_prediction, rollout_update, sample_student, student_clean_sample)


def rms(value):
    return float(value.float().square().mean().sqrt())


@torch.inference_mode()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--manifest', default='data/repo-t2i/validation.jsonl')
    parser.add_argument('--cache_dir', default='.cache/flux-klein-t2i-conditioning')
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args(argv)
    checkpoint, run, output = Path(args.checkpoint), Path(args.run_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = read_export_config(checkpoint / 'student')
    if config['family'] != 'flux_klein' or config['task'] != 't2i':
        parser.error('Expected a FLUX T2I diagnostic snapshot')
    records = read_manifest(args.manifest, 't2i', config['height'], config['width'])[:8]
    adapter = FlowAdapter(config['family'], config['model_id'], config['revision'],
        teacher_guidance=config['teacher_guidance'], scheduler_config=config['scheduler_config'])
    adapter.saved_schedules = config['schedules']
    paths = [Path(args.cache_dir) / (cache_key(record, adapter.identity) + '.pt') for record in records]
    samples = [CachedConditioning(paths)[i] for i in range(len(paths))]
    report = dict(checkpoint=str(checkpoint.resolve()), completed=False, cache_checks=[], adapter_checks=[],
        prompt_response=[], critic_risk=[], references=None)
    def save():
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    # Independently encode the exact recorded prompts, avoiding all cached values.
    adapter.keep_encoders_on_device = True
    fresh_samples = []
    for index, (record, cached) in enumerate(zip(records, samples)):
        fresh = adapter.prepare(record)
        fresh_samples.append(fresh)
        checks = {}
        for key in ['prompt_embeds', 'negative_prompt_embeds', 'text_ids', 'negative_text_ids', 'latent_ids']:
            a, b = cached['condition'][key].to('cuda'), fresh['condition'][key]
            checks[key] = dict(equal=torch.equal(a, b), max_abs=float((a.float()-b.float()).abs().max()),
                relative_rms=rms(a-b)/max(rms(b), 1e-12))
        report['cache_checks'].append(dict(index=index, prompt=record['prompt'], cache=str(paths[index]), tensors=checks))
        save()
    adapter.keep_encoders_on_device = False
    adapter.offload_encoders()
    print('CACHE_CHECK_COMPLETE', flush=True)
    student = load_export(adapter, checkpoint / 'student', config)
    fake = load_export(adapter, checkpoint / 'fake')
    teacher = adapter.load_transformer().requires_grad_(False).eval()
    report['references'] = prepare_flux_teacher_references(adapter, teacher, records,
        run / 'teacher-references', [42, 43])
    save()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        reference_map = {(row['index'], row['seed']): row for row in report['references']}
        for index in [0, 2, 4, 6]:
            for seed in [42, 43]:
                sample, fresh = samples[index], fresh_samples[index]
                generator = lambda: torch.Generator(device='cuda').manual_seed(seed)
                current = sample_student(adapter, student, sample, 4, generator())
                reencoded = sample_student(adapter, student, fresh, 4, generator())
                actual = adapter.decode(current, sample)
                actual.save(output / f'{index:03d}-seed{seed}-student.png')
                adapter.offload_encoders()
                student.disable_adapters()
                try:
                    disabled = sample_student(adapter, student, sample, 4, generator())
                finally:
                    student.enable_adapters()
                disabled_image = adapter.decode(disabled, sample)
                disabled_image.save(output / f'{index:03d}-seed{seed}-adapter-disabled.png')
                with Image.open(reference_map[index, seed]['image']) as native:
                    save_comparison(output / f'{index:03d}-seed{seed}-comparison.png',
                        [('Teacher, 50 steps', native), ('Student, 4 steps', actual),
                         ('Student adapter disabled, 4 steps', disabled_image)], records[index]['prompt'], seed, 4500)
                result = dict(index=index, seed=seed,
                    fresh_vs_cached_latent_relative_rms=rms(reencoded-current)/max(rms(current),1e-12),
                    adapter_disabled_vs_enabled_relative_rms=rms(disabled-current)/max(rms(current),1e-12))
                original = run / 'validation-004496' / f'{index:03d}-seed{seed}.png'
                if original.exists():
                    with Image.open(original) as image:
                        result['replay_vs_training_png_mean_abs'] = float(np.abs(np.asarray(actual).astype(float)-np.asarray(image).astype(float)).mean())
                report['adapter_checks'].append(result)
                # Inspect all clean heads on the exact inference trajectory.
                if seed == 42 and index in [0, 6]:
                    sigmas = adapter.schedule(sample, 4)
                    state = rollout_noise(adapter, student, sample['shape'], generator())
                    panels = []
                    for step in range(4):
                        velocity = rollout_prediction(adapter, student, state, sigmas[step], sample['condition'])
                        clean = reconstruct_clean(state, velocity, sigmas[step])
                        panels.append((f'Clean head {step}, sigma {float(sigmas[step]):.3f}', adapter.decode(clean, sample)))
                        state = rollout_update(adapter, state, velocity, sigmas[step], sigmas[step+1])
                    save_comparison(output / f'{index:03d}-clean-heads.png', panels, records[index]['prompt'], seed, 4500)
                adapter.offload_encoders()
                save()
                print('ADAPTER_CHECK', index, seed, flush=True)
        # Same latent and noise for all prompts: quantify conditional responses,
        # and whether the trained score tracks its own current student samples.
        for index, sample in enumerate(samples):
            other = samples[(index+1) % len(samples)]
            clean = student_clean_sample(adapter, student, sample, 4,
                torch.Generator(device='cuda').manual_seed(700+index), selected_step=2).detach()
            for sigma_index, sigma in enumerate([.05, .2, .5, .8, .95]):
                noise = torch.randn(clean.shape, device='cuda',
                    generator=torch.Generator(device='cuda').manual_seed(9000+index*10+sigma_index))
                noisy = interpolate(clean, noise, sigma)
                f = adapter.predict(fake, noisy, sigma, sample['condition'])
                fw = adapter.predict(fake, noisy, sigma, other['condition'])
                t = adapter.predict(teacher, noisy, sigma, sample['condition'])
                tw = adapter.predict(teacher, noisy, sigma, other['condition'])
                guided = adapter.predict(teacher, noisy, sigma, sample['condition'], config['teacher_guidance'])
                sv = adapter.predict(student, noisy, sigma, sample['condition'])
                sw = adapter.predict(student, noisy, sigma, other['condition'])
                report['critic_risk'].append(dict(index=index, sigma=sigma,
                    trained_mse=float(fake_score_loss(f, clean, noise)), base_mse=float(fake_score_loss(t, clean, noise)),
                    trained_wrong_prompt_mse=float(fake_score_loss(fw, clean, noise)),
                    base_wrong_prompt_mse=float(fake_score_loss(tw, clean, noise))))
                report['prompt_response'].append(dict(index=index, sigma=sigma,
                    teacher_prompt_delta=rms(t-tw), fake_prompt_delta=rms(f-fw), student_prompt_delta=rms(sv-sw),
                    student_update_from_base=rms(sv-t), fake_update_from_base=rms(f-t),
                    teacher_cfg4_minus_cfg1=rms(guided-t)))
            save()
            print('CRITIC_PROBE', index, flush=True)
    risks = report['critic_risk']
    report['risk_summary'] = {str(sigma): dict(
        trained_mse=float(np.mean([r['trained_mse'] for r in risks if r['sigma']==sigma])),
        base_mse=float(np.mean([r['base_mse'] for r in risks if r['sigma']==sigma])),
        wrong_prompt_penalty=float(np.mean([r['trained_wrong_prompt_mse']-r['trained_mse'] for r in risks if r['sigma']==sigma])))
        for sigma in [.05,.2,.5,.8,.95]}
    report['completed'] = True
    report['scope'] = 'Read-only cache/adapter replay and conditional denoising diagnostics. Denoising risk and prompt-response norms do not establish oracle-score accuracy or a unique cause of collapse.'
    save()
    print('DIAGNOSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
