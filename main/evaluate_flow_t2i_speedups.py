"""Generate text-to-image comparisons of a flow export against published Qwen Image 2.1 speedups.

Each variant runs its own published recipe (sigmas, CFG, KV cache) on identical prompts and seeds, in its own
process, so one model's patches never reach another. The teacher and E go through the same code as
main.evaluate_flow_t2i, which lets --check_against prove pixel identity with an existing evaluation.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from main.flow.adapters import FlowAdapter
from main.flow.checkpoints import load_export, read_export_config
from main.flow.comparisons import native_teacher_guidance, native_teacher_steps
from main.flow.data import read_manifest
from main.flow.losses import sample_student
from main.flow.validation import image_statistics

BASE_ID, BASE_REVISION = 'Qwen/Qwen-Image-2.1', 'd26bb61231c349cf6b7896fa83353113880e1ba3'
VARIANTS = ('teacher', 'e_stochastic', 'e_euler', 'lightning', 'fun_acc', 'fun_acc_2048', 'dmd2_2step', 'turbo')


def adapter_runner(args, variant, height, width):
    """Teacher and E: the code path of main.evaluate_flow_t2i."""
    config = read_export_config(args.export)
    adapter = FlowAdapter(config['family'], config['model_id'], config['revision'], 'cuda',
                          teacher_guidance=1, scheduler_config=config['scheduler_config'])
    adapter.saved_schedules = config.get('schedules', {})
    if variant == 'teacher':
        transformer = adapter.load_transformer().requires_grad_(False).eval()
        adapter.pipe.transformer = transformer
        adapter.pipe.scheduler = adapter.pipe.scheduler.__class__.from_config(adapter.pipe.scheduler.config)
        adapter.pipe.to('cuda')
        adapter.pipe.set_progress_bar_config(disable=True)

        def run(prompt, seed):
            return adapter.pipe(prompt=prompt, height=height, width=width,
                **native_teacher_guidance(config['family'], config['teacher_guidance']),
                num_inference_steps=native_teacher_steps(config['family']),
                generator=torch.Generator(device='cuda').manual_seed(seed)).images[0]
        return run, dict(steps=native_teacher_steps(config['family']), cfg=config['teacher_guidance'])
    student = load_export(adapter, args.export, config)
    adapter.rollout_mode = 'stochastic' if variant == 'e_stochastic' else 'euler'

    def run(prompt, seed):
        record = dict(task='t2i', prompt=prompt, reference_images=[], height=height, width=width)
        sample = adapter.prepare(record)
        generator = torch.Generator(device='cuda').manual_seed(seed)
        latents = sample_student(adapter, student, sample, config['num_steps'], generator)
        if not torch.isfinite(latents).all():
            raise FloatingPointError('Nonfinite text-to-image output')
        image = adapter.decode(latents, sample)
        adapter.offload_encoders()
        return image
    return run, dict(steps=config['num_steps'], cfg=1, rollout=adapter.rollout_mode, sigmas=config['schedules'])


def native_pipe(model=None):
    from diffusers import QwenImage21Pipeline
    return QwenImage21Pipeline.from_pretrained(model or BASE_ID, revision=None if model else BASE_REVISION,
                                               torch_dtype=torch.bfloat16)


def lightning_runner(args, height, width):
    """Verbatim from the lightx2v/Qwen-Image-2.1-Lightning model card."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    pipe = native_pipe()
    pipe.load_lora_weights(args.lightning, weight_name='pytorch_lora_weights.safetensors', adapter_name='lightning')
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, num_train_timesteps=1000,
        use_dynamic_shifting=True, time_shift_type='exponential', base_image_seq_len=256, max_image_seq_len=8192,
        base_shift=0.5, max_shift=0.9, shift_terminal=None)
    pipe.to('cuda')
    pipe.set_progress_bar_config(disable=True)
    sigmas = [1.0, 0.75, 0.5, 0.25]

    def run(prompt, seed):
        return pipe(prompt=prompt, width=width, height=height, num_inference_steps=4, sigmas=sigmas,
            true_cfg_scale=1.0, use_kv_cache=False, generator=torch.Generator('cuda').manual_seed(seed)).images[0]
    return run, dict(steps=4, cfg=1, sigmas=sigmas, use_kv_cache=False)


def dmd2_runner(args, height, width):
    """Verbatim from the linoyts/Qwen-Image-2.1-DMD2-2steps model card, with its newest (v1.5) weights."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    pipe = native_pipe()
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, use_dynamic_shifting=False,
        shift=1.0, shift_terminal=None)
    pipe.load_lora_weights(args.dmd2, weight_name='pytorch_lora_weights_v1_5.safetensors')
    pipe.to('cuda')
    pipe.set_progress_bar_config(disable=True)
    sigmas = [1.0, 0.5]

    def run(prompt, seed):
        return pipe(prompt=prompt, height=height, width=width, num_inference_steps=2, sigmas=sigmas,
            true_cfg_scale=1.0, generator=torch.Generator('cuda').manual_seed(seed)).images[0]
    return run, dict(steps=2, cfg=1, sigmas=sigmas, weights='v1_5')


def fun_acc_runner(args, height, width):
    """predict_t2i.py from alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs; its grid is fixed for pdd_sample_size."""
    root = Path(args.fun_acc)
    sys.path.append(str(root))  # lora_utils_pdd is imported by name from the reviewed release snapshot.
    spec = importlib.util.spec_from_file_location('qwenimage21_pdd', root / 'qwenimage21_pdd.py')
    pdd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pdd)
    pipe = native_pipe()
    config = pdd.load_pdd_lora(pipe.transformer, str(root / 'models/Qwen-Image-2.1-Fun-Acc-4Step.safetensors'))
    pipe.transformer.eval()
    pipe.scheduler = pdd.QwenImage21PDDScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.register_to_config(**config)
    sigmas = torch.tensor(config['pdd_sigmas'], dtype=torch.float32)
    pipe.to('cuda')
    pipe.set_progress_bar_config(disable=True)

    def run(prompt, seed):
        return pipe(prompt=prompt, height=height, width=width, true_cfg_scale=1.0, use_kv_cache=False,
            generator=torch.Generator('cuda').manual_seed(seed), num_inference_steps=4,
            callback_on_step_end=pdd.pdd_step_callback(pipe.transformer, sigmas, config['pdd_block_size'])).images[0]
    return run, dict(steps=4, cfg=1, sigmas=config['pdd_sigmas'], trained_size=config['pdd_sample_size'],
                     use_kv_cache=False)


def turbo_runner(args, height, width):
    """Official Turbo with its own full pipeline; sigmas are passed as main.evaluate_flow_turbo_comparison does."""
    pipe = native_pipe(args.turbo)
    sigmas = json.loads((Path(args.turbo) / 'model_index.json').read_text())['sample_sigmas']
    pipe.to('cuda')
    pipe.set_progress_bar_config(disable=True)

    def run(prompt, seed):
        return pipe(prompt=prompt, height=height, width=width, num_inference_steps=len(sigmas), sigmas=list(sigmas),
            true_cfg_scale=1.0, use_kv_cache=True, generator=torch.Generator('cuda').manual_seed(seed)).images[0]
    return run, dict(steps=len(sigmas), cfg=1, sigmas=sigmas, use_kv_cache=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=VARIANTS)
    parser.add_argument('--manifest', default='data/repo-t2i/validation.jsonl')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--export', help='Student export with flow_config.json; needed for teacher and e_* variants')
    parser.add_argument('--lightning')
    parser.add_argument('--dmd2')
    parser.add_argument('--fun_acc')
    parser.add_argument('--turbo')
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--plan', nargs='+', default=['42:64', '43:16'], help='seed:prompt_count pairs')
    parser.add_argument('--check_against', help='Directory of 000-student.png / 000-teacher.png to match exactly')
    args = parser.parse_args(argv)
    output = Path(args.output_dir) / args.variant
    output.mkdir(parents=True, exist_ok=True)
    records = read_manifest(args.manifest, 't2i', args.size, args.size)
    started = time.time()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.variant in ('teacher', 'e_stochastic', 'e_euler')):
        if args.variant in ('teacher', 'e_stochastic', 'e_euler'):
            run, recipe = adapter_runner(args, args.variant, args.size, args.size)
        else:
            builder = dict(lightning=lightning_runner, dmd2_2step=dmd2_runner, fun_acc=fun_acc_runner,
                           fun_acc_2048=fun_acc_runner, turbo=turbo_runner)[args.variant]
            run, recipe = builder(args, args.size, args.size)
        load_seconds = time.time() - started
        path = output / 'results.json'
        results = json.loads(path.read_text()) if path.exists() else []
        done = {(item['index'], item['seed']) for item in results}
        for pair in args.plan:
            seed, count = map(int, pair.split(':'))
            for index, record in enumerate(records[:count]):
                if (index, seed) in done:
                    continue
                begin = time.time()
                image = run(record['prompt'], seed)
                torch.cuda.synchronize()
                seconds = time.time() - begin
                image.convert('RGB').save(output / f'{index:03d}-seed{seed}.png')
                results.append(dict(index=index, seed=seed, prompt=record['prompt'], seconds=seconds,
                                    statistics=image_statistics(image)))
                path.write_text(json.dumps(results, indent=2) + '\n')
                print(f'{args.variant} {index} seed{seed} {seconds:.1f}s', flush=True)
    summary = dict(variant=args.variant, size=args.size, recipe=recipe, load_seconds=load_seconds,
                   median_seconds=float(np.median([item['seconds'] for item in results])),
                   peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024 ** 3, images=len(results),
                   diffusers=__import__('diffusers').__version__, quality_verified=False)
    if args.check_against:
        name = '000-teacher.png' if args.variant == 'teacher' else '000-student.png'
        expected = np.asarray(Image.open(Path(args.check_against) / name).convert('RGB'))
        actual = np.asarray(Image.open(output / '000-seed42.png').convert('RGB'))
        summary['pixel_identical_to_existing'] = bool(np.array_equal(expected, actual))
        summary['max_pixel_difference'] = int(np.abs(expected.astype(int) - actual.astype(int)).max())
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'recipe'}), flush=True)


if __name__ == '__main__':
    main()
