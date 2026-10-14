"""Audit completed diagnostic pilots against optimizers, configuration and PNGs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from main.audit_flow_run import audit_checkpoint
from main.flow.data import cache_key, read_manifest
from main.flow.validation import artifact_reasons, image_statistics


def audit_pilot(directory):
    directory = Path(directory)
    summary = json.loads((directory / 'pilot_summary.json').read_text())
    completed = summary['completed_iterations']
    checkpoint = directory / f'checkpoint-{completed:06d}'
    config = json.loads((checkpoint / 'training_config.json').read_text())
    objective = config.get('loss_weighting', 'normalized')
    if 'raw-flow-kl' in directory.name:
        if objective != 'raw_flow_kl' or config.get('objective_code_sha256') != hashlib.sha256(Path('main/flow_kl_diagnostic.py').read_bytes()).hexdigest():
            raise ValueError('Raw-weighted pilot lacks its actual objective identity/code hash')
    checkpoint_result = audit_checkpoint(checkpoint, completed)
    if checkpoint_result['completed_student_updates'] != summary['completed_student_updates']:
        raise ValueError('Pilot summary differs from actual student optimizer count')
    if summary['reached_requested_updates'] != (completed == config['train_iters']):
        raise ValueError('Pilot completion flag differs from the requested update budget')
    if not summary['reached_requested_updates'] and not summary['stopped_for_artifacts']:
        raise ValueError('Pilot stopped without reaching its budget or triggering its artifact gate')
    train = read_manifest(config['manifest'], config['task'], config['height'], config['width'])
    validation = read_manifest(config['validation_manifest'], config['task'], config['height'], config['width'])[:config['validation_limit']]
    def input_identity(record):
        if record['task'] == 't2i':
            return ' '.join(record['prompt'].casefold().split())
        return hashlib.sha256(Path(record['reference_images'][0]).read_bytes()).hexdigest()
    train_hashes = {input_identity(r) for r in train}
    validation_hashes = {input_identity(r) for r in validation}
    if len(train_hashes) != len(train) or len(validation_hashes) != len(validation) or not train_hashes.isdisjoint(validation_hashes):
        raise ValueError('Pilot inputs are duplicated or overlap across splits')
    export = json.loads((checkpoint / 'student/flow_config.json').read_text())
    identity = {name: export[name] for name in ('format_version', 'family', 'model_id', 'revision', 'diffusers_version', 'teacher_guidance', 'dtype')}
    if [cache_key(record, identity) for record in train] != config['data_keys']:
        raise ValueError('Pilot inputs differ from the conditioning used during training')
    if export.get('sampling_precision', 'fp32') != config.get('sampling_precision', 'fp32'):
        raise ValueError('Pilot export lacks its actual sampling precision policy')
    if export.get('rollout_mode', 'euler') != config.get('rollout_mode', 'euler'):
        raise ValueError('Pilot export lacks its actual rollout mode')
    fake_export = json.loads((checkpoint / 'fake/flow_config.json').read_text())
    if export['lora_config']['r'] != config['lora_rank'] or fake_export['lora_config']['r'] != config['fake_lora_rank']:
        raise ValueError('Student/fake adapter ranks differ from the trained configuration')
    baseline = json.loads((directory / 'validation-000000/summary.json').read_text())
    expected_records = [(index, seed) for index in range(len(validation)) for seed in config['validation_seeds']]
    evaluations = []
    for path in sorted(directory.glob('validation-*/summary.json')):
        result = json.loads(path.read_text())
        layout_path = path.parent / 'comparison_metadata.json'
        layouts = {}
        if layout_path.exists():
            layout_report = json.loads(layout_path.read_text())
            layouts = {(row['index'], row['seed']): row for row in layout_report['results']}
            if not layout_report['completed'] or set(layouts) != set(expected_records):
                raise ValueError('Incomplete refreshed comparison layouts')
        if not result['completed'] or result['samples'] != len(expected_records) or len(result['results']) != len(expected_records):
            raise ValueError(f'Incomplete validation at {path}')
        rejected = False
        for position, (row, expected_record) in enumerate(zip(result['results'], expected_records)):
            index, seed = expected_record
            if (row['index'], row['seed']) != expected_record or row['prompt'] != validation[index]['prompt'] or row['reference_images'] != validation[index]['reference_images']:
                raise ValueError(f'Validation inputs differ at {path}, record {position}')
            image_path = Path(row['image'])
            if image_path.parent.resolve() != path.parent.resolve():
                raise ValueError('Validation image path points outside its own evaluation directory')
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                if image.size != (512, 512):
                    raise ValueError('Unexpected validation output dimensions')
                actual_stats = image_statistics(image)
                pixels = np.asarray(image.convert('RGB'))
            for name, value in actual_stats.items():
                if abs(row['statistics'][name] - value) > 1e-8:
                    raise ValueError('Stored artifact statistics differ from actual image pixels')
            reasons = [] if path.parent.name == 'validation-000000' else artifact_reasons(actual_stats, baseline['results'][position]['statistics'])
            if reasons != row['artifact_reasons']:
                raise ValueError('Stored artifact verdict differs from its declared pixel thresholds')
            rejected |= bool(reasons)
            comparison_path = path.parent / f'{index:03d}-seed{seed}-comparison.png'
            with Image.open(comparison_path) as image:
                image.verify()
            with Image.open(comparison_path) as image:
                comparison = np.asarray(image.convert('RGB'))
                layout = layouts.get(expected_record, row.get('comparison'))
                if layout is not None:
                    if layout['prompt'] != row['prompt'] or layout['seed'] != seed or image.info.get('Prompt') != row['prompt']:
                        raise ValueError('Comparison prompt/seed differs from the generated sample')
                    def panel_pixels(position):
                        x0, y0, x1, y1 = layout['panels'][position]['box']
                        return comparison[y0:y1, x0:x1]
                    if not np.array_equal(panel_pixels(-1), pixels):
                        raise ValueError('Comparison student differs from the actual generated PNG')
                    references = validation[index]['reference_images']
                    if references:
                        with Image.open(references[0]) as source:
                            expected_pixels = np.asarray(ImageOps.pad(source.convert('RGB'), (512,512)))
                        if not np.array_equal(panel_pixels(0), expected_pixels):
                            raise ValueError('Comparison source differs from the actual held-out image')
                    teacher_path = layout.get('teacher_image')
                    if teacher_path:
                        teacher_report = json.loads((Path(teacher_path).parent / 'summary.json').read_text())
                        identity = teacher_report['identity']
                        if layout.get('teacher_steps') != 50 or identity['teacher_steps'] != 50 or any(
                            identity[name] != config[name] for name in ['model_id','revision','teacher_guidance']):
                            raise ValueError('Comparison teacher differs from the trained base/CFG or 50-step protocol')
                        teacher_row = next(r for r in teacher_report['results'] if (r['index'],r['seed']) == expected_record)
                        if teacher_row['prompt'] != row['prompt'] or teacher_row['image'] != teacher_path:
                            raise ValueError('Comparison uses a teacher from a different prompt')
                        with Image.open(teacher_path) as teacher_image:
                            expected_pixels = np.asarray(ImageOps.pad(teacher_image.convert('RGB'), (512,512)))
                        if not np.array_equal(panel_pixels(1 if references else 0), expected_pixels):
                            raise ValueError('Comparison teacher panel differs from the generated native reference')
                    elif path.parent.name != 'validation-000000':
                        with Image.open(baseline['results'][position]['image']) as base_image:
                            expected_pixels = np.asarray(ImageOps.pad(base_image.convert('RGB'), (512,512)))
                        if not np.array_equal(panel_pixels(1 if references else 0), expected_pixels):
                            raise ValueError('Comparison baseline differs from the initial student image')
                    continue
                # Verify that the displayed current panel is the actual generated PNG.
                if not np.array_equal(comparison[32:544, -512:], pixels):
                    raise ValueError('Comparison panel differs from the generated image')
                references = validation[index]['reference_images']
                if references:
                    with Image.open(references[0]) as source:
                        source_pixels = np.asarray(ImageOps.pad(source.convert('RGB'), (512, 512)))
                    if not np.array_equal(comparison[32:544, :512], source_pixels):
                        raise ValueError('Comparison source differs from the actual held-out image')
                if path.parent.name != 'validation-000000':
                    with Image.open(baseline['results'][position]['image']) as base_image:
                        base_pixels = np.asarray(ImageOps.pad(base_image.convert('RGB'), (512, 512)))
                    base_offset = 512 if references else 0
                    if not np.array_equal(comparison[32:544, base_offset:base_offset+512], base_pixels):
                        raise ValueError('Comparison baseline differs from the actual untrained image')
        if result['artifact_gate_passed'] != (not rejected):
            raise ValueError('Evaluation summary differs from its per-image artifact verdicts')
        evaluations.append(dict(iteration=int(path.parent.name.split('-')[-1]), artifact_gate_passed=not rejected,
                                samples=len(expected_records)))
    if not evaluations or evaluations[-1]['iteration'] != completed:
        raise ValueError('Final evaluation does not match the checkpoint and stop status')
    if summary.get('validation_policy') == 'report_only':
        if summary['stopped_for_artifacts'] or not summary['reached_requested_updates']:
            raise ValueError('Report-only validation must not interrupt the requested training budget')
    elif evaluations[-1]['artifact_gate_passed'] == summary['stopped_for_artifacts']:
        raise ValueError('Historical artifact stop differs from the final evaluation')
    required = {0, completed}
    for student_update in range(config['validation_every'], summary['completed_student_updates']+1, config['validation_every']):
        required.add(config.get('fake_warmup_iters', 0) + (student_update-1)*config['fake_gen_ratio']+1)
    if not required.issubset({row['iteration'] for row in evaluations}):
        raise ValueError('A required intermediate held-out evaluation is missing')
    random_states = torch.load(checkpoint / 'random_states_0.pkl', weights_only=False, map_location='cpu')
    cuda_states = random_states.get('torch_cuda_manual_seed')
    noise_rng_hash = None
    if cuda_states is not None:
        digest = hashlib.sha256()
        for state in cuda_states:
            digest.update(state.cpu().numpy().tobytes())
        noise_rng_hash = digest.hexdigest()
    return dict(checkpoint=checkpoint_result, summary=summary, evaluations=evaluations,
                train_examples=len(train), validation_examples=len(validation),
                sampling_precision=config.get('sampling_precision', 'fp32'), student_lr=config['student_lr'],
                fake_lr=config['fake_lr'], student_rank=config['lora_rank'], fake_rank=config['fake_lora_rank'],
                seed=config['seed'], noise_rng_hash=noise_rng_hash, artifacts_verified=True,
                loss_weighting=objective,
                rollout_mode=config.get('rollout_mode', 'euler'), pilot_fixed_noise=config.get('pilot_fixed_noise', False),
                scope='Execution, state, inputs, evaluations and obvious-artifact decisions; visual editing-quality review is independent.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', default='outputs/flux-validation')
    parser.add_argument('--variants', nargs='+', default=['baseline', 'native-precision', 'slow-student', 'critic-rank128'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    results = {name: audit_pilot(output / f'pilot-{name}-seed{args.seed}') for name in args.variants}
    result = dict(completed=True, artifacts_verified=True, seed=args.seed, pilots=results,
                  editing_quality_verified=False, note='Artifact verification is not a convergence claim.')
    (output / f'pilot_audit_seed{args.seed}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: dict(fake_updates=row['checkpoint']['completed_fake_updates'],
          student_updates=row['checkpoint']['completed_student_updates'], artifacts_verified=row['artifacts_verified'])
          for name, row in results.items()}, indent=2))


if __name__ == '__main__':
    main()
