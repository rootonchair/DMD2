"""Portable student exports plus training state managed by Accelerate."""

import json
from pathlib import Path
import shutil

import torch


def configure_trainable(model, mode, rank=32, alpha=32, gradient_checkpointing=True):
    if mode == 'lora':
        from peft import LoraConfig
        model.requires_grad_(False)
        # Select actual linear modules, including FLUX's fused single-stream QKV/MLP.
        projection_names = {'to_q', 'to_k', 'to_v', 'to_qkv', 'to_qkv_mlp_proj',
                            'to_out', 'add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out'}
        targets = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
                   and (name.split('.')[-1] in projection_names or name.endswith('to_out.0'))]
        if not targets:
            raise ValueError('No supported attention projections found for LoRA')
        model.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets, lora_dropout=0.0))
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    else:
        model.float().requires_grad_(True)
    if gradient_checkpointing:
        model.enable_gradient_checkpointing()
    model.train()
    return model


def export_model(model, directory, metadata, mode):
    from safetensors.torch import save_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(metadata, train_mode=mode, format_version=1)
    if mode == 'lora':
        from peft import get_peft_model_state_dict
        config['lora_config'] = model.peft_config['default'].to_dict()
        state = {name: value.detach().cpu().contiguous() for name, value in get_peft_model_state_dict(model).items()}
        save_file(state, str(directory / 'adapter.safetensors'))
    else:
        model.save_pretrained(directory / 'transformer', safe_serialization=True)
    (directory / 'flow_config.json').write_text(json.dumps(config, indent=2, default=_json_default) + '\n')


def _json_default(value):
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f'Cannot serialize {type(value)}')


def read_export_config(directory):
    path = Path(directory) / 'flow_config.json'
    if not path.is_file():
        raise ValueError(f'Expected a student export containing {path.name}: {directory}')
    config = json.loads(path.read_text())
    if config.get('format_version') != 1 or config.get('schedule_policy') != 'native-linear-sigmas-v1':
        raise ValueError('Unsupported flow checkpoint format or sampling policy')
    return config


def restore_lora_weights(model, directory):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    config = read_export_config(directory)
    if config['train_mode'] != 'lora':
        raise ValueError('Expected a LoRA adapter checkpoint')
    current = json.loads(json.dumps(model.peft_config['default'].to_dict(), default=_json_default))
    if current != config['lora_config']:
        raise ValueError('LoRA configuration differs from the checkpoint')
    result = set_peft_model_state_dict(model, load_file(str(Path(directory) / 'adapter.safetensors')))
    missing = [key for key in result.missing_keys if 'lora_' in key]
    if result.unexpected_keys or missing:
        raise ValueError(f'Adapter weight mismatch: unexpected={result.unexpected_keys}, missing={missing}')


def register_compact_lora_hooks(accelerator, metadata):
    """Accelerate retains optimizers/RNG; frozen base weights are reconstructed."""
    def save_hook(models, weights, output_dir):
        if len(models) != 2:
            raise ValueError('Compact flow checkpoints expect student and fake-score models')
        if accelerator.is_main_process:
            for name, model in zip(('student', 'fake'), models):
                export_model(accelerator.unwrap_model(model), Path(output_dir) / name, metadata, 'lora')
            (Path(output_dir) / 'compact_lora.json').write_text(json.dumps({'format_version': 1}) + '\n')
        weights.clear()

    def load_hook(models, input_dir):
        marker = Path(input_dir) / 'compact_lora.json'
        if not marker.exists():
            return  # Legacy Accelerate checkpoints include complete model weights.
        if json.loads(marker.read_text()).get('format_version') != 1 or len(models) != 2:
            raise ValueError('Unsupported compact flow checkpoint')
        for name, model in zip(('student', 'fake'), models):
            restore_lora_weights(accelerator.unwrap_model(model), Path(input_dir) / name)
        models.clear()

    return (accelerator.register_save_state_pre_hook(save_hook),
            accelerator.register_load_state_pre_hook(load_hook))


def prune_checkpoints(output_dir, keep, run_id):
    """Only remove this run's completed numeric checkpoint directories."""
    if keep is None:
        return
    candidates = []
    for directory in Path(output_dir).glob('checkpoint-*'):
        marker = directory / 'checkpoint_complete.json'
        if not directory.is_dir() or directory.is_symlink() or not marker.is_file():
            continue
        suffix = directory.name.removeprefix('checkpoint-')
        if not suffix.isdigit():
            continue
        info = json.loads(marker.read_text())
        if info.get('run_id') == run_id and info.get('completed_iterations') == int(suffix):
            candidates.append((int(suffix), directory))
    for _, directory in sorted(candidates)[:-keep]:
        shutil.rmtree(directory)


def load_export(adapter, directory, config=None):
    from safetensors.torch import load_file
    config = config or read_export_config(directory)
    adapter.sampling_precision = config.get('sampling_precision', 'fp32')
    if adapter.sampling_precision not in ('fp32', 'native'):
        raise ValueError('Unsupported sampling precision policy')
    adapter.rollout_mode = config.get('rollout_mode', 'euler')
    if adapter.rollout_mode not in ('euler', 'stochastic'):
        raise ValueError('Unsupported rollout mode')
    if config['family'] != adapter.family or config['model_id'] != adapter.model_id or config['revision'] != adapter.revision:
        raise ValueError('Student export and base model identity do not match')
    if config.get('local_fingerprint') is not None and config['local_fingerprint'] != adapter.identity.get('local_fingerprint'):
        raise ValueError('The local base model has changed since the student was exported')
    directory = Path(directory)
    if config['train_mode'] == 'lora':
        from peft import LoraConfig
        model = adapter.load_transformer()
        model.add_adapter(LoraConfig(**config['lora_config']))
        # Keep adapter weights at the same precision used during training.
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        restore_lora_weights(model, directory)
    else:
        model = adapter.transformer_class.from_pretrained(directory / 'transformer', torch_dtype=adapter.dtype).to(adapter.device)
    return model.requires_grad_(False).eval()
