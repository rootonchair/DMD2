# FLUX Klein and Qwen Image flow distillation

Train one-step or four-step students for **T2I and native reference-image editing (I2I)**.
The two supported base teachers are `black-forest-labs/FLUX.2-klein-base-4B` and
`Qwen/Qwen-Image-2.1`. The implementation uses distribution matching and a learned
fake-score flow model. **There is no GAN loss or target-image requirement.**

This is a new backend; the older SDXL/SD1.5 commands still use their original code.
The student retains its base architecture and parameter count. Distillation reduces
sampling steps. Do not use the already-distilled FLUX Klein checkpoint as the base teacher.

## Setup

Run commands from the repository root in a separate Python 3.10+ environment.
Install a CUDA-enabled PyTorch build suitable for your GPU, then:

```bash
python3 -m pip install -r requirements-flow.txt
```

The Diffusers source revision is pinned because Qwen **2.1** has different conditioning
and VAE code from earlier Qwen Image versions. The tested environment has PyTorch
2.11.0+cu128, Transformers 5.17.0, Accelerate 1.15.0, and PEFT 0.21.2.
Authenticate with Hugging Face if your checkpoint requires it. First use downloads the
base model's weights; GPU tests with tiny random transformers do not need downloads.

Training loads three transformers (teacher, student, fake-score). Inference memory
requirements do not represent training memory requirements. Defaults target one GPU
using BF16 and LoRA; full-weight training can require more memory or multiple GPUs.

## Status of the FLUX recipe

These are experimental recipes, not established convergence settings. Correct
conditioning, finite losses, and successful checkpoint/resume tests do not establish
generation or editing quality. I2I pilots can preserve the source while missing the
requested edit. Full-size Qwen training quality has not been validated.

Machine-local handover notes, execution reports, datasets, caches and generated
artifacts are excluded from version control. The recipes below are self-contained.

## Input data

JSONL records use image paths relative to their manifest:

```json
{"task":"t2i","prompt":"A mountain at sunset","reference_images":[]}
{"task":"i2i","prompt":"Change the jacket to red","reference_images":["images/person.png"]}
```

Optional `height`/`width` override CLI defaults per record. Outputs must align to the
native VAE/packing factor: 16 pixels for FLUX and 32 pixels for Qwen. T2I has no
references; I2I has one or more. Qwen supports up to ten references and preserves RGBA
input/output. FLUX references must pass its native minimum size/aspect-ratio checks.

Examples are provided in `t2i.jsonl`, `i2i.jsonl`, and `mixed.jsonl`. For editing examples,
put your own source photograph in `experiments/flow/images/person.png`, or edit those
manifest paths. The small example datasets demonstrate execution, not convergence.

`--task mixed` samples the selected manifest's records uniformly. Set the desired task
ratio through the number of T2I/I2I records. Independent references and variable prompt
lengths are handled per record; batching never shares one edit's references with another.

## Training examples

These four commands produce separate checkpoints. Run the desired command; they do
not depend on each other's results. Each supports both tasks with the mixed manifest.

```bash
# FLUX, four steps
accelerate launch --num_processes 1 --num_machines 1 --mixed_precision bf16 -m main.train_flow \
  --model_family flux_klein --manifest experiments/flow/mixed.jsonl \
  --task mixed --num_steps 4 --output_dir outputs/flux-4step

# FLUX, one step
accelerate launch --num_processes 1 --num_machines 1 --mixed_precision bf16 -m main.train_flow \
  --model_family flux_klein --manifest experiments/flow/mixed.jsonl \
  --task mixed --num_steps 1 --output_dir outputs/flux-1step

# Qwen, four steps
accelerate launch --num_processes 1 --num_machines 1 --mixed_precision bf16 -m main.train_flow \
  --model_family qwen_image21 --manifest experiments/flow/mixed.jsonl \
  --task mixed --num_steps 4 --output_dir outputs/qwen-4step

# Qwen, one step
accelerate launch --num_processes 1 --num_machines 1 --mixed_precision bf16 -m main.train_flow \
  --model_family qwen_image21 --manifest experiments/flow/mixed.jsonl \
  --task mixed --num_steps 1 --output_dir outputs/qwen-1step
```

For **T2I only**, use `--manifest experiments/flow/t2i.jsonl --task t2i`.
For **I2I only**, use `--manifest experiments/flow/i2i.jsonl --task i2i`.
For your own compatible base/fine-tune use `--model_id PATH_OR_HF_ID`; remote revisions
are resolved to a commit and saved. An SDXL model or earlier Qwen model is incompatible.

Defaults: 512x512, batch size 1, LoRA rank/alpha 32 on both trainable models,
gradient checkpointing, AdamW learning rate `1e-5`, 10,000 fake-score optimizer updates,
one student update every five fake updates, and checkpoints every 500 updates.
These learning rates are starting points, not validated convergence recipes.

Useful flags:

- `--student_train_mode full --fake_train_mode full` enables full weights (default LR `5e-7`).
- Each training mode can be selected independently. Override `--student_lr` and `--fake_lr` as needed.
- `--gradient_accumulation_steps 4` accumulates four microbatches per optimizer update.
- `--batch_size 2` processes two independent records per microbatch.
- `--height 1024 --width 1024` changes default output dimensions.
- `--teacher_guidance 4` changes teacher CFG; defaults are FLUX=4, Qwen=1. Student/fake CFG is 1.
- `--fake_lora_rank 128 --fake_lora_alpha 128` increases only fake-score capacity.
- `--fake_warmup_iters 200` delays the first student update until 200 fake-only updates finish.
- `--rollout_mode stochastic` predicts a clean image and adds fresh noise between steps, matching FastVideo DMD training/inference. The generic default remains `euler`; exports preserve the selected mode for inference.
- `--score_noise_sampling shifted --score_noise_shift 8` matches FastVideo's modular Wan score-noise sampler: uniform integer timesteps over the scheduler's training range, then sigma=8u/(1+7u), without an extra sigma clamp. Both critic training and normalized DMD scoring use it. The generic default remains the earlier uniform sigma.02-.98 sampler.
- `--student_ema_decay .98` enables FastVideo-style student EMA from the start, updated every outer iteration, including critic-only iterations. Validation uses averaged weights; `student-ema/` contains the averaged inference export while `student/` retains raw weights. EMA state and counters resume with checkpoints. EMA is disabled by default in the generic CLI.
- `--pilot_fixed_noise` separates rollout, step-selection, DMD-scoring and fake-scoring random streams for controlled fresh quality pilots.
- `--sampling_precision native` uses the pinned FLUX pipeline's carry/timestep rounding; exports preserve this policy. The default `fp32` preserves the previous backend behavior.
- `--validation_manifest PATH --validation_every 20 --validation_limit 8 --validation_seeds 42 43` records fixed held-out comparisons and artifact warnings during a fresh single-GPU run. Image-quality warnings do not stop training. Inspect the comparison images to assess quality.
- `--warmup_step 500` matches the original DMD2 linear learning-rate warm-up for both optimizers. Both schedules advance every outer iteration, including critic-only iterations. Scheduler state resumes with the checkpoint. The default `0` preserves earlier constant-LR runs; the supervised T2I recipe explicitly uses `500`.
- `--train_iters 5 --save_every 5 --log_every 1` makes a short execution smoke test.

The effective batch size is `batch_size * gradient_accumulation_steps * GPU_count`.
Use Accelerate's distributed launch configuration for multiple GPUs, for example
`accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 -m main.train_flow ...`.
Each GPU holds its own model copies (DDP); this backend does not implement model sharding
or optimizer offloading. Full-weight training must fit the per-GPU memory available.

## Conditioning cache and resume

The trainer automatically caches frozen prompt/reference conditioning before loading
the three transformers. To preprocess separately and share a cache between runs:

```bash
python3 -m main.prepare_flow_data \
  --model_family flux_klein --manifest experiments/flow/mixed.jsonl \
  --cache_dir data/flux-conditioning
```

Then supply `--cache_dir data/flux-conditioning` to training. Use the same dimensions,
model revision, and teacher guidance. Cache identities include reference contents,
prompt, preprocessing settings, and Diffusers version. Cache files contain tensors and
metadata and may consume substantial storage for large datasets.

Checkpoints are `OUTPUT/checkpoint-000500`, etc. `OUTPUT/student` is the latest portable
inference export. Resume with the same training options and a larger final iteration count:

```bash
accelerate launch --num_processes 1 --num_machines 1 --mixed_precision bf16 -m main.train_flow \
  --model_family flux_klein --manifest experiments/flow/mixed.jsonl \
  --task mixed --num_steps 4 --output_dir outputs/flux-4step \
  --resume outputs/flux-4step/checkpoint-000500 --train_iters 20000
```

Resume restores student/fake weights, optimizers, RNG state, and iteration counters.
It rejects changes to model, schedule, conditioning data, world size, or update settings.
Checkpoint directories include training state; student exports do not. When both trainable
models use LoRA, compact checkpoints store both adapters and optimizer/RNG state without
duplicating the frozen base transformers. Existing full Accelerate checkpoints remain loadable.
Older checkpoints are retained unless `--keep_checkpoints N` is supplied; retention only
prunes this run's successfully completed checkpoints. Full-weight checkpoints still need ample disk.

## Repository-caption T2I recipe

Prepare the original repository's LAION caption source and held-out COCO prompts:

```bash
python3 -m main.prepare_repo_t2i --output_dir data/repo-t2i
python3 -u scripts/run_flux_klein_t2i.py \
  --data_dir data/repo-t2i --output_dir outputs/flux-klein-t2i
```

The supervisor uses the pinned FLUX Klein Base 4B, four stochastic steps, both
LoRAs at rank64/alpha8, both learning rates5e-5, clipping10,500-iteration LR warm-up,
batch1/accumulation8, and10,000 critic/2,000 student updates. Teacher CFG is4;
student and critic use conditional predictions without external CFG. Score-noise
sampling uses shift8 and student EMA decay is0.98, updated every outer iteration.
These sigma settings follow FastVideo's Wan reference recipe; they have not been
established as optimal for FLUX. Training and inference use the same saved native
FLUX sigma grid and fresh re-noising transitions.

Training samples from the full prepared caption corpus using a bounded on-demand
conditioning cache; no LAION images are needed. Fixed held-out comparisons show
the complete prompt and seed alongside native50-step teacher and four-step EMA
student images. Artifact warnings are report-only and do not stop training.

Check `outputs/flux-klein-t2i/run_status.json` and `train.log` for progress.
Once a checkpoint is saved, generate with the averaged student:

```bash
HF_HOME="$PWD/.cache/huggingface" python3 -m demo.flow_image \
  --checkpoint outputs/flux-klein-t2i/student-ema \
  --prompt "A ceramic teapot on a wooden table beside a sunlit window" \
  --num_steps 4 --seed 42 --output outputs/teapot.png
```

Raw weights remain in `student/`. Checkpoints also contain EMA state, both
optimizers, LR schedules and RNG state. Run `python3 -m main.audit_flow_validation
--help` for the completed-run artifact audit. Training losses and learned quality
metrics are not substitutes for held-out image review.

## HQ-Edit I2I run

Stream just a bounded source-image/instruction subset from `UCSC-VLAA/HQ-Edit`:

```bash
HF_HOME="$PWD/.cache/huggingface" HF_DATASETS_CACHE="$PWD/.cache/datasets" \
  python3 -m main.prepare_hqedit \
  --output_dir data/flux-klein-i2i-hqedit \
  --train_size 1024 --validation_size 64 --seed 42
```

This projects only `input_image` and `edit` from remote Parquet shards using streaming;
it never materializes the full dataset or its edited-target image column. Source hashes
prevent train/validation overlap. References preserve aspect ratio and are downscaled to
a maximum side of 512 for this run. Dataset revision, selected source hashes, and dimensions
are recorded alongside the manifests. HQ-Edit uses synthesized images.

The example I2I supervisor can be launched with:

```bash
python3 -u scripts/run_flux_klein_i2i.py
```

It uses the workspace's `.cache/` for all download/temporary/conditioning paths, requires
50 GiB free space before starting, runs five smoke updates and four held-out inference
checks, then resumes to 10,000 fake-score updates (2,000 student updates). Both models use
rank/alpha 32 LoRA, LR `1e-5`, BF16, 512x512, batch size 1, and gradient accumulation 4.
It retains the latest two complete checkpoints and generates 64 held-out student edits plus
eight 50-step teacher comparisons after training. Do not launch a duplicate supervisor
against an existing run; direct `main.train_flow --resume ...` is available for recovery.

Run state and logs are in `outputs/flux-klein-i2i-4step-lora/`, including `run_status.json`,
`smoke.log`, `smoke_inference.log`, `train.log`, and `evaluation.log`. `stage=complete`
means both training and final evaluation finished; `stage=failed` records the failure.
After the run finishes, verify saved update counts, adapters, data identity, and every
held-out image artifact with `python3 -m main.audit_flow_run`. A successful check writes
`completion_audit.json` in the run directory. The audit requires the full 10,000-update
run and all 64 student / eight teacher comparisons; smoke results cannot satisfy it.

## Inference examples

The same CLI handles either family and both tasks, loading the base identity and trained
step count from the export. It reuses saved sigma sequences for training resolutions;
new output resolutions use the saved scheduler configuration and native schedule policy.
One-step schedules use exact `[1, 0]` sigma endpoints; this avoids Qwen's native terminal
stretching becoming undefined when only one denoising point is supplied.

```bash
# FLUX T2I
python3 -m demo.flow_image --checkpoint outputs/flux-4step/student \
  --prompt "A mountain at sunset" --seed 42 --output generated/flux-t2i.png

# FLUX I2I
python3 -m demo.flow_image --checkpoint outputs/flux-4step/student \
  --prompt "Change the jacket to red" \
  --reference_images experiments/flow/images/person.png --seed 42 --output generated/flux-i2i.png

# Qwen T2I
python3 -m demo.flow_image --checkpoint outputs/qwen-4step/student \
  --prompt "A mountain at sunset" --seed 42 --output generated/qwen-t2i.png

# Qwen I2I
python3 -m demo.flow_image --checkpoint outputs/qwen-4step/student \
  --prompt "Change the jacket to red" \
  --reference_images experiments/flow/images/person.png --seed 42 --output generated/qwen-i2i.png
```

For one-step inference substitute `outputs/flux-1step/student` or
`outputs/qwen-1step/student`. `--num_steps` is optional and must match the export if
provided. Use `--height`/`--width` for a different aligned resolution. New resolutions
and few-step quality require evaluation; schedule compatibility does not guarantee quality.
Multiple references follow `--reference_images ref1.png ref2.png` in their intended order.
Qwen outputs should use PNG, WebP, or TIFF to preserve alpha. I2I means reference editing,
not a strength-based partial-noise img2img or mask-inpainting interface.

## Validation

```bash
OMP_NUM_THREADS=1 python3 -m pytest -q tests/test_flow.py
```

Tests use tiny native random transformers, covering both tasks and both sampling targets,
LoRA/full gradients, frozen teacher parameters, exports, and CUDA accumulation/resume.
They establish execution/math correctness; they do not establish pretrained image quality.
Real-checkpoint training/inference needs downloaded base weights and a subsequent quality
evaluation on held-out T2I prompts and I2I source/instruction pairs.
