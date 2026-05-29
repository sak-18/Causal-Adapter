# Causal-Adapter (SD1.5)

Stable Diffusion v1.5 implementation of Causal-Adapter. The trainer fine-tunes
a Causal ControlNet head plus a small set of MCPL pseudo-token embeddings on
top of a frozen SD1.5 / miniSD backbone.

> Some internal symbols and scripts still carry legacy `MCPL` / `presudo`
> names. They're kept for backward compatibility.

## Workflow

1. **Prepare or download the dataset** for your target domain (Pendulum,
   ADNI, CelebA, CelebA-HQ, MorphoMNIST, …). See the loaders in
   `causal_datasets/` for the expected on-disk layout.
2. **Prepare the pretrained diffusion model.** Either point at a local
   Stable Diffusion / miniSD folder (recommended behind a firewall) or a
   HuggingFace model id.
3. **Train or load an SCM** with the scripts under `SCM_modeling/`. The
   training script consumes the resulting checkpoint via `--scm_path`.
4. **Run Causal-Adapter / MCPL training** with `train.py` (examples below).

## Installation

Create your Python / Conda environment, then install the local modified
`diffusers` package in editable mode:

```bash
pip install -e diffusers
```

## Required paths

`train.py` takes three path-style arguments. Two are required, one is
optional but commonly supplied:

| Argument | Required | What it points at |
| --- | --- | --- |
| `--pretrained_model_name_or_path` | yes | HuggingFace model id (e.g. `runwayml/stable-diffusion-v1-5`) **or** a local SD / miniSD folder containing `unet/`, `vae/`, `text_encoder/`, `tokenizer/`, `scheduler/`. |
| `--train_data_dir` | yes | Dataset root. Layout depends on `--dataset`. |
| `--scm_path` | optional | Path to a pretrained SCM checkpoint produced by `SCM_modeling`. Required when you want to start from a trained SCM head (the typical setup; `--causal_training` defaults to `False`). |

### Behind a firewall

If your machine cannot reach `huggingface.co`, download the SD1.5 / miniSD
checkpoint once on a connected machine and copy the snapshot folder over.
Pass the absolute path to `--pretrained_model_name_or_path`. Tested local
checkpoint:

```
.../models--lambdalabs--miniSD-diffusers/snapshots/<hash>/
```

### Open-source users

If you do have internet access, you can pass any HuggingFace SD1.5 model id
directly:

```bash
--pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5"
```

The first run will cache the snapshot under `~/.cache/huggingface/`.

## Quick start

Train on Pendulum (matches `test_commands.md`):

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --pretrained_model_name_or_path "/path/to/miniSD-diffusers" \
  --train_data_dir "/path/to/pendulum/train/" \
  --dataset "pendulum" \
  --resolution 256 \
  --train_batch_size 2 \
  --max_train_steps 20000 \
  --placeholder_string 'a image of @ and * and & and !' \
  --presudo_words '@,*,&,!' \
  --presudo_words_infonce '@,*,&,!' \
  --scm_path "/path/to/scm/best_model.pt" \
  --output_dir ./logs/logs_pendulum_all \
  --output_name causal-adapter
```

The full set of tested commands (Pendulum / ADNI / CelebA-complex) lives in
`test_commands.md`.

## Inference (counterfactual notebooks)

Once you have a trained Causal-Adapter run (a `controlnet-steps-*.safetensors`
+ matching `learned_embeds-steps-*.safetensors` pair, optionally an SCM
checkpoint), the three notebooks under `notebook_benchmarks/` reproduce the
DDIM-inversion / intervention / cross-attention figures used in the paper.

| Notebook | Dataset |
| --- | --- |
| `notebook_benchmarks/counterfactuals_pendulum.ipynb` | Pendulum |
| `notebook_benchmarks/counterfactuals_celeba.ipynb`   | CelebA (complex) |
| `notebook_benchmarks/counterfactuals_ADNI.ipynb`     | ADNI |

All three share `notebook_benchmarks/inference_utils.py`, which centralises
the pipeline assembly (`Causal_ControlNetModel` load → SCM head load →
adjacency mask → MCPL textual-inversion embeddings → `DDIMScheduler`
configured for inversion). Only dataset-specific pieces (sample loading,
intervention encoding, attention plotting) live in each notebook.

To run one of them:

1. Open the notebook in JupyterLab / VSCode.
2. Edit the **Configuration** cell. Two roots drive every other path:
   - `MODEL_CACHE` — local HuggingFace snapshot cache holding the frozen
     SD1.5 / miniSD backbone.
   - `LOGS_ROOT` — directory holding the training-run sub-folders produced
     by `train.py` and the SCM pretraining runs from `SCM_modeling/`.

   The notebook then derives `BASE_MODEL_PATH`, `RUN_DIR`, the
   ControlNet / pseudo-token / SCM checkpoint paths, and the dataset root
   from those two roots.
3. Restart the kernel and run all cells.

`inference_utils.py` ships with the same per-dataset metadata that
`train.py` uses (adjacency masks, default training prompt, pseudo-word set,
dataset-specific torchvision prefix), so training and inference stay in
sync — when you add a new dataset to `train.py`, mirror the entry in
`inference_utils.A_MATRICES` / `DATASET_PROMPTS`.

### Important defaults

| Flag | Default | Notes |
| --- | --- | --- |
| `--mcpl_training` | `True` | Trains the pseudo-token embeddings. |
| `--causal_training` | `False` | When `False` (the default), the SCM head is frozen and you must pass `--scm_path`. |
| `--task_cond` | `generation_text_global_after` | Where the causal vector is injected. All tested runs use this. |
| `--learning_rate` | `1e-5` | CelebA-complex uses `5e-6`; override per-dataset. |
| `--lr_scheduler` | `constant` | |
| `--lr_warmup_steps` | `0` | |
| `--gradient_accumulation_steps` | `1` | ADNI uses `2`. |
| `--mixed_precision` | `no` | Pass `fp16` or `bf16` to halve activation memory. |

Run `python train.py --help` for the full surface.

## GPU memory

These are **approximate** recommendations measured against the tested
commands above. Real usage depends on the batch size, resolution, mixed
precision, gradient checkpointing, and whether ControlNet / SCM /
contrastive losses are enabled. Treat the numbers as a starting point and
profile your own setup.

| Dataset | Resolution | Batch size | Precision | Recommended GPU memory (train)| Recommended GPU memory (inference)|
| --- | --- | --- | --- | --- |
| Pendulum | 256 | 2 | fp32 | ~16 GB |6234MiB|
| CelebA-complex | 256 | 2 | fp32 | ~16 GB |6234MiB|
| ADNI | 256 | 16 (`grad_accum=2`) | fp32 | ~24 GB |6234MiB|
| CelebA-HQ-simple | 512 | 2 | fp32 | ~24 GB |6234MiB|
| Any of the above | 512 | 4 | fp16 / bf16 | ~24 GB |6234MiB|

To shrink the footprint:

- `--mixed_precision fp16` (or `bf16`) — halves activations.
- `--gradient_checkpointing` — trades compute for memory on the UNet and
  text encoder.
- Lower `--train_batch_size` and raise `--gradient_accumulation_steps` to
  keep the effective batch size constant.
- Disable contrastive training (`--presudo_words_infonce ""`) if you don't
  need the InfoNCE term.

## Pretrained weights

Tested SD1.5 backbones:

- `lambdalabs/miniSD-diffusers`
- `runwayml/stable-diffusion-v1-5`

Project checkpoints (Causal-Adapter heads, SCMs) are released separately on
HuggingFace under `LeiTong02/Causal-Adapter`. Do **not** commit weights into
this repo — the `.gitignore` already excludes `*.safetensors`, `*.ckpt`,
`*.pt`, etc.

## Notes

- If you previously installed a conflicting `diffusers` version, uninstall
  and reinstall the local editable package (`pip install -e diffusers`).
- Output runs are written to
  `<output_dir>/<timestamp>-<output_name><task_cond>/` and contain the
  TensorBoard logs, periodic `learned_embeds-steps-*.safetensors`, and the
  controlnet checkpoint.
