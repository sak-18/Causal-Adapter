# Causal-Adapter (SD1.5)

This repository contains the Stable Diffusion v1.5 implementation used in the
Causal-Adapter project.

Note: internal symbols and scripts may still contain legacy `MCPL` names for
backward compatibility.

## Installation

Create your Python/Conda environment, then install the local modified
`diffusers` package in editable mode:

```bash
pip install -e diffusers
```

## Pretrained Weights

Recommended model IDs:

- `stable-diffusion-v1-5/stable-diffusion-v1-5`
- `stabilityai/stable-diffusion-2`
- `lambdalabs/miniSD-diffusers`
- `Nihirc/Prompt2MedImage` (medical domain)

Project weights/checkpoints are published separately on Hugging Face:

- `LeiTong02/Causal-Adapter`

Do not commit model weights/checkpoints to git.

## Quick Start

Set these environment variables first:

```bash
export PROJECT_ROOT=/path/to/MCPL-diffuser
export DATA_ROOT=/path/to/dataset
export MODEL_ID=stable-diffusion-v1-5/stable-diffusion-v1-5
```

### Train (example)

```bash
accelerate launch MCPL.py \
  --output_name "causal-adapter-sd15" \
  --output_dir "${PROJECT_ROOT}/logs" \
  --pretrained_model_name_or_path "${MODEL_ID}" \
  --train_data_dir "${DATA_ROOT}/causal_4_concepts/pendulum/3" \
  --learnable_property "object" \
  --resolution 512 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 6100 \
  --checkpointing_steps 6100 \
  --learning_rate 5.0e-04 \
  --lr_scheduler "constant" \
  --lr_warmup_steps 0
```

### Inference (example)

```bash
python scripts/txt2img_inference.py
```

### Attention Maps (example)

```bash
python scripts/show_attn_maps/plt_attn_maps.py \
  --prompts_string "orange @ and red * and black & and black !" \
  --ckpt_path "${MODEL_ID}" \
  --embedding_path "${PROJECT_ROOT}/logs/<run>/learned_embeds-steps-XXXX.safetensors" \
  --out_base "${PROJECT_ROOT}/outputs/attention_maps" \
  --img_size 512
```

## Notes

- If you previously installed a conflicting package version, uninstall/reinstall
  your local editable `diffusers` package.
- Many scripts currently accept legacy argument names containing `mcpl`.
