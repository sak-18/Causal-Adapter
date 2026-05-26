# Causal-Adapter (SD3 / Flux) 🚀

This repository contains the Stable Diffusion 3 / Flux implementation used in
the Causal-Adapter project.

Note: internal symbols and scripts may still contain legacy `MCPL` names for
backward compatibility.

## Installation 🔧

Create your Python/Conda environment, then install the local modified
`diffusers` package in editable mode:

```bash
pip install -e diffusers
```

## Pretrained Weights 🔥

Examples:

- `stabilityai/stable-diffusion-3-medium-diffusers`

Project weights/checkpoints are published separately on Hugging Face:

- `LeiTong02/Causal-Adapter`

Do not commit model weights/checkpoints to git.

## Quick Start ⚡

Set environment variables first:

```bash
export PROJECT_ROOT=/path/to/MCPL-diffuser-flux
export DATA_ROOT=/path/to/counterfactual-benchmark/datasets
export MODEL_ID=stabilityai/stable-diffusion-3-medium-diffusers
```

### Train SD3 (example) 🧪

```bash
accelerate launch train_causalnet_SD3.py \
  --output_name "causal-adapter-sd3" \
  --output_dir "${PROJECT_ROOT}/logs/logs_celebahq_simple_all" \
  --pretrained_model_name_or_path "${MODEL_ID}" \
  --dataset "celebahq_simple" \
  --train_data_dir "${DATA_ROOT}" \
  --learnable_property "object" \
  --resolution 512 \
  --train_batch_size 4 \
  --dataloader_num_workers 8 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 100 \
  --task_cond "generation_text_global_after" \
  --learning_rate 1e-5 \
  --lr_scheduler "constant" \
  --lr_warmup_steps 0 \
  --mixed_precision "bf16" \
  --gradient_checkpointing \
  --use_8bit_adam
```

## Notes 📝

- Keep this implementation separate from SD1.5 (`MCPL-diffuser`) to preserve
  reproducibility.
- Many scripts currently accept legacy argument names containing `mcpl`.