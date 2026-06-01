#!/usr/bin/env bash
# SD3 Causal-Adapter training on CelebA-HQ-simple (7 attributes).
# Edit the variables below to point at your local checkpoints / dataset.

# <local SD3-medium snapshot folder, OR "stabilityai/stable-diffusion-3-medium-diffusers">
PRETRAINED_PATH="<set me>"

# <root of the counterfactual-benchmark CelebA-HQ datasets> can be None if you don't want to use it.
DATA_ROOT="<set me>"

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --num_machines=1 train_SD3.py \
    --output_name "causal-adapter" \
    --output_dir "./logs/logs_celebahq_simple_all" \
    --pretrained_model_name_or_path "${PRETRAINED_PATH}" \
    --train_data_dir "${DATA_ROOT}" \
    --dataset 'celebahq_simple' \
    --learnable_property "object" \
    --resolution 512 \
    --train_batch_size 2 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 200000 \
    --save_steps 5000 \
    --task_cond 'generation_text_global_after' \
    --checkpointing_steps 999999 \
    --learning_rate 1e-5 \
    --lr_scheduler "constant" \
    --lr_warmup_steps 0 \
    --num_validation_images 1 \
    --placeholder_string 'a human of @ and * and mouth and gender and ** and $ and #' \
    --presudo_words '@,*,mouth,gender,**,$,#' \
    --presudo_words_infonce '@,*,mouth,gender,**,$,#' \
    --mcpl_training "True" \
    --causal_training "False" \
    --random_prompt_template "False" \
    --mixed_precision "bf16" \
    --T5_injection "False" \
    --gradient_checkpointing \
    --use_8bit_adam
