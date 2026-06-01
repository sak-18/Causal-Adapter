#!/usr/bin/env bash
# SD1.5 Causal-Adapter training on ADNI (6 attributes).
# Edit the variables below to point at your local checkpoints / dataset.

# <local SD1.5 / miniSD snapshot folder, OR a HF id>
PRETRAINED_PATH="<set me>"

# <ADNI preprocessing root, e.g. .../adni/preprocessing>
DATA_ROOT="<set me>"

# <SCM checkpoint produced by scripts/scm_training/train_scm_adni.sh> can be None if you don't want to use it.
SCM_PATH="<set me>"

CUDA_VISIBLE_DEVICES=0 python train.py \
    --output_name "controlnet_textcond_contrast_nohorizonflip" \
    --output_dir "./logs/logs_ADNI_all" \
    --pretrained_model_name_or_path "${PRETRAINED_PATH}" \
    --train_data_dir "${DATA_ROOT}" \
    --dataset "ADNI" \
    --resolution 256 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 100000 \
    --placeholder_string 'a mri image of @ and * and &' \
    --presudo_words '@,*,&' \
    --presudo_words_infonce '@,*,&' \
    --scm_path "${SCM_PATH}"
