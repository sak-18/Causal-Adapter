#!/usr/bin/env bash
# SD1.5 Causal-Adapter training on Pendulum (4 attributes).
# Edit the variables below to point at your local checkpoints / dataset.

# <local SD1.5 / miniSD snapshot folder, OR a HF id like "runwayml/stable-diffusion-v1-5">
PRETRAINED_PATH="<set me>"

# <pendulum dataset root, parent folder of "train"/"test" subfolders>
DATA_ROOT="<set me>"

# <SCM checkpoint produced by scripts/scm_training/train_scm_pendulum.sh> can be None if you don't want to use it.
SCM_PATH="<set me>"

CUDA_VISIBLE_DEVICES=0 python train.py \
    --output_name "causal-adapter" \
    --output_dir "./logs/logs_pendulum_all" \
    --pretrained_model_name_or_path "${PRETRAINED_PATH}" \
    --train_data_dir "${DATA_ROOT}" \
    --dataset "pendulum" \
    --resolution 256 \
    --train_batch_size 2 \
    --max_train_steps 20000 \
    --placeholder_string 'a image of @ and * and & and !' \
    --presudo_words '@,*,&,!' \
    --presudo_words_infonce '@,*,&,!' \
    --scm_path "${SCM_PATH}"
