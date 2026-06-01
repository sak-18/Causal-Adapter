#!/usr/bin/env bash
# SD1.5 Causal-Adapter training on CelebA (complex SCM, 4 attributes).
# Edit the variables below to point at your local checkpoints / dataset.

# <local SD1.5 / miniSD snapshot folder, OR a HF id>
PRETRAINED_PATH="<set me>"

# <CelebA root expected by torchvision.datasets.CelebA(root=...)>
DATA_ROOT="<set me>"

# <SCM checkpoint produced by SCM_modeling/train_scm.py> can be None if you don't want to use it.
SCM_PATH="<set me>"

CUDA_VISIBLE_DEVICES=0 python train.py \
    --output_name "causal-adapter" \
    --output_dir "./logs/logs_celeA_complex_all" \
    --pretrained_model_name_or_path "${PRETRAINED_PATH}" \
    --train_data_dir "${DATA_ROOT}" \
    --dataset "celeA_complex" \
    --resolution 256 \
    --train_batch_size 2 \
    --max_train_steps 200000 \
    --learning_rate 5e-6 \
    --placeholder_string 'an image of @ * & !' \
    --presudo_words '@,*,&,!' \
    --presudo_words_infonce '@,*,&,!' \
    --scm_path "${SCM_PATH}"
