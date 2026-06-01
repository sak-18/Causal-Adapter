#!/usr/bin/env bash
# Train SCM CausalNet on ADNI.
# Edit --data-root to point at your local ADNI dataset.

python SCM_modeling/train_scm.py \
    --dataset ADNI \
    --device cpu \
    --data-root  --data-root <ADNI dataset root, e.g. .adni/preprocessing>
