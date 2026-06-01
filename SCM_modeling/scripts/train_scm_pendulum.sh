#!/usr/bin/env bash
# Train SCM CausalNet on pendulum.
# Edit --data-root to point at your local pendulum dataset.

python SCM_modeling/train_scm.py \
    --dataset pendulum \
    --device cpu \
    --data-root <pendulum dataset root, parent folder of "train"/"test" subfolders> 
