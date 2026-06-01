#!/usr/bin/env bash
# Causal discovery on pendulum. Pass method as $1 (default: sdcd).
# Edit --data-root to point at your local pendulum dataset.

python SCM_modeling/discover_causal.py \
    --dataset pendulum \
    --method-name "${1:-sdcd}" \
    --device cpu \
    --data-root <pendulum dataset root, parent folder of "train"/"test" subfolders> 
