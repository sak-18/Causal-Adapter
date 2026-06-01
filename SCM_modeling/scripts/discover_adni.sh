#!/usr/bin/env bash
# Causal discovery on ADNI. Pass method as $1 (default: sdcd).
# Edit --data-root to point at your local ADNI dataset.

python SCM_modeling/discover_causal.py \
    --dataset ADNI \
    --method-name "${1:-sdcd}" \
    --device cpu \
    --data-root <ADNI dataset root, e.g. .adni/preprocessing>
