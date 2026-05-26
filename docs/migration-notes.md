# Migration Notes

## Completed

- Migrated three repositories into monorepo subfolders:
  - causal-adapter-sd15
  - causal-adapter-sd3
  - counterfactual-benchmark
- Added root-level open-source files:
  - README.md
  - .gitignore
  - CITATION.cff
  - docs/reproducibility.md
  - envs/README.md
- Replaced hardcoded path usage in key entry scripts:
  - causal-adapter-sd15/scripts/txt2img_inference.py
  - causal-adapter-sd3/scripts/txt2img_inference.py
  - causal-adapter-sd3/scripts/control_editing.py
  - causal-adapter-sd3/scripts/controlnet_sampling.py
  - causal-adapter-sd15/causal_modules/ddim_modules.py
  - causal-adapter-sd15/causal_modules/__init__.py

## Remaining Work

A larger set of legacy absolute paths still exists in:

- show_attn_maps scripts
- kubectl yaml templates
- some training/evaluation scripts in sd15/sd3 and benchmark modules

These should be migrated to environment-variable-driven configuration using:

- PROJECT_ROOT
- DATA_ROOT
- BASE_MODEL_PATH
- CONTROLNET_PATH
- MCPL_EMBEDDING_PATH
- OUTPUT_DIR

## Recommendation

Proceed with staged replacement by priority:

1. Training/evaluation entrypoints
2. Production/benchmark configs
3. Utility/visualization scripts
4. Notebook examples (last)
