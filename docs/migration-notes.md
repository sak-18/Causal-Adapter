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
- Extended portability cleanup across non-notebook code/config:
  - benchmark evaluation/metrics entrypoints in `counterfactual-benchmark/counterfactual_benchmark/methods/deepscm`
  - benchmark dataset and embedding modules under `counterfactual-benchmark/counterfactual_benchmark/ctf_datasets` and `evaluation/embeddings`
  - SD1.5/SD3 core train/inference helper scripts (`MCPL_linear*`, `causalnet*`, `causal_modules/*`)
  - kubectl YAML and command/config templates by replacing machine-specific absolute paths with placeholders

## Remaining Work

Legacy absolute paths now remain primarily in:

- notebook files (`*.ipynb`) with historical experiment cells/outputs

These should be migrated to environment-variable-driven configuration using:

- PROJECT_ROOT
- DATA_ROOT
- BASE_MODEL_PATH
- CONTROLNET_PATH
- MCPL_EMBEDDING_PATH
- OUTPUT_DIR

## Recommendation

Proceed with staged replacement by priority:

1. Notebook cleanup (remove machine-specific literals from code and markdown cells)
2. Optional notebook metadata/output stripping before public release
3. Final smoke tests for sd15/sd3/benchmark commands in fresh environments
