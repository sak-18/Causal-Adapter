# Causal-Adapter 🚀

Official implementation of ICML 2026 paper:
"Causal-Adapter: Taming Text-to-Image Diffusion for Faithful Counterfactual Generation."

## Repository Structure 🔧

This repository is organized as a monorepo with three major components:

- `causal-adapter-sd15/`: Stable Diffusion v1.5 implementation.
- `causal-adapter-sd3/`: Stable Diffusion 3 / Flux implementation.
- `counterfactual-benchmark/`: Benchmark and evaluation pipeline.

## Quick Start ⚡

1. Choose one component folder above.
2. Follow the folder-level README for environment setup and commands.
3. Install local modified `diffusers` in editable mode where required:

```bash
pip install -e diffusers
```

## Pretrained Weights 🔥

Project checkpoints are published on Hugging Face:

- `LeiTong02/Causal-Adapter`

Do not commit large model artifacts/checkpoints to git.

## Reproducibility Notes ✅

- SD1.5 and SD3 implementations are kept separate by design.
- Keep separate environments for SD1.5 and SD3 for reproducibility.

## Citation 📚

See `CITATION.cff` for citation metadata.
