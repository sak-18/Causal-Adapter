<p align="center">
  <img src="./CA_logo.png" alt="Causal-Adapter logo" width="220"/>
</p>

<h1 align="center">Causal-Adapter</h1>

<p align="center">
  <b>Taming Text-to-Image Diffusion for Faithful Counterfactual Generation</b>
  <br/>
  <em>ICML 2026</em>
</p>

<p align="center">
  <a href="#repository-structure">Structure</a> ·
  <a href="#getting-started">Getting Started</a> ·
  <a href="#pretrained-weights">Pretrained Weights</a> ·
  <a href="#reproducibility">Reproducibility</a> ·
  <a href="#citation">Citation</a>
</p>

---

## Overview

Causal-Adapter equips text-to-image diffusion models with a lightweight causal
ControlNet head and a small set of MCPL pseudo-token embeddings, enabling
faithful counterfactual generation under interventions on causal attributes.
This repository hosts the official reference implementations for both Stable
Diffusion v1.5 and Stable Diffusion 3 / Flux backbones, together with the
benchmark used for evaluation.

## Repository Structure

This repository is organized as a monorepo with three components:

| Directory | Purpose |
| --- | --- |
| [`causal-adapter-sd15/`](./causal-adapter-sd15) | Stable Diffusion v1.5 implementation (training, datasets, SCM head). |
| [`causal-adapter-sd3/`](./causal-adapter-sd3) | Stable Diffusion 3 / Flux implementation. |
| [`counterfactual-benchmark/`](./counterfactual-benchmark) | Benchmark and evaluation pipeline. |

Each component is self-contained and ships its own README with installation
notes, dataset layout, and example commands.

## Getting Started

1. Pick a component above based on the diffusion backbone you target.
2. Follow that component's README for environment setup, dataset preparation,
   and example training/evaluation commands.
3. Where required, install the local modified `diffusers` in editable mode:

   ```bash
   pip install -e diffusers
   ```

We recommend keeping a separate Python environment per component (SD1.5 vs
SD3/Flux) to avoid dependency clashes.

## Pretrained Weights

Project checkpoints — Causal-Adapter heads, pretrained SCMs, and learned
MCPL embeddings — are released on Hugging Face:

- [`LeiTong02/Causal-Adapter`](https://huggingface.co/LeiTong02/Causal-Adapter)

Please do **not** commit large model artifacts or dataset binaries to git;
the per-component `.gitignore` files already exclude common formats
(`*.safetensors`, `*.ckpt`, `*.pt`, `*.npy`, …).

## Reproducibility

- SD1.5 and SD3/Flux implementations are kept separate by design; their
  training entry points, dataset adapters, and `diffusers` versions differ.
- Hyperparameters, dataset splits, and tested commands are documented inside
  each component's README and `test_commands.md`.
- See [`CHANGELOG.md`](./CHANGELOG.md) for notable repository-level changes.

## Citation

If you find this work useful, please cite our paper. Citation metadata is
provided in [`CITATION.cff`](./CITATION.cff).
