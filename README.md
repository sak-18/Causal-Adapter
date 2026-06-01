<p align="center">
  <img src="./CA_logo.png" alt="Causal-Adapter logo" width="280"/>
</p>

<!--<h1 align="center">Causal-Adapter</h1>-->

<p align="center">
  <b>Taming Text-to-Image Diffusion for Faithful Counterfactual Generation</b>
  <br/>
  <em>ICML 2026</em>
</p>

<p align="center">
  <a href="#news">News</a> ·
  <a href="#repository-structure">Structure</a> ·
  <a href="#getting-started">Getting Started</a> ·
  <a href="#pretrained-weights">Pretrained Weights</a> ·
  <a href="#reproducibility">Reproducibility</a> ·
  <a href="#citation">Citation</a>
</p>

---

## News

- **2026-05-29** — SD1.5: shipped reproducible counterfactual inference notebooks (Pendulum / CelebA / ADNI) sharing a common `inference_utils.py`. See [`causal-adapter-sd15/README.md`](./causal-adapter-sd15/README.md#inference-counterfactual-notebooks).
- **2026-05-28** — Open-sourced Causal-Adapter training code across three datasets (Pendulum, ADNI, CelebA).
- **2026-05-01** — [Causal-Adapter](https://icml.cc/virtual/2026/poster/61202) accepted for a poster presentation at **ICML 2026** 🔥.

## Overview

Causal-Adapter equips text-to-image diffusion models with a lightweight causal
ControlNet head and a small set of MCPL pseudo-token embeddings, enabling
faithful counterfactual generation under interventions on causal attributes.
This repository hosts the official reference implementations for both Stable
Diffusion v1.5 and Stable Diffusion 3 / Flux backbones, together with the
benchmark used for evaluation.

## Repository Structure

SD1.5 and SD3/Flux now share a single project layout and a single
`diffusers/` install:

| Path | Purpose |
| --- | --- |
| `train.py` | SD1.5 training entrypoint. |
| `train_SD3.py` | SD3 training entrypoint. |
| `causal_datasets/` | Dataset adapters (CelebA-HQ, ADNI, MorphoMNIST, Pendulum, ...) — shared by both backbones. |
| `causal_modules/` | ControlNet heads, DDIM/Flow modules (`ddim_modules.py`, `ddim_modules_sd3.py`, `ddim_modules_flux.py`), SCM pretraining, p2p edits. |
| `SCM_modeling/` | Causal discovery (DAGMA, NoTEARS, SDCD) + SCM training. |
| `notebook_benchmarks/` | Counterfactual inference notebooks (Pendulum / CelebA / ADNI / SD3 CelebA-HQ). |
| `diffusers/` | Project fork of `diffusers` (0.36.0.dev0) with `Causal_ControlNetModel` and `Causal_SD3ControlNetModel`. |
| `counterfactual-benchmark/` | Benchmark and evaluation pipeline. |
| `test_commands.md` | SD1.5 reference commands. |
| `commands_training_sd3.md` | SD3 reference commands. |

## Getting Started

1. Install the shared `diffusers` fork in editable mode:

   ```bash
   pip install -e diffusers
   ```

2. Make sure the project root is on `PYTHONPATH` so the patched
   `Causal_ControlNetModel` can lazily import
   `causal_modules.scm_pretraining.load_dataset_model`.

3. Run training:
   - SD1.5 — see `test_commands.md`.
   - SD3 — see `commands_training_sd3.md`.

4. Run inference notebooks under `notebook_benchmarks/`.

## Pretrained Weights

Project checkpoints — Causal-Adapter heads, pretrained SCMs, and learned
MCPL embeddings — are released on Hugging Face:

- [`LeiTong02/Causal-Adapter`](https://huggingface.co/LeiTong02/Causal-Adapter)

Please do **not** commit large model artifacts or dataset binaries to git;
the per-component `.gitignore` files already exclude common formats
(`*.safetensors`, `*.ckpt`, `*.pt`, `*.npy`, …).

## Reproducibility

- SD1.5 (`train.py`) and SD3 (`train_SD3.py`) share a single dataset/p2p
  layer (`causal_datasets/`, `causal_modules/p2p_edits/`) and a single
  `diffusers/` fork.
- Hyperparameters, dataset splits, and tested commands live in
  `test_commands.md` (SD1.5) and `commands_training_sd3.md` (SD3).
- See [`CHANGELOG.md`](./CHANGELOG.md) for notable repository-level changes.

## Citation

If you find this work useful, please cite our paper.

@inproceedings{tong2026causaladapter,
  title     = {Causal-Adapter: Taming Text-to-Image Diffusion for Faithful Counterfactual Generation},
  author    = {Tong, Lei and Liu, Zhihua and Lu, Chaochao and Oglic, Dino and Diethe, Tom and Teare, Philip and Tsaftaris, Sotirios A. and Jin, Chen},
  booktitle = {Proceedings of the Forty-third International Conference on Machine Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=si8F5lk6Kg},
  note      = {arXiv:2509.24798}
}