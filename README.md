<p align="center">
  <img src="./CA_logo.png" alt="Causal-Adapter logo" width="280"/>
</p>

<p align="center">
  <b>Taming Text-to-Image Diffusion for Faithful Counterfactual Generation</b>
  <br/>
  <em>ICML 2026</em>
</p>

<p align="center">
  <a href="#news">News</a> ·
  <a href="#overview">Overview</a> ·
  <a href="#repository-structure">Structure</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#training">Training</a> ·
  <a href="#inference">Inference</a> ·
  <a href="#gpu-memory">GPU Memory</a> ·
  <a href="#pretrained-weights">Pretrained Weights</a> ·
  <a href="#citation">Citation</a>
</p>

---

## News

- ⏳ **TBD** — Release pretrained weights (Causal-Adapter heads, SCMs, MCPL embeddings) on [`LeiTong/Causal-Adapter`](https://huggingface.co/LeiTong/Causal-Adapter).
- ⏳ **TBD** — Open-source the benchmark / evaluation pipeline under `counterfactual-benchmark/` (Effectiveness / Composition / Reverse / FID).
- ✅ **2026-06-01** — SD1.5 and SD3 codebases unified into a single project root with a shared `diffusers/` fork (0.36.0.dev0).
- ✅ **2026-05-29** — SD1.5: shipped reproducible counterfactual inference notebooks (Pendulum / CelebA / ADNI) sharing a common `inference_utils.py`.
- ✅ **2026-05-28** — Open-sourced Causal-Adapter training code across Pendulum, ADNI, and CelebA.
- ✅ **2026-05-01** — [Causal-Adapter](https://icml.cc/virtual/2026/poster/61202) accepted as a poster at **ICML 2026**.

## Overview

Causal-Adapter equips text-to-image diffusion models with a lightweight causal
ControlNet head and a small set of MCPL pseudo-token embeddings, enabling
faithful counterfactual generation under interventions on causal attributes.
This repository hosts the official reference implementation for both Stable
Diffusion v1.5 and Stable Diffusion 3 backbones, together with the benchmark
used for evaluation.

> Some internal symbols and scripts still carry legacy `MCPL` / `presudo`
> names — kept for backward compatibility.

## Repository Structure

SD1.5 and SD3 share a single project layout and a single `diffusers/` install:

| Path | Purpose |
| --- | --- |
| `train.py` | SD1.5 training entrypoint. |
| `train_SD3.py` | SD3 training entrypoint. |
| `causal_datasets/` | Dataset adapters (CelebA / CelebA-HQ / ADNI / MorphoMNIST / Pendulum / …) — shared by both backbones. |
| `causal_modules/` | Causal ControlNet heads, DDIM/Flow modules (`ddim_modules.py`, `ddim_modules_sd3.py`, `ddim_modules_flux.py`), SCM pretraining, and `p2p_edits/`. |
| `SCM_modeling/` | Causal discovery (DAGMA / NoTEARS / SDCD) and SCM training. |
| `notebook_benchmarks/` | Counterfactual inference notebooks (Pendulum / CelebA / ADNI / SD3 CelebA-HQ) sharing `inference_utils.py`. |
| `diffusers/` | Project fork of `diffusers` (0.36.0.dev0) exporting `Causal_ControlNetModel` (SD1.5) and `Causal_SD3ControlNetModel` (SD3). |
| `counterfactual-benchmark/` | Benchmark and evaluation pipeline. |
| `scripts/` | Shell wrappers — `scripts/adapter_training/*.sh` (Causal-Adapter training) and `scripts/scm_training/*.sh` (SCM pretraining + causal discovery). |
| `environment.yml` | Pinned conda environment (PyTorch 2.4.1 / CUDA 12.1 / diffusers 0.36 / transformers 4.46). |
| `utils_sd3.py` | SD3 train/val split + SCM metric helpers. |
| `pendulum.py` | Pendulum data generation script. |

## Installation

```bash
conda env create -f environment.yml
conda activate causal-adapter
pip install -e diffusers
```

`environment.yml` is pinned to the versions actually exercised on the
cluster (PyTorch 2.4.1 + CUDA 12.1 + diffusers 0.36 + transformers 4.46);
the editable `diffusers` install above is what `train.py` / `train_SD3.py`
load at runtime.

Make sure the project root is on `PYTHONPATH` so the patched
`Causal_ControlNetModel` can lazily import
`causal_modules.scm_pretraining.load_dataset_model`.

## Workflow

1. **Prepare or download a dataset** for your target domain. See the loaders
   under `causal_datasets/` for the expected on-disk layout.
2. **Prepare a pretrained backbone** — either a local SD / miniSD / SD3 folder
   (recommended behind a firewall) or a HuggingFace model id.
3. **Train or load an SCM** with the scripts under `SCM_modeling/`. The
   training scripts consume the resulting checkpoint via `--scm_path`.
4. **Run Causal-Adapter / MCPL training** with `train.py` (SD1.5) or
   `train_SD3.py` (SD3).
5. **Run inference** with the notebooks under `notebook_benchmarks/`.

## Training

### Required paths

| Argument | Required | What it points at |
| --- | --- | --- |
| `--pretrained_model_name_or_path` | yes | HuggingFace model id (e.g. `runwayml/stable-diffusion-v1-5`, `stabilityai/stable-diffusion-3-medium-diffusers`) **or** a local snapshot folder. |
| `--train_data_dir` | yes | Dataset root. Layout depends on `--dataset`. |
| `--scm_path` | optional | Pretrained SCM checkpoint produced by `SCM_modeling/`. Required when `--causal_training False` (the default) — the SCM head is frozen and loaded from this path. |

### Shell scripts

Edit the `<set me>` placeholders at the top of each script and run from
the repo root:

```bash
# SD1.5
bash scripts/adapter_training/train_pendulum.sh
bash scripts/adapter_training/train_adni.sh
bash scripts/adapter_training/train_celeba_complex.sh

# SD3 (uses accelerate + bf16)
bash scripts/adapter_training/train_sd3_celebahq.sh
```

See [`scripts/README.md`](./scripts/README.md) for the script index and
the recommended SCM-then-Causal-Adapter sequence.

### Important defaults

| Flag | Default | Notes |
| --- | --- | --- |
| `--mcpl_training` | `True` | Trains the pseudo-token embeddings. |
| `--causal_training` | `False` | When `False`, SCM head is frozen — pass `--scm_path`. |
| `--task_cond` | `generation_text_global_after` | Where the causal vector is injected. All tested runs use this. |
| `--learning_rate` | `1e-5` | CelebA-complex uses `5e-6`; override per-dataset. |
| `--lr_scheduler` | `constant` | |
| `--lr_warmup_steps` | `0` | |
| `--gradient_accumulation_steps` | `1` | ADNI (SD1.5) uses `2`. |
| `--mixed_precision` | `no` (SD1.5) / `bf16` (SD3) | Override to halve activation memory. |

Run `python train.py --help` or `python train_SD3.py --help` for the full surface.

## Inference

Once you have a trained run (a `controlnet-steps-*.safetensors` plus matching
`learned_embeds-*.safetensors`, optionally an SCM checkpoint), the notebooks
under `notebook_benchmarks/` reproduce the inversion / intervention /
attention-map figures used in the paper.

| Notebook | Backbone | Dataset |
| --- | --- | --- |
| `counterfactuals_pendulum.ipynb` | SD1.5 | Pendulum |
| `counterfactuals_celeba.ipynb`   | SD1.5 | CelebA (complex) |
| `counterfactuals_ADNI.ipynb`     | SD1.5 | ADNI |
| `counterfactuals_celebahq_SD3.ipynb` | SD3 | CelebA-HQ (simple) |

The three SD1.5 notebooks share `notebook_benchmarks/inference_utils.py`,
which centralises pipeline assembly (`Causal_ControlNetModel` load → SCM head
load → adjacency mask → MCPL textual-inversion embeddings → `DDIMScheduler`
configured for inversion). The SD3 notebook follows the same configuration
pattern.

To run one:

1. Open the notebook in JupyterLab / VSCode.
2. Edit the **Configuration** cell. Each notebook is shipped with empty
   strings — set `MODEL_CACHE`, `LOGS_ROOT`, `DATA_ROOT`, the `BASE_MODEL_PATH`,
   the ControlNet / pseudo-token / SCM checkpoint paths, and (for SD3) the
   three `EMBEDDING_*_PATH` files. They can also be set via env vars.
3. Restart the kernel and run all cells.

## GPU Memory

Approximate guidance against the tested commands. Real usage depends on
batch size, resolution, mixed precision, gradient checkpointing, and which
loss terms are enabled — profile your own setup.

| Dataset | Backbone | Resolution | Batch size | Precision | Train | Inference |
| --- | --- | --- | --- | --- | --- | --- |
| Pendulum | SD1.5 | 256 | 2 | fp32 | ~16 GB | ~6.2 GB |
| CelebA-complex | SD1.5 | 256 | 2 | fp32 | ~16 GB | ~6.2 GB |
| ADNI | SD1.5 | 256 | 2 | fp32 | ~16 GB | ~6.2 GB |
| CelebA-HQ | SD3 | 512 | 2 | bf16 | ~36 GB | ~21 GB |

To shrink the footprint:

- `--mixed_precision fp16` (or `bf16`).
- `--gradient_checkpointing` — trades compute for activation memory.
- Lower `--train_batch_size` and raise `--gradient_accumulation_steps`.
- Disable contrastive training with `--presudo_words_infonce ""` if the
  InfoNCE term is not needed.

## Pretrained Weights

Tested SD1.5 backbones:
- `lambdalabs/miniSD-diffusers`
- `runwayml/stable-diffusion-v1-5`

Tested SD3 backbone:
- `stabilityai/stable-diffusion-3-medium-diffusers`

Project checkpoints — Causal-Adapter heads, pretrained SCMs, and learned MCPL
embeddings — will be released on Hugging Face under
[`LeiTong02/Causal-Adapter`](https://huggingface.co/LeiTong02/Causal-Adapter)
(see [News](#news)).

Do **not** commit large model artifacts or dataset binaries to git; the
top-level `.gitignore` already excludes common formats (`*.safetensors`,
`*.ckpt`, `*.pt`, `*.npy`, …).

## Reproducibility

- SD1.5 (`train.py`) and SD3 (`train_SD3.py`) share a single dataset / p2p
  layer (`causal_datasets/`, `causal_modules/p2p_edits/`) and a single
  `diffusers/` fork.
- Tested commands live in [`scripts/adapter_training/`](./scripts/adapter_training/)
  (Causal-Adapter) and [`scripts/scm_training/`](./scripts/scm_training/)
  (SCM pretraining + causal discovery).
- See [`CHANGELOG.md`](./CHANGELOG.md) for repository-level changes.
- Output runs are written to
  `<output_dir>/<timestamp>-<output_name><task_cond>/` and contain TensorBoard
  logs, periodic `learned_embeds-*.safetensors`, and the controlnet
  checkpoint.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{tong2026causaladapter,
  title     = {Causal-Adapter: Taming Text-to-Image Diffusion for Faithful Counterfactual Generation},
  author    = {Tong, Lei and Liu, Zhihua and Lu, Chaochao and Oglic, Dino and Diethe, Tom and Teare, Philip and Tsaftaris, Sotirios A. and Jin, Chen},
  booktitle = {Proceedings of the Forty-third International Conference on Machine Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=si8F5lk6Kg},
  note      = {arXiv:2509.24798}
}
```

## License

Original code in this repository is released under the **Apache License 2.0**:

```
Copyright AstraZeneca UK Ltd. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
```

Source files authored by us carry this SPDX header. This repository also builds
on third-party code, which retains its original license and copyright:

| Component | Source | License |
|---|---|---|
| `diffusers/` | [huggingface/diffusers](https://github.com/huggingface/diffusers) (vendored fork) | Apache-2.0 |
| `train.py`, `train_SD3.py` | HuggingFace `diffusers` training scripts (modified) | Apache-2.0 |
| `causal_modules/p2p_edits/{ptp_utils,seq_aligner,p2p_ldm_utils}.py` | [google/prompt-to-prompt](https://github.com/google/prompt-to-prompt) (modified) | Apache-2.0 |
| `causal_modules/sdcd_modules.py`, `SCM_modeling/causal_discovery/**` | DCDI / DCDFG / NOTEARS / DAGMA reference implementations | MIT / authors' licenses |
| `counterfactual-benchmark/` | [gulnazaki/counterfactual-benchmark](https://github.com/gulnazaki/counterfactual-benchmark) (NeurIPS 2024, modified) | MIT |
| `pendulum.py` | Huawei Technologies Co., Ltd. | MIT |

Files derived from upstream sources keep the upstream copyright header; where we
made substantive changes we prepend a `Modifications Copyright AstraZeneca` line
above the original header. Vendored upstream code (e.g. `diffusers/`) is left
unmodified and is **not** relicensed.
