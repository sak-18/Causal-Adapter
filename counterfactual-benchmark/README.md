# 🧪 Counterfactual Evaluation for Causal-Adapter

> Stable Diffusion-based counterfactual image generation evaluation for **Causal-Adapter**.

This subproject is adapted from the **Benchmarking Counterfactual Image Generation** benchmark  
(**NeurIPS 2024 Datasets & Benchmarks**) and extended to evaluate the **Causal-Adapter** Stable Diffusion counterfactual generator.

The original Deep-SCM training and evaluation code is preserved, while a Stable Diffusion + Causal-ControlNet evaluation pipeline is added on top. The generated counterfactual images are evaluated using the same benchmark metrics:

- ✅ **Effectiveness**
- 🔁 **Reverse**
- 🧩 **Composition**
- 📊 **FID / Minimality**

> For the original benchmark, including the paper, project page, Deep-SCM training pipeline, full dataset instructions, and citation, please refer to the upstream repository:  
> **https://github.com/gulnazaki/counterfactual-benchmark**  
> Paper: https://arxiv.org/abs/2403.20287  
>
> This README documents only the **Causal-Adapter Stable Diffusion extensions**.

---

## ✨ What was adapted?

This fork adapts the original counterfactual benchmark to support **Causal-Adapter** evaluation.

The main changes are:

- The `causal_modules` package is now expected at the **Causal-Adapter project root**, one level above `counterfactual-benchmark/`.
- The diffusion editing primitives are provided by `causal_modules`, including:
  - `ddim_editing`
  - `P2P_editing`
  - MCPL embedding loading
  - Causal-ControlNet conditioning utilities
- The Deep-SCM `SCM` class from `methods/deepscm/model.py` is reused for:
  - abducting exogenous noise;
  - decoding intervened parent variables;
  - producing counterfactual labels used to condition Causal-ControlNet.

`causal_modules` should now be imported as a normal package from the project root.

---

## 📁 Relevant files

Main files under:

```text
counterfactual_benchmark/methods/deepscm/
```

| File / Folder | Purpose |
|---|---|
| `evaluate_SD_DSCM.py` | Main Stable Diffusion counterfactual evaluation script with Deep-SCM intervention. Supports Effectiveness, Composition, Reverse, and FID / Minimality. |
| `evaluate_SD.py` | Legacy Stable Diffusion baseline mainly used for Pendulum runs without Deep-SCM intervention. It uses the Causal-Adapter causal-matrix style intervention. |
| `compute_FID.py` | Computes FID and minimality from counterfactual tensors saved by `evaluate_SD_DSCM.py`. |
| `_paths.py` | Shared path bootstrap utility. Import this before any first-party project modules. |
| `compute_final_metrics/` | Aggregates `final_results.pkl` files into final quantitative tables. See its own `README.md`. |

---

## 🚀 Main evaluation script

The main entry point is:

```bash
evaluate_SD_DSCM.py
```

It supports two orthogonal configuration axes:

### 1. Editing backend

#### `--editing standard`

Uses DDIM-based editing through `ddim_editing`.

This mode first inverts the factual image using DDIM inversion, then regenerates the image under the intervened causal conditioning.

It is the default option and is generally faster.

#### `--editing p2p`

Uses Prompt-to-Prompt editing through `P2P_editing`.

This mode additionally manipulates cross-attention and self-attention maps using per-attribute blend and cross-replace schedules. It is designed to better localise the edit and preserve identity, at the cost of additional computation.

---

### 2. Evaluation mode

#### `--reverse`

Runs both:

```text
factual → counterfactual
counterfactual → factual
```

This enables the **Reverse** metric, which measures reversibility. It also reports identity-preservation and composition-related LPIPS / L1 scores.

#### `--anti-classifier`

Uses the anti-causal CelebA-HQ predictor and the corresponding `*_anti.json` configuration files.

#### `--output-root`

Specifies the root directory for saved samples and metric outputs.

Default:

```text
saved_benchmark
```

#### `--max-batches`

Limits the number of evaluated batches for debugging.

Default:

```text
None
```

which means the full evaluation set is used.

Both editing backends share the same Deep-SCM abduction / decoding logic and the same metric computation. Only the image editing step differs.

To view the full argument list:

```bash
python evaluate_SD_DSCM.py -h
```

---

## 🧰 Prerequisites

For each dataset, please prepare the following components.

### Model checkpoints

- A trained **Causal-ControlNet** checkpoint:

```bash
--controlnet_path
```

- The learned **MCPL embeddings**:

```bash
--mcpl_embedding_path
```

- Anti-causal predictor checkpoints.

These are read from the classifier config field:

```text
ckpt_path
```

Checkpoints for CelebA, ADNI, and MorphoMNIST are bundled under:

```text
methods/deepscm/checkpoints/
```

- SCM mechanism checkpoints referenced by the corresponding `vae*.json` config files.

---

### Base Stable Diffusion model

By default, the code uses:

```text
lambdalabs/miniSD-diffusers
```

You can override this with:

```bash
export CAUSAL_ADAPTER_SD15_BASE_MODEL=/path/to/your/base/model
```

or another Hugging Face model id.

---

## 📦 Output structure

By default, outputs are saved under:

```text
<output_root>/<dataset>/DSCM_effectiveness_step<steps>_scale<scale>_<NTI|Blend><NTI>_seed<seed>/
```

A typical output folder looks like:

```text
<output_root>/<dataset>/DSCM_effectiveness_step<steps>_scale<scale>_<NTI|Blend><NTI>_seed<seed>/
│
├── composition/
│   └── compositon.txt              # legacy filename for composition metric log
│
├── <do_attr>/
│   └── final_results.pkl           # per-intervention effectiveness, LPIPS, reverse arrays
│
└── FID/
    ├── counterfactual_tensors.pt    # saved tensors for FID computation
    └── minimality.pt                # minimality-related tensors / scores
```

To aggregate the saved results into final tables, use the utilities in:

```text
methods/deepscm/compute_final_metrics/
```

See the corresponding `README.md` in that folder for details.

---

### 🔁 Effectiveness + Reverse

#### CelebA-HQ with P2P editing

```bash
accelerate launch evaluate_SD_DSCM.py \
  --dataset celebahq_simple \
  --editing p2p \
  --reverse \
  --anti-classifier \
  --metrics effectiveness \
  --embeddings lpips \
  --bs 10 \
  --guidance_scale 1.5 \
  --num_steps 100 \
  --NTI True \
  --controlnet_path /logs/.../controlnet-steps-200000.safetensors \
  --mcpl_embedding_path /logs/.../learned_embeds-steps-200000.safetensors
```

---

### ✅ Effectiveness

#### CelebA complex with standard DDIM editing

```bash
accelerate launch evaluate_SD_DSCM.py \
  --dataset celeA_complex \
  --editing standard \
  --metrics effectiveness \
  --embeddings lpips \
  --bs 100 \
  --guidance_scale 2.0 \
  --num_steps 100 \
  --NTI False \
  --controlnet_path /logs/.../controlnet-steps-300000.safetensors \
  --mcpl_embedding_path /logs/.../learned_embeds-steps-300000.safetensors
```

---

### 📊 FID / Minimality

#### CelebA complex

```bash
accelerate launch evaluate_SD_DSCM.py \
  --dataset celeA_complex \
  --metrics minimality \
  --embeddings lpips \
  --bs 10 \
  --guidance_scale 3.0 \
  --num_steps 50 \
  --NTI True \
  --controlnet_path /logs/.../controlnet-steps-300000.safetensors \
  --mcpl_embedding_path /logs/.../learned_embeds-steps-300000.safetensors
```

Then aggregate the saved tensors with:

```bash
python compute_FID.py
```

---

## ⚙️ Local paths to configure

Before running evaluation, please check the following path-dependent settings.

| Item | Where to configure |
|---|---|
| Causal-ControlNet checkpoint | `--controlnet_path` |
| MCPL embeddings | `--mcpl_embedding_path` |
| Predictor checkpoint folder | `ckpt_path` inside classifier config JSON files |
| SCM mechanism checkpoints | `checkpoint_dir` inside `vae*.json` config files |
| CelebA complex pretrained embedding path | `dataset_prompt_and_graph()` |
| Base SD model | `CAUSAL_ADAPTER_SD15_BASE_MODEL` |

---

## 📝 Notes for reproducibility

- The Stable Diffusion editing backend can affect identity preservation and local edit quality.
- `standard` and `p2p` share the same causal intervention logic but differ in the image editing mechanism.
- For fair comparison, keep the following settings consistent across runs:
  - `--num_steps`
  - `--guidance_scale`
  - `--NTI`
  - `--seed`
  - classifier checkpoint
  - SCM checkpoint
  - Causal-ControlNet checkpoint
  - MCPL embedding checkpoint
- If using local models due to firewall or offline environments, set `CAUSAL_ADAPTER_SD15_BASE_MODEL` explicitly.

---

##   Citation & credits

This benchmark is based on the work of Melistas et al. Please cite the original paper and refer to the upstream repository for the full benchmark details and third-party credits.

```bibtex
@inproceedings{melistas2024benchmarking,
  title={Benchmarking Counterfactual Image Generation},
  author={Thomas Melistas and Nikos Spyrou and Nefeli Gkouti and Pedro Sanchez and Athanasios Vlontzos and Yannis Panagakis and Giorgos Papanastasiou and Sotirios A. Tsaftaris},
  booktitle={The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2024},
  url={https://openreview.net/forum?id=0T8xRFrScB}
}
```

Upstream repository:

```text
https://github.com/gulnazaki/counterfactual-benchmark
```