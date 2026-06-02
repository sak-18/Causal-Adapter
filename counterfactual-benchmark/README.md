# Counterfactual evaluation for Causal-Adapter (Stable Diffusion)

This is a fork of the **Benchmarking Counterfactual Image Generation** benchmark
(NeurIPS 2024 D&B), adapted to evaluate the **Causal-Adapter** Stable-Diffusion
counterfactual generator. The original Deep-SCM training/evaluation code is kept,
and a Stable-Diffusion + Causal-ControlNet evaluation pipeline is added on top,
scored with the same metrics (Effectiveness, Composition, Reverse, FID).

> For the original benchmark (paper, project page, Deep-SCM model training, the
> full datasets/fine-tuning instructions and citation), refer to the upstream
> repository: **https://github.com/gulnazaki/counterfactual-benchmark**
> ([paper](https://arxiv.org/abs/2403.20287)). This README only documents the
> Causal-Adapter SD additions.

## How it was adapted
- The `causal_modules` package (at the **Causal-Adapter project root**, one level
  above `counterfactual-benchmark/`) provides the diffusion editing primitives
  (`ddim_editing`, `P2P_editing`, MCPL embedding loading, ...).
- The Deep-SCM `SCM` class (`methods/deepscm/model.py`) is reused to abduct
  exogenous noise and decode the intervened parents; those labels condition the
  Causal-ControlNet.
- All SD evaluation scripts resolve imports through `methods/deepscm/_paths.py`
  (`from _paths import bootstrap; bootstrap()`), so they run identically whether
  launched from the project root or from `methods/deepscm`. There are no
  `sys.path.append("../../")` / `causal-adapter-sd15` hacks — `causal_modules` is
  imported as a normal package from the project root.

## Relevant files (`methods/deepscm/`)

| File | Purpose |
|---|---|
| `evaluate_SD_DSCM.py` | Main SD counterfactual evaluation: Effectiveness, Composition, Reverse, FID/minimality. |
| `evaluate_SD.py` | Legacy SD baseline used for the ADNI / pendulum runs (no celeba-HQ / anti-causal support). |
| `compute_FID.py` | Computes FID + minimality from the counterfactual tensors saved by `evaluate_SD_DSCM.py`. |
| `_paths.py` | Shared path bootstrap (`bootstrap()`); import before any first-party module. |
| `compute_final_metrics/` | Aggregates `final_results.pkl` into the final tables; see its `README.md`. |

## The main evaluation script

`evaluate_SD_DSCM.py` selects behaviour with flags along two orthogonal axes:

- **`--editing standard`** — DDIM-based editing (`ddim_editing`). Inverts the
  factual image with DDIM and regenerates it under the intervened conditioning.
  Simple and fast; the default.
- **`--editing p2p`** — Prompt-to-Prompt editing (`P2P_editing`). Additionally
  manipulates cross-/self-attention maps with a per-attribute blend / cross-
  replace schedule, which better localises the edit and preserves identity, at
  extra compute cost.
- **`--reverse`** — also runs the backward edit (counterfactual → factual) and
  reports the **Reverse** (reversibility) metric alongside identity-preservation
  (IDP) and composition LPIPS/L1.
- **`--anti-classifier`** — use the anti-causal celeba-HQ predictor and the
  `*_anti.json` configs.
- **`--output-root`** — root folder for saved samples/metrics (default
  `saved_benchmark`).
- **`--max-batches`** — cap the number of evaluated batches (debugging). Default
  is `None` = evaluate the full set.

Both editing backends share the same abduction/decoding and the same metrics;
only the image-edit step differs. Run `python evaluate_SD_DSCM.py -h` for the
full argument list.

## Prerequisites
Per dataset you need:
- a trained **Causal-ControlNet** checkpoint → `--controlnet_path`
- the learned **MCPL embeddings** → `--mcpl_embedding_path`
- **anti-causal predictor** checkpoints (path read from the classifier config
  `ckpt_path`; checkpoints for celeba/adni/morphomnist are bundled under
  `methods/deepscm/checkpoints/`)
- the **SCM mechanism checkpoints** referenced by the `vae*.json` config
- the dataset itself (CelebA / celeba-HQ / ADNI / pendulum)
- the base SD model: defaults to `lambdalabs/miniSD-diffusers`, override with the
  `CAUSAL_ADAPTER_SD15_BASE_MODEL` environment variable
- for `celeA_complex`, a pretrained causal-net embedding at
  `<project_root>/logs/logs_celeA_complex_all/.../best_model.pt` (loaded if present)

## Where outputs go
```
<output_root>/<dataset>/DSCM_effectiveness_step<steps>_scale<scale>_<NTI|Blend><NTI>_seed<seed>/
    composition/compositon.txt        # composition metric log
    <do_attr>/final_results.pkl       # per-intervention effectiveness + LPIPS/reverse arrays
    FID/counterfactual_tensors.pt     # tensors for FID
    FID/minimality.pt
```
Aggregate these into the final tables with the helpers in
`methods/deepscm/compute_final_metrics/` (see its `README.md`).

## Example commands
Run from `counterfactual_benchmark/methods/deepscm` (or with the full path from
the project root — both work). Use `accelerate launch` for multi-GPU.

**Effectiveness + Reverse (celeba-HQ, P2P editing):**
```
accelerate launch evaluate_SD_DSCM.py --dataset celebahq_simple \
  --editing p2p --reverse --anti-classifier \
  --metrics effectiveness --embeddings lpips \
  --bs 10 --guidance_scale 1.5 --num_steps 100 --NTI True \
  --controlnet_path /logs/.../controlnet-steps-200000.safetensors \
  --mcpl_embedding_path /logs/.../learned_embeds-steps-200000.safetensors
```

**Effectiveness (celeba complex, standard DDIM editing):**
```
accelerate launch evaluate_SD_DSCM.py --dataset celeA_complex \
  --editing standard --metrics effectiveness --embeddings lpips \
  --bs 100 --guidance_scale 2.0 --num_steps 100 --NTI False \
  --controlnet_path /logs/.../controlnet-steps-300000.safetensors \
  --mcpl_embedding_path /logs/.../learned_embeds-steps-300000.safetensors
```

**FID / minimality (celeba complex):**
```
accelerate launch evaluate_SD_DSCM.py --dataset celeA_complex \
  --metrics minimality --embeddings lpips --bs 10 \
  --guidance_scale 3.0 --num_steps 50 --NTI True \
  --controlnet_path ... --mcpl_embedding_path ...
# then aggregate the saved tensors with compute_FID.py
```

## Local paths you may need to change
- `--controlnet_path` / `--mcpl_embedding_path`: your checkpoint locations.
- `ckpt_path` inside the classifier config JSONs: predictor checkpoint folder.
- `checkpoint_dir` inside the `vae*.json` configs: SCM mechanism checkpoints.
- the `celeA_complex` pretrained-embedding path in `dataset_prompt_and_graph()`.
- `CAUSAL_ADAPTER_SD15_BASE_MODEL`: base SD model id/path.
- the `accelerate launch ...` lines and checkpoint paths inside `kubectl_p4d/`.

## Setup
```
virtualenv -p python3.10 venv
. venv/bin/activate
pip install -r requirements.txt
```

## Citation & credits
This benchmark is the work of Melistas et al. (NeurIPS 2024 D&B). Please cite the
original paper and see the [upstream repository](https://github.com/gulnazaki/counterfactual-benchmark)
for the full citation and third-party credits.
```
@inproceedings{melistas2024benchmarking,
  title={Benchmarking Counterfactual Image Generation},
  author={Thomas Melistas and Nikos Spyrou and Nefeli Gkouti and Pedro Sanchez and Athanasios Vlontzos and Yannis Panagakis and Giorgos Papanastasiou and Sotirios A. Tsaftaris},
  booktitle={The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2024},
  url={https://openreview.net/forum?id=0T8xRFrScB}
}
```
