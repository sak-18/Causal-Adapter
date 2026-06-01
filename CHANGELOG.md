# Changelog

All notable changes to this project are documented in this file.

## 2026-06-01

### Changed
- Merged `causal-adapter-sd3/` into the unified project layout. SD1.5 and SD3
  now share a single `diffusers/` install (sd3 fork, 0.36.0.dev0), a single
  `causal_datasets/` package, and a single `causal_modules/p2p_edits/` package.
- Promoted SD3 entrypoint to `train_SD3.py` at the repo root (alongside SD1.5
  `train.py`).
- Added `commands_training_sd3.md` (replaces sd3's `commands_training.txt`)
  with portable `$SD3_MODEL_PATH` / `$DATASET_ROOT` placeholders.
- Migrated `notebook_benchmarks/control_edit_celebahq_simple_SD3.ipynb` into
  the shared notebook directory; paths are now configured via env vars at the
  top of the notebook.

### Added
- `diffusers/src/diffusers/models/controlnets/controlnet_causal.py` — sd1.5
  `Causal_ControlNetModel` ported into the new `controlnets/` package
  structure; exported from the top-level `diffusers` namespace.
- `diffusers/src/diffusers/pipelines/controlnet/pipeline_causal_controlnet.py`
  — sd1.5 `StableDiffusionCausalControlNetPipeline` re-exported in the new
  fork.
- `causal_modules/ddim_modules_sd3.py`, `causal_modules/ddim_modules_flux.py`
  — SD3/Flux-specific prompt/Flow utilities.
- `causal_modules/p2p_edits/mcpl_utils_sd3.py` — SD3 variant of
  `prompt_contrastive_loss` (handles the `RELATE` placeholder branch).
- `utils_sd3.py` — SD3 train/val split + SCM metrics helpers.

### Repository State
- `causal-adapter-sd15/` and `causal-adapter-sd3/` directories are kept on
  disk as read-only references until SD3 GPU smoke tests pass; they will be
  removed once verified end-to-end.

### Testing
- diffusers smoke import: `Causal_ControlNetModel`,
  `StableDiffusionCausalControlNetPipeline`, `Causal_SD3ControlNetModel`,
  `StableDiffusion3InpaintPipeline_Adapter` all resolve in 0.36.0.dev0.
- `train.py` (SD1.5) and `train_SD3.py` (SD3) module-level imports execute
  without error on CPU.
- `causal_datasets.TextualInversionDataset(dataset='celebahq_simple', ...)`
  yields valid `pixel_values` / `input_ids` / `label` tensors with a local
  CLIP tokenizer.
- GPU end-to-end runs (training step / inference notebook) are pending on a
  GPU node.

## 2026-05-29

### Added
- SD1.5 counterfactual inference notebooks under `causal-adapter-sd15/notebook_benchmarks/`:
  - `counterfactuals_pendulum.ipynb`
  - `counterfactuals_celeba.ipynb`
  - `counterfactuals_ADNI.ipynb`
- Shared helper `causal-adapter-sd15/notebook_benchmarks/inference_utils.py` centralising pipeline assembly (ControlNet + SCM head + MCPL embeddings + DDIM scheduler).
- README section documenting the inference workflow and the `MODEL_CACHE` / `LOGS_ROOT` configuration roots.

### Changed
- Replaced the five legacy `control_edit_*.ipynb` notebooks with the three dataset-aligned `counterfactuals_*.ipynb` above.
- Notebooks now derive every path from two roots and add per-section comments (DDIM inversion, intervention sweep, attention maps).

### Repository State
- Main branch includes commits:
  - `b526120` refactor(notebooks): consolidate counterfactual inference notebooks
  - `715c01d` docs(notebooks): factor CONFIG paths into shared roots and explain steps

### Testing
- Each notebook re-validated end-to-end against its corresponding pretrained checkpoint (Pendulum 20k, CelebA 200k, ADNI 100k); all figure-producing cells render without errors.

## 2026-05-27

### Added
- Added reproducible environment files under `envs/`:
  - `envs/causal-adapter-sd15.yml`
  - `envs/causal-adapter-sd3.yml`
  - `envs/benchmark.yml`
- Added concrete setup instructions in `envs/README.md`.

### Changed
- Aligned environment mapping with actual cluster usage:
  - `flux` for SD3 / Flux workflows.
  - `mcpl` for SD1.5 and benchmark workflows.
- Completed non-notebook hardcoded-path portability cleanup across scripts/configs/templates.
- Updated migration notes and work log for release readiness tracking.

### Repository State
- Main branch includes commits:
  - `aa42454` chore(env): add reproducible conda env files
  - `e633d35` fix(env): align names with flux and mcpl mapping
- Remaining path cleanup scope is notebook files (`*.ipynb`) only.

### Testing
- Functional smoke tests are pending and planned on a separate GPU node.
