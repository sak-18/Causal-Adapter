# Changelog

All notable changes to this project are documented in this file.

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
