# Changelog

All notable changes to this project are documented in this file.

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
