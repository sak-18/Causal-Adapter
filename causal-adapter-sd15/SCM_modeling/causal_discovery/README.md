# Vendored causal-discovery libraries

This directory contains lightly-modified copies of three upstream projects so
that scripts under `SCM_modeling/` can run from this repository without
external checkouts.

| Package  | Upstream                                                | Local license file   |
|----------|---------------------------------------------------------|----------------------|
| `sdcd/`  | https://github.com/azizilab/sdcd                        | `LICENSE_SDCD.md`    |
| `dagma/` | https://github.com/kevinsbello/dagma                    | `LICENSE_DAGMA`      |
| `notears/` | https://github.com/xunzheng/notears                   | `LICENSE_NOTEARS`    |

## Local modifications

* `sdcd/__init__.py` and `sdcd/models/__init__.py` were trimmed so optional
  backends (DAGMA/DCDFG/DCDI/GIES/NoBEARS/NOTEARS variants inside SDCD,
  `_sortnregress`) are no longer imported eagerly. Use explicit submodule
  imports such as `from sdcd.models._sdcd import SDCD` if you need them.
* Hard-coded `sys.path.append("/home/jovyan/...")` lines in
  `dagma/nonlinear.py`, `dagma/nolinear_gpu.py`, `notears/nonlinear.py`,
  `notears/nonlinear_gpu.py`, and `notears/nonlinear_gpu_backup.py` were
  removed. The vendored packages rely on `sys.path` being set by
  `SCM_modeling/common.add_causal_discovery_paths()`.

No other source files were modified. License terms of each upstream project
continue to apply to the corresponding subdirectory.
