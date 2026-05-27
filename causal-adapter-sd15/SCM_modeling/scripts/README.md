# SCM modelling shell scripts

Thin wrappers around `SCM_modeling/discover_causal.py` (causal discovery via
SDCD / DAGMA / NOTEARS) and `SCM_modeling/train_scm.py` (SCM CausalNet
training). Each script documents its assumed dataset layout in the header
and forwards extra CLI flags via `"$@"` so you can override hyper-parameters
without editing the file:

```bash
bash SCM_modeling/scripts/discover_pendulum.sh --quick --device cpu
bash SCM_modeling/scripts/train_scm_pendulum.sh --quick
```

## Conventions

* `PYTHON` defaults to the active interpreter; export `PYTHON=/path/to/python`
  to pin a specific env (e.g. the `mcpl` conda env used in this repo).
* Dataset roots default to the sibling-project layout under
  `DSAI_CAUSAL_WORKSPACE` (auto-detected by `common.get_default_data_root`).
  Override with `--data-root /custom/path`.
* Outputs land under `SCM_modeling/saved_mtx/{dataset}_{tag}/`.

## Smoke test (no real data)

`smoke_test.sh` runs the synthetic-data path for every method on CPU and
should finish in well under a minute. Use it to verify the vendored
SDCD / DAGMA / NOTEARS packages are importable end-to-end.
