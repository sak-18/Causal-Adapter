# SCM modelling shell scripts

Minimal wrappers around `SCM_modeling/discover_causal.py` and
`SCM_modeling/train_scm.py`. Edit `--data-root` in each script to point at
your local dataset, then run:

```bash
bash scripts/scm_training/discover_pendulum.sh
bash scripts/scm_training/train_scm_adni.sh
```

Outputs land under `SCM_modeling/saved_mtx/{dataset}_{tag}/`.
