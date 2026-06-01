# Training scripts

Two flavours of shell scripts. Each script declares a few `<set me>`
placeholder variables at the top — edit those to point at your local
checkpoints / dataset, then run.

## Layout

```
scripts/
├── adapter_training/        # Causal-Adapter training (SD1.5 + SD3)
│   ├── train_pendulum.sh
│   ├── train_adni.sh
│   ├── train_celeba_complex.sh
│   └── train_sd3_celebahq.sh
└── scm_training/            # SCM pretraining + causal discovery
    ├── train_scm_pendulum.sh
    ├── train_scm_adni.sh
    ├── discover_pendulum.sh
    └── discover_adni.sh
```

## Usage

Run from the repo root so `python train.py` / `python SCM_modeling/...`
resolve correctly:

```bash
# 1) SCM pretraining (produces the checkpoint consumed by --scm_path below)
bash scripts/scm_training/train_scm_pendulum.sh

# 2) Causal-Adapter training
bash scripts/adapter_training/train_pendulum.sh
```

SCM outputs land under `SCM_modeling/saved_mtx/{dataset}_{tag}/`.
Causal-Adapter outputs land under
`<output_dir>/<timestamp>-<output_name><task_cond>/`.

See the top-level [`README.md`](../README.md) for the full workflow,
required-paths reference, and GPU memory guidance.
