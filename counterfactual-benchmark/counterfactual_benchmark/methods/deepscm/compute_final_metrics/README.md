# Computing final quantitative metrics

This folder turns the raw `final_results.pkl` files produced by
`evaluate_SD_DSCM.py` (under `<output_root>/<dataset>/<run_name>/<do_attr>/`)
into the tables reported in the paper.

- **`metrics_utils.py`** — all metric computation lives here (version-controlled,
  testable). Import from it; do not copy-paste the logic back into notebooks.
- **`compute_final_metrics.ipynb`** — a single notebook with one section per
  dataset (CelebA, CelebA-HQ, CelebA-HQ reverse, ADNI, pendulum, FID/minimality).
  It replaces the old per-dataset notebooks (`vis_celeba`, `vis_celebahq`,
  `vis_celeba_ablation`, `vis_celebahq_fulltest`, `vis_adni`, `vis_pendulum`).
  Each section only sets a `root_dir` and prints a table.

## `metrics_utils.py` API

| function | dataset | purpose |
|----------|---------|---------|
| `evaluate_by_batch(loaded, batch_size=None)` | CelebA / CelebA-HQ | per-attribute binary F1 / accuracy (with calibrated thresholds) |
| `evaluate_adni_by_batch(loaded, batch_size, num_slices)` | ADNI | mixed classification (`apoE`/`sex`/`slice`) + regression (`age`/`brain_vol`/`vent_vol`) |
| `evaluate_regression_by_batch(loaded, batch_size=None)` | pendulum | per-attribute L1 error |
| `effectiveness_table(root_dir, target_attrs)` | any | full `{target_attr: {do_attr: scores}}` table |
| `lpips_summary(root_dir)` | any | mean LPIPS / reverse / composition distances per `do(attr)` |
| `sigmoid`, `load_results`, `list_do_folders` | — | small shared helpers |

```python
from metrics_utils import effectiveness_table, lpips_summary

root_dir = "../saved_benchmark/celebahq_simple/DSCM_effectiveness_step100.0_scale1.5_BlendTrue_seed40"
table = effectiveness_table(root_dir, target_attrs=["Smiling", "Eyeglasses"])
print(lpips_summary(root_dir))
```

## Notes
- The notebook is a thin **result-inspection / plotting** layer — keep reusable
  computation in `metrics_utils.py`.
- The ADNI section needs `torch` / `torchmetrics` and the ADNI label codecs; the
  notebook's setup cell calls `_paths.bootstrap()` so `ctf_datasets` is importable.
- **TODO:** the old `vis_celebahq*` notebooks also contained standalone classifier
  re-evaluation cells that loaded CelebA-HQ via the removed `edit_modules` package.
  Those were dropped in the merge; if you need them, port the dataset loading onto
  the current `ctf_datasets.celeba_hq` package.
