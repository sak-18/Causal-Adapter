"""Train an SCM (Structural Causal Model) on tabular attribute labels.

This script is the Python-script equivalent of the original
``scm_training.ipynb`` notebook. It loads a dataset's attribute matrix
(pendulum / CelebA / MorphoMNIST / ADNI), builds an SDCD ``CausalNet``
initialized from a (ground-truth or user-supplied) adjacency matrix, runs the
two-stage SDCD training schedule, and finally saves the learned adjacency
matrices to ``saved_mtx/{dataset}_{tag}/``.

Example::

    python SCM_modeling/train_scm.py --dataset pendulum --device cpu \\
        --data-root /path/to/causal_data2

See ``scripts/scm_training/`` for ready-to-run shell wrappers.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Make sibling modules (``common``) importable when this script is launched
# directly via ``python SCM_modeling/train_scm.py``.
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from common import (
    REPO_ROOT,
    WORKSPACE_ROOT,
    add_causal_discovery_paths,
    apply_max_samples,
    build_intervention_dataset,
    count_accuracy,
    ensure_output_dir,
    is_dag,
    import_sdcd_attr,
    load_attr_from_file,
    load_dataset_splits,
    make_synthetic_data,
    maybe_plot_matrices,
    save_matrices,
    set_random_seed,
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_causalnet_class():
    """Load the project-local ``CausalNet`` class from ``causal_modules``.

    We import by file path (instead of ``import causal_modules.pretraining``)
    so the script does not depend on ``causal_modules`` being a regular
    importable package on ``sys.path`` at call time.
    """
    local_pretraining = REPO_ROOT / "causal_modules" / "scm_pretraining.py"
    return load_attr_from_file(
        local_pretraining, "CausalNet", module_name="local_pretraining_causalnet"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SCM CausalNet from notebook-equivalent tabular labels"
    )
    parser.add_argument("--dataset", default="ADNI", help="Dataset name")
    parser.add_argument("--data-root", default=None, help="Optional dataset root override")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "SCM_modeling" / "saved_mtx"),
        help="Root directory for matrix artifacts",
    )
    parser.add_argument("--output-tag", default="minmax", help="Suffix in saved_mtx/{dataset}_{tag}")
    parser.add_argument("--method-name", default="scm", help="Saved matrix prefix")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--adj-matrix-path", default=None, help="Optional initial adjacency CSV")
    parser.add_argument("--dry-run", action="store_true", help="Only load data and print shapes")
    parser.add_argument("--quick", action="store_true", help="Use short training schedule for smoke runs")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Use generated data instead of disk dataset")
    parser.add_argument("--synthetic-samples", type=int, default=256)
    parser.add_argument("--synthetic-dim", type=int, default=4)
    parser.add_argument("--finetune", dest="finetune", action="store_true")
    parser.add_argument("--no-finetune", dest="finetune", action="store_false")
    parser.set_defaults(finetune=True)
    return parser.parse_args()


def resolve_input_adjacency(args: argparse.Namespace, gt_adj: np.ndarray) -> np.ndarray:
    """Return the adjacency matrix used to initialise ``CausalNet``.

    Priority:
      1. CSV file passed via ``--adj-matrix-path`` (user override).
      2. The dataset's ground-truth adjacency returned by ``load_dataset_splits``.
    """
    if args.adj_matrix_path:
        return np.loadtxt(args.adj_matrix_path, delimiter=",")
    return gt_adj


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Load the train / test attribute tensors and the GT adjacency
    # ------------------------------------------------------------------
    if args.synthetic:
        train_set, gt_adj = make_synthetic_data(
            n_samples=args.synthetic_samples,
            n_features=args.synthetic_dim,
            seed=args.seed,
        )
        test_set, _ = make_synthetic_data(
            n_samples=max(32, args.synthetic_samples // 4),
            n_features=args.synthetic_dim,
            seed=args.seed + 11,
        )
    else:
        train_set, test_set, gt_adj = load_dataset_splits(dataset=args.dataset, data_root=args.data_root)
    train_set = apply_max_samples(train_set, args.max_samples, args.seed)
    if test_set is not None:
        test_set = apply_max_samples(test_set, args.max_samples, args.seed + 1)

    print(f"Train shape: {train_set.shape}")
    if test_set is not None:
        print(f"Test shape: {test_set.shape}")

    if args.dry_run:
        return

    # ------------------------------------------------------------------
    # 2. Decide which adjacency seeds the model and prepare output dirs
    # ------------------------------------------------------------------
    adj_matrix = resolve_input_adjacency(args, gt_adj)

    output_root = ensure_output_dir(args.output_dir)
    run_dir = output_root / "logs" / f"{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}_causal_discovered_matrix"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger_path = run_dir / "logger.txt"

    # ------------------------------------------------------------------
    # 3. Build the SDCD-style intervention dataset and instantiate model
    # ------------------------------------------------------------------
    train_dataset = build_intervention_dataset(train_set)
    CausalNet = resolve_causalnet_class()

    # ``CausalNet`` may have evolved across branches; only forward kwargs it
    # actually accepts to stay compatible with older signatures.
    model_kwargs = {"use_gumbel": False}
    if "dataset_name" in CausalNet.__init__.__code__.co_varnames:
        model_kwargs["dataset_name"] = args.dataset
    if "logger_path" in CausalNet.__init__.__code__.co_varnames:
        model_kwargs["logger_path"] = str(logger_path)

    model = CausalNet(**model_kwargs)

    # ``--quick`` cuts the SDCD two-stage schedule down to a handful of
    # epochs so smoke tests finish in seconds instead of minutes.
    stage1_kwargs = None
    stage2_kwargs = None
    if args.quick:
        batch_size = min(64, len(train_set))
        stage1_kwargs = {"n_epochs": 5, "batch_size": batch_size}
        stage2_kwargs = {"n_epochs": 5, "batch_size": batch_size, "n_epochs_check": 1}

    # ------------------------------------------------------------------
    # 4. Train SCM (stage 1 + optional finetune stage 2)
    # ------------------------------------------------------------------
    model.train(
        train_dataset,
        finetune=args.finetune,
        device=args.device,
        input_matrix=adj_matrix,
        val_fraction=args.val_fraction,
        stage1_kwargs=stage1_kwargs,
        stage2_kwargs=stage2_kwargs,
    )

    matrix_unthreshold = model.get_adjacency_matrix(threshold=False)
    matrix_threshold = model.get_adjacency_matrix(threshold=True)

    # ------------------------------------------------------------------
    # 5. Optional evaluation: test-set NLL and structural metrics
    # ------------------------------------------------------------------
    if test_set is not None:
        test_dataset = build_intervention_dataset(test_set)
        nll = model.compute_nll(test_dataset)
        print(f"Test NLL: {nll}")

    threshold_path, matrix_path = save_matrices(
        matrix_threshold=matrix_threshold,
        matrix_unthreshold=matrix_unthreshold,
        output_root=output_root,
        dataset=args.dataset,
        method_name=args.method_name,
        output_tag=args.output_tag,
    )
    print(f"Saved threshold matrix: {threshold_path}")
    print(f"Saved weighted matrix: {matrix_path}")

    dag_ok = is_dag((matrix_threshold != 0).astype(int))
    print(f"Is DAG (thresholded): {dag_ok}")

    # ``count_accuracy`` only makes sense when the ground-truth adjacency and
    # the learned matrix share the same shape (e.g. pendulum 4x4). For
    # datasets where one-hot expansion changes the dimensionality (ADNI,
    # MorphoMNIST) we skip the metric block silently.
    if gt_adj is not None and gt_adj.shape == matrix_threshold.shape:
        metrics = count_accuracy(gt_adj.astype(int), (matrix_threshold != 0).astype(int))
        print("Accuracy:", json.dumps(metrics, indent=2))

    if args.plot:
        fig_path = output_root / f"{args.dataset}_{args.output_tag}" / f"{args.method_name}_matrices.png"
        maybe_plot_matrices(matrix_unthreshold, matrix_threshold, args.method_name.upper(), fig_path)
        print(f"Saved plot: {fig_path}")


if __name__ == "__main__":
    main()
