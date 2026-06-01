"""Causal-discovery driver for SDCD / DAGMA / NOTEARS.

This script replaces the original ``causal_discovery.ipynb`` notebook. Given
a dataset name (``pendulum`` / ``ADNI`` / ``celeA_simple`` / ``celeA_complex``
/ ``MorphoMNIST``) it loads the attribute table, runs the requested method
on the observational sample, and writes both the raw weighted adjacency and
its thresholded version under ``saved_mtx/{dataset}_{tag}/``.

Example::

    python SCM_modeling/discover_causal.py --dataset pendulum --method-name sdcd \\
        --device cpu --data-root /path/to/causal_data2

See ``scripts/scm_training/`` for ready-to-run shell wrappers.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Make sibling ``common`` importable when this script is launched directly.
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
    load_train_dataset,
    make_synthetic_data,
    maybe_plot_matrices,
    save_matrices,
    set_random_seed,
)


def run_sdcd(data: np.ndarray, args: argparse.Namespace):
    """Run the SDCD continuous DAG-learning model on observational data."""
    add_causal_discovery_paths()
    SDCD = import_sdcd_attr("models._sdcd", "SDCD")
    dataset = build_intervention_dataset(data)
    model = SDCD()

    # ``--quick`` shrinks both training stages to ~5 epochs for smoke tests.
    stage1_kwargs = None
    stage2_kwargs = None
    if args.quick:
        stage1_kwargs = {"n_epochs": 5, "batch_size": min(64, len(data))}
        stage2_kwargs = {"n_epochs": 5, "batch_size": min(64, len(data)), "n_epochs_check": 1}

    model.train(
        dataset,
        finetune=args.finetune,
        device=args.device,
        val_fraction=args.val_fraction,
        stage1_kwargs=stage1_kwargs,
        stage2_kwargs=stage2_kwargs,
    )
    matrix_unthreshold = model.get_adjacency_matrix(threshold=False)
    matrix_threshold = model.get_adjacency_matrix(threshold=True)
    return matrix_threshold, matrix_unthreshold


def run_notears(data: np.ndarray, args: argparse.Namespace):
    """Run the nonlinear NOTEARS DAG learner."""
    add_causal_discovery_paths()
    from notears.nonlinear import NotearsMLP, notears_nonlinear

    # NOTEARS upstream code assumes float64 tensors.
    torch.set_default_dtype(torch.double)
    data = data.astype(np.double)
    d = data.shape[1]
    model = NotearsMLP(dims=[d, args.hidden_dim, 1], bias=True)
    max_iter = args.max_iter if args.max_iter is not None else (20 if args.quick else 100)
    matrix_threshold, matrix_unthreshold = notears_nonlinear(
        model,
        data,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        max_iter=max_iter,
        w_threshold=args.w_threshold,
    )
    return matrix_threshold, matrix_unthreshold


def run_dagma(data: np.ndarray, args: argparse.Namespace):
    """Run the nonlinear DAGMA DAG learner."""
    add_causal_discovery_paths()
    from dagma.nonlinear import DagmaMLP, DagmaNonlinear

    # DAGMA upstream code assumes float64 tensors.
    torch.set_default_dtype(torch.double)
    d = data.shape[1]
    eq_model = DagmaMLP(dims=[d, args.hidden_dim, 1], bias=True)
    model = DagmaNonlinear(eq_model)

    if args.quick:
        matrix_threshold, matrix_unthreshold = model.fit(
            data,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            T=2,
            warm_iter=300,
            max_iter=400,
            lr=0.0002,
            w_threshold=args.w_threshold,
        )
    else:
        matrix_threshold, matrix_unthreshold = model.fit(
            data,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            w_threshold=args.w_threshold,
        )
    return matrix_threshold, matrix_unthreshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SDCD/DAGMA/NOTEARS causal discovery")
    parser.add_argument("--dataset", default="pendulum", help="Dataset name")
    parser.add_argument("--method-name", default="sdcd", choices=["sdcd", "dagma", "notears"])
    parser.add_argument("--data-root", default=None, help="Optional dataset root override")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "SCM_modeling" / "saved_mtx"),
        help="Root directory for matrix artifacts",
    )
    parser.add_argument("--output-tag", default="causaldata2", help="Suffix in saved_mtx/{dataset}_{tag}")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lambda1", type=float, default=0.02)
    parser.add_argument("--lambda2", type=float, default=0.005)
    parser.add_argument("--w-threshold", type=float, default=0.3)
    parser.add_argument("--hidden-dim", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=None, help="NOTEARS max_iter override")
    parser.add_argument("--finetune", action="store_true", help="Only used by SDCD")
    parser.add_argument("--quick", action="store_true", help="Use short optimization schedule")
    parser.add_argument("--dry-run", action="store_true", help="Only load data and print shapes")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Use generated data instead of disk dataset")
    parser.add_argument("--synthetic-samples", type=int, default=256)
    parser.add_argument("--synthetic-dim", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Load observational data (real dataset or synthetic smoke sample)
    # ------------------------------------------------------------------
    if args.synthetic:
        data, gt_adj = make_synthetic_data(
            n_samples=args.synthetic_samples,
            n_features=args.synthetic_dim,
            seed=args.seed,
        )
    else:
        data, gt_adj = load_train_dataset(dataset=args.dataset, data_root=args.data_root)
    data = apply_max_samples(data, args.max_samples, args.seed)
    print(f"Loaded {args.dataset} with shape {data.shape}")

    if args.dry_run:
        return

    # ------------------------------------------------------------------
    # 2. Dispatch to the requested discovery method
    # ------------------------------------------------------------------
    method = args.method_name.lower()
    if method == "sdcd":
        matrix_threshold, matrix_unthreshold = run_sdcd(data, args)
    elif method == "dagma":
        matrix_threshold, matrix_unthreshold = run_dagma(data, args)
    else:
        matrix_threshold, matrix_unthreshold = run_notears(data, args)

    # ------------------------------------------------------------------
    # 3. Persist matrices and report structural metrics if possible
    # ------------------------------------------------------------------
    output_root = ensure_output_dir(args.output_dir)
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

    if gt_adj is not None and gt_adj.shape == matrix_threshold.shape:
        metrics = count_accuracy(gt_adj.astype(int), (matrix_threshold != 0).astype(int))
        print("Accuracy:", json.dumps(metrics, indent=2))

    if args.plot:
        fig_path = output_root / f"{args.dataset}_{args.output_tag}" / f"{args.method_name}_matrices.png"
        maybe_plot_matrices(matrix_unthreshold, matrix_threshold, args.method_name.upper(), fig_path)
        print(f"Saved plot: {fig_path}")


if __name__ == "__main__":
    main()
