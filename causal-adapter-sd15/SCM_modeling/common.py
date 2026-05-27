"""Shared helpers for the SCM training / causal-discovery scripts.

This module centralises:

* Dataset loading for pendulum / CelebA / MorphoMNIST / ADNI (and an in-memory
  synthetic generator used by smoke tests).
* Per-dataset ground-truth adjacency matrices.
* Path-bootstrap for the vendored SDCD / DAGMA / NOTEARS packages so the
  project works without external git checkouts.
* Small structural-learning utilities (DAG check, SHD / TPR / FPR metrics,
  matrix plotting and on-disk persistence).

It is imported by both ``train_scm.py`` and ``discover_causal.py``; keep it
free of CLI / argument-parsing code.
"""

import importlib
import importlib.util
import os
import random
import sys
import types
from pathlib import Path
from typing import Dict, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.datasets import CelebA

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
# This file lives at ``causal-adapter-sd15/SCM_modeling/common.py``.
#   parents[0] = SCM_modeling/
#   parents[1] = causal-adapter-sd15/   <- the repository root
#   parents[3] = the workspace root that contains all sibling projects
# ``DSAI_CAUSAL_WORKSPACE`` lets callers override the workspace root when the
# auxiliary datasets (pendulum / ADNI / CelebA / MorphoMNIST) live elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(
    os.environ.get("DSAI_CAUSAL_WORKSPACE", str(REPO_ROOT.parents[1]))
).resolve()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edit_modules.load_datasets_adni import (  # noqa: E402
    load_data,
    load_extra_attributes,
    normalize as adni_normalize,
)
from edit_modules.load_datasets_morphominist import load_morphomnist_like  # noqa: E402

# ---------------------------------------------------------------------------
# Per-dataset constants
# ---------------------------------------------------------------------------
# ``PENDULUM_GAUSSIAN_SCALE`` stores (mean, std) used to z-normalise the four
# pendulum attribute columns: pendulum angle, light source angle, shadow
# length, shadow position.
PENDULUM_GAUSSIAN_SCALE = np.array([[2, 42], [104, 44], [7.5, 4.5], [11, 8]])

# Ground-truth adjacency matrices follow the convention ``A[i, j] == 1`` iff
# node ``i`` is a parent of node ``j`` (rows = parents, columns = children).
PENDULUM_DEFAULT_ADJ = np.array(
    [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 1, 0]],
    dtype=float,
)
MORPHO_DEFAULT_ADJ = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
ADNI_DEFAULT_ADJ = np.array(
    [
        [0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
    dtype=float,
)


# Vendored copies of SDCD / DAGMA / NOTEARS live next to this file so the
# project is self-contained and does not require external git checkouts.
VENDORED_CAUSAL_ROOT = REPO_ROOT / "SCM_modeling" / "causal_discovery"


def add_causal_discovery_paths() -> None:
    """Expose the vendored SDCD/DAGMA/NOTEARS packages on ``sys.path``.

    Sources live under ``SCM_modeling/causal_discovery/`` so the project can
    be used without external repositories. See ``LICENSE_*`` files in that
    directory for upstream attribution.
    """
    if VENDORED_CAUSAL_ROOT.exists():
        path_str = str(VENDORED_CAUSAL_ROOT)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_label_gaussian(label: np.ndarray) -> np.ndarray:
    norm_label = np.zeros(label.shape, dtype=np.float32)
    for i in range(label.shape[0]):
        norm_label[i] = (label[i] - PENDULUM_GAUSSIAN_SCALE[i][0]) / PENDULUM_GAUSSIAN_SCALE[i][1]
    return norm_label


def normalize_minist(metrics: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    min_max = {
        "thickness": [0.87598526, 6.255515],
        "intensity": [66.601204, 254.90317],
    }
    normalized = {}
    for key, limits in min_max.items():
        value = metrics[key]
        normalized[key] = (value - limits[0]) / (limits[1] - limits[0])
        normalized[key] = 2 * normalized[key] - 1
    return normalized["intensity"], normalized["thickness"]


def get_default_data_root(dataset: str) -> Path:
    dataset = dataset.lower()
    if dataset == "pendulum":
        return WORKSPACE_ROOT / "MCPL-diffuser" / "dataset" / "causal_data2" / "pendulum"
    if "celea" in dataset:
        return WORKSPACE_ROOT / "counterfactual-benchmark" / "datasets"
    if dataset == "morphomnist":
        return (
            WORKSPACE_ROOT
            / "counterfactual-benchmark"
            / "counterfactual_benchmark"
            / "ctf_datasets"
            / "morphomnist"
            / "data"
        )
    if dataset == "adni":
        return (
            WORKSPACE_ROOT
            / "counterfactual-benchmark"
            / "counterfactual_benchmark"
            / "ctf_datasets"
            / "adni"
            / "preprocessing"
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def _pendulum_load(split_root: Path) -> np.ndarray:
    image_paths = [str(split_root / name) for name in sorted(os.listdir(split_root))]
    labels = np.array(
        [list(map(float, p[:-4].split("/")[-1].split("_")[1:])) for p in image_paths],
        dtype=np.float32,
    )
    return np.apply_along_axis(normalize_label_gaussian, 1, labels)


def _celeba_load(data_root: Path, dataset: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    data = CelebA(root=str(data_root), split=split, transform=None, download=False)
    if "simple" in dataset:
        selected_item = ["Smiling", "Eyeglasses"]
        adj_matrix = np.array([[0, 0], [0, 0]], dtype=float)
    elif "complex" in dataset:
        selected_item = ["Young", "Male", "No_Beard", "Bald"]
        adj_matrix = np.array(
            [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=float
        )
    else:
        raise ValueError(f"Unknown CelebA dataset variant: {dataset}")

    attribute_ids = [data.attr_names.index(attr) for attr in selected_item]
    metrics = {
        attr: torch.as_tensor(data.attr[:, attr_id], dtype=torch.float32)
        for attr, attr_id in zip(selected_item, attribute_ids)
    }
    attrs = torch.cat([metrics[attr].unsqueeze(1) for attr in selected_item], dim=1)
    return np.asarray(attrs), adj_matrix


def _morphomnist_load(data_root: Path, train: bool) -> np.ndarray:
    attribute_size = {"thickness": 1, "intensity": 1, "digit": 10}
    columns = [att for att in attribute_size if att != "digit"]
    _, labels, metrics_df = load_morphomnist_like(str(data_root), train=train, columns=columns)
    labels = F.one_hot(torch.as_tensor(labels.copy(), dtype=torch.long), num_classes=10)
    metrics = {col: torch.as_tensor(metrics_df[col], dtype=torch.float32) for col in columns}
    metrics["intensity"], metrics["thickness"] = normalize_minist(metrics)
    metrics["digit"] = torch.argmax(labels, dim=1)
    attrs = torch.cat([metrics[attr].unsqueeze(1) for attr in attribute_size.keys()], dim=1)
    return np.asarray(attrs)


def _adni_load(data_root: Path, split: str) -> np.ndarray:
    num_of_slices = 10
    keep_only_screening = False
    data_dir = data_root / "preprocessed_data"
    _, attribute_dict, subject_dates_dict = load_data(
        str(data_dir),
        num_of_slices=num_of_slices,
        split=split,
        keep_only_screening=keep_only_screening,
    )
    csv_path = list(data_root.glob("ADNIMERGE*.csv"))[0]
    attribute_size = {
        "apoE": 2,
        "age": 1,
        "sex": 1,
        "brain_vol": 1,
        "vent_vol": 1,
        "slice": 10,
    }
    attributes, indices_to_remove = load_extra_attributes(
        csv_path,
        attributes=attribute_size.keys(),
        attribute_dict=attribute_dict,
        subject_dates_dict=subject_dates_dict,
        keep_only_screening=keep_only_screening,
    )
    attributes["slice"] = np.delete(attributes["slice"], indices_to_remove, axis=0)
    attributes = {
        attr: adni_normalize(torch.tensor(np.array(values), dtype=torch.float32), attr)
        for attr, values in attributes.items()
    }
    attrs = torch.cat(
        [
            attributes[attr].unsqueeze(1) if len(attributes[attr].shape) == 1 else attributes[attr]
            for attr in attribute_size.keys()
        ],
        dim=1,
    )
    return np.asarray(attrs)


def _resolve_pendulum_root(root: Path) -> Path:
    """Accept either ``.../pendulum`` or its parent (e.g. ``.../causal_data2``)."""
    if (root / "train").exists() or (root / "test").exists():
        return root
    nested = root / "pendulum"
    if nested.exists():
        return nested
    return root


def load_train_dataset(dataset: str, data_root: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    root = Path(data_root).resolve() if data_root else get_default_data_root(dataset)
    if dataset == "pendulum":
        root = _resolve_pendulum_root(root)
        train = _pendulum_load(root / "train" if (root / "train").exists() else root)
        return train, PENDULUM_DEFAULT_ADJ.copy()
    if "celeA" in dataset:
        train, adj = _celeba_load(root, dataset, split="train")
        return train, adj
    if dataset == "MorphoMNIST":
        return _morphomnist_load(root, train=True), MORPHO_DEFAULT_ADJ.copy()
    if dataset == "ADNI":
        return _adni_load(root, split="train"), ADNI_DEFAULT_ADJ.copy()
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_dataset_splits(
    dataset: str,
    data_root: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    root = Path(data_root).resolve() if data_root else get_default_data_root(dataset)
    if dataset == "pendulum":
        root = _resolve_pendulum_root(root)
        train = _pendulum_load(root / "train" if (root / "train").exists() else root)
        test_root = root / "test"
        test = _pendulum_load(test_root) if test_root.exists() else None
        return train, test, PENDULUM_DEFAULT_ADJ.copy()
    if "celeA" in dataset:
        train, adj = _celeba_load(root, dataset, split="train")
        test, _ = _celeba_load(root, dataset, split="test")
        return train, test, adj
    if dataset == "MorphoMNIST":
        train = _morphomnist_load(root, train=True)
        test = _morphomnist_load(root, train=False)
        return train, test, MORPHO_DEFAULT_ADJ.copy()
    if dataset == "ADNI":
        train = _adni_load(root, split="train")
        return train, None, ADNI_DEFAULT_ADJ.copy()
    raise ValueError(f"Unsupported dataset: {dataset}")


def apply_max_samples(data: np.ndarray, max_samples: Optional[int], seed: int) -> np.ndarray:
    if max_samples is None or max_samples <= 0 or max_samples >= len(data):
        return data
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(data), size=max_samples, replace=False)
    return data[np.sort(indices)]


def build_intervention_dataset(data: np.ndarray):
    add_causal_discovery_paths()
    create_intervention_dataset = import_sdcd_attr("utils.train_utils", "create_intervention_dataset")

    df = pd.DataFrame(data=data, columns=list(range(data.shape[1])))
    df["perturbation_label"] = "obs"
    return create_intervention_dataset(df, perturbation_colname="perturbation_label")


def ensure_output_dir(path: str) -> Path:
    output = Path(path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_matrices(
    matrix_threshold: np.ndarray,
    matrix_unthreshold: np.ndarray,
    output_root: Path,
    dataset: str,
    method_name: str,
    output_tag: str,
) -> Tuple[Path, Path]:
    target_dir = output_root / f"{dataset}_{output_tag}"
    target_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = target_dir / f"{method_name}_threshold.csv"
    matrix_path = target_dir / f"{method_name}.csv"
    np.savetxt(threshold_path, matrix_threshold, delimiter=",")
    np.savetxt(matrix_path, matrix_unthreshold, delimiter=",")
    return threshold_path, matrix_path


def is_dag(adj: np.ndarray) -> bool:
    graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    return nx.is_directed_acyclic_graph(graph)


def count_accuracy(B_true: np.ndarray, B_est: np.ndarray) -> Dict[str, float]:
    if (B_est == -1).any():
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError("B_est should take value in {0,1,-1}")
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError("Undirected edge should only appear once")
    else:
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError("B_est should take value in {0,1}")
        if not is_dag(B_est):
            raise ValueError("B_est should be a DAG")

    d = B_true.shape[0]
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])

    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])

    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])

    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)

    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)

    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)

    return {
        "fdr": fdr,
        "tpr": tpr,
        "fpr": fpr,
        "shd": float(shd),
        "nnz": float(pred_size),
    }


def maybe_plot_matrices(
    original: np.ndarray,
    thresholded: np.ndarray,
    title_prefix: str,
    save_path: Optional[Path] = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    matrices = [original, thresholded]
    titles = [f"{title_prefix} Original", f"{title_prefix} Thresholded"]
    cmaps = ["Reds", "YlOrBr"]

    for i, (ax, matrix, title) in enumerate(zip(axes, matrices, titles)):
        vmin = np.min(matrix)
        vmax = np.max(matrix)
        if vmin == vmax:
            vmax = 1.0
        im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmaps[i], aspect="auto")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, f"{matrix[row, col]:.1e}", ha="center", va="center", color="black")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_title(title)
        fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def make_synthetic_data(
    n_samples: int = 256,
    n_features: int = 4,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a small linear-Gaussian DAG dataset for smoke tests."""
    rng = np.random.default_rng(seed)
    # Upper triangular weights guarantee acyclicity.
    weights = rng.uniform(0.2, 0.9, size=(n_features, n_features))
    weights = np.triu(weights, k=1)
    noise = rng.normal(0.0, 1.0, size=(n_samples, n_features)).astype(np.float32)
    data = np.zeros_like(noise, dtype=np.float32)

    for j in range(n_features):
        parents = data[:, :j] @ weights[:j, j]
        data[:, j] = parents + noise[:, j]

    return data, (weights != 0).astype(float)


def load_attr_from_file(file_path: Path, attr_name: str, module_name: Optional[str] = None):
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot load {attr_name}; missing file: {file_path}")

    name = module_name or f"dynamic_{file_path.stem}_{attr_name}"
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build module spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, attr_name):
        raise AttributeError(f"{attr_name} not found in {file_path}")
    return getattr(module, attr_name)


def import_sdcd_attr(module_suffix: str, attr_name: str):
    """Import ``attr_name`` from the vendored ``sdcd.<module_suffix>`` module."""
    add_causal_discovery_paths()
    module = importlib.import_module(f"sdcd.{module_suffix}")
    if not hasattr(module, attr_name):
        raise AttributeError(f"{attr_name} not found in sdcd.{module_suffix}")
    return getattr(module, attr_name)
