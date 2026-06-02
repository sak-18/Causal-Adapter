"""Shared helpers for turning the raw ``final_results.pkl`` files written by
``evaluate_SD_DSCM.py`` into the final quantitative tables.

These functions were previously copy-pasted into every ``vis_*.ipynb`` notebook
under this folder. They now live here so the core computation is version-
controlled and testable; the notebooks should ``from metrics_utils import ...``
and stay as thin result-inspection / plotting layers.

A run directory produced by ``evaluate_SD_DSCM.py --metrics effectiveness`` looks
like::

    <output_root>/<dataset>/<run_name>/
        <do_attr_1>/final_results.pkl
        <do_attr_2>/final_results.pkl
        ...

Each ``final_results.pkl`` is a dict with keys ``predictions`` / ``targets``
(per attribute), plus optional ``lpips`` (non-reverse runs) or
``IDP_*`` / ``reverse_*`` / ``compos_*`` arrays (``--reverse`` runs).
"""
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# Default per-attribute decision thresholds, calibrated on the validation set.
# (Logits are sigmoid-squashed to probabilities before thresholding.)
DEFAULT_THRESHOLDS = {
    "Smiling": 0.136,
    "Eyeglasses": 0.159,
}


def sigmoid(x):
    """Numerically stable logistic sigmoid."""
    x = np.clip(np.asarray(x, dtype=np.float64), -80, 80)
    return 1.0 / (1.0 + np.exp(-x))


def load_results(result_path: str) -> Optional[dict]:
    """Load a single ``final_results.pkl``; return None if it is missing."""
    if not os.path.exists(result_path):
        return None
    with open(result_path, "rb") as f:
        return pickle.load(f)


def evaluate_by_batch(loaded_results: dict, batch_size: Optional[int] = None,
                      thresholds: Optional[Dict[str, float]] = None) -> dict:
    """Compute per-attribute F1 / accuracy for one ``final_results.pkl``.

    Args:
        loaded_results: dict with ``predictions`` and ``targets`` per attribute.
        batch_size: if None, compute a single global F1/accuracy; otherwise
            compute the mean +/- std of per-chunk F1/accuracy (chunked by
            ``batch_size``), matching the original notebook behaviour.
        thresholds: per-attribute decision thresholds; falls back to
            :data:`DEFAULT_THRESHOLDS` then 0.5.
    """
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    attributes = list(loaded_results['predictions'].keys())
    results_summary = {}

    for attr in attributes:
        preds = np.asarray(loaded_results['predictions'][attr]).squeeze()
        targets = np.asarray(loaded_results['targets'][attr]).squeeze()

        probs = sigmoid(preds)
        threshold = thresholds.get(attr, 0.5)
        binary_preds = (probs >= threshold).astype(int)
        binary_targets = targets.astype(int)

        if batch_size is None:
            results_summary[attr] = {
                'F1-score': round(f1_score(binary_targets, binary_preds) * 100, 3),
                'Accuracy': round(accuracy_score(binary_targets, binary_preds) * 100, 3),
            }
        else:
            f1_list, acc_list = [], []
            for i in range(0, len(binary_targets), batch_size):
                batch_preds = binary_preds[i:i + batch_size]
                batch_targets = binary_targets[i:i + batch_size]
                if len(np.unique(batch_targets)) > 1:
                    f1 = f1_score(batch_targets, batch_preds)
                else:
                    # single-class chunk: F1 is 1.0 only if perfectly predicted
                    f1 = 1.0 if (batch_targets == batch_preds).all() else 0.0
                f1_list.append(f1)
                acc_list.append(accuracy_score(batch_targets, batch_preds))
            results_summary[attr] = {
                'F1-score': round(float(np.mean(f1_list)), 3),
                'F1-std': round(float(np.std(f1_list)), 2),
                'Accuracy': round(float(np.mean(acc_list)), 3),
                'Acc-std': round(float(np.std(acc_list)), 2),
            }
    return results_summary


def evaluate_regression_by_batch(loaded_results: dict, batch_size: Optional[int] = None,
                                 attrs: Optional[List[str]] = None) -> dict:
    """Per-attribute L1 error for regression targets (e.g. pendulum SCM attrs).

    Used by the pendulum experiments where every attribute is continuous.
    """
    from sklearn.metrics import mean_absolute_error
    target_attrs = attrs or list(loaded_results['predictions'].keys())
    results_summary = {}
    for attr in target_attrs:
        if attr not in loaded_results['predictions']:
            continue
        preds = np.asarray(loaded_results['predictions'][attr]).squeeze()
        targets = np.asarray(loaded_results['targets'][attr]).squeeze()
        if batch_size is None:
            results_summary[attr] = {'L1-loss': round(float(mean_absolute_error(targets, preds)), 3)}
        else:
            l1_list = [mean_absolute_error(targets[i:i + batch_size], preds[i:i + batch_size])
                       for i in range(0, len(preds), batch_size)]
            results_summary[attr] = {
                'L1-loss': round(float(np.mean(l1_list)), 3),
                'L1-std': round(float(np.std(l1_list)), 2),
            }
    return results_summary


def evaluate_adni_by_batch(loaded_results: dict, batch_size: Optional[int] = None,
                           num_slices: int = 10) -> dict:
    """Per-attribute metrics for the ADNI graph (mixed classification + regression).

    - ``apoE`` (3-class) / ``slice`` (``num_slices``-class) / ``sex`` (binary):
      F1 + accuracy, after decoding the ordinal/binary label encodings.
    - ``age`` / ``brain_vol`` / ``vent_vol``: L1 error.

    Requires ``torch`` / ``torchmetrics`` and the ADNI label codecs
    (``bin_array`` / ``ordinal_array``); imported lazily so the binary/regression
    helpers above stay dependency-light.
    """
    import torch
    import torch.nn as nn
    from torchmetrics.classification import F1Score, BinaryF1Score
    from sklearn.metrics import mean_absolute_error
    from ctf_datasets.adni.dataset_SD import bin_array, ordinal_array  # needs _paths bootstrap / sys.path

    results_summary = {}
    for attr in list(loaded_results['predictions'].keys()):
        preds = torch.sigmoid(torch.tensor(loaded_results['predictions'][attr]).squeeze())
        targets = torch.tensor(loaded_results['targets'][attr]).squeeze()

        if attr == "apoE":
            preds_label = preds.argmax(dim=-1)
            targets_label = bin_array(targets, reverse=True).long()
            metric = F1Score(task="multiclass", num_classes=3).to(preds.device)
        elif attr == "sex":
            preds_label = (preds >= 0.5).int()
            targets_label = targets.int()
            metric = BinaryF1Score().to(preds.device)
        elif attr == "slice":
            preds_label = preds.argmax(dim=-1)
            targets_label = ordinal_array(targets, m=num_slices, reverse=True).long()
            metric = F1Score(task="multiclass", num_classes=num_slices).to(preds.device)
        elif attr in ['age', 'brain_vol', 'vent_vol']:
            metric = None  # regression
        else:
            continue

        if metric is not None:  # classification
            if batch_size is None:
                results_summary[attr] = {
                    'F1-score': round(metric(preds_label, targets_label).item(), 3),
                    'Accuracy': round(accuracy_score(targets_label.cpu().numpy(), preds_label.cpu().numpy()), 3),
                }
            else:
                f1_list, acc_list = [], []
                for i in range(0, len(preds_label), batch_size):
                    p, t = preds_label[i:i + batch_size], targets_label[i:i + batch_size]
                    f1_list.append(metric(p, t).item())
                    acc_list.append(accuracy_score(t.cpu().numpy(), p.cpu().numpy()))
                results_summary[attr] = {
                    'F1-score': round(float(np.mean(f1_list)), 3), 'F1-std': round(float(np.std(f1_list)), 2),
                    'Accuracy': round(float(np.mean(acc_list)), 3), 'Acc-std': round(float(np.std(acc_list)), 2),
                }
        else:  # regression
            preds_val, targets_val = preds.cpu().numpy(), targets.cpu().numpy()
            if batch_size is None:
                results_summary[attr] = {'L1-loss': round(float(mean_absolute_error(targets_val, preds_val)), 3)}
            else:
                l1_list = [mean_absolute_error(targets_val[i:i + batch_size], preds_val[i:i + batch_size])
                           for i in range(0, len(preds_val), batch_size)]
                results_summary[attr] = {
                    'L1-loss': round(float(np.mean(l1_list)), 3), 'L1-std': round(float(np.std(l1_list)), 2),
                }
    return results_summary


def list_do_folders(root_dir: str, candidates: Optional[List[str]] = None) -> List[str]:
    """Return the present ``do(attr)`` sub-folders under a run directory.

    If ``candidates`` is given (e.g. ``["Smiling", "Eyeglasses"]``) only those
    are considered; otherwise every sub-directory is returned.
    """
    if candidates is not None:
        return [name for name in candidates if os.path.isdir(os.path.join(root_dir, name))]
    return sorted(name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name)))


def effectiveness_table(root_dir: str, target_attrs: List[str], do_folders: Optional[List[str]] = None,
                        batch_size: Optional[int] = None) -> Dict[str, Dict[str, dict]]:
    """Build the effectiveness table: ``{target_attr: {do_attr: scores}}``.

    Reads every ``<do_attr>/final_results.pkl`` under ``root_dir`` and evaluates
    each target attribute. Missing folders / attributes are skipped.
    """
    if do_folders is None:
        do_folders = list_do_folders(root_dir, candidates=target_attrs)
    table: Dict[str, Dict[str, dict]] = {t: {} for t in target_attrs}
    for do_attr in do_folders:
        loaded = load_results(os.path.join(root_dir, do_attr, "final_results.pkl"))
        if loaded is None:
            continue
        results = evaluate_by_batch(loaded, batch_size=batch_size)
        for target_attr in target_attrs:
            if target_attr in results:
                table[target_attr][do_attr] = results[target_attr]
    return table


def lpips_summary(root_dir: str, do_folders: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """Aggregate the LPIPS / reverse / composition distances per ``do(attr)``.

    Returns ``{do_attr: {metric_name: mean_value}}`` for whichever distance
    arrays are present in each ``final_results.pkl`` (``lpips`` for non-reverse
    runs, ``IDP_*`` / ``reverse_*`` / ``compos_*`` for ``--reverse`` runs).
    """
    if do_folders is None:
        do_folders = list_do_folders(root_dir)
    distance_keys = ['lpips', 'IDP_l1', 'IDP_lpips', 'reverse_l1', 'reverse_lpips', 'compos_l1', 'compos_lpips']
    summary: Dict[str, Dict[str, float]] = {}
    for do_attr in do_folders:
        loaded = load_results(os.path.join(root_dir, do_attr, "final_results.pkl"))
        if loaded is None:
            continue
        summary[do_attr] = {k: float(np.mean(loaded[k])) for k in distance_keys if k in loaded}
    return summary
