"""Per-dataset balanced samplers used during MCPL / Causal-Adapter training.

The CelebA-style datasets are heavily imbalanced on a few attributes (e.g.
``Bald``/``No_Beard`` for ``celeA_complex`` and ``Eyeglasses`` for
``celebahq_simple``). Training with uniform sampling lets the rare classes
collapse, so the original training script switched to a
``WeightedRandomSampler`` for the first part of training and back to uniform
shuffling for the rest.

This module isolates that policy from ``train.py``:

* :func:`build_balanced_sampler` returns a sampler (or ``None`` for datasets
  that don't need rebalancing) given the dataset name and the
  ``TextualInversionDataset`` instance.
* :func:`needs_uniform_switch` reports whether the dataset is one of the
  CelebA-style datasets whose schedule swaps to uniform shuffling later in
  training.

Behaviour is intentionally kept identical to the legacy inline block so the
move is a pure refactor.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


# Datasets whose dataloader switches from a WeightedRandomSampler to a
# uniformly-shuffled one once ``epoch`` exceeds ``UNIFORM_SWITCH_EPOCH``.
_UNIFORM_SWITCH_DATASETS = {"celeA_complex", "celebahq_simple"}
UNIFORM_SWITCH_EPOCH = 12


def needs_uniform_switch(dataset_name: str) -> bool:
    """Whether ``dataset_name`` follows the balanced→uniform schedule."""
    return dataset_name in _UNIFORM_SWITCH_DATASETS


def build_balanced_sampler(
    dataset_name: str, train_dataset: Dataset
) -> Optional[WeightedRandomSampler]:
    """Return a ``WeightedRandomSampler`` for imbalanced datasets, else ``None``.

    Args:
        dataset_name: The ``--dataset`` flag (e.g. ``"celeA_complex"``).
        train_dataset: The :class:`TextualInversionDataset`. It must expose an
            ``imglabel`` tensor whose column layout matches the adapter for
            ``dataset_name`` (see :mod:`causal_datasets._adapters`).

    Returns:
        A weighted sampler whose per-sample weights inverse-frequency-balance
        the relevant class column(s), or ``None`` if no balancing is needed.
    """
    imglabel = getattr(train_dataset, "imglabel", None)

    if dataset_name == "celeA_complex":
        # Adapter column layout: [Young, Male, No_Beard, Bald].
        # Combine ``No_Beard`` (col -2) and ``Bald`` (col -1) into a 4-class
        # joint label and inverse-frequency-balance over those classes.
        no_beard = imglabel[:, -2]
        bald = imglabel[:, -1]
        combined = (no_beard * 2 + bald).long()  # 0..3

        # The original code used hard-coded counts measured from the training
        # split; preserve them so sample weights are bit-identical to the
        # legacy run.
        class_counts = np.array([25301, 1690, 133756, 2023], dtype=np.float64)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[combined.numpy()]
        return WeightedRandomSampler(
            sample_weights, num_samples=len(train_dataset), replacement=True
        )

    if dataset_name == "celebahq_simple":
        # Adapter column layout: [Smiling, Eyeglasses, Mouth_Slightly_Open,
        # Male, Bald, Wearing_Lipstick, Wearing_Hat]. Balance on
        # ``Eyeglasses`` (col 1) using tempered inverse frequency.
        glass_label = imglabel[:, 1].long()
        class_counts = torch.bincount(glass_label).float()
        # alpha=0 → uniform, alpha=1 → full inverse-frequency. 0.5 was tuned
        # empirically in the original run.
        alpha = 0.5
        class_weights = (1.0 / class_counts) ** alpha
        sample_weights = class_weights[glass_label]
        return WeightedRandomSampler(
            sample_weights, num_samples=len(train_dataset), replacement=True
        )

    return None
