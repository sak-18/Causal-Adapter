# Copyright AstraZeneca UK Ltd. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Morpho-MNIST adapter for the textual-inversion pipeline.

Loads the IDX-format images and the per-image metrics CSV via
:func:`load_morphomnist_like`. ``thickness`` and ``intensity`` are z-scored;
``label`` (digit class) is min-max normalized to ``[0, 1]``.
"""

from __future__ import annotations

import os

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from .._normalization import zscore_continu_minmax_categorical
from ..morphomnist_dataset import _get_paths, load_idx
from .base import DatasetAdapter, make_normalize_transform


class MorphoMNISTAdapter(DatasetAdapter):
    dataset_name = "MorphoMNIST"

    def __init__(self, data_root: str, size: int, set_: str, **_: object):
        super().__init__()
        split_dir = os.path.join(data_root, set_)
        split_csv = os.path.join(data_root, "splits", f"{set_}.csv")
        if not os.path.isfile(split_csv):  # pre-reorg fallback
            split_csv = os.path.join(data_root, f"{set_}.csv")
        if os.path.isdir(split_dir) and os.path.isfile(split_csv):
            # Per-split layout: PNGs in <set_>/ named by row index + a csv with
            # columns index,thickness,intensity,label (supports a val split).
            metric = pd.read_csv(split_csv, index_col="index")
            self.data = [os.path.join(split_dir, f"{i}.png") for i in metric.index]
        else:
            # Back-compat: idx arrays (train/t10k only, no val), under raw/ or root.
            train_bool = set_ == "train"
            raw_root = os.path.join(data_root, "raw")
            if not os.path.isdir(raw_root):
                raw_root = data_root
            images_path, labels_path, metrics_path = _get_paths(raw_root, train=train_bool)
            images = load_idx(images_path)
            labels = load_idx(labels_path)
            metric = pd.read_csv(metrics_path, index_col="index")
            metric["label"] = labels
            self.data = [Image.fromarray(images[i]) for i in range(images.shape[0])]

        metric = zscore_continu_minmax_categorical(
            metric,
            z_score_columns=["thickness", "intensity"],
            min_max_columns=["label"],
        )

        self.num_images = len(self.data)
        # Column order in imglabel: thickness, intensity, label.
        self.imglabel = torch.from_numpy(metric.values)

        self.image_transforms = transforms.Compose(
            [
                transforms.Pad(padding=2),
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        self.normalize_transforms = make_normalize_transform()

    def load_image(self, idx: int):
        src = self.data[idx]
        return src if isinstance(src, Image.Image) else Image.open(src)
