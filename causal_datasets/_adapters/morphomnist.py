# Copyright AstraZeneca UK Ltd. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Morpho-MNIST adapter for the textual-inversion pipeline.

Loads the IDX-format images and the per-image metrics CSV via
:func:`load_morphomnist_like`. ``thickness`` and ``intensity`` are z-scored;
``label`` (digit class) is min-max normalized to ``[0, 1]``.
"""

from __future__ import annotations

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
        train_bool = set_ == "train"
        images_path, labels_path, metrics_path = _get_paths(data_root, train=train_bool)
        images = load_idx(images_path)
        labels = load_idx(labels_path)
        metric = pd.read_csv(metrics_path, index_col="index")

        metric["label"] = labels
        metric = zscore_continu_minmax_categorical(
            metric,
            z_score_columns=["thickness", "intensity"],
            min_max_columns=["label"],
        )

        self.num_images = len(images)
        self.data = [Image.fromarray(images[i]) for i in range(images.shape[0])]
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
        return self.data[idx]
