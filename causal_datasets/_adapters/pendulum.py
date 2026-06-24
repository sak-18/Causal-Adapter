# Copyright AstraZeneca UK Ltd. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pendulum dataset adapter (CausalVAE-style images named by their attributes).

Each image filename encodes the four pendulum attribute values separated by
underscores. Labels are min-max normalized using the fixed ranges in
:data:`PENDULUM_MINMAX_SCALE`.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .._constants import PENDULUM_MINMAX_SCALE
from .base import DatasetAdapter, make_normalize_transform


class PendulumAdapter(DatasetAdapter):
    dataset_name = "pendulum"

    def __init__(self, data_root: str, size: int, set_: str = "train", **_: object):
        super().__init__()
        # Prefer the per-split subdir (train/ val/ test/); fall back to a flat
        # dir of PNGs for backward compatibility with the upstream layout.
        split_dir = os.path.join(data_root, set_)
        self.data_root = split_dir if os.path.isdir(split_dir) else data_root

        listing = sorted(fp for fp in os.listdir(self.data_root) if fp.endswith(".png"))
        self.image_paths = [os.path.join(self.data_root, fp) for fp in listing]
        self.image_names = [fp.split(".")[0] for fp in listing]
        self.num_images = len(self.image_paths)

        # Filename pattern: "<prefix>_<a1>_<a2>_<a3>_<a4>.<ext>".
        labels = np.asarray(
            [list(map(float, p[:-4].split("/")[-1].split("_")[1:])) for p in self.image_paths],
            dtype=np.float32,
        )
        lo = PENDULUM_MINMAX_SCALE[:, 0]
        hi = PENDULUM_MINMAX_SCALE[:, 1]
        self.imglabel = torch.from_numpy(((labels - lo) / (hi - lo)).astype(np.float32))

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        self.normalize_transforms = make_normalize_transform()

    def load_image(self, idx: int):
        return Image.open(self.image_paths[idx])
