# Copyright AstraZeneca UK Ltd. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CelebA dataset adapter for the textual-inversion pipeline.

Reads the aligned CelebA images from a per-split folder layout
(``<data_root>/{train,val,test}/<file>.jpg``) and the attribute annotations
from ``<data_root>/list_attr_celeba.txt``. ``celeA_simple`` selects ``Smiling``
and ``Eyeglasses``; ``celeA_complex`` selects ``Young``, ``Male``,
``No_Beard``, ``Bald``. Attribute values are remapped from {-1, 1} to {0, 1}.
"""

from __future__ import annotations

import os

import torch
from PIL import Image
from torchvision import transforms

from .base import DatasetAdapter, make_normalize_transform


_SIMPLE_ATTRS = ["Smiling", "Eyeglasses"]
_COMPLEX_ATTRS = ["Young", "Male", "No_Beard", "Bald"]
_SPLIT_DIR = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def parse_attr_file(path):
    """Return (attr_names, {filename: [int x40]}) from a list_attr_celeba.txt."""
    with open(path) as f:
        f.readline()  # leading count line
        names = f.readline().split()
        attrs = {}
        for line in f:
            parts = line.split()
            attrs[parts[0]] = [int(v) for v in parts[1:]]
    return names, attrs


class CelebAAdapter(DatasetAdapter):
    dataset_name = "celeA"

    def __init__(self, data_root: str, size: int, dataset: str, set_: str = "train", **_: object):
        super().__init__()
        if "simple" in dataset:
            selected = _SIMPLE_ATTRS
        elif "complex" in dataset:
            selected = _COMPLEX_ATTRS
        else:
            raise ValueError(f"Unknown CelebA variant: {dataset!r}")

        self.img_dir = os.path.join(data_root, _SPLIT_DIR.get(set_, set_))
        names, attrs = parse_attr_file(os.path.join(data_root, "list_attr_celeba.txt"))
        self.filenames = sorted(fn for fn in os.listdir(self.img_dir) if fn.endswith(".jpg"))
        self.num_images = len(self.filenames)

        col = [names.index(attr) for attr in selected]
        # {-1, 1} -> {0, 1}; column order matches ``selected``.
        self.imglabel = torch.tensor(
            [[(attrs[fn][c] + 1) // 2 for c in col] for fn in self.filenames],
            dtype=torch.float32,
        )

        self.image_transforms = transforms.Compose(
            [
                transforms.CenterCrop(150),
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        self.normalize_transforms = make_normalize_transform()

    def load_image(self, idx: int):
        return Image.open(os.path.join(self.img_dir, self.filenames[idx]))
