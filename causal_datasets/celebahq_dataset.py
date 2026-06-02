# Copyright AstraZeneca UK Ltd. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CelebA-HQ dataset wrapper.

Extends ``torchvision.datasets.CelebA`` so the high-resolution CelebAMask-HQ
images can be paired with their original CelebA attribute annotations. The
dataset returns ``(image, attribute_vector)`` pairs where the attribute vector
is a float tensor in ``{0, 1}`` (the original ``{-1, 1}`` labels are remapped
in-class).

Expected layout under ``<root>/celebahq``::

    list_eval_partition.txt
    CelebAMask-HQ/
        CelebA-HQ-to-CelebA-mapping.txt
        CelebAMask-HQ-attribute-anno.txt
        CelebA-HQ-img/
            0.jpg, 1.jpg, ...
"""

from __future__ import annotations

import csv
import os
import re

import PIL
import torch
from torchvision.datasets import CelebA
from torchvision.datasets.utils import check_integrity, verify_str_arg


class CelebAHQ(CelebA):
    base_folder = "celebahq"
    file_list = [
        # (Google Drive file id, MD5, filename)
        ("1badu11NqxGf6qM3PTTooQDJvQbejgbTv", "b08032b342a8e0cf84c273db2b52eef3", "CelebAMask-HQ.zip"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
        self,
        root,
        split="train",
        target_type="attr",
        attributes=None,
        transform=None,
        target_transform=None,
        download=False,
    ):
        super(CelebA, self).__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.target_type = target_type if isinstance(target_type, list) else [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

        split_map = {"train": 0, "valid": 1, "test": 2, "all": None}
        split_id = split_map[verify_str_arg(split.lower(), "split", tuple(split_map.keys()))]

        splits = self._load_csv("list_eval_partition.txt")
        id_map = self._load_id_map("CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt", header=0)
        attr = self._load_csv("CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt", header=1)
        self.id_map_file_name = "CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt"

        if split_id is None:
            self.filename = list(splits.index)
        else:
            mask = (splits.data == split_id).squeeze()
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        self.filename = [id_map[f] for f in self.filename if f in id_map]

        self.attr = torch.zeros((len(self.filename), attr.data.shape[1]), dtype=torch.int64)
        if split_id is not None:
            for i, f in enumerate(self.filename):
                num = int(re.sub(r"[^0-9]", "", f))
                self.attr[i] = attr.data[num]
        # CelebA labels are {-1, 1}; map to {0, 1}.
        self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")
        self.attr_names = attr.header

        if attributes is not None:
            self.attr_idxs = [self.attr_names.index(name) for name in attributes]

    def _check_integrity(self):
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Original archives may be deleted after extraction.
            if ext not in (".zip", ".7z") and not check_integrity(fpath, md5):
                return False
        return os.path.isdir(os.path.join(self.root, self.base_folder, "CelebAMask-HQ/CelebA-HQ-img"))

    def download(self):
        # Disabled by design: CelebA-HQ archives must be staged manually because
        # Google Drive throttling makes scripted downloads unreliable. Follow
        # the dataset README for the expected file layout instead.
        return

    def _load_id_map(self, filename, header=None):
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            data = data[header + 1:]

        indices = [row[0] for row in data]
        orig_files = [row[2] for row in data]

        return {orig: f"{idx}.jpg" for idx, orig in zip(indices, orig_files)}

    def __getitem__(self, index):
        img_path = os.path.join(
            self.root, self.base_folder, "CelebAMask-HQ/CelebA-HQ-img", self.filename[index]
        )
        x = PIL.Image.open(img_path)

        target = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            else:
                raise ValueError(f"Target type '{t}' is not recognized.")

        if self.transform is not None:
            x = self.transform(x)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        if hasattr(self, "attr_idxs"):
            target = target[self.attr_idxs]

        return x, target.float()
