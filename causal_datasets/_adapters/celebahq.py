# Copyright AstraZeneca UK Ltd. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CelebA-HQ adapter for the textual-inversion pipeline.

Uses :class:`CelebAHQ` (the local extension of torchvision's CelebA dataset).
``celebahq_simple`` exposes seven facial attributes; the ``complex`` variant is
reserved for future use.
"""

from __future__ import annotations

import torch
from torchvision import transforms

from ..celebahq_dataset import CelebAHQ
from .base import DatasetAdapter, make_normalize_transform


_SIMPLE_ATTRS = [
    "Smiling",
    "Eyeglasses",
    "Mouth_Slightly_Open",
    "Male",
    "Bald",
    "Wearing_Lipstick",
    "Wearing_Hat",
]


class CelebAHQAdapter(DatasetAdapter):
    dataset_name = "celebahq"

    def __init__(self, data_root: str, size: int, dataset: str, **_: object):
        super().__init__()
        if "simple" in dataset:
            selected = _SIMPLE_ATTRS
        elif "complex" in dataset:
            raise NotImplementedError("celebahq_complex variant is not implemented yet")
        else:
            raise ValueError(f"Unknown CelebA-HQ variant: {dataset!r}")

        pre_transforms = transforms.Compose(
            [transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR)]
        )
        self.data = CelebAHQ(root=data_root, split="train", transform=pre_transforms, download=False)
        self.num_images = len(self.data)

        attribute_ids = [self.data.attr_names.index(attr) for attr in selected]
        metrics = {
            attr: torch.as_tensor(self.data.attr[:, attr_id], dtype=torch.float32)
            for attr, attr_id in zip(selected, attribute_ids)
        }
        self.imglabel = torch.cat([metrics[attr].unsqueeze(1) for attr in selected], dim=1)

        # The pre-resize is already done inside ``CelebAHQ``; here we only flip
        # and tensorize.
        self.image_transforms = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        )
        self.normalize_transforms = make_normalize_transform()

    def load_image(self, idx: int):
        return self.data[idx][0]
