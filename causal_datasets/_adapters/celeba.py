"""CelebA dataset adapter for the textual-inversion pipeline.

Wraps ``torchvision.datasets.CelebA`` and exposes a configurable subset of
attributes. ``celeA_simple`` selects ``Smiling`` and ``Eyeglasses``;
``celeA_complex`` selects ``Young``, ``Male``, ``No_Beard``, ``Bald``.
"""

from __future__ import annotations

import torch
from torchvision import transforms
from torchvision.datasets import CelebA

from .base import DatasetAdapter, make_normalize_transform


_SIMPLE_ATTRS = ["Smiling", "Eyeglasses"]
_COMPLEX_ATTRS = ["Young", "Male", "No_Beard", "Bald"]


class CelebAAdapter(DatasetAdapter):
    dataset_name = "celeA"

    def __init__(self, data_root: str, size: int, dataset: str, **_: object):
        super().__init__()
        if "simple" in dataset:
            selected = _SIMPLE_ATTRS
        elif "complex" in dataset:
            selected = _COMPLEX_ATTRS
        else:
            raise ValueError(f"Unknown CelebA variant: {dataset!r}")

        self.data = CelebA(root=data_root, split="train", transform=None, download=False)
        self.num_images = len(self.data)

        attribute_ids = [self.data.attr_names.index(attr) for attr in selected]
        metrics = {
            attr: torch.as_tensor(self.data.attr[:, attr_id], dtype=torch.float32)
            for attr, attr_id in zip(selected, attribute_ids)
        }
        self.imglabel = torch.cat([metrics[attr].unsqueeze(1) for attr in selected], dim=1)

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
        return self.data[idx][0]
