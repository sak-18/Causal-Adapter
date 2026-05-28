"""'Human' adapter: a simple image-folder loader with placeholder labels.

Used for unconditional textual-inversion runs over arbitrary face crops where
no causal labels are available; ``imglabel`` is a single-row constant tensor
just to satisfy the dispatcher contract.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .base import DatasetAdapter, make_normalize_transform


class HumanAdapter(DatasetAdapter):
    dataset_name = "human"

    def __init__(self, data_root: str, size: int, **_: object):
        super().__init__()
        listing = os.listdir(data_root)
        self.image_paths = [os.path.join(data_root, fp) for fp in listing]
        self.num_images = len(self.image_paths)
        self.imglabel = torch.from_numpy(np.asarray([[1, 1, 1, 1]]))

        self.image_transforms = transforms.Compose(
            [
                transforms.CenterCrop(150),
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        self.normalize_transforms = make_normalize_transform()

    def load_image(self, idx: int):
        return Image.open(self.image_paths[idx])
