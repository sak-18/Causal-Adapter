"""CheXpert chest-X-ray adapter.

Uses a sampling subdir layout where ``data_root`` ends in
``sampling_<id>``; the matching ``meta_<id>.csv`` lives in the parent
directory. Selected attribute columns: ``Sex``, ``Age``, ``Pleural Effusion``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

from .base import DatasetAdapter, make_normalize_transform


_SELECTED_COLUMNS = ["Sex", "Age", "Pleural Effusion"]
_SEX_MAPPING = {"Female": 0, "Male": 1}


class CheXpertAdapter(DatasetAdapter):
    dataset_name = "chexpert"

    def __init__(self, data_root: str, size: int, **_: object):
        super().__init__()
        dataset_id = data_root.split("/")[-1].split("_")[-1]
        root_parent = os.path.dirname(data_root)
        csv_path = os.path.join(root_parent, f"meta_{dataset_id}.csv")
        df = pd.read_csv(csv_path)

        img_names = np.asarray(df.Path)
        self.image_paths = [os.path.join(data_root, fp) for fp in img_names]
        label_list = np.asarray(df[_SELECTED_COLUMNS])
        label_list[:, 0] = np.vectorize(_SEX_MAPPING.get)(label_list[:, 0])
        self.imglabel = label_list.astype(int)
        self.num_images = len(self.image_paths)

        # Per-column (mean, std) for downstream consumers that expect it.
        col_means = self.imglabel.mean(axis=0)
        col_stds = self.imglabel.std(axis=0)
        self.scale = np.column_stack((col_means, col_stds))

        self.image_transforms = transforms.Compose(
            [
                transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.3),
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        self.normalize_transforms = make_normalize_transform()

    def load_image(self, idx: int):
        return Image.open(self.image_paths[idx])
