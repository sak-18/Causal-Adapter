"""Morpho-MNIST dataset loader.

Morpho-MNIST is MNIST augmented with morphological metrics (thickness,
intensity, etc.) extracted from each digit. This module loads the IDX-format
images/labels and the per-image metrics CSV, optionally normalizing both into
``[-1, 1]``.

The IDX read helpers are adapted from
https://github.com/dccastro/Morpho-MNIST/morphomnist/io.py.
"""

from __future__ import annotations

import gzip
import os
import struct

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from ._normalization import morphomnist_normalize


def _load_uint8(f):
    idx_dtype, ndim = struct.unpack("BBBB", f.read(4))[2:]
    shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data


def load_idx(path: str) -> np.ndarray:
    open_fcn = gzip.open if path.endswith(".gz") else open
    with open_fcn(path, "rb") as f:
        return _load_uint8(f)


def _get_paths(root_dir, train):
    prefix = "train" if train else "t10k"
    return (
        os.path.join(root_dir, f"{prefix}-images-idx3-ubyte.gz"),
        os.path.join(root_dir, f"{prefix}-labels-idx1-ubyte.gz"),
        os.path.join(root_dir, f"{prefix}-morpho.csv"),
    )


def load_morphomnist_like(root_dir, train=True, columns=None):
    """Read the images, labels and metrics CSV for one Morpho-MNIST split."""
    images_path, labels_path, metrics_path = _get_paths(root_dir, train)
    images = load_idx(images_path)
    labels = load_idx(labels_path)

    if columns is not None and "index" not in columns:
        usecols = ["index"] + list(columns)
    else:
        usecols = columns

    metrics = pd.read_csv(metrics_path, usecols=usecols, index_col="index")
    return images, labels, metrics


_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


class MorphoMNISTLike(Dataset):
    """Morpho-MNIST style dataset returning ``(image, attribute_vector)`` pairs.

    ``attribute_size`` is the ordered mapping ``{name: width}``. ``digit`` is
    treated specially: it is loaded from the labels file and one-hot encoded
    with width 10. All other names must be columns in the morphological
    metrics CSV.
    """

    def __init__(
        self,
        attribute_size,
        split="train",
        normalize_=True,
        transform=None,
        data_dir=_DEFAULT_DATA_DIR,
    ):
        self.has_valid_set = False
        self.root_dir = data_dir
        self.train = split == "train"
        self.transform = transform
        # 28x28 -> 32x32 to align with the diffusion backbones.
        self.pad = transforms.Pad(padding=2)

        # ``digit`` is handled separately from morphological metrics.
        columns = [att for att in attribute_size.keys() if att != "digit"]

        images, labels, metrics_df = load_morphomnist_like(data_dir, self.train, columns)

        self.images = self.pad(torch.as_tensor(images.copy(), dtype=torch.float32))
        self.labels = F.one_hot(torch.as_tensor(labels.copy(), dtype=torch.long), num_classes=10)

        if columns is None:
            columns = metrics_df.columns
        self.metrics = {col: torch.as_tensor(metrics_df[col], dtype=torch.float32) for col in columns}
        self.columns = columns
        assert len(self.images) == len(self.labels) == len(metrics_df)

        if normalize_:
            self.images, self.metrics["intensity"], self.metrics["thickness"] = morphomnist_normalize(
                self.images, self.metrics["intensity"], self.metrics["thickness"]
            )

        if "digit" in attribute_size:
            self.metrics["digit"] = self.labels

        self.attrs = torch.cat(
            [
                self.metrics[attr].unsqueeze(1) if attr != "digit" else self.metrics[attr]
                for attr in attribute_size.keys()
            ],
            dim=1,
        )

        self.possible_values = {attr: torch.unique(values, dim=0) for attr, values in self.metrics.items()}

        # Bin continuous metrics into 9 buckets across [-1, 1] for qualitative grids.
        bins = np.linspace(-1, 1, 10)
        self.bins = {}
        for attr, values in self.metrics.items():
            if attr != "digit":
                data = values.numpy()
                digitized = np.digitize(data, bins)
                self.bins[attr] = [data[digitized == i].mean() for i in range(1, len(bins))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {col: values[idx] for col, values in self.metrics.items()}
        item["image"] = self.images[idx].unsqueeze(0)
        item["attrs"] = self.attrs[idx]
        if self.transform:
            return self.transform(item["image"], item["attrs"])
        return item["image"], item["attrs"]
