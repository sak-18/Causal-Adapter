"""ADNI brain-MRI dataset loader.

The dataset stores per-subject 2D MRI slices and pulls clinical attributes
(age, sex, brain volume, ventricle volume, APOE genotype) from the ADNIMERGE
CSV. Image tensors land in ``[0, 1]`` and are padded from 180x180 to 192x192.

Layout assumed under ``dataset_dir``::

    preprocessed_data/
        <subject_id>/<.../.../*.tiff>   # one TIFF per slice
    ADNIMERGE*.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ._normalization import normalize as adni_normalize


ATTRIBUTE_MAPPING = {
    "apoE": "APOE4",
    "age": "AGE",
    "sex": "PTGENDER",
    "brain_vol": "WholeBrain",
    "vent_vol": "Ventricles",
}

TOTAL_SLICES = 30

# Train/val/test split is computed on the sorted list of subject directories;
# percentages are applied in order.
SPLIT_TRAIN_VAL_TEST = [
    ("train", 0.7),
    ("valid", 0.15),
    ("test", 0.15),
]


def split_subjects(subjects, split):
    offset = 0
    for split_, percent in SPLIT_TRAIN_VAL_TEST:
        split_size = int(percent * len(subjects))
        if split_ == split:
            return subjects[offset:offset + split_size]
        offset += split_size
    return subjects[offset:offset + split_size]


def _binary_encode_scalar(value: int, width: int) -> np.ndarray:
    """Encode a non-negative integer as a fixed-width binary float vector."""
    return np.array(list(np.binary_repr(value).zfill(width))).astype(np.float32)


def _ordinal_encode_scalar(value: int, width: int) -> np.ndarray:
    """Encode an integer as a thermometer (ordinal) float vector of length ``width``."""
    return np.pad(np.ones(value), (width - value, 0), "constant").astype(np.float32)


def encode_attribute(value, name, num_of_slices=30):
    if name == "sex":
        return 0 if value == "Female" else 1
    if name == "apoE":
        return _binary_encode_scalar(int(value), 2)
    if name == "slice":
        return _ordinal_encode_scalar(int(value), num_of_slices)
    return float(value)


def fix_age(initial_age, visit):
    """Convert ADNI ``VISCODE`` (e.g. ``m12``) into a corrected age in years."""
    if visit == "bl":
        return initial_age
    return initial_age + float(visit[1:]) / 12


def load_data(data_dir, normalize_=True, num_of_slices=30, split="train", keep_only_screening=False):
    """Load ADNI MRI slices for ``split`` and return ``(images, attributes, subject_dates)``.

    Only the centred ``num_of_slices`` slices around the volume midpoint are
    kept (the dataset stores 30 such slices per visit).
    """

    def img_to_0_1(img):
        # Stored TIFFs are in [-1, 1]; convert to [0, 1] to match downstream code.
        return (img + 1) / 2 if normalize_ else img

    leave_out_first = (TOTAL_SLICES - num_of_slices) // 2
    valid_range = range(leave_out_first, leave_out_first + num_of_slices)

    attributes = {"subject": [], "slice": [], "date": []}
    subject_dates_dict = {}
    images = []
    subject_paths = split_subjects(sorted(Path(data_dir).glob("*")), split)

    for subject_path in subject_paths:
        dates = sorted(list(subject_path.glob("*/*")), key=lambda d: d.name)
        if keep_only_screening:
            dates = dates[:1]
            # These two subjects only have volume measurements at the second visit,
            # so the first visit would yield nulls when joined with ADNIMERGE.
            if subject_path.name in ("123_S_0050", "137_S_0825"):
                dates = dates[1:2]
        subject_dates_dict[subject_path.name] = []
        for date_path in dates:
            date = date_path.name
            subject_dates_dict[subject_path.name].append(date)

            for image_path in sorted(date_path.glob("*/*.tiff")):
                slice_idx = int(image_path.stem.split("slice")[1])
                if slice_idx in valid_range:
                    images.append(img_to_0_1(np.array(Image.open(image_path))))
                    attributes["subject"].append(subject_path.name)
                    attributes["slice"].append(
                        encode_attribute(slice_idx - leave_out_first, "slice", num_of_slices)
                    )
                    attributes["date"].append(date)

    return np.array(images), attributes, subject_dates_dict


def load_extra_attributes(csv_path, attributes, attribute_dict, subject_dates_dict, keep_only_screening=False):
    """Join image-side metadata with ADNIMERGE clinical attributes.

    Returns the augmented attribute dict and the list of indices to drop
    (subjects/dates with missing entries in ADNIMERGE).
    """
    index_col = "PTID"
    usecols = [index_col, "VISCODE", "EXAMDATE"] + [
        ATTRIBUTE_MAPPING[att] for att in attributes if att in ATTRIBUTE_MAPPING
    ]
    df = pd.read_csv(csv_path, usecols=usecols, index_col=index_col).sort_index()

    for att in ATTRIBUTE_MAPPING:
        if att in attributes:
            attribute_dict[att] = []

    indices_to_remove = []

    for idx, (subject, date) in enumerate(zip(attribute_dict["subject"], attribute_dict["date"])):
        subject_df = df.loc[subject].sort_values(by="EXAMDATE")

        # Same exception as in load_data: these subjects have volume measures
        # only at the second exam.
        if keep_only_screening and subject in ("123_S_0050", "137_S_0825"):
            date_idx = 1
        else:
            date_idx = subject_dates_dict[subject].index(date)

        if date_idx >= len(subject_df) or subject_df.iloc[date_idx].isnull().any().any():
            indices_to_remove.append(idx)
            continue

        for att, csv_att in ATTRIBUTE_MAPPING.items():
            if att in attributes:
                value = encode_attribute(subject_df[csv_att].iloc[date_idx], att)
                if att == "age":
                    value = fix_age(initial_age=value, visit=subject_df["VISCODE"].iloc[date_idx])
                attribute_dict[att].append(value)

    del attribute_dict["subject"]
    del attribute_dict["date"]
    return attribute_dict, indices_to_remove


_DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "preprocessing")


class ADNI(Dataset):
    """ADNI MRI dataset returning ``(image, attribute_vector)`` pairs.

    ``attribute_size`` is an ordered mapping ``{name: width}`` describing which
    attributes to expose and the encoded width of each (e.g. ``apoE`` is
    binary-encoded with width 2; ``slice`` is thermometer-encoded with width
    ``num_of_slices``). The order of keys also defines the column order in the
    output ``attrs`` tensor.
    """

    def __init__(
        self,
        attribute_size,
        split="train",
        normalize_=True,
        transform=None,
        transform_cls=None,
        num_of_slices=30,
        keep_only_screening=False,
        dataset_dir=_DEFAULT_DATASET_DIR,
    ):
        super().__init__()
        self.has_valid_set = True
        self.transform = transform
        self.transform_cls = transform_cls
        # 180x180 -> 192x192 to match downstream model resolutions.
        self.pad = transforms.Pad(padding=6)

        num_of_slices = attribute_size["slice"]
        assert num_of_slices <= 30, "The 30 middle slices have been saved"

        data_dir = os.path.join(dataset_dir, "preprocessed_data")
        images, attribute_dict, subject_dates_dict = load_data(
            data_dir,
            num_of_slices=num_of_slices,
            split=split,
            keep_only_screening=keep_only_screening,
        )

        csv_paths = list(Path(dataset_dir).glob("ADNIMERGE*.csv"))
        assert csv_paths and csv_paths[0].is_file(), "Provide ADNIMERGE csv path"
        csv_path = csv_paths[0]

        self.attributes, indices_to_remove = load_extra_attributes(
            csv_path,
            attributes=attribute_size.keys(),
            attribute_dict=attribute_dict,
            subject_dates_dict=subject_dates_dict,
            keep_only_screening=keep_only_screening,
        )
        self.attributes["slice"] = np.delete(
            self.attributes["slice"], indices_to_remove, axis=0
        )

        kept_images = np.delete(images, indices_to_remove, axis=0).copy()
        self.images = self.pad(
            torch.as_tensor(kept_images, dtype=torch.float32).unsqueeze(1)
        )

        if normalize_:
            self.attributes = {
                attr: adni_normalize(torch.tensor(np.array(values), dtype=torch.float32), attr)
                for attr, values in self.attributes.items()
            }
        else:
            self.attributes = {
                attr: torch.tensor(np.array(values), dtype=torch.float32)
                for attr, values in self.attributes.items()
            }

        self.attrs = torch.cat(
            [
                self.attributes[attr].unsqueeze(1)
                if len(self.attributes[attr].shape) == 1
                else self.attributes[attr]
                for attr in attribute_size.keys()
            ],
            dim=1,
        )

        self.possible_values = {
            attr: torch.unique(values, dim=0) for attr, values in self.attributes.items()
        }

        # Bin continuous attributes into 4 quartile-style buckets for downstream
        # qualitative grids; categorical attributes are skipped.
        bins = np.linspace(0, 1, 5)
        self.bins = {}
        for attr, values in self.attributes.items():
            if attr not in ("sex", "apoE", "slice"):
                data = values.numpy()
                digitized = np.digitize(data, bins)
                self.bins[attr] = [data[digitized == i].mean() for i in range(1, len(bins))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.images[idx], self.attrs[idx])
        if self.transform_cls:
            return self.transform_cls(self.images[idx]), self.attrs[idx]
        return self.images[idx], self.attrs[idx]
