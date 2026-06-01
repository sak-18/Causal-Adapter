"""ADNI adapter for the textual-inversion pipeline.

Loads ADNI MRI slices via :func:`load_data` and joins them with ADNIMERGE
clinical attributes via :func:`load_extra_attributes`. The resulting
``imglabel`` tensor concatenates each attribute in the order defined by
``_ATTRIBUTE_SIZE`` (so columns are ``[apoE(2), age, sex, brain_vol, vent_vol,
slice(num_of_slices)]``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .._normalization import normalize as adni_normalize
from ..adni_dataset import load_data, load_extra_attributes
from .base import DatasetAdapter, make_normalize_transform


_NUM_OF_SLICES = 10
_ATTRIBUTE_SIZE = {
    "apoE": 2,
    "age": 1,
    "sex": 1,
    "brain_vol": 1,
    "vent_vol": 1,
    "slice": _NUM_OF_SLICES,
}


class ADNIAdapter(DatasetAdapter):
    dataset_name = "ADNI"

    def __init__(self, data_root: str, size: int, set_: str, **_: object):
        super().__init__()
        self.num_of_slices = _NUM_OF_SLICES

        data_dir = Path(data_root) / "preprocessed_data"
        image_arr, attribute_dict, subject_dates_dict = load_data(
            str(data_dir),
            num_of_slices=_NUM_OF_SLICES,
            split=set_,
            keep_only_screening=False,
        )

        csv_paths = list(Path(data_root).glob("ADNIMERGE*.csv"))
        assert csv_paths and csv_paths[0].is_file(), "Provide ADNIMERGE csv path"
        csv_path = csv_paths[0]

        attributes, indices_to_remove = load_extra_attributes(
            csv_path,
            attributes=_ATTRIBUTE_SIZE.keys(),
            attribute_dict=attribute_dict,
            subject_dates_dict=subject_dates_dict,
            keep_only_screening=False,
        )
        attributes["slice"] = np.delete(attributes["slice"], indices_to_remove, axis=0)
        attributes = {
            attr: adni_normalize(torch.tensor(np.array(values), dtype=torch.float32), attr)
            for attr, values in attributes.items()
        }

        self.imglabel = torch.cat(
            [
                attributes[attr].unsqueeze(1)
                if attributes[attr].dim() == 1
                else attributes[attr]
                for attr in _ATTRIBUTE_SIZE.keys()
            ],
            dim=1,
        )

        self._raw_images = [
            item for i, item in enumerate(image_arr) if i not in set(indices_to_remove)
        ]
        self.num_images = len(self._raw_images)

        self.image_transforms = transforms.Compose(
            [
                transforms.Pad(padding=6),
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        self.normalize_transforms = make_normalize_transform()

    def load_image(self, idx: int):
        # Stored slices are float arrays in [0, 1]; convert to a uint8 PIL image.
        image = self._raw_images[idx]
        image_array = (image * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(image_array, mode="L")

    def extend_text(self, text: str) -> str:
        # ADNI uses a one-hot slice encoder, so the placeholder must be repeated
        # ``num_of_slices`` times in the prompt.
        if self.num_of_slices > 1:
            return text + (" " + text[-1]) * (self.num_of_slices - 1)
        return text
