# Copyright AstraZeneca UK Ltd. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base class for textual-inversion dataset adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torchvision import transforms


class DatasetAdapter(ABC):
    """Adapter contract used by :class:`TextualInversionDataset`.

    Subclasses must populate ``num_images``, ``imglabel`` (or set it to
    ``None``), and the two transform pipelines. ``load_image(idx)`` returns a
    raw PIL image for index ``idx`` (already taken modulo ``num_images``).
    """

    dataset_name: str = ""

    def __init__(self):
        self.num_images: int = 0
        self.imglabel: Optional[torch.Tensor] = None
        self.image_transforms: transforms.Compose = transforms.Compose([transforms.ToTensor()])
        self.normalize_transforms: transforms.Compose = transforms.Compose(
            [transforms.Normalize([0.5], [0.5])]
        )

    @abstractmethod
    def load_image(self, idx: int):
        """Return a PIL image for the (already-wrapped) index ``idx``."""

    def extend_text(self, text: str) -> str:
        """Hook for adapters that need to extend the placeholder text.

        Default is a no-op; the ADNI adapter overrides this to repeat the
        placeholder once per slice.
        """
        return text


def make_normalize_transform() -> transforms.Compose:
    """Standard ``[0.5]/[0.5]`` channel normalization shared across adapters."""
    return transforms.Compose([transforms.Normalize([0.5], [0.5])])
