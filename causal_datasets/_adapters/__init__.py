# Copyright AstraZeneca UK Ltd. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-dataset adapters used by :class:`TextualInversionDataset`.

Each adapter encapsulates a single dataset family (loading images, building
the attribute label tensor, and providing the matching torchvision
transforms). The ``ADAPTERS`` mapping below resolves ``dataset`` strings to
the appropriate adapter class.
"""

from .adni import ADNIAdapter
from .base import DatasetAdapter
from .celeba import CelebAAdapter
from .celebahq import CelebAHQAdapter
from .chexpert import CheXpertAdapter
from .human import HumanAdapter
from .morphomnist import MorphoMNISTAdapter
from .pendulum import PendulumAdapter

ADAPTERS = {
    "pendulum": PendulumAdapter,
    "celeA_simple": CelebAAdapter,
    "celeA_complex": CelebAAdapter,
    "celebahq_simple": CelebAHQAdapter,
    "MorphoMNIST": MorphoMNISTAdapter,
    "ADNI": ADNIAdapter,
    "human": HumanAdapter,
    "chexpert": CheXpertAdapter,
}

__all__ = [
    "ADAPTERS",
    "DatasetAdapter",
    "ADNIAdapter",
    "CelebAAdapter",
    "CelebAHQAdapter",
    "CheXpertAdapter",
    "HumanAdapter",
    "MorphoMNISTAdapter",
    "PendulumAdapter",
]
