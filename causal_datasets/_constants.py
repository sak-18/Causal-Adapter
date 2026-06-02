# Copyright AstraZeneca UK Ltd. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared min/max ranges and scales used by the per-dataset loaders."""

import numpy as np


ADNI_MIN_MAX = {
    "image": [0.0, 255.0],
    "age": [55.1, 89.3],
    "brain_vol": [669364.0, 1350180.0],
    "vent_vol": [5834.0, 145115.0],
}

MORPHOMNIST_MIN_MAX = {
    "thickness": [0.87598526, 6.255515],
    "intensity": [66.601204, 254.90317],
    "image": [0.0, 255.0],
}

# (min, max) per pendulum attribute column:
# pendulum angle, light source angle, shadow length, shadow position.
PENDULUM_MINMAX_SCALE = np.array(
    [[-40, 43], [60, 147], [3, 12], [2, 19]],
    dtype=np.float32,
)
