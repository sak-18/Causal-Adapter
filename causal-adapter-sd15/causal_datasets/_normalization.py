"""Shared normalization and label-encoding helpers for the dataset loaders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ._constants import ADNI_MIN_MAX, MORPHOMNIST_MIN_MAX


def normalize(value, name, ranges=None):
    """Min-max normalize ``value`` for an attribute called ``name``.

    Returns ``value`` unchanged if ``name`` has no entry in ``ranges``.
    Defaults to the ADNI ranges to match the legacy ``adni_normalize`` call
    sites.
    """
    ranges = ranges if ranges is not None else ADNI_MIN_MAX
    if name not in ranges:
        return value
    lo, hi = ranges[name]
    return (value - lo) / (hi - lo)


def unnormalize(value, name, ranges=None):
    """Inverse of :func:`normalize`."""
    ranges = ranges if ranges is not None else ADNI_MIN_MAX
    if name not in ranges:
        return value
    lo, hi = ranges[name]
    return value * (hi - lo) + lo


def morphomnist_normalize(images, intensity, thickness):
    """Map MorphoMNIST images and metrics into ``[-1, 1]``.

    The image tensor is expected to be in ``[0, 255]``; the two morphological
    metrics are normalized using ``MORPHOMNIST_MIN_MAX``.
    """
    out = {}
    for key, value in (("image", images), ("intensity", intensity), ("thickness", thickness)):
        lo, hi = MORPHOMNIST_MIN_MAX[key]
        scaled = (value - lo) / (hi - lo)
        out[key] = 2.0 * scaled - 1.0
    return out["image"], out["intensity"], out["thickness"]


def morphomnist_unnormalize(value, name):
    """Inverse of the per-attribute path of :func:`morphomnist_normalize`."""
    lo, hi = MORPHOMNIST_MIN_MAX[name]
    value = (value + 1.0) / 2.0
    return value * (hi - lo) + lo


def bin_array(num, m=None, reverse=False):
    """Encode integers as binary vectors (or decode back).

    ``num`` is a torch tensor. When ``reverse=False`` (default) it must be a
    1-D int tensor; the result has shape ``(len(num), m)`` with each row a
    binary representation of the corresponding entry. When ``reverse=True``
    the inverse mapping is applied row-wise.
    """
    if reverse:
        if num.dim() == 1:
            num = num.unsqueeze(dim=0)
        width = num.shape[1]
        weights = 2 ** torch.arange(
            width - 1, -1, -1, dtype=torch.float32, device=num.device
        )
        return torch.sum(num * weights, dim=1)

    if m is None:
        m = int(torch.ceil(torch.log2(num.max().float() + 1)).item())
    powers = 2 ** torch.arange(m - 1, -1, -1, device=num.device)
    return ((num.unsqueeze(1).long() & powers) > 0).float()


def ordinal_array(num, m=None, reverse=False, scale=1):
    """Encode an integer ``num`` in ``{0, ..., m}`` as a thermometer vector.

    ``reverse=True`` decodes a thermometer-coded torch tensor back to an
    integer count (multiplied by ``scale``).
    """
    if reverse:
        return scale * torch.count_nonzero(num, dim=1).to(num.device)
    return np.pad(np.ones(num), (m - num, 0), "constant").astype(np.float32)


def zscore_continu_minmax_categorical(attr_df: pd.DataFrame, z_score_columns, min_max_columns):
    """Z-score the listed continuous columns and min-max the categorical ones.

    Mutates ``attr_df`` in place and also returns it.
    """
    if z_score_columns:
        attr_df[z_score_columns] = StandardScaler().fit_transform(attr_df[z_score_columns])
    if min_max_columns:
        attr_df[min_max_columns] = MinMaxScaler().fit_transform(attr_df[min_max_columns])
    return attr_df
