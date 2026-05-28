"""Unified control-head implementation.

This module consolidates the previous per-dataset modules
``causal_modules/control_heads/{pendulum,celeA,ADNI,MorphoMNIST}.py``
into a single file with one ``ControlNetConditioningEmbedding`` class.

Dataset-specific differences are dispatched internally based on the
``dataset_name`` constructor argument:

* ``pendulum`` / ``MorphoMNIST``: continuous SCM head, MSE loss.
* ``celeA*``: binary SCM head, sigmoid + BCE loss, 0.5 hard threshold,
  plus the celeA OOD rule (``attr2==0 ==> -1`` when intervening on idx 2).
* ``ADNI*``: continuous SCM head running on a 6-dim semantic vector;
  16-dim inputs are first compressed via ``transform_attributes`` and
  outputs of ``inference`` are expanded back to 16 dims.  Interventions
  on indices 0/5 are decoded via ``bin_array`` / ``ordinal_array``.
* ``celebahq*``: treated as continuous (matches prior pendulum-style behavior).

The helpers ``bin_array``, ``ordinal_array``, ``transform_attributes``
remain top-level exports because notebooks import them directly.
"""

from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


__all__ = [
    "ControlNetConditioningEmbedding",
    "DispatcherLayer",
    "MLP",
    "LinearParallel",
    "get_activation",
    "zero_module",
    "bin_array",
    "ordinal_array",
    "transform_attributes",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def get_activation(activation: Literal["relu", "sigmoid", "tanh", "linear", "leakyrelu"]):
    """Return an ``nn.Module`` activation by name."""
    if activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sin":
        return torch.sin()
    elif activation == "linear":
        return nn.Identity()
    elif activation == "leakyrelu":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unknown activation function: {activation}")


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


# ---------------------------------------------------------------------------
# ADNI-specific encoding helpers (kept at top-level: notebooks import them)
# ---------------------------------------------------------------------------


def bin_array(num: torch.Tensor, m: int = None, reverse: bool = False):
    if reverse:
        if num.dim() == 1:
            num = num.unsqueeze(dim=0)
        bs, width = num.shape
        weights = 2 ** torch.arange(width - 1, -1, -1, dtype=torch.float32, device=num.device)
        return torch.sum(num * weights, dim=1)
    else:
        if m is None:
            m = int(torch.ceil(torch.log2(num.max().float() + 1)).item())
        powers = 2 ** torch.arange(m - 1, -1, -1, device=num.device)
        num = num.unsqueeze(1).long()
        return ((num & powers) > 0).float()


def ordinal_array(num: torch.Tensor, m: int = 10, reverse: bool = False, scale: float = 1.0):
    if reverse:
        if num.dim() == 1:
            num = num.unsqueeze(dim=0)
        return scale * torch.count_nonzero(num, dim=1).to(num.dtype)
    else:
        bs = num.shape[0]
        device = num.device
        out = torch.zeros((bs, m), dtype=torch.float32, device=device)
        for i in range(bs):
            count = int(num[i].item())
            if count > 0:
                out[i, m - count: m] = scale
        return out


def transform_attributes(tensor: torch.Tensor, reverse: bool = False) -> torch.Tensor:
    """Convert between 16-dim and 6-dim ADNI feature representations.

    Field order: [apoE (2 or scalar), age, sex, brain_vol, ventricle_vol,
    slice (10 or scalar)].
    """
    if not reverse:
        apoE_labels = bin_array(tensor[:, 0:2], reverse=True)
        slice_labels = ordinal_array(tensor[:, -10:], reverse=True)
        other_features = tensor[:, 2:6]
        return torch.cat([
            apoE_labels.unsqueeze(1),
            other_features,
            slice_labels.unsqueeze(1),
        ], dim=1)
    else:
        apoE_labels = tensor[:, 0].long()
        other_features = tensor[:, 1:5]
        slice_labels = tensor[:, 5].long()
        apoE_encoding = bin_array(apoE_labels, m=2, reverse=False)
        slice_encoding = ordinal_array(slice_labels, m=10, reverse=False)
        return torch.cat([
            apoE_encoding,
            other_features,
            slice_encoding,
        ], dim=1)


# ---------------------------------------------------------------------------
# Shared low-level layers
# ---------------------------------------------------------------------------


class DispatcherLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, adjacency_p=2.0, mask=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.adjacency_p = adjacency_p

        if mask is not None:
            self.register_buffer("mask", torch.tensor(mask).float())
        else:
            self.register_buffer("mask", torch.ones((in_dim, out_dim)))

        self._weight = nn.Parameter(torch.zeros(in_dim, out_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim, hidden_dim))
        self.reset_parameters_bounded_eigenvalues()

    @property
    def weight(self):
        if self.mask is not None:
            return self._weight * self.mask[:, :, None]
        return self._weight

    def forward(self, x):
        """(batch_size, in_dim) -> (batch_size, out_dim, hidden_dim)."""
        x = torch.einsum("ni, ioh -> noh", x, self.weight) + self.bias
        return x

    @torch.no_grad()
    def reset_parameters(self):
        self.reset_parameters_bounded_eigenvalues()

    @torch.no_grad()
    def reset_parameters_bounded_eigenvalues(self, scale=1.0):
        if self._weight.device.type == "meta":
            print("Skipping initialization on meta device.")
            return
        bound = scale / self.in_dim / self.hidden_dim ** (1.0 / self.adjacency_p)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def get_adjacency_matrix(self):
        # same as Dagma sum(pow)
        return torch.linalg.vector_norm(self.weight, dim=2, ord=self.adjacency_p)

    def __repr__(self):
        return (
            f"DispatcherLayer(in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"hidden_dim={self.hidden_dim}, adjacency_p={self.adjacency_p})"
        )


class MLP(nn.Module):
    """A simple 4-layer MLP used as the per-variable nonlinearity."""

    def __init__(self, latent_dim, num_var, middle_dim=64, use_bias=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_var = num_var

        self.net = nn.Sequential(
            nn.Linear(self.latent_dim // self.num_var, middle_dim, bias=use_bias),
            nn.LeakyReLU(),
            nn.Linear(middle_dim, middle_dim, bias=use_bias),
            nn.LeakyReLU(),
            nn.Linear(middle_dim, self.latent_dim // self.num_var, bias=use_bias),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim // self.num_var, 1, bias=use_bias),
        )

    def forward(self, x):
        return self.net(x)


class LinearParallel(nn.Module):
    def __init__(self, in_dim, out_dim, parallel_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.parallel_dim = parallel_dim

        self.weight = nn.Parameter(torch.zeros(parallel_dim, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(parallel_dim, out_dim))
        self.reset_parameters()

    def forward(self, x):
        """(batch_size, parallel_dim, in_dim) -> (batch_size, parallel_dim, out_dim)."""
        x = torch.einsum("npi, pio -> npo", x, self.weight) + self.bias
        return x

    @torch.no_grad()
    def reset_parameters(self):
        if self.weight.device.type == "meta":
            print("Skipping initialization on meta device.")
            return
        bound = 1.0 / self.in_dim**0.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def __repr__(self):
        return (
            f"LinearParallel(in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"parallel_dim={self.parallel_dim})"
        )


# ---------------------------------------------------------------------------
# Unified ControlNetConditioningEmbedding
# ---------------------------------------------------------------------------


def _resolve_strategy(dataset_name: str):
    """Return (head_type, pre_transform, intervention_decoder, ood_hook) for ``dataset_name``.

    * ``head_type``: ``"continuous"`` -> MSE, raw output;
      ``"binary"`` -> sigmoid + BCE + hard 0/1 threshold output.
    * ``pre_transform``: callable applied to ``label`` when ``label.shape[1] == 16``
      before running the SCM (and reverse-applied to the output of ``inference``).
      ``None`` if no reshape is needed.
    * ``intervention_decoder``: ``dict[int, Callable]`` mapping
      ``intervention_indx -> decoder(intervention_values)``.  Empty dict if
      no decoding is needed.
    * ``ood_hook``: optional callable ``(preds, label, intervention_indx) -> preds``
      applied after the SCM in ``inference`` to encode dataset-specific
      out-of-distribution rules (only celeA today).
    """
    name = (dataset_name or "").lower()

    head_type = "continuous"
    pre_transform = None
    intervention_decoder = {}
    ood_hook = None

    if "celea" in name or "celebahq" in name:
        # NOTE: ``celebahq*`` historically used the *same* class as ``celeA*``
        # in ``load_dataset_model`` (``autoencoder = celeA.ControlNetConditioningEmbedding``),
        # i.e. the binary head with the same OOD rule.
        head_type = "binary"
        ood_hook = _celeA_ood_hook

    if "adni" in name:
        pre_transform = transform_attributes
        intervention_decoder = {
            0: lambda v: bin_array(v, reverse=True),
            5: lambda v: ordinal_array(v, reverse=True),
        }

    return head_type, pre_transform, intervention_decoder, ood_hook


def _celeA_ood_hook(preds: torch.Tensor, label: torch.Tensor, intervention_indx):
    """celeA OOD rule: when intervening on attribute 2, mark rows where
    label[:, 1] == 0 with -1 to flag invalid conditioning."""
    if intervention_indx == 2:
        row_index = torch.where(label[:, 1] == 0)[0]
        if row_index.numel() > 0:
            preds = preds.clone()
            preds[row_index, intervention_indx] = -1
    return preds


class ControlNetConditioningEmbedding(nn.Module):
    """Unified SCM-style control head used by all datasets.

    Parameters
    ----------
    in_dim, hidden_dims, activation, adjacency_p, mask
        Same meaning as the legacy per-dataset modules.
    dataset_name
        Selects dataset-specific behavior (see ``_resolve_strategy``).
        Defaults to ``"pendulum"`` to preserve the legacy default.
    """

    def __init__(
        self,
        in_dim,
        hidden_dims=16,
        activation="leakyrelu",
        adjacency_p: float = 2.0,
        mask=None,
        dataset_name: str = "pendulum",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.activation = get_activation(activation)
        self.adjacency_p = adjacency_p
        self.dataset_name = dataset_name

        (
            self._head_type,
            self._pre_transform,
            self._intervention_decoder,
            self._ood_hook,
        ) = _resolve_strategy(dataset_name)

        if mask is not None:
            mask = (mask.astype(bool) & (1 - np.eye(self.in_dim)).astype(bool)).astype(int)
        else:
            mask = 1 - np.eye(self.in_dim)

        self.dispatcher_layer = DispatcherLayer(
            self.in_dim,
            self.in_dim,
            hidden_dims,
            adjacency_p=self.adjacency_p,
            mask=mask,
        )

        self.identity = torch.eye(self.in_dim)
        self.nonlinearities = nn.ModuleDict()
        for i in range(self.in_dim):
            self.nonlinearities[str(i)] = MLP(
                latent_dim=self.hidden_dims * self.in_dim,
                num_var=self.in_dim,
                use_bias=False,
            )

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Shared utilities (byte-identical across legacy files)
    # ------------------------------------------------------------------

    def update_mask(self, A):
        if isinstance(A, torch.Tensor):
            eye = torch.eye(self.in_dim, device=A.device)
            mask = (A.bool() & (~eye.bool())).int()
            self.mask = mask.to(self.device)
            self.dispatcher_layer.mask = self.mask
        else:
            mask = (A.astype(bool) & (1 - np.eye(self.in_dim)).astype(bool)).astype(int)
            self.mask = torch.tensor(mask).to(self.device)
            self.dispatcher_layer.mask = self.mask

    def get_adjacency_matrix(self):
        return self.dispatcher_layer.get_adjacency_matrix()

    @torch.no_grad()
    def reset_parameters(self):
        self.dispatcher_layer.reset_parameters()

    def get_result_variable_indices(self, A):
        reason_variable_indices = []
        result_variable_indices = []
        for i in range(A.size(1)):
            col = A[:, i]
            if torch.all(col == 0):
                reason_variable_indices.append(i)
            if torch.any(col != 0):
                result_variable_indices.append(i)
        return reason_variable_indices, result_variable_indices

    @property
    def device(self):
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    # Forward / pretrain / inference
    # ------------------------------------------------------------------

    def forward(self, label):
        device = self.device
        mu = label.to(device)
        loss = 0
        output = mu.unsqueeze(2)
        return output, loss

    def _scm_rollout(self, mu, apply_sigmoid: bool):
        """Run the dispatcher + per-variable MLPs over ``mu``.

        Returns ``(z_post, result_v_indices)``.
        ``z_post[:, i]`` is the MLP output (optionally passed through sigmoid)
        for child variables ``i in result_v_indices`` and ``mu[:, i]`` otherwise.
        """
        device = mu.device
        z_pre = self.dispatcher_layer(mu)
        z_post = torch.zeros_like(mu).to(device)
        _, result_v_indices = self.get_result_variable_indices(self.mask)

        for i in range(self.in_dim):
            z_i = z_pre[:, i, :]
            z_i_out = self.nonlinearities[str(i)](z_i).squeeze(-1)
            if len(result_v_indices) > 0 and i in result_v_indices:
                if apply_sigmoid:
                    z_post[:, i] = torch.sigmoid(z_i_out)
                else:
                    z_post[:, i] = z_i_out
            else:
                z_post[:, i] = mu[:, i]

        return z_post, result_v_indices

    def pretrain(self, label):
        device = self.device
        mu = label.to(device)

        # ADNI: collapse 16-dim raw attributes down to 6-dim SCM space.
        if self._pre_transform is not None and mu.shape[1] == 16:
            mu = self._pre_transform(mu)

        is_binary = self._head_type == "binary"
        z_post, result_v_indices = self._scm_rollout(mu, apply_sigmoid=is_binary)

        if len(result_v_indices) > 0:
            if is_binary:
                loss = F.binary_cross_entropy(
                    z_post[:, result_v_indices], mu[:, result_v_indices], reduction="none"
                )
            else:
                loss = F.mse_loss(
                    z_post[:, result_v_indices], mu[:, result_v_indices], reduction="none"
                )
        else:
            loss = torch.tensor(0.0, device=mu.device)

        if is_binary:
            preds = (z_post >= 0.5).float()
            output = preds.unsqueeze(2)
        else:
            output = z_post.unsqueeze(2)
        return output, loss

    def inference(
        self,
        label,
        sample=False,
        intervention_indx=None,
        intervention_values=None,
        disentangle=False,
    ):
        device = self.device
        mu = label.to(device).clone()

        if intervention_indx is None or intervention_values is None:
            # Reconstruction pass-through.
            return mu.unsqueeze(2), None

        if disentangle:
            mu[:, intervention_indx] = intervention_values
            return mu.unsqueeze(2), None

        # ADNI: collapse 16-dim raw attributes down to 6-dim SCM space.
        original_was_wide = self._pre_transform is not None and mu.shape[1] == 16
        if original_was_wide:
            mu = self._pre_transform(mu)

        # Decode intervention value (ADNI uses binary/ordinal encodings for
        # specific indices; other datasets pass the value through as-is).
        decoder = self._intervention_decoder.get(intervention_indx)
        if decoder is not None:
            intervention_values = decoder(intervention_values)

        reason_v_indices, result_v_indices = self.get_result_variable_indices(self.mask)

        # When intervening on a root variable, overwrite mu BEFORE the SCM
        # rollout so the change propagates to its children.
        if self._pre_transform is None:
            # Pendulum / celeA / MorphoMNIST: only override for root vars.
            if len(reason_v_indices) > 0 and intervention_indx in reason_v_indices:
                mu[:, intervention_indx] = intervention_values
        else:
            # ADNI historically overwrites unconditionally before the rollout.
            mu[:, intervention_indx] = intervention_values

        is_binary = self._head_type == "binary"
        z_post, result_v_indices = self._scm_rollout(mu, apply_sigmoid=is_binary)
        z = z_post

        # Post-rollout override: re-apply the intervention on result vars.
        if self._pre_transform is None:
            if len(result_v_indices) > 0 and intervention_indx in result_v_indices:
                z = label.to(device)
                z[:, intervention_indx] = intervention_values
        else:
            # ADNI: indices 0/1/2/3/6 are scalar overrides on the SCM output;
            # indices 4/5 require rebuilding from the original label.
            if intervention_indx in (0, 1, 2, 3, 6):
                z[:, intervention_indx] = intervention_values
            elif intervention_indx in (4, 5):
                z = self._pre_transform(label.to(device))
                z[:, intervention_indx] = intervention_values

        if is_binary:
            preds = (z > 0.5).float()
            # celeA OOD rule.
            if (
                self._ood_hook is not None
                and len(result_v_indices) > 0
                and intervention_indx in result_v_indices
            ):
                preds = self._ood_hook(preds, label, intervention_indx)
            output = preds.unsqueeze(2)
        elif original_was_wide:
            # ADNI: expand back to 16 dims for the controlnet.
            output = self._pre_transform(z, reverse=True).unsqueeze(2)
        else:
            output = z.unsqueeze(2)

        return output, None
