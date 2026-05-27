#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Train an SCM CausalNet on the pendulum attribute table.
#
# Uses the ground-truth 4x4 pendulum adjacency as the input mask for SDCD's
# CausalNet. The learned weighted/threshold adjacencies are written to
# ``SCM_modeling/saved_mtx/pendulum_minmax/``.
#
# Forward extra flags via "$@", e.g. ``--quick`` for smoke tests or
# ``--device cuda:0`` for GPU training.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"

DATA_ROOT="${PENDULUM_DATA_ROOT:-${REPO_ROOT}/../causaledit/MCPL-diffuser/dataset/causal_data2}"

"${PYTHON}" "${REPO_ROOT}/SCM_modeling/train_scm.py" \
    --dataset pendulum \
    --device cpu \
    --data-root "${DATA_ROOT}" \
    "$@"
