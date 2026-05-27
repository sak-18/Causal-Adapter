#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Causal discovery on the pendulum dataset (4 attributes).
#
# Runs the requested method (SDCD by default) on the observational pendulum
# attribute table and writes the learned adjacency matrix under
# ``SCM_modeling/saved_mtx/pendulum_causaldata2/``.
#
# Dataset layout expected:
#   <DATA_ROOT>/pendulum/{train,test}/*.png   (labels parsed from filenames)
#   OR  <DATA_ROOT>/{train,test}/*.png        (already pointed at pendulum/)
#
# Override the method with the first positional argument, e.g.
#   bash discover_pendulum.sh dagma
#   bash discover_pendulum.sh notears --quick
# ---------------------------------------------------------------------------
set -euo pipefail

# Resolve repository root regardless of the caller's working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Allow ``PYTHON=/path/to/python`` to pin a specific interpreter (e.g. the
# ``mcpl`` conda env shipped with this workspace).
PYTHON="${PYTHON:-python}"

# Default pendulum data root: sibling MCPL-diffuser project.
DATA_ROOT="${PENDULUM_DATA_ROOT:-${REPO_ROOT}/../causaledit/MCPL-diffuser/dataset/causal_data2}"

# First positional arg = method name (sdcd / dagma / notears); rest forwarded.
METHOD="${1:-sdcd}"
shift || true

"${PYTHON}" "${REPO_ROOT}/SCM_modeling/discover_causal.py" \
    --dataset pendulum \
    --method-name "${METHOD}" \
    --device cpu \
    --data-root "${DATA_ROOT}" \
    "$@"
