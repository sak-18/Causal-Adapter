#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Causal discovery on ADNI brain-imaging attributes.
#
# Loads the 16-column attribute table (apoE one-hot + age + sex + brain_vol +
# vent_vol + 10-class slice one-hot) and runs the requested discovery method.
#
# Dataset layout expected:
#   <DATA_ROOT>/preprocessed_data/...
#   <DATA_ROOT>/ADNIMERGE*.csv
#
# NOTE: ADNI gt adjacency is provided at the group level (6x6). Structural
# metrics are therefore only reported when the learned matrix happens to also
# be 6x6 (it usually is not -- the learner operates on the expanded 16x16
# feature graph). The matrices are still saved for downstream use.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"

# Default ADNI root: sibling counterfactual-benchmark project.
DATA_ROOT="${ADNI_DATA_ROOT:-${REPO_ROOT}/../counterfactual-benchmark/counterfactual_benchmark/ctf_datasets/adni/preprocessing}"

METHOD="${1:-sdcd}"
shift || true

"${PYTHON}" "${REPO_ROOT}/SCM_modeling/discover_causal.py" \
    --dataset ADNI \
    --method-name "${METHOD}" \
    --device cpu \
    --data-root "${DATA_ROOT}" \
    "$@"
