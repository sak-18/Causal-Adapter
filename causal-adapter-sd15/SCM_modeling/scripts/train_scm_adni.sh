#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Train an SCM CausalNet on the ADNI attribute table.
#
# Loads the 16-column ADNI feature table and trains an SDCD CausalNet seeded
# with the project's default ADNI adjacency (or a custom CSV via
# ``--adj-matrix-path``). Outputs land under
# ``SCM_modeling/saved_mtx/ADNI_minmax/``.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"

DATA_ROOT="${ADNI_DATA_ROOT:-${REPO_ROOT}/../counterfactual-benchmark/counterfactual_benchmark/ctf_datasets/adni/preprocessing}"

"${PYTHON}" "${REPO_ROOT}/SCM_modeling/train_scm.py" \
    --dataset ADNI \
    --device cpu \
    --data-root "${DATA_ROOT}" \
    "$@"
