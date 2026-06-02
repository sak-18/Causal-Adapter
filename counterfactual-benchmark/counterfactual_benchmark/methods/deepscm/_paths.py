"""Shared path bootstrap for the Causal-Adapter Stable-Diffusion evaluation scripts.

Historically every ``evaluate_SD*`` script started with a block such as::

    REPO_ROOT = Path(__file__).resolve().parents[4]
    sys.path.append("../../")                                   # relies on CWD
    sys.path.append(str(REPO_ROOT / "causal-adapter-sd15"))     # dead path

The ``causal-adapter-sd15`` directory no longer exists: ``causal_modules`` now
lives at the Causal-Adapter project root.  The ``../../`` append also silently
depends on the current working directory, so the scripts only imported
correctly when launched from the ``deepscm`` folder.

This module centralises path resolution in one place.  Import it and call
:func:`bootstrap` *before* importing any first-party package (``causal_modules``,
``models``, ``ctf_datasets``, ``evaluation``, ``model``)::

    from _paths import bootstrap, REPO_ROOT, DEEPSCM_CONFIG_DIR
    bootstrap()

Resolution is anchored to this file's location, so the scripts run identically
whether invoked from the Causal-Adapter project root or from the ``deepscm``
directory.
"""
import sys
from pathlib import Path

# This file lives at: <REPO_ROOT>/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/_paths.py
DEEPSCM_DIR = Path(__file__).resolve().parent
CB_PACKAGE_DIR = DEEPSCM_DIR.parents[1]     # .../counterfactual_benchmark  (top-level package)
REPO_ROOT = DEEPSCM_DIR.parents[3]          # .../Causal-Adapter            (holds causal_modules/)
DEEPSCM_CONFIG_DIR = DEEPSCM_DIR / "configs"


def bootstrap() -> None:
    """Make the first-party packages importable regardless of the launch CWD.

    Adds, in priority order:
      * ``REPO_ROOT``     -> enables ``import causal_modules``
      * ``CB_PACKAGE_DIR``-> enables ``import models / ctf_datasets / evaluation``
      * ``DEEPSCM_DIR``   -> enables ``import model`` (the local SCM definition)
    """
    for path in (REPO_ROOT, CB_PACKAGE_DIR, DEEPSCM_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
