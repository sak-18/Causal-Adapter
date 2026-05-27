"""Vendored subset of SDCD (https://github.com/azizilab/sdcd).

Heavy/optional submodules are loaded on demand via direct imports such as
``from sdcd.models._sdcd import SDCD``. The package __init__ stays lightweight
so projects depending only on the SCM/SDCD core do not pull every backend.
"""

__all__ = []
