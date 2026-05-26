"""Shared, cross-module primitives.

Currently exposes:
  - ``BIOMETRIC_COLUMNS``: canonical PHI column set so every module
    redacts the same biometrics.
  - ``dumps_signed`` / ``loads_signed`` / ``SignedPickleError``: ECDSA-
    signed pickle I/O so the model artefacts that Module 3/4 load at
    inference time are tamper-evident under the Module 5 audit key.

Add new shared definitions here only when at least two modules need
them — keep this package small on purpose.
"""

from . import split_paths
from .phi import BIOMETRIC_COLUMNS
from .signed_pickle import SignedPickleError, dumps_signed, loads_signed
from .split_paths import Split

__all__ = [
    "BIOMETRIC_COLUMNS",
    "dumps_signed",
    "loads_signed",
    "SignedPickleError",
    "Split",
    "split_paths",
]
