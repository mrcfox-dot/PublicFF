"""Exceptions the simulation layer raises deliberately and clearly.

Every one of these is meant to be caught by a caller (CLI or test) and
turned into a clean, human-readable failure - never a bare stack trace
from deep inside NumPy code, and never a silently wrong result.
"""

from __future__ import annotations


class SimulationError(Exception):
    """Base class for all Stage 3 simulation errors."""


class InvalidCaptainError(SimulationError):
    """Raised when a candidate captain is not eligible (e.g. not in the XI)."""


class MissingProjectionError(SimulationError):
    """Raised when a player referenced by a squad has no projection supplied."""


class InvalidCaptainDistributionError(SimulationError):
    """Raised when a rival captain probability distribution is malformed."""


class InvalidManagerStateError(SimulationError):
    """Raised when a ManagerState is structurally invalid (e.g. empty XI)."""
