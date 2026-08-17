"""Exceptions the prediction layer raises deliberately and clearly."""

from __future__ import annotations


class PredictionError(Exception):
    """Base class for all Stage 4 prediction errors."""


class EmptyCandidateSetError(PredictionError):
    """Raised when a rival's current squad (the candidate pool) is empty."""


class InvalidExternalInputError(PredictionError):
    """Raised when an optional external input (global consensus, expected
    points) is malformed."""
