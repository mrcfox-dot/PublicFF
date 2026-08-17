"""The one deliberate seam to Stage 3: turns a Stage 4 ``CaptainPrediction``
into the ``{player_id: probability}`` shape Stage 3's
``ManagerState.captain_probabilities`` already accepts.

This does NOT rewrite Stage 3 - it only reuses Stage 3's own
``rival_behaviour.validate_captain_distribution`` (the exact validation
Stage 3 already runs internally in "probabilistic" rival-captain mode) to
confirm a Stage 4 prediction is immediately usable, before handing it
back. Imports are local/lazy, matching the pattern
``fpl_rival/simulation/relevance_adapter.py`` already uses to reuse Stage 2
code without a hard import-time coupling between the two packages.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict

from .models import CaptainPrediction


def to_captain_probabilities(prediction: CaptainPrediction) -> Dict[int, float]:
    """Extracts and validates the probability distribution for direct use
    as a Stage 3 ``ManagerState.captain_probabilities`` value."""
    from fpl_rival.simulation.rival_behaviour import validate_captain_distribution

    distribution = dict(prediction.probabilities)
    validate_captain_distribution(distribution, prediction.rival_name)
    return distribution


def apply_prediction_to_manager_state(manager_state, prediction: CaptainPrediction):
    """Returns a copy of a Stage 3 ``ManagerState`` with
    ``captain_probabilities`` set from this Stage 4 prediction, and
    ``captain_id`` set to the most likely captain (used only as the
    "fixed mode" fallback - Stage 3's probabilistic mode uses the
    distribution, not this single value). Does not mutate the input.

    Note: the prediction's candidate set should match the manager's
    starting_xi (predict from the rival's starting XI, not their full 15)
    - Stage 3 only ever captains a starting-XI player, and ManagerState
    validates that at construction time.
    """
    distribution = to_captain_probabilities(prediction)
    return replace(
        manager_state,
        captain_id=prediction.most_likely_captain,
        captain_probabilities=distribution,
    )
