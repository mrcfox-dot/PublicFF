"""Simple benchmark captain predictors - what Stage 4's model has to beat.

Every baseline returns either a valid probability distribution over
``candidates`` (summing to 1), or ``None`` when the signal it needs isn't
legitimately available for that prediction - never a fabricated number.
``harness.py`` treats a ``None`` result as "this baseline is not evaluated
for this prediction" and reports coverage (how many predictions it could
actually make) alongside its accuracy metrics.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from fpl_rival.prediction.models import CaptainObservation


def _uniform(candidates: Tuple[int, ...]) -> Dict[int, float]:
    n = len(candidates)
    return {cid: 1.0 / n for cid in candidates}


def _one_hot(candidates: Tuple[int, ...], winner: int) -> Dict[int, float]:
    return {cid: (1.0 if cid == winner else 0.0) for cid in candidates}


def predict_last_captain(
    candidates: Tuple[int, ...], rival_history: Tuple[CaptainObservation, ...]
) -> Dict[int, float]:
    """Baseline A: repeat the previous gameweek's captain, if still owned.
    Falls back to uniform (not a fabricated guess) when there is no prior
    gameweek, or the previous captain isn't in the current candidate set."""
    if not rival_history:
        return _uniform(candidates)
    last = max(rival_history, key=lambda o: o.gameweek)
    if last.captain_id in candidates:
        return _one_hot(candidates, last.captain_id)
    return _uniform(candidates)


def predict_personal_frequency(
    candidates: Tuple[int, ...], rival_history: Tuple[CaptainObservation, ...]
) -> Dict[int, float]:
    """Baseline B: captain probability proportional to how often this
    manager has captained each CURRENTLY OWNED candidate, historically."""
    if not rival_history:
        return _uniform(candidates)
    counts = {cid: sum(1 for o in rival_history if o.captain_id == cid) for cid in candidates}
    total = sum(counts.values())
    if total == 0:
        return _uniform(candidates)
    return {cid: c / total for cid, c in counts.items()}


def predict_consensus(
    candidates: Tuple[int, ...], other_managers_history: Dict[int, Tuple[CaptainObservation, ...]]
) -> Optional[Dict[int, float]]:
    """Baseline C: the mini-league's historical consensus favourite among
    current candidates. Returns None (skip - do not fabricate a consensus
    signal) if there are no other managers in the dataset at all."""
    if not other_managers_history:
        return None
    all_observations = [o for obs in other_managers_history.values() for o in obs]
    if not all_observations:
        return None
    counts = {cid: sum(1 for o in all_observations if o.captain_id == cid) for cid in candidates}
    total = sum(counts.values())
    if total == 0:
        return _uniform(candidates)
    return {cid: c / total for cid, c in counts.items()}


def predict_expected_points_favourite(
    candidates: Tuple[int, ...], expected_points: Optional[Dict[int, float]]
) -> Optional[Dict[int, float]]:
    """Baseline D: 100% on the highest-projected owned candidate. Returns
    None (skip - do not fabricate a projection) if no expected-points
    input is supplied for this gameweek."""
    if not expected_points:
        return None
    owned_ep = {cid: expected_points[cid] for cid in candidates if cid in expected_points}
    if not owned_ep:
        return None
    return _one_hot(candidates, max(owned_ep, key=owned_ep.get))


def predict_uniform(candidates: Tuple[int, ...]) -> Dict[int, float]:
    """Baseline E: equal probability across every candidate - a
    deliberately weak calibration floor."""
    return _uniform(candidates)


# name -> (display label, callable, requires_other_managers, requires_expected_points)
BASELINE_REGISTRY = {
    "last_captain": ("Last Captain", predict_last_captain, False, False),
    "personal_frequency": ("Personal Frequency", predict_personal_frequency, False, False),
    "consensus": ("Consensus", predict_consensus, True, False),
    "expected_points": ("Expected-Points Favourite", predict_expected_points_favourite, False, True),
    "uniform": ("Uniform", predict_uniform, False, False),
}
