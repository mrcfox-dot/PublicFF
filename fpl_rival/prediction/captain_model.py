"""The interpretable scoring model: features -> score -> probability.

    captain_score(candidate) =
          shrinkage_weight * [ w_personal * personal_loyalty_rate
                              + w_recency  * recency_weighted_rate
                              + w_concentration * concentration_share ]
        +                     [ w_consensus * league_consensus_rate
                              + w_global_consensus * global_consensus_value   (if supplied)
                              + w_expected_points   * normalized_expected_points (if supplied) ]

    probabilities = softmax(scores / temperature)

The personal-behaviour components (loyalty, recency, concentration) are
scaled by ``shrinkage_weight`` - 0 with zero personal history, rising
towards 1 as decisions accumulate (see ``PredictionConfig.personal_shrinkage_k``
and models.py's docstring). The prior components (consensus, global
consensus, expected points) always contribute at their full configured
weight - they remain legitimately relevant information regardless of how
much personal history exists. This is what makes "predictions fall back
toward the prior with zero personal history" true, and what makes the
manager-specific signal "progressively matter more" as history
accumulates, WITHOUT any hard gameweek-based switch.

All weights and thresholds live in ``PredictionConfig`` - nothing here is
an unlabelled magic number.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from .models import (
    BEHAVIOUR_DRIVEN,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MODERATE,
    DataSufficiency,
    MIXED,
    PRIOR_DRIVEN,
    PredictionConfig,
)


def shrinkage_weight(num_personal_decisions: int, config: PredictionConfig) -> float:
    """n / (n + k) - 0 at n=0, 0.5 at n=k, -> 1 as n grows. Gradual by
    construction; there is no gameweek at which this "switches on"."""
    n = num_personal_decisions
    k = config.personal_shrinkage_k
    return n / (n + k) if (n + k) > 0 else 0.0


def _minmax_normalize(values: Dict[int, float]) -> Dict[int, float]:
    """Rescales a set of candidate values to [0, 1] (min -> 0, max -> 1) so
    e.g. expected-points figures in "points" units land on the same scale
    as the rate-based features (personal loyalty, consensus, ... all
    naturally in [0, 1]) instead of z-scores, whose magnitude is
    unbounded and can swamp a well-established personal-history signal
    for a small candidate set. Falls back to a flat 0.5 for every
    candidate (no relative signal - deliberately not 0, since "equal EP"
    is not evidence against any candidate) if every candidate has the
    same value."""
    if not values:
        return {}
    vals = list(values.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 0.5 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def score_candidates(
    raw_features: Dict[int, dict],
    global_consensus: Optional[Dict[int, float]],
    expected_points: Optional[Dict[int, float]],
    shrinkage: float,
    config: PredictionConfig,
) -> Dict[int, dict]:
    """Turns raw per-candidate feature dicts (as produced by
    prediction_engine.py from features.py's functions) into a score plus a
    labelled contributions breakdown, for every candidate.

    ``raw_features[candidate_id]`` must contain: personal_loyalty_rate,
    recency_weighted_rate, concentration_share.
    """
    candidate_ids = list(raw_features.keys())

    normalized_global = _minmax_normalize(global_consensus) if global_consensus else {}
    normalized_ep = _minmax_normalize(expected_points) if expected_points else {}

    results: Dict[int, dict] = {}
    for cid in candidate_ids:
        f = raw_features[cid]
        contributions = {
            "personal_history": shrinkage * config.weight_personal_loyalty * f["personal_loyalty_rate"],
            "recent_captaincy": shrinkage * config.weight_recency * f["recency_weighted_rate"],
            "concentration": shrinkage * config.weight_concentration * f["concentration_share"],
            "league_consensus": config.weight_consensus * f["league_consensus_rate"],
        }
        if cid in normalized_global:
            contributions["global_consensus"] = config.weight_global_consensus * normalized_global[cid]
        if cid in normalized_ep:
            contributions["expected_points"] = config.weight_expected_points * normalized_ep[cid]

        score = sum(contributions.values())
        results[cid] = {"score": score, "contributions": contributions}

    return results


def softmax_probabilities(scores: Dict[int, float], temperature: float) -> Dict[int, float]:
    if not scores:
        return {}
    temperature = max(temperature, 1e-6)
    values = {k: v / temperature for k, v in scores.items()}
    max_val = max(values.values())  # numerical stability
    exp_values = {k: math.exp(v - max_val) for k, v in values.items()}
    total = sum(exp_values.values())
    return {k: v / total for k, v in exp_values.items()}


def classify_data_sufficiency(
    gameweeks_observed: int,
    captain_decisions_observed: int,
    concentration_idx: float,
    config: PredictionConfig,
) -> DataSufficiency:
    weight = shrinkage_weight(captain_decisions_observed, config)

    if weight <= config.prior_driven_max_shrinkage:
        data_state = PRIOR_DRIVEN
    elif weight >= config.behaviour_driven_min_shrinkage:
        data_state = BEHAVIOUR_DRIVEN
    else:
        data_state = MIXED

    low_personal_data = gameweeks_observed < config.low_personal_data_gw_threshold

    if data_state == PRIOR_DRIVEN or low_personal_data:
        confidence = CONFIDENCE_LOW
    elif data_state == BEHAVIOUR_DRIVEN and gameweeks_observed >= config.high_confidence_min_gw:
        confidence = CONFIDENCE_HIGH
    else:
        confidence = CONFIDENCE_MODERATE

    return DataSufficiency(
        gameweeks_observed=gameweeks_observed,
        captain_decisions_observed=captain_decisions_observed,
        shrinkage_weight=weight,
        data_state=data_state,
        low_personal_data=low_personal_data,
        confidence_label=confidence,
        concentration_index=concentration_idx,
    )
