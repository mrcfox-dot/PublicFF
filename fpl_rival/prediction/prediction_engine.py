"""Orchestrates candidate generation, feature computation and scoring into
one ``CaptainPrediction``.

THE LEAKAGE BOUNDARY: this module calls ``league_history.before(target_gameweek)``
exactly once, at the top of ``predict_captain``, before any feature is
computed. Every downstream function (features.py, captain_model.py) only
ever sees that already-safe slice. This means even a caller who
accidentally passes a ``LeagueHistory`` containing gameweeks at or after
``target_gameweek`` (e.g. a backtest driver with an off-by-one) cannot leak
the future into a prediction - see tests/test_prediction_engine.py's
defensive leakage test.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from . import captain_model, features
from .exceptions import EmptyCandidateSetError
from .models import CandidateFeatures, CaptainPrediction, LeagueHistory, PredictionConfig


def predict_captain(
    rival_entry_id: int,
    rival_name: str,
    current_squad: Tuple[int, ...],
    target_gameweek: int,
    league_history: LeagueHistory,
    config: PredictionConfig = PredictionConfig(),
    global_consensus: Optional[Dict[int, float]] = None,
    expected_points: Optional[Dict[int, float]] = None,
) -> CaptainPrediction:
    """Predicts P(rival captains X) for every X in ``current_squad``.

    Candidates are restricted to ``current_squad`` by construction - a
    player the rival does not currently own is never scored, let alone
    proposed as a candidate (Stage 4 brief item 2).
    """
    if not current_squad:
        raise EmptyCandidateSetError(f"{rival_name} (entry {rival_entry_id}): current_squad is empty.")

    safe_history = league_history.before(target_gameweek)
    rival_history = safe_history.for_entry(rival_entry_id)
    other_history = safe_history.other_entries(rival_entry_id)

    candidates = list(dict.fromkeys(current_squad))  # de-duplicate, preserve order

    raw_features: Dict[int, dict] = {}
    for cid in candidates:
        times_owned, times_captained, loyalty_rate = features.personal_loyalty_rate(rival_history, cid)
        recency_rate = features.recency_weighted_rate(rival_history, cid, target_gameweek, config.recency_decay_rate)
        _, conc_share = features.concentration_share(rival_history, cid)
        consensus_rate = features.league_consensus_rate(other_history, cid)
        recent_captained, recent_window = features.recent_window_stats(rival_history, cid, config.recent_window_size)
        raw_features[cid] = {
            "times_owned": times_owned,
            "times_captained": times_captained,
            "personal_loyalty_rate": loyalty_rate,
            "recency_weighted_rate": recency_rate,
            "concentration_share": conc_share,
            "league_consensus_rate": consensus_rate,
            "recent_window_captained": recent_captained,
            "recent_window_size": recent_window,
        }

    num_decisions = len(rival_history)
    shrinkage = captain_model.shrinkage_weight(num_decisions, config)

    relevant_global = {cid: global_consensus[cid] for cid in candidates if global_consensus and cid in global_consensus}
    relevant_ep = {cid: expected_points[cid] for cid in candidates if expected_points and cid in expected_points}

    scored = captain_model.score_candidates(raw_features, relevant_global or None, relevant_ep or None, shrinkage, config)
    probabilities = captain_model.softmax_probabilities(
        {cid: scored[cid]["score"] for cid in candidates}, config.softmax_temperature
    )

    candidate_features: Dict[int, CandidateFeatures] = {}
    for cid in candidates:
        f = raw_features[cid]
        candidate_features[cid] = CandidateFeatures(
            player_id=cid,
            times_owned=f["times_owned"],
            times_captained=f["times_captained"],
            personal_loyalty_rate=f["personal_loyalty_rate"],
            recency_weighted_rate=f["recency_weighted_rate"],
            concentration_share=f["concentration_share"],
            league_consensus_rate=f["league_consensus_rate"],
            global_consensus_value=relevant_global.get(cid),
            expected_points_value=relevant_ep.get(cid),
            recent_window_captained=f["recent_window_captained"],
            recent_window_size=f["recent_window_size"],
            contributions=scored[cid]["contributions"],
            raw_score=scored[cid]["score"],
        )

    gameweeks_observed = len({o.gameweek for o in rival_history})
    conc_idx = features.concentration_index(rival_history)
    data_sufficiency = captain_model.classify_data_sufficiency(gameweeks_observed, num_decisions, conc_idx, config)

    return CaptainPrediction(
        rival_entry_id=rival_entry_id,
        rival_name=rival_name,
        target_gameweek=target_gameweek,
        probabilities=probabilities,
        candidate_features=candidate_features,
        data_sufficiency=data_sufficiency,
        config_version=config.version,
    )
