"""Chronological backtesting: replay a manager's historical decisions one
gameweek at a time, predicting each from only the gameweeks before it.

Critical rule (Stage 4 brief item 14): when predicting GWn, the model may
only use information available through GWn-1. This module enforces that
by construction - it calls ``prediction_engine.predict_captain`` exactly
the way live usage would, passing the FULL ``LeagueHistory`` and letting
its own ``.before(target_gameweek)`` boundary do the filtering (see
prediction_engine.py's docstring). The backtester itself never manually
slices history, so there's only one leakage boundary in the whole system
to get right and test.

Assumption, stated plainly: this backtester assumes gameweek N's SQUAD
(who the rival owned) is already known when predicting gameweek N's
captain - that's an input field of the historical dataset (Stage 4 brief
item 13), not something the model predicts. It tests the captaincy
*scoring* model, not squad-visibility timing.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .models import LeagueHistory, PredictionConfig
from .prediction_engine import predict_captain


def run_backtest(
    rival_entry_id: int,
    rival_name: str,
    league_history: LeagueHistory,
    config: PredictionConfig = PredictionConfig(),
    global_consensus_by_gw: Optional[Dict[int, Dict[int, float]]] = None,
    expected_points_by_gw: Optional[Dict[int, Dict[int, float]]] = None,
) -> List[dict]:
    """Returns one result dict per observed gameweek, each containing the
    ``CaptainPrediction`` made using only prior gameweeks, and the actual
    captain revealed afterwards - ready for evaluation.py/calibration.py."""
    observations = sorted(league_history.for_entry(rival_entry_id), key=lambda o: o.gameweek)

    results = []
    for observation in observations:
        gw = observation.gameweek
        prediction = predict_captain(
            rival_entry_id=rival_entry_id,
            rival_name=rival_name,
            current_squad=observation.owned_player_ids,
            target_gameweek=gw,
            league_history=league_history,  # full history is safe - predict_captain slices it itself
            config=config,
            global_consensus=(global_consensus_by_gw or {}).get(gw),
            expected_points=(expected_points_by_gw or {}).get(gw),
        )
        results.append(
            {
                "gameweek": gw,
                "prediction": prediction,
                "candidate_probabilities": dict(prediction.probabilities),
                "actual_captain_id": observation.captain_id,
            }
        )
    return results
