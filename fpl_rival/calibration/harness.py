"""Runs either the Stage 4 model or a baseline chronologically over a whole
dataset, with no future leakage, and tags every prediction with its
train/validation/test phase.

Reuses Stage 4's ``LeagueHistory`` / ``predict_captain`` completely
unmodified - this module only groups records by season (gameweek numbers
reset each season, so a season boundary is also a personal-history
boundary here: history does not carry over between seasons, a deliberate,
documented simplification - see the Stage 4.5 completion report) and
loops chronologically. The leakage boundary itself (``LeagueHistory.before``)
is Stage 4's, used here for baselines exactly the same way
``predict_captain`` uses it internally for the model - one boundary,
tested once, trusted everywhere.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from fpl_rival.prediction.models import LeagueHistory, PredictionConfig
from fpl_rival.prediction.prediction_engine import predict_captain

from .baselines import BASELINE_REGISTRY
from .importer import records_by_manager_and_season
from .models import DatasetSplit, HistoricalRecord

ExpectedPointsLookup = Dict[Tuple[str, int], Dict[int, float]]


def _season_league_history(season_records: List[HistoricalRecord]) -> LeagueHistory:
    by_manager = defaultdict(list)
    for r in season_records:
        by_manager[r.manager_id].append(r.to_captain_observation())
    return LeagueHistory({mid: tuple(sorted(obs, key=lambda o: o.gameweek)) for mid, obs in by_manager.items()})


def _iter_seasons(records: List[HistoricalRecord], max_managers: Optional[int]):
    seasons = sorted({r.season for r in records})
    for season in seasons:
        season_records = [r for r in records if r.season == season]
        league_history = _season_league_history(season_records)
        by_manager = records_by_manager_and_season(season_records)
        manager_ids = sorted({r.manager_id for r in season_records})
        if max_managers:
            manager_ids = manager_ids[:max_managers]
        for manager_id in manager_ids:
            for record in by_manager.get((season, manager_id), []):
                yield season, manager_id, record, league_history


def run_stage4_over_dataset(
    records: List[HistoricalRecord],
    config: PredictionConfig,
    split: Optional[DatasetSplit] = None,
    expected_points: Optional[ExpectedPointsLookup] = None,
    max_managers: Optional[int] = None,
) -> List[dict]:
    """One result dict per (manager, gameweek), each holding the full
    Stage 4 ``CaptainPrediction`` (so ablation/history-depth/calibration
    correction can all reuse the same run)."""
    results = []
    for season, manager_id, record, league_history in _iter_seasons(records, max_managers):
        ep = (expected_points or {}).get((season, record.gameweek))
        prediction = predict_captain(
            rival_entry_id=manager_id,
            rival_name=record.manager_name or f"Manager {manager_id}",
            current_squad=record.starting_xi,
            target_gameweek=record.gameweek,
            league_history=league_history,
            config=config,
            expected_points=ep,
        )
        phase = split.phase_of(season, record.gameweek) if split else None
        results.append(
            {
                "season": season,
                "gameweek": record.gameweek,
                "manager_id": manager_id,
                "phase": phase,
                "candidate_probabilities": dict(prediction.probabilities),
                "actual_captain_id": record.captain,
                "gameweeks_observed": prediction.data_sufficiency.gameweeks_observed,
                "prediction": prediction,
            }
        )
    return results


def run_baseline_over_dataset(
    records: List[HistoricalRecord],
    baseline_name: str,
    split: Optional[DatasetSplit] = None,
    expected_points: Optional[ExpectedPointsLookup] = None,
    max_managers: Optional[int] = None,
) -> List[dict]:
    if baseline_name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline {baseline_name!r}; choose from {sorted(BASELINE_REGISTRY)}.")
    _, predictor_fn, requires_other, requires_ep = BASELINE_REGISTRY[baseline_name]

    results = []
    for season, manager_id, record, league_history in _iter_seasons(records, max_managers):
        safe_history = league_history.before(record.gameweek)
        rival_history = safe_history.for_entry(manager_id)
        candidates = record.starting_xi

        if baseline_name == "consensus":
            probabilities = predictor_fn(candidates, safe_history.other_entries(manager_id))
        elif baseline_name == "expected_points":
            ep = (expected_points or {}).get((season, record.gameweek))
            probabilities = predictor_fn(candidates, ep)
        elif baseline_name == "uniform":
            probabilities = predictor_fn(candidates)
        else:
            probabilities = predictor_fn(candidates, rival_history)

        if probabilities is None:
            continue  # this baseline legitimately has nothing to say here - skipped, not faked

        phase = split.phase_of(season, record.gameweek) if split else None
        results.append(
            {
                "season": season,
                "gameweek": record.gameweek,
                "manager_id": manager_id,
                "phase": phase,
                "candidate_probabilities": probabilities,
                "actual_captain_id": record.captain,
                "gameweeks_observed": len(rival_history),
            }
        )
    return results


def filter_phase(results: List[dict], phase: str) -> List[dict]:
    return [r for r in results if r["phase"] == phase]


def to_eval_pairs(results: List[dict]) -> List[dict]:
    return [{"candidate_probabilities": r["candidate_probabilities"], "actual_captain_id": r["actual_captain_id"]} for r in results]
