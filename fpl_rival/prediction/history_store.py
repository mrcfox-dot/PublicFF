"""Local JSON(L) storage for predictions and their eventual actual outcomes.

No database - a plain append-only JSON-Lines file per season, exactly like
Stage 1/2's plain-JSON convention. Each line is one prediction record.
``record_actual_outcome`` later fills in the actual captain once the
gameweek deadline has passed, turning the record into a
prediction -> outcome pair usable by evaluation.py / calibration.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from .models import CaptainPrediction

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "predictions"


def default_path(season: str) -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return DATA_ROOT / f"{season}.jsonl"


def prediction_to_record(prediction: CaptainPrediction, season: str) -> dict:
    return {
        "season": season,
        "gameweek": prediction.target_gameweek,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rival_entry_id": prediction.rival_entry_id,
        "rival_team_name": prediction.rival_name,
        "candidate_probabilities": {str(pid): p for pid, p in prediction.probabilities.items()},
        "features": {
            str(pid): {
                "times_owned": f.times_owned,
                "times_captained": f.times_captained,
                "personal_loyalty_rate": f.personal_loyalty_rate,
                "recency_weighted_rate": f.recency_weighted_rate,
                "concentration_share": f.concentration_share,
                "league_consensus_rate": f.league_consensus_rate,
                "global_consensus_value": f.global_consensus_value,
                "expected_points_value": f.expected_points_value,
                "contributions": f.contributions,
                "raw_score": f.raw_score,
            }
            for pid, f in prediction.candidate_features.items()
        },
        "data_sufficiency": {
            "gameweeks_observed": prediction.data_sufficiency.gameweeks_observed,
            "captain_decisions_observed": prediction.data_sufficiency.captain_decisions_observed,
            "shrinkage_weight": prediction.data_sufficiency.shrinkage_weight,
            "data_state": prediction.data_sufficiency.data_state,
            "low_personal_data": prediction.data_sufficiency.low_personal_data,
            "confidence_label": prediction.data_sufficiency.confidence_label,
            "concentration_index": prediction.data_sufficiency.concentration_index,
        },
        "config_version": prediction.config_version,
        "actual_captain_id": None,
        "outcome_recorded_at": None,
    }


def save_prediction(record: dict, path: Path) -> None:
    """Appends one prediction record as a new JSON line. Never overwrites
    an existing prediction - each call to predict_captain -> save_prediction
    is a new immutable log entry, even if called twice for the same
    rival/gameweek (the caller may deliberately want a re-run recorded)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def load_predictions(
    path: Path,
    season: Optional[str] = None,
    gameweek: Optional[int] = None,
    rival_entry_id: Optional[int] = None,
) -> List[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if season is not None:
        records = [r for r in records if r["season"] == season]
    if gameweek is not None:
        records = [r for r in records if r["gameweek"] == gameweek]
    if rival_entry_id is not None:
        records = [r for r in records if r["rival_entry_id"] == rival_entry_id]
    return records


def record_actual_outcome(
    path: Path, season: str, gameweek: int, rival_entry_id: int, actual_captain_id: int
) -> int:
    """Fills in ``actual_captain_id`` on every matching, not-yet-resolved
    prediction record. Returns how many records were updated (0 if none
    matched - callers should treat that as worth investigating, not
    silently ignore it)."""
    if not path.exists():
        return 0
    records = load_predictions(path)
    updated = 0
    for record in records:
        if (
            record["season"] == season
            and record["gameweek"] == gameweek
            and record["rival_entry_id"] == rival_entry_id
            and record["actual_captain_id"] is None
        ):
            record["actual_captain_id"] = actual_captain_id
            record["outcome_recorded_at"] = datetime.now(timezone.utc).isoformat()
            updated += 1
    if updated:
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, default=str) + "\n")
    return updated


def prediction_outcome_pairs(records: Iterable[dict]) -> List[dict]:
    """Filters to records that have a recorded actual outcome, in the
    {candidate_probabilities, actual_captain_id} shape evaluation.py and
    calibration.py expect (probability keys converted back to int)."""
    pairs = []
    for r in records:
        if r.get("actual_captain_id") is None:
            continue
        pairs.append(
            {
                "candidate_probabilities": {int(k): v for k, v in r["candidate_probabilities"].items()},
                "actual_captain_id": r["actual_captain_id"],
            }
        )
    return pairs
