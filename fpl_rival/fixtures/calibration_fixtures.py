"""Synthetic historical dataset for Stage 4.5 - SYNTHETIC DATA ONLY.

No real historical FPL manager dataset was supplied for this Stage 4.5
build (none exists in this environment, and per the brief this module
never downloads or scrapes one). This generates a synthetic dataset from
Stage 4's own behavioural personas and writes it through the REAL CSV
importer format (``fpl_rival.calibration.importer.import_csv``) - so
running the calibration CLI without ``--dataset`` still exercises the
actual import path, not a shortcut around it. Every result produced from
this data is clearly labelled synthetic; see the Stage 4.5 completion
report for what real data would be needed to answer the underlying
question for real.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

from fpl_rival.fixtures.prediction_fixtures import (
    PLAYER_NAMES,
    differential_manager_history,
    erratic_manager_history,
    expected_points_follower_history,
    loyal_captain_manager_history,
    template_manager_history,
    weekly_expected_points_series,
)
from fpl_rival.fixtures.simulation_fixtures import TEMPLATE_IDS
from fpl_rival.calibration.importer import DELIMITER
from fpl_rival.prediction.models import CaptainObservation

SEASON = "2026-27-SYNTHETIC"
SQUAD_PADDING = tuple(TEMPLATE_IDS[:7])  # pads the 4-player captain pool to a full 11-man XI

CSV_FIELDS = ("season", "manager_id", "manager_name", "gameweek", "starting_xi", "captain", "vice_captain")

PERSONA_BUILDERS = {
    "template": template_manager_history,
    "loyal": loyal_captain_manager_history,
    "differential": differential_manager_history,
    "erratic": erratic_manager_history,
}


def build_synthetic_dataset(num_gameweeks: int = 30, managers_per_persona: int = 2) -> Tuple[List[dict], Dict[Tuple[str, int], Dict[int, float]]]:
    """Returns (rows in native CSV-schema dict form, expected_points lookup).

    Several independently-seeded managers per persona (not just one of
    each) so the dataset has enough managers for a real consensus signal
    and enough sample size for meaningful history-depth buckets.
    """
    rows: List[dict] = []
    manager_id = 1
    ep_series = weekly_expected_points_series(num_gameweeks, seed=999)
    expected_points = {(SEASON, gw): values for gw, values in ep_series.items()}

    for persona_index, (persona_name, builder) in enumerate(PERSONA_BUILDERS.items()):
        for i in range(managers_per_persona):
            # Deterministic seed derivation - NOT Python's hash() on a str/tuple,
            # which is randomized per-process (PYTHONHASHSEED) and would silently
            # make the whole synthetic dataset non-reproducible between runs.
            seed = 1000 * persona_index + 10 * i + 1
            history = builder(num_gameweeks, seed=seed)
            rows.extend(_history_to_rows(manager_id, f"[SYNTHETIC] {persona_name.title()} {i + 1}", history))
            manager_id += 1

    ep_follower_history, _ = expected_points_follower_history(num_gameweeks, seed=4)
    rows.extend(_history_to_rows(manager_id, "[SYNTHETIC] EP Follower 1", ep_follower_history))
    manager_id += 1

    return rows, expected_points


def _history_to_rows(manager_id: int, manager_name: str, history: Tuple[CaptainObservation, ...]) -> List[dict]:
    rows = []
    for obs in history:
        starting_xi = obs.owned_player_ids + SQUAD_PADDING
        rows.append(
            {
                "season": SEASON,
                "manager_id": manager_id,
                "manager_name": manager_name,
                "gameweek": obs.gameweek,
                "starting_xi": DELIMITER.join(str(p) for p in starting_xi),
                "captain": obs.captain_id,
                "vice_captain": "",
            }
        )
    return rows


def write_synthetic_dataset_csv(path: Path, num_gameweeks: int = 30, managers_per_persona: int = 2) -> Dict[Tuple[str, int], Dict[int, float]]:
    """Writes the synthetic dataset to ``path`` in the native CSV schema
    and returns the accompanying expected-points lookup (kept separate,
    since EP is an optional external input, not part of the historical
    decision record itself - see Stage 4 brief item 3)."""
    rows, expected_points = build_synthetic_dataset(num_gameweeks, managers_per_persona)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return expected_points


PLAYER_NAMES_FOR_REPORTING = dict(PLAYER_NAMES)
