"""Builds REAL ``HistoricalRecord``s from live FPL data - the one bridge
from retrieval into Stage 4.5's historical dataset format.

This is different from ``importer.py``'s "local file only, never
downloaded automatically" rule: that rule is about arbitrary third-party
historical datasets. This module calls the exact same public FPL endpoints
Stage 1 already uses (via Stage 2's ``intelligence.collect``, reused
unmodified), for a league the user has legitimate real-time access to -
exactly like Stage 1's own CLI does every time it runs. It never touches
a third-party dataset and never modifies Stage 1 or Stage 2.

The output is written to a growing local CSV, in the exact native schema
``importer.import_csv`` already reads - so every existing Stage 4.5 tool
(baselines, harness, fitting, ablation, calibration) works on real data
completely unchanged, the moment enough real gameweeks exist.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from fpl_rival.api import FPLClient
from fpl_rival.intelligence.collect import collect_league_data

from .importer import DELIMITER, import_csv
from .models import HistoricalRecord

REAL_CSV_FIELDS = (
    "season",
    "manager_id",
    "manager_name",
    "gameweek",
    "starting_xi",
    "squad",
    "captain",
    "vice_captain",
    "transfers",
    "chip",
    "total_points",
    "gameweek_points",
    "rank",
)


def infer_season_label(bootstrap: dict) -> str:
    """Derives "YYYY-YY" from Gameweek 1's deadline (the FPL season starts
    in August and is conventionally labelled by its start/end years)."""
    gw1 = next((e for e in bootstrap.get("events", []) if e.get("id") == 1), None)
    if gw1 is None or not gw1.get("deadline_time"):
        return "unknown-season"
    dt = datetime.fromisoformat(gw1["deadline_time"].replace("Z", "+00:00"))
    if dt.month >= 7:
        return f"{dt.year}-{(dt.year + 1) % 100:02d}"
    return f"{dt.year - 1}-{dt.year % 100:02d}"


def build_real_historical_records(
    client: FPLClient,
    league_id: int,
    bootstrap: dict,
    chris_entry_id: Optional[int] = None,
    max_managers: int = 50,
    season: Optional[str] = None,
) -> Tuple[List[HistoricalRecord], List[str]]:
    """Fetches live and converts every finished gameweek's picks for every
    manager in the league into ``HistoricalRecord``s. Errors already
    surfaced by the live fetch (e.g. one manager's picks not public yet)
    are returned alongside, never silently dropped - a manager with no
    usable gameweek simply contributes no records, rather than a
    fabricated one.
    """
    league_data, _ctx, errors = collect_league_data(
        client, league_id, bootstrap, chris_entry_id=chris_entry_id, max_managers=max_managers, fetch_full_picks_history=True,
    )
    season = season or infer_season_label(bootstrap)

    records: List[HistoricalRecord] = []
    for manager in league_data.managers:
        history_by_event = {row.event: row for row in manager.gameweek_history}
        for event, squad in sorted(manager.squads_by_event.items()):
            starting_xi = tuple(p.element for p in squad.starting_xi)
            captain = squad.captain_element
            if not starting_xi or captain is None or captain not in starting_xi:
                errors.append(
                    f"entry {manager.entry_id} GW{event}: no valid captain in starting XI - "
                    "skipped (not a usable historical decision record)."
                )
                continue
            history_row = history_by_event.get(event)
            full_squad = tuple(p.element for p in squad.picks)
            records.append(
                HistoricalRecord(
                    season=season,
                    manager_id=manager.entry_id,
                    gameweek=event,
                    starting_xi=starting_xi,
                    captain=captain,
                    manager_name=manager.manager_name,
                    squad=full_squad or None,
                    vice_captain=squad.vice_captain_element,
                    transfers=(history_row.event_transfers if history_row and history_row.event_transfers is not None else 0),
                    chip=squad.active_chip,
                    total_points=(history_row.total_points if history_row else None),
                    gameweek_points=(squad.points if squad.points is not None else (history_row.points if history_row else None)),
                    rank=(history_row.rank if history_row else None),
                )
            )
    return records, errors


def _record_to_row(record: HistoricalRecord) -> dict:
    return {
        "season": record.season,
        "manager_id": record.manager_id,
        "manager_name": record.manager_name or "",
        "gameweek": record.gameweek,
        "starting_xi": DELIMITER.join(str(p) for p in record.starting_xi),
        "squad": DELIMITER.join(str(p) for p in record.squad) if record.squad else "",
        "captain": record.captain,
        "vice_captain": record.vice_captain if record.vice_captain is not None else "",
        "transfers": record.transfers,
        "chip": record.chip or "",
        "total_points": record.total_points if record.total_points is not None else "",
        "gameweek_points": record.gameweek_points if record.gameweek_points is not None else "",
        "rank": record.rank if record.rank is not None else "",
    }


def merge_and_write_csv(new_records: List[HistoricalRecord], path: Path) -> dict:
    """Appends/updates ``new_records`` into the CSV at ``path``, keyed by
    (season, manager_id, gameweek) - re-running for a gameweek that's
    already stored simply refreshes that row (e.g. if a correction came
    through) rather than duplicating it. Returns a summary of what changed.
    """
    path = Path(path)
    existing = import_csv(path) if path.exists() else []
    merged = {(r.season, r.manager_id, r.gameweek): r for r in existing}

    added, updated = 0, 0
    for record in new_records:
        key = (record.season, record.manager_id, record.gameweek)
        if key in merged:
            updated += 1
        else:
            added += 1
        merged[key] = record

    ordered = sorted(merged.values(), key=lambda r: (r.season, r.manager_id, r.gameweek))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REAL_CSV_FIELDS)
        writer.writeheader()
        for record in ordered:
            writer.writerow(_record_to_row(record))

    return {
        "path": str(path),
        "records_added": added,
        "records_updated": updated,
        "total_records": len(ordered),
        "managers": len({r.manager_id for r in ordered}),
        "gameweeks": sorted({r.gameweek for r in ordered}),
    }
