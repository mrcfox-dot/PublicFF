"""Local-file-only historical dataset importer.

NEVER downloads or scrapes anything - every function here takes a path
already present on disk. Two "native" formats are supported directly
(CSV and JSON-Lines, both in the same wide one-row-per-manager-gameweek
shape as ``HistoricalRecord``), plus one example ADAPTOR
(``import_long_format_csv``) showing how a differently-shaped raw export
(one row per manager-gameweek-PLAYER) gets translated into the same
normalized records before anything downstream ever sees it. A real
dataset in a third shape needs one more adaptor function like it - nothing
else in the system changes.

Provenance: every dataset loaded through this module should be documented
by the CALLER (e.g. in the CLI invocation or a README note) - this module
deliberately does not embed assumptions about any specific dataset's
license or source (Stage 4.5 brief item 15).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .models import DatasetError, DatasetSummary, HistoricalRecord

REQUIRED_FIELDS = ("season", "manager_id", "gameweek", "starting_xi", "captain")
DELIMITER = ";"


def _parse_id_list(raw: Optional[str]) -> Optional[tuple]:
    if raw is None or raw == "":
        return None
    return tuple(int(x) for x in str(raw).split(DELIMITER) if x.strip() != "")


def _row_to_record(row: dict, source: str, line_no: int) -> HistoricalRecord:
    missing = [f for f in REQUIRED_FIELDS if row.get(f) in (None, "")]
    if missing:
        raise DatasetError(f"{source} line {line_no}: missing required field(s) {missing}.")

    starting_xi = row["starting_xi"]
    if isinstance(starting_xi, str):
        starting_xi = _parse_id_list(starting_xi)
    squad = row.get("squad")
    if isinstance(squad, str):
        squad = _parse_id_list(squad)

    try:
        return HistoricalRecord(
            season=str(row["season"]),
            manager_id=int(row["manager_id"]),
            gameweek=int(row["gameweek"]),
            starting_xi=tuple(int(x) for x in starting_xi),
            captain=int(row["captain"]),
            manager_name=row.get("manager_name") or None,
            squad=tuple(int(x) for x in squad) if squad else None,
            vice_captain=int(row["vice_captain"]) if row.get("vice_captain") not in (None, "") else None,
            transfers=int(row["transfers"]) if row.get("transfers") not in (None, "") else 0,
            chip=row.get("chip") or None,
            total_points=int(row["total_points"]) if row.get("total_points") not in (None, "") else None,
            gameweek_points=int(row["gameweek_points"]) if row.get("gameweek_points") not in (None, "") else None,
            rank=int(row["rank"]) if row.get("rank") not in (None, "") else None,
        )
    except (ValueError, TypeError) as exc:
        raise DatasetError(f"{source} line {line_no}: malformed field - {exc}") from exc


def import_csv(path: Path) -> List[HistoricalRecord]:
    """Native wide CSV: one row per (manager, gameweek). Required columns:
    season, manager_id, gameweek, starting_xi, captain (starting_xi/squad
    are ';'-delimited player ids). All other columns optional."""
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):  # header is line 1
            if not any(row.values()):
                continue
            records.append(_row_to_record(row, str(path), i))
    return records


def import_jsonl(path: Path) -> List[HistoricalRecord]:
    """Native JSON-Lines: one JSON object per line, same fields as the CSV
    (starting_xi/squad as JSON arrays of ints, not delimited strings)."""
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records.append(_row_to_record(row, str(path), i))
    return records


def import_long_format_csv(path: Path) -> List[HistoricalRecord]:
    """EXAMPLE ADAPTOR for a raw export shaped one row per
    (manager, gameweek, player) instead of the native wide shape. Expected
    columns: season, manager_id, gameweek, player_id, is_starting,
    is_captain, is_vice_captain, plus optionally manager_name, transfers,
    chip, total_points, gameweek_points, rank (repeated identically on
    every row of a group - the last non-empty value seen wins).

    This function's only job is to PIVOT into the native wide shape and
    hand off to the same ``_row_to_record`` validation every other format
    uses - all dataset-specific logic stays here, not in the prediction
    engine or anywhere downstream.
    """
    groups: Dict[tuple, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not any(row.values()):
                continue
            key = (row["season"], row["manager_id"], row["gameweek"])
            group = groups.setdefault(
                key,
                {
                    "season": row["season"],
                    "manager_id": row["manager_id"],
                    "gameweek": row["gameweek"],
                    "starting_xi": [],
                    "squad": [],
                    "captain": None,
                    "vice_captain": None,
                },
            )
            player_id = row.get("player_id")
            if player_id in (None, ""):
                continue
            group["squad"].append(player_id)
            if str(row.get("is_starting", "")).strip().lower() in ("1", "true", "yes"):
                group["starting_xi"].append(player_id)
            if str(row.get("is_captain", "")).strip().lower() in ("1", "true", "yes"):
                group["captain"] = player_id
            if str(row.get("is_vice_captain", "")).strip().lower() in ("1", "true", "yes"):
                group["vice_captain"] = player_id
            for optional_field in ("manager_name", "transfers", "chip", "total_points", "gameweek_points", "rank"):
                value = row.get(optional_field)
                if value not in (None, ""):
                    group[optional_field] = value

    records = []
    for i, group in enumerate(groups.values(), start=1):
        group["starting_xi"] = DELIMITER.join(group["starting_xi"])
        group["squad"] = DELIMITER.join(group["squad"])
        records.append(_row_to_record(group, str(path) + " (long format)", i))
    return records


IMPORTERS_BY_SUFFIX = {
    ".csv": import_csv,
    ".jsonl": import_jsonl,
    ".ndjson": import_jsonl,
}


def load_dataset(path: Path, long_format: bool = False) -> List[HistoricalRecord]:
    """Loads a single file (dispatched by extension) or every supported
    file in a directory. Pass ``long_format=True`` to use the example
    per-player-row adaptor for CSV files instead of the native wide shape.
    """
    path = Path(path)
    if path.is_dir():
        records: List[HistoricalRecord] = []
        for child in sorted(path.iterdir()):
            if child.is_file() and child.suffix.lower() in (".csv", ".jsonl", ".ndjson"):
                records.extend(load_dataset(child, long_format=long_format))
        if not records:
            raise DatasetError(f"No supported dataset files (.csv/.jsonl) found in directory {path}.")
        return records

    if not path.exists():
        raise DatasetError(f"Dataset path does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv" and long_format:
        return import_long_format_csv(path)
    importer = IMPORTERS_BY_SUFFIX.get(suffix)
    if importer is None:
        raise DatasetError(f"Unsupported dataset file type {suffix!r} for {path} - expected .csv or .jsonl.")
    return importer(path)


def summarize_dataset(records: List[HistoricalRecord], is_synthetic: bool = False) -> DatasetSummary:
    if not records:
        return DatasetSummary(
            num_records=0, num_managers=0, num_gameweeks=0, seasons=(), min_gameweek=None, max_gameweek=None,
            records_missing_rank=0, records_missing_total_points=0, records_missing_chip=0,
            warnings=("Dataset is empty.",), is_synthetic=is_synthetic,
        )
    seasons = tuple(sorted({r.season for r in records}))
    gameweeks = [r.gameweek for r in records]
    warnings = []
    missing_rank = sum(1 for r in records if r.rank is None)
    missing_points = sum(1 for r in records if r.total_points is None)
    missing_chip = sum(1 for r in records if r.chip is None)
    if missing_rank == len(records):
        warnings.append("No record has a rank - rank-based analysis is unavailable for this dataset.")
    if missing_points == len(records):
        warnings.append("No record has total_points - points-based analysis is unavailable.")
    num_managers = len({r.manager_id for r in records})
    if num_managers < 2:
        warnings.append("Fewer than 2 managers - league consensus baseline/feature has no signal and will be skipped.")

    return DatasetSummary(
        num_records=len(records),
        num_managers=num_managers,
        num_gameweeks=len({(r.season, r.gameweek) for r in records}),
        seasons=seasons,
        min_gameweek=min(gameweeks),
        max_gameweek=max(gameweeks),
        records_missing_rank=missing_rank,
        records_missing_total_points=missing_points,
        records_missing_chip=missing_chip,
        warnings=tuple(warnings),
        is_synthetic=is_synthetic,
    )


def records_by_manager_and_season(records: List[HistoricalRecord]) -> Dict[tuple, List[HistoricalRecord]]:
    grouped = defaultdict(list)
    for r in records:
        grouped[(r.season, r.manager_id)].append(r)
    for key in grouped:
        grouped[key].sort(key=lambda r: r.gameweek)
    return dict(grouped)
