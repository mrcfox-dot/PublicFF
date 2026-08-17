"""Local JSON storage for a single run. No database - plain files only."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def new_run_dir(entry_id: int, league_id: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = DATA_ROOT / f"run_{entry_id}_league{league_id}_{stamp}"
    (run_dir / "managers").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str, ensure_ascii=False)
