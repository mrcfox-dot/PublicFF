#!/usr/bin/env python3
"""Builds/updates a REAL historical captain-decision dataset for one
mini league, from live FPL data - the growing input Stage 4.5's
calibration harness needs.

    python3 real_history_cli.py --league-id 1132833
    python3 real_history_cli.py --league-id 1132833 --out data/real_history/league_1132833.csv

Safe to re-run every week: it re-fetches whatever gameweeks are finished
and merges them into the existing file (keyed by season/manager/gameweek -
see fpl_rival/calibration/live_import.py), so running it after each
gameweek's deadline grows a real dataset one gameweek at a time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fpl_rival.api import FPLAPIError, FPLClient
from fpl_rival.calibration.importer import import_csv, summarize_dataset
from fpl_rival.calibration.live_import import build_real_historical_records, merge_and_write_csv
from fpl_rival.calibration.report_text import render_dataset_summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/update a real historical captain-decision dataset from live FPL data.")
    parser.add_argument("--league-id", type=int, required=True)
    parser.add_argument("--chris-entry", type=int, default=5234355)
    parser.add_argument("--max-managers", type=int, default=50)
    parser.add_argument("--out", type=str, default=None, help="Defaults to data/real_history/league_<id>.csv")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_path = Path(args.out) if args.out else Path("data/real_history") / f"league_{args.league_id}.csv"

    client = FPLClient()
    print("Fetching bootstrap data ...")
    try:
        bootstrap = client.get_bootstrap_static()
    except FPLAPIError as exc:
        print(f"FATAL: could not retrieve bootstrap-static: {exc}")
        return 1

    print(f"Collecting live picks history for league {args.league_id} (this can take a while for large leagues) ...")
    records, errors = build_real_historical_records(
        client, args.league_id, bootstrap, chris_entry_id=args.chris_entry, max_managers=args.max_managers
    )
    print(f"Converted {len(records)} real (manager, gameweek) captain decisions.")

    summary = merge_and_write_csv(records, out_path)
    print(f"\nWrote {out_path}")
    print(f"  This run: {summary['records_added']} new, {summary['records_updated']} updated")
    print(f"  Dataset total: {summary['total_records']} records, {summary['managers']} managers, gameweeks {summary['gameweeks']}")

    if errors:
        print(f"\n{len(errors)} data gap(s) this run:")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more.")

    dataset = import_csv(out_path)
    print()
    print(render_dataset_summary(summarize_dataset(dataset, is_synthetic=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
