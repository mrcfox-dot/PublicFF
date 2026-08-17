#!/usr/bin/env python3
"""Stage 2 single-command launcher: FPL Rival Intelligence Engine.

    python3 intelligence_cli.py                        # live, interactive league pick
    python3 intelligence_cli.py --league-id 1132827     # live, specific league
    python3 intelligence_cli.py --fixture scenario_a    # synthetic demo, no network at all

Retrieval (fpl_rival.api / .league / .intelligence.collect), analytics
(fpl_rival.intelligence.engine and friends) and presentation
(fpl_rival.intelligence.report_text) are kept in separate modules; this
script only wires them together and prints/saves the result.
"""

from __future__ import annotations

import argparse
import json

from fpl_rival.api import FPLAPIError, FPLClient
from fpl_rival.cli import (
    DEFAULT_ENTRY_ID,
    DEFAULT_MAX_MANAGERS,
    list_manager_leagues,
    print_league_menu,
    prompt_for_league,
)
from fpl_rival.fixtures import synthetic
from fpl_rival.intelligence import engine
from fpl_rival.intelligence.collect import collect_league_data
from fpl_rival.intelligence.config import EngineConfig
from fpl_rival.intelligence.report_text import render_report
from fpl_rival.storage import new_run_dir, save_json

FIXTURES = {
    "scenario_a": synthetic.scenario_a_defend,
    "scenario_b": synthetic.scenario_b_attack,
    "scenario_c": synthetic.scenario_c_desperate,
    "scenario_d": synthetic.scenario_d_identical_squads,
    "scenario_e": synthetic.scenario_e_threats,
    "preseason": synthetic.scenario_preseason,
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FPL Rival Intelligence Engine (Stage 2).")
    parser.add_argument("--entry", type=int, default=DEFAULT_ENTRY_ID, help="FPL manager entry ID.")
    parser.add_argument("--league-id", type=int, default=None, help="Classic league ID to analyse (skips the prompt).")
    parser.add_argument("--max-managers", type=int, default=DEFAULT_MAX_MANAGERS)
    parser.add_argument(
        "--fixture",
        choices=sorted(FIXTURES),
        default=None,
        help="Run against synthetic test data instead of live FPL data (no network calls at all).",
    )
    parser.add_argument("--json", action="store_true", help="Also print the full structured report as JSON.")
    return parser.parse_args(argv)


def run_fixture(name: str, as_json: bool) -> int:
    league, ctx, chris_entry_id = FIXTURES[name]()
    report = engine.run(league, chris_entry_id, ctx, EngineConfig())
    print(render_report(report))
    if as_json:
        print("\n--- JSON ---")
        print(json.dumps(report, indent=2, default=str))
    return 0


def run_live(args: argparse.Namespace) -> int:
    client = FPLClient()
    print(f"Fetching manager entry {args.entry} ...")
    try:
        entry_data = client.get_entry(args.entry)
    except FPLAPIError as exc:
        print(f"FATAL: could not retrieve entry {args.entry}: {exc}")
        return 1
    if entry_data is None:
        print(f"FATAL: entry {args.entry} not found (404). Check the ID.")
        return 1

    leagues = list_manager_leagues(entry_data, include_system=False)
    if not leagues:
        print("No classic mini leagues found for this manager.")
        return 0

    if args.league_id is not None:
        selected = next(
            (l for l in leagues if l["id"] == args.league_id),
            {"id": args.league_id, "name": f"League {args.league_id}"},
        )
    else:
        print_league_menu(leagues)
        selected = prompt_for_league(leagues)
        if selected is None:
            print("No league selected. Exiting.")
            return 0

    league_id = selected["id"]

    print("\nFetching bootstrap data (players, gameweeks, chip rules) ...")
    try:
        bootstrap = client.get_bootstrap_static()
    except FPLAPIError as exc:
        print(f"FATAL: could not retrieve bootstrap-static: {exc}")
        return 1

    print(f"Collecting data for league {league_id} ({selected.get('name')}) ...")
    league_data, ctx, errors = collect_league_data(
        client, league_id, bootstrap, chris_entry_id=args.entry, max_managers=args.max_managers
    )
    print(f"Collected {len(league_data.managers)} manager(s). Running the intelligence engine ...")

    try:
        report = engine.run(league_data, args.entry, ctx, EngineConfig())
    except engine.UnknownManagerError as exc:
        print(f"FATAL: {exc}")
        return 1

    run_dir = new_run_dir(args.entry, league_id)
    save_json(run_dir / "intelligence_report.json", report)
    save_json(run_dir / "collection_errors.json", errors)

    print()
    print(render_report(report))
    print()
    print(f"Full structured report saved to: {run_dir / 'intelligence_report.json'}")
    if errors:
        print(f"{len(errors)} data gap(s)/errors encountered during collection - see collection_errors.json")

    if args.json:
        print("\n--- JSON ---")
        print(json.dumps(report, indent=2, default=str))

    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.fixture:
        return run_fixture(args.fixture, args.json)
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
