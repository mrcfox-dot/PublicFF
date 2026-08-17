"""Orchestration + CLI for the FPL Rival Stage 1 prototype."""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from .analysis import (
    build_element_name_map,
    build_position_summary,
    build_squad_comparisons,
    get_latest_finished_event,
)
from .api import FPLAPIError, FPLClient
from .league import build_manager_list, fetch_league
from .manager import fetch_manager_detail
from .storage import new_run_dir, save_json

DEFAULT_ENTRY_ID = 5234355
DEFAULT_MAX_MANAGERS = 50

GLOBAL_LEAGUE_TYPES = {"s"}  # 'system' leagues (Overall, country, club, gameweek) - not real mini-leagues


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FPL Rival - Stage 1: retrieve and structure public FPL mini-league data."
    )
    parser.add_argument("--entry", type=int, default=DEFAULT_ENTRY_ID, help="FPL manager entry ID.")
    parser.add_argument("--league-id", type=int, default=None, help="Classic league ID to analyse (skips the prompt).")
    parser.add_argument(
        "--max-managers",
        type=int,
        default=DEFAULT_MAX_MANAGERS,
        help=f"Cap on number of managers to deep-fetch per run (default {DEFAULT_MAX_MANAGERS}).",
    )
    parser.add_argument(
        "--include-system-leagues",
        action="store_true",
        help="Also list FPL's automatic system leagues (Overall, country, club, gameweek), not just user mini-leagues.",
    )
    return parser.parse_args(argv)


def list_manager_leagues(entry_data: dict, include_system: bool) -> list[dict]:
    leagues = []
    for league in entry_data.get("leagues", {}).get("classic", []):
        if not include_system and league.get("league_type") in GLOBAL_LEAGUE_TYPES:
            continue
        leagues.append(league)
    return leagues


def print_league_menu(leagues: list[dict]) -> None:
    print("\nClassic mini leagues for this manager:")
    print(f"{'#':<4}{'League ID':<12}{'Name':<32}{'Your Position':<20}{'Managers':<10}")
    for idx, league in enumerate(leagues, start=1):
        position = league.get("entry_rank")
        if not position:
            position = "n/a (pre-season)"
        managers = league.get("rank_count")
        managers = managers if managers is not None else "unknown"
        print(f"{idx:<4}{league['id']:<12}{league['name'][:31]:<32}{str(position):<20}{str(managers):<10}")


def prompt_for_league(leagues: list[dict]) -> Optional[dict]:
    if not sys.stdin.isatty():
        print(
            "\nNo --league-id given and input is not interactive. "
            "Re-run with --league-id <id> to analyse a specific league."
        )
        return None
    while True:
        choice = input(f"\nSelect a league [1-{len(leagues)}] (or 'q' to quit): ").strip()
        if choice.lower() == "q":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(leagues):
            return leagues[int(choice) - 1]
        print("Invalid choice, try again.")


def main(argv=None) -> int:
    args = parse_args(argv)
    client = FPLClient()
    data_gaps: list[str] = []

    print(f"Fetching manager entry {args.entry} ...")
    try:
        entry_data = client.get_entry(args.entry)
    except FPLAPIError as exc:
        print(f"FATAL: could not retrieve entry {args.entry}: {exc}")
        return 1
    if entry_data is None:
        print(f"FATAL: entry {args.entry} was not found (404). Check the ID.")
        return 1

    manager_label = f"{entry_data.get('player_first_name', '')} {entry_data.get('player_last_name', '')}".strip()
    print(f"Manager: {manager_label or 'unknown'} | Team: {entry_data.get('name', 'unknown')}")

    leagues = list_manager_leagues(entry_data, args.include_system_leagues)
    if not leagues:
        print("No classic mini leagues found for this manager (only system leagues, if any).")
        return 0
    print_league_menu(leagues)

    selected_league_meta = None
    if args.league_id is not None:
        selected_league_meta = next((l for l in leagues if l["id"] == args.league_id), None)
        if selected_league_meta is None:
            print(f"League id {args.league_id} is not one of this manager's listed leagues; attempting to fetch it anyway.")
            selected_league_meta = {"id": args.league_id, "name": f"League {args.league_id}"}
    else:
        selected_league_meta = prompt_for_league(leagues)
        if selected_league_meta is None:
            print("No league selected. Exiting.")
            return 0

    league_id = selected_league_meta["id"]
    print(f"\nFetching standings for league {league_id} ({selected_league_meta.get('name')}) ...")
    league_data = fetch_league(client, league_id)
    data_gaps.extend(league_data.errors)

    managers = build_manager_list(league_data)
    print(f"Found {len(managers)} manager(s) in this league.")

    truncated = False
    if len(managers) > args.max_managers:
        truncated = True
        managers_for_deep_fetch = managers[: args.max_managers]
        print(
            f"League has {len(managers)} managers; limiting per-manager deep fetch to the first "
            f"{args.max_managers} by rank (use --max-managers to change). Listing above still shows all."
        )
    else:
        managers_for_deep_fetch = managers

    print("Fetching bootstrap data (players, gameweeks) ...")
    try:
        bootstrap = client.get_bootstrap_static()
    except FPLAPIError as exc:
        print(f"FATAL: could not retrieve bootstrap-static: {exc}")
        return 1

    latest_finished_event = get_latest_finished_event(bootstrap)
    name_map = build_element_name_map(bootstrap)
    if latest_finished_event is None:
        print("No gameweek has finished yet this season - squad/picks data will not be available.")
    else:
        print(f"Latest completed gameweek: {latest_finished_event}")

    run_dir = new_run_dir(args.entry, league_id)
    save_json(run_dir / "entry.json", entry_data)
    save_json(run_dir / "league_standings.json", {
        "league_id": league_id,
        "meta": league_data.meta,
        "standings_raw": league_data.standings,
        "new_entries_raw": league_data.new_entries,
        "managers": managers,
        "errors": league_data.errors,
    })

    print(f"Fetching detailed data for {len(managers_for_deep_fetch)} manager(s) ...")
    manager_details: dict[int, dict] = {}
    for i, m in enumerate(managers_for_deep_fetch, start=1):
        entry_id = m["entry_id"]
        print(f"  [{i}/{len(managers_for_deep_fetch)}] entry {entry_id} ({m.get('team_name') or 'unknown team'})")
        detail = fetch_manager_detail(client, entry_id, latest_finished_event, data_gaps)
        manager_details[entry_id] = detail
        save_json(run_dir / "managers" / f"{entry_id}.json", detail)

    if args.entry not in manager_details:
        print(f"  (fetching your own entry {args.entry} separately - not in the deep-fetch batch)")
        detail = fetch_manager_detail(client, args.entry, latest_finished_event, data_gaps)
        manager_details[args.entry] = detail
        save_json(run_dir / "managers" / f"{args.entry}.json", detail)

    position_summary = build_position_summary(args.entry, managers)
    squad_summary = build_squad_comparisons(args.entry, manager_details, name_map)

    summary = {
        "entry_id": args.entry,
        "league_id": league_id,
        "league_name": (league_data.meta or {}).get("name") or selected_league_meta.get("name"),
        "position_summary": position_summary,
        "squad_summary": squad_summary,
        "managers_listed": len(managers),
        "managers_deep_fetched": len(managers_for_deep_fetch),
        "truncated": truncated,
        "data_gaps": data_gaps,
    }
    save_json(run_dir / "summary.json", summary)

    print_final_report(summary, run_dir)
    return 0


def print_final_report(summary: dict, run_dir) -> None:
    print("\n" + "=" * 72)
    print("FPL RIVAL - STAGE 1 REPORT")
    print("=" * 72)
    print(f"League: {summary['league_name']} (ID {summary['league_id']})")
    print(f"Data saved to: {run_dir}")

    ps = summary["position_summary"]
    print("\n-- Position summary --")
    print(f"Managers in league (all, via membership list): {ps['num_managers']}")
    print(f"Managers with a scored gameweek: {ps['num_managers_with_scores']}")
    if ps["my_position"] is not None:
        print(f"My position: {ps['my_position']}")
        print(f"My total points: {ps['my_total_points']}")
        print(f"Points behind leader: {ps['points_behind_leader']}")
        if ps["points_ahead_of_below"] is not None:
            print(f"Points ahead of manager below: {ps['points_ahead_of_below']}")
    if ps["note"]:
        print(f"Note: {ps['note']}")

    ss = summary["squad_summary"]
    print(f"\n-- Squad comparison (latest completed gameweek: {ss['latest_completed_gameweek']}) --")
    if ss["note"]:
        print(f"Note: {ss['note']}")
    else:
        for c in ss["comparisons"]:
            print(f"\nvs {c['team_name']} (entry {c['entry_id']}):")
            if c.get("note"):
                print(f"  {c['note']}")
                continue
            print(f"  Overlap: {c['overlap_count']} shared players")
            print(f"  Only I own: {', '.join(c['players_only_i_own']) or 'none'}")
            print(f"  Only they own: {', '.join(c['players_only_rival_owns']) or 'none'}")
            print(f"  My captain: {c['my_captain']} | Their captain: {c['rival_captain']}")

    print(f"\n-- Coverage --")
    print(f"Managers listed: {summary['managers_listed']}")
    print(f"Managers deep-fetched (history/picks/transfers): {summary['managers_deep_fetched']}"
          + (" (truncated by --max-managers)" if summary["truncated"] else ""))

    print("\n-- Data classification --")
    print("Public data retrieved via unauthenticated FPL API calls:")
    print("  - Manager entry profile, overall points/rank, league memberships")
    print("  - Classic league standings and membership")
    print("  - Per-manager gameweek-by-gameweek history and chips used")
    print("  - Per-manager transfer history")
    print("  - Per-manager squad picks (captain/vice/bench) for the latest completed gameweek")
    print("Requires FPL authentication (not attempted in this prototype):")
    print("  - Making transfers, changing captain, joining/leaving leagues")
    print("  - Anything under 'My Team' before it becomes public history")
    print("Cannot be obtained via any public or authenticated means shown here:")
    print("  - Another manager's picks for a gameweek before that gameweek's deadline")
    print("    (FPL platform restriction - applies even to authenticated users)")
    print("  - Private contact info, or leagues this manager has not joined")

    if summary["data_gaps"]:
        print(f"\n-- Data gaps encountered this run ({len(summary['data_gaps'])}) --")
        for gap in summary["data_gaps"][:25]:
            print(f"  - {gap}")
        if len(summary["data_gaps"]) > 25:
            print(f"  ... and {len(summary['data_gaps']) - 25} more (see summary.json).")
    else:
        print("\nNo data gaps encountered this run.")

    print("=" * 72)


if __name__ == "__main__":
    raise SystemExit(main())
