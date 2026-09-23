#!/usr/bin/env python3
"""Live Captain Battle: Stage 3's simulator plus Stage 4's rival prediction,
run against a REAL mini league instead of a synthetic fixture.

    python3 captain_battle_cli.py --league-id 1132827
    python3 captain_battle_cli.py --league-id 1132827 --entry 5234355 --simulations 20000

Unlike simulation_cli.py / prediction_cli.py (both synthetic-only by design
- see their own docstrings), this script makes real network calls and its
output is a genuine prediction, not a demo. It only wires together live
retrieval (fpl_rival/live_captain_battle.py), the simulation engine, and the
text renderer; it contains no analytics of its own.
"""

from __future__ import annotations

import argparse
import time

from fpl_rival.api import FPLAPIError, FPLClient
from fpl_rival.live_captain_battle import build_live_captain_battle_inputs
from fpl_rival.simulation.captain_battle import DEFAULT_OBJECTIVE, OBJECTIVES, run_captain_battle
from fpl_rival.simulation.models import SimulationConfig
from fpl_rival.simulation.report_text import render_captain_battle_report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FPL Rival - live Captain Battle (Stage 3 + Stage 4, real league data).")
    parser.add_argument("--league-id", type=int, required=True)
    parser.add_argument("--entry", type=int, default=5234355, help="Your FPL manager entry ID.")
    parser.add_argument("--max-managers", type=int, default=50)
    parser.add_argument("--simulations", type=int, default=20_000, help="Monte Carlo simulation count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (reproducible).")
    parser.add_argument("--objective", default=DEFAULT_OBJECTIVE, choices=sorted(OBJECTIVES), help="Mini-league objective to rank captains by.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    client = FPLClient()

    print("Fetching bootstrap data ...")
    try:
        bootstrap = client.get_bootstrap_static()
    except FPLAPIError as exc:
        print(f"FATAL: could not retrieve bootstrap-static: {exc}")
        return 1

    print(f"Collecting live data for league {args.league_id} (this can take a while for large leagues) ...")
    inputs = build_live_captain_battle_inputs(client, args.league_id, args.entry, bootstrap, max_managers=args.max_managers)

    if inputs.chris is None:
        print("FATAL: could not build a live Captain Battle:")
        for e in inputs.errors:
            print(f"  - {e}")
        return 1

    config = SimulationConfig(num_simulations=args.simulations, random_seed=args.seed, rival_captain_mode="probabilistic")
    start = time.perf_counter()
    result = run_captain_battle(
        inputs.chris, inputs.league, inputs.projections, inputs.candidate_captain_ids, config,
        objective=args.objective, primary_rival_entry_ids=inputs.primary_rival_ids,
    )
    elapsed = time.perf_counter() - start

    print()
    print(render_captain_battle_report(result, num_managers=len(inputs.league.managers), is_synthetic=False))
    print(
        f"\n(runtime: {elapsed:.2f}s for {args.simulations:,} simulations x "
        f"{len(inputs.candidate_captain_ids)} candidates, target GW{inputs.target_gameweek})"
    )

    if inputs.predicted_rival_ids:
        names = [m.name for m in inputs.league.managers if m.entry_id in inputs.predicted_rival_ids]
        print(f"Rivals with a real Stage 4 captain prediction (probabilistic mode): {', '.join(names)}")
    else:
        print("No rival had enough history yet for a Stage 4 prediction - every rival used their actual current captain (fixed mode).")

    if inputs.errors:
        print(f"\n{len(inputs.errors)} data gap(s)/note(s):")
        for e in inputs.errors[:20]:
            print(f"  - {e}")
        if len(inputs.errors) > 20:
            print(f"  ... and {len(inputs.errors) - 20} more.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
