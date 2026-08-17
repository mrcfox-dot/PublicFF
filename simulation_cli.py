#!/usr/bin/env python3
"""Stage 3 single-command launcher: the League Win Simulator (Captain Battle).

    python3 simulation_cli.py --fixture captain-attack
    python3 simulation_cli.py --fixture captain-defend
    python3 simulation_cli.py --fixture shared-squad
    python3 simulation_cli.py --fixture rival-uncertainty
    python3 simulation_cli.py --fixture captain-attack --simulations 5000 --seed 7

Live 2026/27 gameweek data does not exist yet, so every fixture here is
synthetic - see fpl_rival/fixtures/simulation_fixtures.py. This script only
wires together retrieval-free synthetic inputs, the simulation engine, and
the text renderer; it contains no analytics of its own.
"""

from __future__ import annotations

import argparse
import time

from fpl_rival.fixtures import simulation_fixtures as fx
from fpl_rival.simulation.captain_battle import DEFAULT_OBJECTIVE, OBJECTIVES, run_captain_battle
from fpl_rival.simulation.models import SimulationConfig
from fpl_rival.simulation.monte_carlo import collect_required_player_ids
from fpl_rival.simulation.manager_outcomes import simulate_manager_gameweek
from fpl_rival.simulation.player_outcomes import sample_all_players
from fpl_rival.simulation.report_text import render_captain_battle_report, render_shared_squad_report

import numpy as np

CAPTAIN_BATTLE_FIXTURES = {
    "captain-attack": fx.captain_attack,
    "captain-defend": fx.captain_defend,
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FPL Rival Stage 3 - Captain Battle simulator.")
    parser.add_argument(
        "--fixture",
        required=True,
        choices=sorted(CAPTAIN_BATTLE_FIXTURES) + ["shared-squad", "rival-uncertainty"],
        help="Which synthetic scenario to run (no live data exists yet).",
    )
    parser.add_argument("--simulations", type=int, default=50_000, help="Monte Carlo simulation count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (reproducible).")
    parser.add_argument("--objective", default=DEFAULT_OBJECTIVE, choices=sorted(OBJECTIVES), help="Mini-league objective to rank captains by.")
    return parser.parse_args(argv)


def run_captain_battle_fixture(name: str, args: argparse.Namespace) -> int:
    chris, league, projections, candidates = CAPTAIN_BATTLE_FIXTURES[name]()
    config = SimulationConfig(num_simulations=args.simulations, random_seed=args.seed)
    start = time.perf_counter()
    result = run_captain_battle(chris, league, projections, candidates, config, objective=args.objective)
    elapsed = time.perf_counter() - start

    print(render_captain_battle_report(result, num_managers=len(league.managers)))
    print(f"\n(runtime: {elapsed:.2f}s for {args.simulations:,} simulations x {len(candidates)} candidates)")
    return 0


def run_shared_squad_fixture(args: argparse.Namespace) -> int:
    def gap_std(builder):
        chris, rival, league, projections = builder()
        rng = np.random.default_rng(args.seed)
        ids = collect_required_player_ids(league)
        points = sample_all_players(ids, projections, args.simulations, rng)
        chris_scores = simulate_manager_gameweek(chris, points, np.full(args.simulations, chris.captain_id))
        rival_scores = simulate_manager_gameweek(rival, points, np.full(args.simulations, rival.captain_id))
        return float(np.std(chris_scores - rival_scores))

    low_std = gap_std(fx.shared_squad_low_diff)
    high_std = gap_std(fx.shared_squad_high_diff)
    print(render_shared_squad_report("1/11 XI slots differ", low_std, "6/11 XI slots differ", high_std))
    return 0


def run_rival_uncertainty_fixture(args: argparse.Namespace) -> int:
    for label, builder in (
        ("SCENARIO A - leader captain probability: Salah 90% / Palmer 10%", fx.rival_uncertainty_scenario_a),
        ("SCENARIO B - leader captain probability: Salah 50% / Palmer 50%", fx.rival_uncertainty_scenario_b),
    ):
        chris, league, projections, candidates = builder()
        config = SimulationConfig(num_simulations=args.simulations, random_seed=args.seed, rival_captain_mode="probabilistic")
        result = run_captain_battle(chris, league, projections, candidates, config, objective=args.objective)
        print(label)
        print("-" * len(label))
        print(render_captain_battle_report(result, num_managers=len(league.managers)))
        print()
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.fixture in CAPTAIN_BATTLE_FIXTURES:
        return run_captain_battle_fixture(args.fixture, args)
    if args.fixture == "shared-squad":
        return run_shared_squad_fixture(args)
    if args.fixture == "rival-uncertainty":
        return run_rival_uncertainty_fixture(args)
    print(f"Unknown fixture {args.fixture!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
