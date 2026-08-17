#!/usr/bin/env python3
"""Stage 4 single-command launcher: the Rival Prediction Engine.

    python3 prediction_cli.py --fixture template
    python3 prediction_cli.py --fixture loyal
    python3 prediction_cli.py --fixture differential
    python3 prediction_cli.py --fixture ep-follower
    python3 prediction_cli.py --fixture erratic
    python3 prediction_cli.py --fixture pipeline-loyal      # full Stage4 -> Stage3 pipeline demo
    python3 prediction_cli.py --fixture pipeline-balanced   # same, with Dave's history changed

Live 2026/27 gameweek data does not exist yet, so every fixture here is
synthetic - see fpl_rival/fixtures/prediction_fixtures.py.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from fpl_rival.fixtures import prediction_fixtures as fx
from fpl_rival.prediction.models import LeagueHistory, PredictionConfig
from fpl_rival.prediction.prediction_engine import predict_captain
from fpl_rival.prediction.report_text import render_captain_prediction
from fpl_rival.prediction.stage3_adapter import apply_prediction_to_manager_state
from fpl_rival.simulation.captain_battle import run_captain_battle
from fpl_rival.simulation.models import SimulationConfig
from fpl_rival.simulation.report_text import render_captain_battle_report

PERSONA_HISTORY_BUILDERS = {
    "template": fx.template_manager_history,
    "loyal": fx.loyal_captain_manager_history,
    "differential": fx.differential_manager_history,
    "erratic": fx.erratic_manager_history,
}

DEFAULT_EP = {pid: v for pid, v in zip(fx.CAPTAIN_POOL, [7.5, 6.0, 7.0, 5.5])}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FPL Rival Stage 4 - Rival Prediction Engine.")
    parser.add_argument(
        "--fixture",
        required=True,
        choices=sorted(PERSONA_HISTORY_BUILDERS) + ["ep-follower", "pipeline-loyal", "pipeline-balanced"],
    )
    parser.add_argument("--gameweeks", type=int, default=20, help="Number of synthetic historical gameweeks.")
    parser.add_argument("--target-gameweek", type=int, default=None, help="Gameweek to predict (default: gameweeks + 1).")
    return parser.parse_args(argv)


def run_persona_fixture(name: str, args: argparse.Namespace) -> int:
    target_gw = args.target_gameweek or (args.gameweeks + 1)
    if name == "ep-follower":
        history, ep_series = fx.expected_points_follower_history(args.gameweeks)
        expected_points = ep_series.get(target_gw, DEFAULT_EP)
    else:
        history = PERSONA_HISTORY_BUILDERS[name](args.gameweeks)
        expected_points = DEFAULT_EP

    league_history = LeagueHistory({1: history})
    prediction = predict_captain(
        rival_entry_id=1,
        rival_name=f"[SYNTHETIC] {name.replace('-', ' ').title()} Manager",
        current_squad=fx.CAPTAIN_POOL,
        target_gameweek=target_gw,
        league_history=league_history,
        config=PredictionConfig(),
        expected_points=expected_points,
    )
    print(render_captain_prediction(prediction, fx.PLAYER_NAMES))
    return 0


def run_pipeline_fixture(variant: str) -> int:
    dave_history = fx.dave_salah_loyal_history() if variant == "loyal" else fx.dave_balanced_history()
    chris, league, projections, candidates, dave, dave_league_history, expected_points = fx.full_pipeline_scenario(
        dave_history
    )

    prediction = predict_captain(
        rival_entry_id=dave.entry_id,
        rival_name=dave.name,
        current_squad=dave.starting_xi,
        target_gameweek=11,
        league_history=dave_league_history,
        config=PredictionConfig(),
        expected_points=expected_points,
    )
    print("STAGE 4 - RIVAL PREDICTION")
    print("=" * 40)
    print(render_captain_prediction(prediction, {**fx.PLAYER_NAMES}))

    dave_updated = apply_prediction_to_manager_state(dave, prediction)
    managers = tuple(dave_updated if m.entry_id == dave.entry_id else m for m in league.managers)
    league = replace(league, managers=managers)

    config = SimulationConfig(num_simulations=30_000, random_seed=42, rival_captain_mode="probabilistic")
    result = run_captain_battle(chris, league, projections, candidates, config)

    print("\n\nSTAGE 3 - CAPTAIN BATTLE (using Stage 4's predicted Dave captaincy)")
    print("=" * 40)
    print(render_captain_battle_report(result, num_managers=len(league.managers)))
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.fixture in PERSONA_HISTORY_BUILDERS or args.fixture == "ep-follower":
        return run_persona_fixture(args.fixture, args)
    if args.fixture == "pipeline-loyal":
        return run_pipeline_fixture("loyal")
    if args.fixture == "pipeline-balanced":
        return run_pipeline_fixture("balanced")
    print(f"Unknown fixture {args.fixture!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
