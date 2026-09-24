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
from fpl_rival.live_captain_battle import (
    build_live_captain_battle_inputs,
    evaluate_transfer_candidate,
    suggest_top_transfer_candidates,
)
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
    parser.add_argument(
        "--extra-candidate", action="append", default=[], metavar="NAME_OR_ID",
        help="Force a specific player from your starting XI into the comparison (by FPL web name or player id). Repeatable.",
    )
    parser.add_argument(
        "--compare-transfer-in", action="append", default=[], metavar="NAME_OR_ID",
        help=(
            "Compare captaining a player NOT currently in your squad, as if you'd transferred them in "
            "(swapped for your weakest same-position starter - price/budget/squad legality are not checked, "
            "that's left to you). By FPL web name or player id. Repeatable."
        ),
    )
    parser.add_argument(
        "--suggest-transfers", action="store_true",
        help=(
            "Search every player you don't own for the best transfer-in-and-captain options against your "
            "rivals (same hypothetical as --compare-transfer-in, price/budget/legality still not checked - "
            "this never performs a transfer, it only reports what the simulation says)."
        ),
    )
    parser.add_argument("--suggest-transfers-count", type=int, default=3, help="How many suggestions to show.")
    return parser.parse_args(argv)


def resolve_player_ids(values: list, bootstrap: dict) -> tuple:
    """Resolves each value to an element id: an integer is used as-is, a
    name is matched case-insensitively against FPL's web_name. Ambiguous or
    unknown names are reported and skipped, not guessed."""
    by_name: dict = {}
    for element in bootstrap.get("elements", []):
        by_name.setdefault(element.get("web_name", "").lower(), []).append(element["id"])

    resolved = []
    for value in values:
        try:
            resolved.append(int(value))
            continue
        except ValueError:
            pass
        matches = by_name.get(value.strip().lower(), [])
        if len(matches) == 1:
            resolved.append(matches[0])
        elif len(matches) > 1:
            print(f"'{value}' matches {len(matches)} players (ids {matches}) - use the numeric id instead. Skipped.")
        else:
            print(f"'{value}' did not match any player name. Skipped.")
    return tuple(resolved)


def main(argv=None) -> int:
    args = parse_args(argv)
    client = FPLClient()

    print("Fetching bootstrap data ...")
    try:
        bootstrap = client.get_bootstrap_static()
    except FPLAPIError as exc:
        print(f"FATAL: could not retrieve bootstrap-static: {exc}")
        return 1

    extra_candidate_ids = resolve_player_ids(args.extra_candidate, bootstrap)
    transfer_target_ids = resolve_player_ids(args.compare_transfer_in, bootstrap)

    print(f"Collecting live data for league {args.league_id} (this can take a while for large leagues) ...")
    inputs = build_live_captain_battle_inputs(
        client, args.league_id, args.entry, bootstrap, max_managers=args.max_managers, extra_candidate_ids=extra_candidate_ids,
    )

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

    best_in_squad = result.objective_winner
    higher_is_better = OBJECTIVES[args.objective]
    overall_best = (best_in_squad.player_name, best_in_squad.result)

    def print_transfer_result(label: str, comparison) -> None:
        candidate = comparison.result.candidates[0]
        r = candidate.result
        print(f"\n{label} Captain {candidate.player_name.upper()} (transferred in for {comparison.replaced_player_name}):")
        print(f"  {comparison.rationale}")
        print(f"  Expected GW points: {r.expected_gameweek_points:.1f}")
        print(f"  P(finish 1st): {100 * r.prob_finish_first:.1f}%")
        print(f"  P(move up): {100 * r.prob_move_up:.1f}%")
        print(f"  Expected position: {r.expected_position:.1f}")

    def maybe_update_best(overall_best, player_name, r) -> tuple:
        metric = getattr(r, args.objective)
        best_metric = getattr(overall_best[1], args.objective)
        is_better = metric > best_metric if higher_is_better else metric < best_metric
        return (player_name, r) if is_better else overall_best

    if transfer_target_ids:
        print("\n" + "=" * 40)
        print("TRANSFER-IN COMPARISON (hypothetical - price/budget not checked)")
        print("=" * 40)
        for player_id in transfer_target_ids:
            try:
                comparison = evaluate_transfer_candidate(inputs, player_id, config, args.objective)
            except Exception as exc:
                print(f"\nCould not evaluate player {player_id}: {exc}")
                continue
            print_transfer_result("NAMED -", comparison)
            candidate = comparison.result.candidates[0]
            overall_best = maybe_update_best(overall_best, candidate.player_name, candidate.result)

    if args.suggest_transfers:
        print("\n" + "=" * 40)
        print(f"TOP {args.suggest_transfers_count} TRANSFER SUGGESTIONS (auto-searched, hypothetical - price/budget not checked)")
        print("=" * 40)
        suggestions = suggest_top_transfer_candidates(inputs, config, args.objective, top_n=args.suggest_transfers_count)
        if not suggestions:
            print("\nNo eligible transfer candidates could be evaluated.")
        for rank, comparison in enumerate(suggestions, start=1):
            print_transfer_result(f"#{rank} -", comparison)
            candidate = comparison.result.candidates[0]
            overall_best = maybe_update_best(overall_best, candidate.player_name, candidate.result)

    if transfer_target_ids or args.suggest_transfers:
        print(f"\nBest option overall ({args.objective}): {overall_best[0]}")
        if overall_best[0] != best_in_squad.player_name:
            print(f"(Beats your best in-squad option, {best_in_squad.player_name}, on this metric.)")
        else:
            print(f"(Your best in-squad option, {best_in_squad.player_name}, still wins - nothing compared here beat it.)")

    if inputs.errors:
        print(f"\n{len(inputs.errors)} data gap(s)/note(s):")
        for e in inputs.errors[:20]:
            print(f"  - {e}")
        if len(inputs.errors) > 20:
            print(f"  ... and {len(inputs.errors) - 20} more.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
