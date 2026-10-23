#!/usr/bin/env python3
"""FPL Rival - mini web dashboard.

A browser front end on top of the same live Captain Battle engine
``captain_battle_cli.py`` uses - no new analytics live here, this is
presentation only. Deliberately still hardcoded to one entry ID (yours)
rather than built out as a multi-user product; see README for why that's
a separate decision.

    python3 webapp.py
    -> open http://127.0.0.1:5000

This never performs a transfer or any other write action on your FPL
team - same rule as everywhere else in this project. It only calls
unauthenticated, public FPL endpoints and reports what the simulation
says; every change is yours to make manually in the official FPL app.
"""

from __future__ import annotations

from flask import Flask, render_template, request

from fpl_rival.api import FPLAPIError, FPLClient
from fpl_rival.live_captain_battle import (
    build_live_captain_battle_inputs,
    evaluate_bench_candidate,
    suggest_top_transfer_candidates,
)
from fpl_rival.simulation.captain_battle import DEFAULT_OBJECTIVE, OBJECTIVES, run_captain_battle
from fpl_rival.simulation.models import SimulationConfig

app = Flask(__name__)

CHRIS_ENTRY_ID = 5234355  # hardcoded by design - see module docstring

LEAGUES = {
    1132827: "TMHCC",
    400059: "Tokio Marine HCC League",
    1132822: "TMHCC - Past and Present",
    1132833: "League 1132833",
    1348570: "TMHCC Finance",
}
DEFAULT_LEAGUE_ID = 1132827

OBJECTIVE_LABELS = {
    "prob_finish_first": "P(finish 1st)",
    "prob_top3": "P(top 3)",
    "prob_move_up": "P(move up)",
    "expected_position": "Expected position",
}

NUM_SIMULATIONS = 20_000
RANDOM_SEED = 42
SUGGEST_TRANSFERS_COUNT = 3


@app.route("/")
def dashboard():
    try:
        league_id = int(request.args.get("league_id", DEFAULT_LEAGUE_ID))
    except ValueError:
        league_id = DEFAULT_LEAGUE_ID
    if league_id not in LEAGUES:
        league_id = DEFAULT_LEAGUE_ID

    objective = request.args.get("objective", DEFAULT_OBJECTIVE)
    if objective not in OBJECTIVES:
        objective = DEFAULT_OBJECTIVE

    context = {
        "leagues": LEAGUES,
        "selected_league_id": league_id,
        "objectives": OBJECTIVE_LABELS,
        "selected_objective": objective,
        "error": None,
    }

    client = FPLClient()
    try:
        bootstrap = client.get_bootstrap_static()
    except FPLAPIError as exc:
        context["error"] = f"Could not reach the FPL API: {exc}"
        return render_template("dashboard.html", **context)

    inputs = build_live_captain_battle_inputs(client, league_id, CHRIS_ENTRY_ID, bootstrap)
    if inputs.chris is None:
        context["error"] = "Could not build a live Captain Battle for this league: " + "; ".join(inputs.errors)
        return render_template("dashboard.html", **context)

    config = SimulationConfig(num_simulations=NUM_SIMULATIONS, random_seed=RANDOM_SEED, rival_captain_mode="probabilistic")
    result = run_captain_battle(
        inputs.chris, inputs.league, inputs.projections, inputs.candidate_captain_ids, config,
        objective=objective, primary_rival_entry_ids=inputs.primary_rival_ids,
    )

    higher_is_better = OBJECTIVES[objective]
    best_in_squad = result.objective_winner
    overall_best_name = best_in_squad.player_name
    overall_best_result = best_in_squad.result

    def maybe_update_best(name, r):
        nonlocal overall_best_name, overall_best_result
        metric = getattr(r, objective)
        best_metric = getattr(overall_best_result, objective)
        if (metric > best_metric) if higher_is_better else (metric < best_metric):
            overall_best_name, overall_best_result = name, r

    bench_options = []
    for player_id in inputs.chris.bench:
        try:
            comparison = evaluate_bench_candidate(inputs, player_id, config, objective)
        except Exception as exc:  # pragma: no cover - defensive, shown to the user instead of hidden
            inputs.errors.append(f"bench player {player_id}: could not be simulated ({exc}).")
            continue
        bench_options.append(comparison)
        maybe_update_best(comparison.result.candidates[0].player_name, comparison.result.candidates[0].result)

    suggestions = suggest_top_transfer_candidates(inputs, config, objective, top_n=SUGGEST_TRANSFERS_COUNT)
    for comparison in suggestions:
        maybe_update_best(comparison.result.candidates[0].player_name, comparison.result.candidates[0].result)

    rival_names = [m.name for m in inputs.league.managers if m.entry_id in inputs.predicted_rival_ids]
    by_id = {c.player_id: c for c in result.candidates}
    ranked_candidates = [by_id[pid] for pid in result.ranking_by_objective]

    context.update(
        league_name=inputs.league.league_name,
        num_managers=len(inputs.league.managers),
        target_gameweek=inputs.target_gameweek,
        result=result,
        ranked_candidates=ranked_candidates,
        best_in_squad=best_in_squad,
        bench_options=bench_options,
        suggestions=suggestions,
        rival_names=rival_names,
        overall_best_name=overall_best_name,
        overall_best_result=overall_best_result,
        beats_in_squad=(overall_best_name != best_in_squad.player_name),
        num_simulations=NUM_SIMULATIONS,
        errors=inputs.errors,
    )
    return render_template("dashboard.html", **context)


if __name__ == "__main__":
    app.run(debug=True)
