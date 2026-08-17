"""Stage 4.5 brief item 13: does fitting Stage 4's parameters actually
change what Stage 3 recommends?

Runs the SAME set of Stage 3 scenarios twice - once with the rival's
captain probabilities predicted by the manual (default) Stage 4 config,
once with the fitted config - and reports whether Captain Battle's
objective winner for Chris changes. This does not modify Stage 3; it only
calls its existing public functions (``run_captain_battle``) exactly the
way Stage 4's own ``stage3_adapter`` already does.

No causal claim from a single gameweek's outcome is made anywhere here -
this only measures whether the DECISION is sensitive to the fitting, not
whether the fitted model is "right" about any one real result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

from fpl_rival.prediction.models import LeagueHistory, PredictionConfig
from fpl_rival.prediction.prediction_engine import predict_captain
from fpl_rival.prediction.stage3_adapter import apply_prediction_to_manager_state
from fpl_rival.simulation.captain_battle import run_captain_battle
from fpl_rival.simulation.models import LeagueState, ManagerState, SimulationConfig


@dataclass(frozen=True)
class Stage3Scenario:
    label: str
    chris: ManagerState
    league: LeagueState
    projections: dict
    candidate_captain_ids: List[int]
    rival: ManagerState
    rival_history: LeagueHistory
    target_gameweek: int
    expected_points: Optional[Dict[int, float]] = None


@dataclass(frozen=True)
class Stage3ImpactRow:
    scenario_label: str
    manual_probabilities: Dict[int, float]
    fitted_probabilities: Dict[int, float]
    manual_objective_winner: str
    fitted_objective_winner: str
    recommendation_changed: bool


def _run_one(scenario: Stage3Scenario, config: PredictionConfig, sim_config: SimulationConfig):
    prediction = predict_captain(
        rival_entry_id=scenario.rival.entry_id,
        rival_name=scenario.rival.name,
        current_squad=scenario.rival.starting_xi,
        target_gameweek=scenario.target_gameweek,
        league_history=scenario.rival_history,
        config=config,
        expected_points=scenario.expected_points,
    )
    rival_updated = apply_prediction_to_manager_state(scenario.rival, prediction)
    managers = tuple(rival_updated if m.entry_id == scenario.rival.entry_id else m for m in scenario.league.managers)
    league = replace(scenario.league, managers=managers)
    battle = run_captain_battle(scenario.chris, league, scenario.projections, scenario.candidate_captain_ids, sim_config)
    return prediction, battle


def compare_stage3_impact(
    scenarios: List[Stage3Scenario],
    manual_config: PredictionConfig,
    fitted_config: PredictionConfig,
    sim_config: SimulationConfig = SimulationConfig(num_simulations=20_000, random_seed=42, rival_captain_mode="probabilistic"),
) -> List[Stage3ImpactRow]:
    rows = []
    for scenario in scenarios:
        manual_prediction, manual_battle = _run_one(scenario, manual_config, sim_config)
        fitted_prediction, fitted_battle = _run_one(scenario, fitted_config, sim_config)
        rows.append(
            Stage3ImpactRow(
                scenario_label=scenario.label,
                manual_probabilities=dict(manual_prediction.probabilities),
                fitted_probabilities=dict(fitted_prediction.probabilities),
                manual_objective_winner=manual_battle.objective_winner.player_name,
                fitted_objective_winner=fitted_battle.objective_winner.player_name,
                recommendation_changed=manual_battle.objective_winner.player_name != fitted_battle.objective_winner.player_name,
            )
        )
    return rows
