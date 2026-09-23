"""Bridges live FPL retrieval (Stage 1/2) into Stage 3's Captain Battle
simulator and, for the automatically-identified primary rivals, Stage 4's
prediction engine.

Stage 3 and Stage 4's own CLIs (simulation_cli.py, prediction_cli.py) are
synthetic-only by design - this module is the one place their live inputs
get assembled, reusing every stage's existing retrieval/logic unmodified:
Stage 2's collect_league_data for retrieval, Stage 3's own relevance_adapter
for picking primary rivals, Stage 4's prediction_engine + stage3_adapter for
turning a rival's real history into a captain probability distribution.

Player projections are a simple heuristic built from the FPL API's own
``ep_next`` figure (not a Stage 3 statistical model - see PlayerProjection's
docstring for why that distinction matters), with a position-based spread:
attacking returns are lumpier than defensive ones, so MID/FWD get a wider
standard deviation than GK/DEF for the same mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from fpl_rival.api import FPLClient
from fpl_rival.intelligence.collect import collect_league_data
from fpl_rival.intelligence.models import GameContext, ManagerData
from fpl_rival.prediction.models import CaptainObservation, LeagueHistory, PredictionConfig
from fpl_rival.prediction.prediction_engine import predict_captain
from fpl_rival.prediction.stage3_adapter import apply_prediction_to_manager_state
from fpl_rival.simulation.models import LeagueState, ManagerState, PlayerProjection
from fpl_rival.simulation.relevance_adapter import default_primary_rival_ids

MIN_STDDEV = 1.5
FALLBACK_MEAN = 2.0
STDDEV_RATIO_BY_ELEMENT_TYPE = {1: 0.55, 2: 0.65, 3: 0.85, 4: 0.90}  # GK, DEF, MID, FWD
DEFAULT_STDDEV_RATIO = 0.8


@dataclass
class LiveCaptainBattleInputs:
    chris: Optional[ManagerState]
    league: Optional[LeagueState]
    projections: Dict[int, PlayerProjection]
    candidate_captain_ids: List[int]
    primary_rival_ids: List[int]
    predicted_rival_ids: List[int]  # subset of primary_rival_ids with a real Stage 4 prediction applied
    target_gameweek: int
    errors: List[str]


def find_target_gameweek(bootstrap: dict) -> int:
    """The gameweek to run the battle for: FPL's own "next" event, or one
    past the latest finished event if nothing is flagged (e.g. season end)."""
    events = bootstrap.get("events", [])
    next_event = next((e["id"] for e in events if e.get("is_next")), None)
    if next_event is not None:
        return next_event
    finished = [e["id"] for e in events if e.get("finished")]
    return (max(finished) + 1) if finished else 1


def _project_player(element_id: int, ctx: GameContext, ep_by_element: Dict[int, float]) -> PlayerProjection:
    mean = ep_by_element.get(element_id) or FALLBACK_MEAN
    if mean <= 0:
        mean = FALLBACK_MEAN
    ratio = STDDEV_RATIO_BY_ELEMENT_TYPE.get(ctx.element_types.get(element_id), DEFAULT_STDDEV_RATIO)
    return PlayerProjection(
        player_id=element_id,
        player_name=ctx.player_name(element_id),
        projected_mean_points=float(mean),
        projected_standard_deviation=max(MIN_STDDEV, mean * ratio),
    )


def _manager_state_from_latest_squad(manager: ManagerData, errors: List[str]) -> Optional[ManagerState]:
    squad = manager.latest_squad
    if squad is None or not squad.starting_xi:
        errors.append(f"entry {manager.entry_id} ({manager.team_name}): no squad available - excluded from the simulation.")
        return None
    starting_ids = tuple(p.element for p in squad.starting_xi)
    captain_id = squad.captain_element
    if captain_id is None or captain_id not in starting_ids:
        errors.append(f"entry {manager.entry_id} ({manager.team_name}): no valid captain in starting XI - excluded from the simulation.")
        return None
    return ManagerState(
        entry_id=manager.entry_id,
        name=manager.team_name or f"Manager {manager.entry_id}",
        current_total_points=float(manager.total_points or 0),
        current_league_position=manager.league_position or 0,
        starting_xi=starting_ids,
        captain_id=captain_id,
        vice_captain_id=squad.vice_captain_element,
        bench=tuple(p.element for p in squad.bench),
    )


def _captain_observations(manager: ManagerData) -> Tuple[CaptainObservation, ...]:
    """Every one of this manager's past finished gameweeks as a
    CaptainObservation - the same conversion live_import.py does for Stage
    4.5's calibration CSVs, targeting Stage 4's own history shape instead."""
    history_by_event = {row.event: row for row in manager.gameweek_history}
    observations = []
    for event, squad in sorted(manager.squads_by_event.items()):
        starting_ids = tuple(p.element for p in squad.starting_xi)
        captain_id = squad.captain_element
        if not starting_ids or captain_id is None or captain_id not in starting_ids:
            continue
        row = history_by_event.get(event)
        observations.append(
            CaptainObservation(
                gameweek=event,
                owned_player_ids=starting_ids,
                captain_id=captain_id,
                vice_captain_id=squad.vice_captain_element,
                transfers=(row.event_transfers if row and row.event_transfers is not None else 0),
                points=(squad.points if squad.points is not None else (row.points if row else None)),
                rank=(row.rank if row else None),
            )
        )
    return tuple(observations)


def build_live_captain_battle_inputs(
    client: FPLClient,
    league_id: int,
    chris_entry_id: int,
    bootstrap: dict,
    max_managers: int = 50,
    max_candidates: int = 4,
    prediction_config: PredictionConfig = PredictionConfig(),
) -> LiveCaptainBattleInputs:
    """Assembles everything ``run_captain_battle`` needs from live FPL data."""
    league_data, ctx, errors = collect_league_data(
        client, league_id, bootstrap, chris_entry_id=chris_entry_id, max_managers=max_managers, fetch_full_picks_history=True,
    )
    target_gameweek = find_target_gameweek(bootstrap)

    manager_states: Dict[int, ManagerState] = {}
    for manager in league_data.managers:
        state = _manager_state_from_latest_squad(manager, errors)
        if state is not None:
            manager_states[manager.entry_id] = state

    if chris_entry_id not in manager_states:
        errors.append(f"entry {chris_entry_id}: no usable current squad - cannot run a live captain battle.")
        return LiveCaptainBattleInputs(None, None, {}, [], [], [], target_gameweek, errors)

    ep_by_element: Dict[int, float] = {}
    for element in bootstrap.get("elements", []):
        try:
            ep_by_element[element["id"]] = float(element.get("ep_next") or 0)
        except (TypeError, ValueError):
            ep_by_element[element["id"]] = 0.0

    required_ids: set = set()
    for state in manager_states.values():
        required_ids.update(state.starting_xi)
        if state.vice_captain_id is not None:
            required_ids.add(state.vice_captain_id)
    projections = {eid: _project_player(eid, ctx, ep_by_element) for eid in required_ids}

    chris = manager_states[chris_entry_id]
    league = LeagueState(league_id=league_data.league_id, league_name=league_data.league_name, managers=tuple(manager_states.values()))

    ranked_xi = sorted(chris.starting_xi, key=lambda eid: -projections[eid].projected_mean_points)
    candidate_captain_ids = list(dict.fromkeys([chris.captain_id] + ranked_xi[:max_candidates]))[:max_candidates]

    primary_rival_ids = default_primary_rival_ids(chris, league)
    expected_points = {eid: proj.projected_mean_points for eid, proj in projections.items()}
    league_history = LeagueHistory({m.entry_id: _captain_observations(m) for m in league_data.managers})

    predicted_rival_ids: List[int] = []
    updated_states = dict(manager_states)
    for rival_id in primary_rival_ids:
        rival_state = manager_states.get(rival_id)
        if rival_state is None or not league_history.for_entry(rival_id):
            continue  # no real history yet - stays in fixed mode with their actual current captain
        try:
            prediction = predict_captain(
                rival_entry_id=rival_id,
                rival_name=rival_state.name,
                current_squad=rival_state.starting_xi,
                target_gameweek=target_gameweek,
                league_history=league_history,
                config=prediction_config,
                expected_points=expected_points,
            )
            updated_states[rival_id] = apply_prediction_to_manager_state(rival_state, prediction)
            predicted_rival_ids.append(rival_id)
        except Exception as exc:
            errors.append(f"entry {rival_id}: Stage 4 prediction failed ({exc}) - kept in fixed captain mode.")

    league = LeagueState(league_id=league_data.league_id, league_name=league_data.league_name, managers=tuple(updated_states.values()))
    chris = updated_states[chris_entry_id]

    return LiveCaptainBattleInputs(
        chris=chris,
        league=league,
        projections=projections,
        candidate_captain_ids=candidate_captain_ids,
        primary_rival_ids=primary_rival_ids,
        predicted_rival_ids=predicted_rival_ids,
        target_gameweek=target_gameweek,
        errors=errors,
    )
