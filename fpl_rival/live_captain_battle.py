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
docstring for why that distinction matters):

* a position-based spread - attacking returns are lumpier than defensive
  ones, so MID/FWD get a wider standard deviation than GK/DEF for the same
  mean;
* a fixture-difficulty adjustment - FPL's own ``ep_next`` does NOT reliably
  discount for a hard fixture (observed directly: a GW6 Leeds player away
  at Arsenal, FDR 5, still carried an ep_next *above* his season points-per-
  game), so this module fetches that gameweek's fixtures and scales the
  mean by each team's difficulty rating, summed across fixtures (so a
  double gameweek roughly doubles the mean and a blank gameweek collapses
  it to near zero). This is still a simple heuristic, not a real model of
  clean-sheet or goal-involvement probability - it exists because ignoring
  fixture difficulty entirely was clearly wrong, not because this is
  precise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from fpl_rival.api import FPLAPIError, FPLClient
from fpl_rival.intelligence.collect import collect_league_data
from fpl_rival.intelligence.models import GameContext, ManagerData
from fpl_rival.prediction.models import CaptainObservation, LeagueHistory, PredictionConfig
from fpl_rival.prediction.prediction_engine import predict_captain
from fpl_rival.prediction.stage3_adapter import apply_prediction_to_manager_state
from fpl_rival.simulation.captain_battle import OBJECTIVES, CaptainBattleResult, run_captain_battle
from fpl_rival.simulation.models import LeagueState, ManagerState, PlayerProjection, SimulationConfig
from fpl_rival.simulation.relevance_adapter import default_primary_rival_ids

MIN_STDDEV = 1.5
FALLBACK_MEAN = 2.0
STDDEV_RATIO_BY_ELEMENT_TYPE = {1: 0.55, 2: 0.65, 3: 0.85, 4: 0.90}  # GK, DEF, MID, FWD
DEFAULT_STDDEV_RATIO = 0.8

FDR_ADJUSTMENT_PER_LEVEL = 0.08  # fraction of mean shifted per FDR point away from neutral (3)
FDR_FACTOR_FLOOR = 0.4  # even FDR 5 never discounts a player to less than 40% of their base mean
BLANK_GAMEWEEK_MEAN = 0.1  # team confirmed to have no fixture this gameweek


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
    # Carried for evaluate_transfer_candidate, so it can project an
    # arbitrary player without re-fetching bootstrap/fixtures data.
    ctx: Optional[GameContext] = None
    ep_by_element: Dict[int, float] = field(default_factory=dict)
    team_by_element: Dict[int, int] = field(default_factory=dict)
    fixture_difficulty_by_team: Dict[int, List[int]] = field(default_factory=dict)
    status_by_element: Dict[int, str] = field(default_factory=dict)  # FPL 'status': a=available, d=doubtful, i/s/u=unavailable


def find_target_gameweek(bootstrap: dict) -> int:
    """The gameweek to run the battle for: FPL's own "next" event, or one
    past the latest finished event if nothing is flagged (e.g. season end)."""
    events = bootstrap.get("events", [])
    next_event = next((e["id"] for e in events if e.get("is_next")), None)
    if next_event is not None:
        return next_event
    finished = [e["id"] for e in events if e.get("finished")]
    return (max(finished) + 1) if finished else 1


def _fixture_difficulty_by_team(client: FPLClient, target_gameweek: int, errors: List[str]) -> Dict[int, List[int]]:
    """team_id -> difficulty rating(s) (1 easiest..5 hardest) for every
    fixture that team plays in the target gameweek. A team missing from the
    result has a confirmed blank gameweek IF fixtures loaded at all -
    an empty overall dict instead means the fetch failed, so callers must
    not apply any adjustment in that case (see _fixture_adjusted_mean)."""
    try:
        fixtures = client.get_fixtures(event=target_gameweek)
    except FPLAPIError as exc:
        errors.append(f"fixtures for GW{target_gameweek}: {exc} - projections will not be fixture-adjusted.")
        return {}
    by_team: Dict[int, List[int]] = {}
    for f in fixtures:
        by_team.setdefault(f["team_h"], []).append(f.get("team_h_difficulty", 3))
        by_team.setdefault(f["team_a"], []).append(f.get("team_a_difficulty", 3))
    return by_team


def _fixture_adjusted_mean(base_mean: float, team_id: Optional[int], difficulty_by_team: Dict[int, List[int]]) -> float:
    if not difficulty_by_team or team_id is None:
        return base_mean  # no fixture data available at all - leave unadjusted rather than guess
    difficulties = difficulty_by_team.get(team_id)
    if difficulties is None:
        return BLANK_GAMEWEEK_MEAN  # confirmed blank gameweek for this team
    total = 0.0
    for fdr in difficulties:
        factor = max(FDR_FACTOR_FLOOR, 1 - FDR_ADJUSTMENT_PER_LEVEL * (fdr - 3))
        total += base_mean * factor
    return total


def _project_player(
    element_id: int,
    ctx: GameContext,
    ep_by_element: Dict[int, float],
    team_by_element: Dict[int, int],
    fixture_difficulty_by_team: Dict[int, List[int]],
) -> PlayerProjection:
    mean = ep_by_element.get(element_id) or FALLBACK_MEAN
    if mean <= 0:
        mean = FALLBACK_MEAN
    mean = max(0.0, _fixture_adjusted_mean(mean, team_by_element.get(element_id), fixture_difficulty_by_team))
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
    extra_candidate_ids: Tuple[int, ...] = (),
    prediction_config: PredictionConfig = PredictionConfig(),
) -> LiveCaptainBattleInputs:
    """Assembles everything ``run_captain_battle`` needs from live FPL data.

    ``extra_candidate_ids`` lets a caller force specific players into the
    comparison (e.g. one you're considering that didn't make the top
    ``max_candidates`` by projected points) - they must already be in your
    starting XI; Stage 3 never evaluates a transfer-in from your own squad
    comparison, only players you currently own (see run_captain_battle's own
    restriction) - see ``evaluate_transfer_candidate`` for comparing in an
    external player instead.
    """
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
    team_by_element: Dict[int, int] = {}
    status_by_element: Dict[int, str] = {}
    for element in bootstrap.get("elements", []):
        eid = element["id"]
        team_by_element[eid] = element.get("team")
        status_by_element[eid] = element.get("status", "a")
        try:
            ep_by_element[eid] = float(element.get("ep_next") or 0)
        except (TypeError, ValueError):
            ep_by_element[eid] = 0.0
    fixture_difficulty_by_team = _fixture_difficulty_by_team(client, target_gameweek, errors)

    required_ids: set = set()
    for state in manager_states.values():
        required_ids.update(state.starting_xi)
        if state.vice_captain_id is not None:
            required_ids.add(state.vice_captain_id)
    projections = {
        eid: _project_player(eid, ctx, ep_by_element, team_by_element, fixture_difficulty_by_team) for eid in required_ids
    }

    chris = manager_states[chris_entry_id]
    league = LeagueState(league_id=league_data.league_id, league_name=league_data.league_name, managers=tuple(manager_states.values()))

    valid_extra_ids = tuple(eid for eid in extra_candidate_ids if eid in chris.starting_xi)
    for eid in extra_candidate_ids:
        if eid not in chris.starting_xi:
            errors.append(f"extra candidate {eid} ({ctx.player_name(eid)}) is not in your starting XI - ignored.")

    ranked_xi = sorted(chris.starting_xi, key=lambda eid: -projections[eid].projected_mean_points)
    candidate_captain_ids = list(dict.fromkeys(list(valid_extra_ids) + [chris.captain_id] + ranked_xi[:max_candidates]))
    candidate_captain_ids.sort(key=lambda eid: -projections[eid].projected_mean_points)

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
        ctx=ctx,
        ep_by_element=ep_by_element,
        team_by_element=team_by_element,
        fixture_difficulty_by_team=fixture_difficulty_by_team,
        status_by_element=status_by_element,
    )


POSITION_LABELS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


@dataclass
class TransferComparisonResult:
    result: CaptainBattleResult  # single-candidate: the external player, captained
    replaced_player_id: int
    replaced_player_name: str
    rationale: str  # explains WHY that particular player was chosen to be replaced


def evaluate_transfer_candidate(
    inputs: LiveCaptainBattleInputs,
    external_player_id: int,
    config: SimulationConfig,
    objective: str,
) -> TransferComparisonResult:
    """Answers "if I transferred X in and captained them, instead of my
    current squad's best option, how would that compare?" - a hypothetical,
    not a transfer recommendation: it does NOT check price, budget, or squad
    legality (formation, club limits). It swaps ``external_player_id`` into
    your starting XI in place of whichever current starter of the same
    position has the lowest projection (falling back to the lowest
    projection overall if no same-position starter exists), sets them as
    captain, and runs the SAME live league/rivals context through Stage 3.
    """
    if inputs.chris is None or inputs.ctx is None:
        raise ValueError("inputs must come from a successful build_live_captain_battle_inputs() call.")

    ctx = inputs.ctx
    projection = _project_player(
        external_player_id, ctx, inputs.ep_by_element, inputs.team_by_element, inputs.fixture_difficulty_by_team
    )
    external_type = ctx.element_types.get(external_player_id)
    position_label = POSITION_LABELS.get(external_type, "player")

    same_position = [eid for eid in inputs.chris.starting_xi if ctx.element_types.get(eid) == external_type]
    pool = same_position or list(inputs.chris.starting_xi)
    pool_sorted = sorted(pool, key=lambda eid: inputs.projections.get(eid, projection).projected_mean_points)
    replaced_id = pool_sorted[0]
    replaced_name = ctx.player_name(replaced_id)

    if same_position:
        comparisons = ", ".join(f"{ctx.player_name(eid)} {inputs.projections[eid].projected_mean_points:.1f}" for eid in pool_sorted)
        rationale = (
            f"Replaced {replaced_name} - your weakest projected {position_label} this gameweek "
            f"(fixture-adjusted points, lowest to highest: {comparisons})."
        )
    else:
        rationale = (
            f"You have no {position_label} in your starting XI, so {replaced_name} - your weakest "
            f"projected starter overall ({inputs.projections[replaced_id].projected_mean_points:.1f} pts) - was replaced instead."
        )

    new_starting_xi = tuple(external_player_id if eid == replaced_id else eid for eid in inputs.chris.starting_xi)
    hypothetical_chris = ManagerState(
        entry_id=inputs.chris.entry_id,
        name=inputs.chris.name,
        current_total_points=inputs.chris.current_total_points,
        current_league_position=inputs.chris.current_league_position,
        starting_xi=new_starting_xi,
        captain_id=external_player_id,
        vice_captain_id=inputs.chris.vice_captain_id if inputs.chris.vice_captain_id != replaced_id else None,
        bench=inputs.chris.bench,
    )
    hypothetical_league = LeagueState(
        league_id=inputs.league.league_id,
        league_name=inputs.league.league_name,
        managers=tuple(hypothetical_chris if m.entry_id == inputs.chris.entry_id else m for m in inputs.league.managers),
    )
    extended_projections = dict(inputs.projections)
    extended_projections[external_player_id] = projection

    result = run_captain_battle(
        hypothetical_chris, hypothetical_league, extended_projections, [external_player_id],
        config=config, objective=objective, primary_rival_entry_ids=inputs.primary_rival_ids,
    )
    return TransferComparisonResult(result=result, replaced_player_id=replaced_id, replaced_player_name=replaced_name, rationale=rationale)


UNAVAILABLE_STATUSES = {"i", "s", "u", "n"}  # injured, suspended, unavailable, not registered (excludes 'a'/'d')
DEFAULT_SHORTLIST_SIZE = 15


def suggest_top_transfer_candidates(
    inputs: LiveCaptainBattleInputs,
    config: SimulationConfig,
    objective: str,
    top_n: int = 3,
    shortlist_size: int = DEFAULT_SHORTLIST_SIZE,
) -> List[TransferComparisonResult]:
    """Searches every player you don't already own for the ``top_n`` who
    would most improve your chances against your rivals if captained,
    exactly as ``evaluate_transfer_candidate`` evaluates one named player -
    same hypothetical, same caveat: price, budget and squad legality are
    NOT checked here either, and this never performs a transfer - it only
    ever reports what a full simulation says, for you to act on or not.

    Full Stage 3 simulations are expensive to run for every eligible
    player, so this pre-ranks everyone by their (cheap) fixture-adjusted
    projection first and only fully simulates the top ``shortlist_size`` -
    a genuinely elite differential could in principle rank outside that
    shortlist on projection alone but still win a simulation; widen
    ``shortlist_size`` if you want a more exhaustive (slower) search.
    """
    if inputs.chris is None or inputs.ctx is None:
        raise ValueError("inputs must come from a successful build_live_captain_battle_inputs() call.")

    ctx = inputs.ctx
    owned_ids = set(inputs.chris.starting_xi) | set(inputs.chris.bench)
    eligible_ids = [
        eid for eid in inputs.ep_by_element
        if eid not in owned_ids and inputs.status_by_element.get(eid, "a") not in UNAVAILABLE_STATUSES
    ]

    shortlist = sorted(
        eligible_ids,
        key=lambda eid: -_project_player(eid, ctx, inputs.ep_by_element, inputs.team_by_element, inputs.fixture_difficulty_by_team).projected_mean_points,
    )[:shortlist_size]

    higher_is_better = OBJECTIVES[objective]
    evaluated: List[TransferComparisonResult] = []
    for player_id in shortlist:
        try:
            evaluated.append(evaluate_transfer_candidate(inputs, player_id, config, objective))
        except Exception as exc:
            inputs.errors.append(f"player {player_id} ({ctx.player_name(player_id)}): could not be simulated as a transfer candidate ({exc}).")

    evaluated.sort(key=lambda c: getattr(c.result.candidates[0].result, objective), reverse=higher_is_better)
    return evaluated[:top_n]
