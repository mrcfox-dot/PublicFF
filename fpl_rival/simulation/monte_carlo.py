"""The Monte Carlo engine: sample -> score every manager -> rerank -> aggregate.

Tie rule (documented, not silently invented): a manager's simulated league
position is ``1 + (number of other managers with a STRICTLY greater
simulated total)``. Tied managers therefore share the same (better)
position - this is "competition ranking" (1, 1, 3, 4, ...), the same
convention as a sports league table before any secondary tiebreaker is
applied. FPL's own real tiebreakers (e.g. total team value) are NOT
implemented - a genuine points tie is left as a tie. Chris is considered
to "finish 1st" whenever his rank computed this way equals 1, including a
shared 1st place.

Performance note: this module computes NumPy arrays of shape
``(num_simulations,)`` per manager and only ever aggregates them down to
scalars for the returned result - per the brief, individual simulation
draws are not retained after a call returns (nothing here is "unnecessarily
storing every simulation" beyond the arrays needed for that one call).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np

from .manager_outcomes import simulate_manager_gameweek
from .models import LeagueState, PlayerProjection, SimulationConfig
from .player_outcomes import sample_all_players
from .rival_behaviour import resolve_captain_ids_by_entry


def collect_required_player_ids(league: LeagueState, extra_ids: Iterable[int] = ()) -> set:
    """Every player_id whose projection is needed to simulate this league:
    every manager's starting XI, vice-captain (for the fallback rule), any
    rival captain-distribution players, plus any extra ids the caller needs
    (e.g. Captain Battle's candidate list, which might not all be in every
    manager's XI)."""
    ids: set = set(extra_ids)
    for manager in league.managers:
        ids.update(manager.starting_xi)
        if manager.vice_captain_id is not None:
            ids.add(manager.vice_captain_id)
        if manager.captain_probabilities:
            ids.update(manager.captain_probabilities.keys())
    return ids


@dataclass(frozen=True)
class SimulationResult:
    entry_id: int
    expected_gameweek_points: float
    prob_finish_first: float
    prob_top3: float
    prob_move_up: float
    prob_move_down: float
    expected_position: float
    expected_leader_gap: float
    rival_overtake_prob: Dict[int, float] = field(default_factory=dict)  # P(chris finishes ahead of rival)
    rival_ahead_prob: Dict[int, float] = field(default_factory=dict)  # P(rival finishes ahead of chris)
    num_simulations: int = 0
    random_seed: int = 0


def compute_league_outcome(
    league: LeagueState,
    player_points: Dict[int, np.ndarray],
    captain_ids_by_entry: Dict[int, np.ndarray],
    chris_entry_id: int,
    num_simulations: int,
    primary_rival_entry_ids: Optional[Iterable[int]] = None,
    random_seed: int = 0,
) -> SimulationResult:
    """Steps 3-6 of the Monte Carlo loop (score everyone, add to current
    total, rerank, aggregate), given already-sampled player outcomes and
    already-resolved captain choices for every manager."""
    gw_scores: Dict[int, np.ndarray] = {}
    for manager in league.managers:
        gw_scores[manager.entry_id] = simulate_manager_gameweek(
            manager, player_points, captain_ids_by_entry[manager.entry_id]
        )

    totals = {
        m.entry_id: m.current_total_points + gw_scores[m.entry_id] for m in league.managers
    }

    chris = league.manager(chris_entry_id)
    chris_totals = totals[chris_entry_id]

    totals_matrix = np.stack(list(totals.values()))  # (n_managers, n_sims)
    leader_totals = np.max(totals_matrix, axis=0)

    chris_rank = np.ones(num_simulations, dtype=int)
    for entry_id, tot in totals.items():
        if entry_id == chris_entry_id:
            continue
        chris_rank += (tot > chris_totals).astype(int)

    if primary_rival_entry_ids is None:
        primary_rival_entry_ids = [m.entry_id for m in league.managers if m.entry_id != chris_entry_id]

    rival_overtake: Dict[int, float] = {}
    rival_ahead: Dict[int, float] = {}
    for rival_id in primary_rival_entry_ids:
        if rival_id not in totals:
            continue
        rival_totals = totals[rival_id]
        rival_overtake[rival_id] = float(np.mean(chris_totals > rival_totals))
        rival_ahead[rival_id] = float(np.mean(rival_totals > chris_totals))

    return SimulationResult(
        entry_id=chris_entry_id,
        expected_gameweek_points=float(np.mean(gw_scores[chris_entry_id])),
        prob_finish_first=float(np.mean(chris_rank == 1)),
        prob_top3=float(np.mean(chris_rank <= 3)),
        prob_move_up=float(np.mean(chris_rank < chris.current_league_position)),
        prob_move_down=float(np.mean(chris_rank > chris.current_league_position)),
        expected_position=float(np.mean(chris_rank)),
        expected_leader_gap=float(np.mean(leader_totals - chris_totals)),
        rival_overtake_prob=rival_overtake,
        rival_ahead_prob=rival_ahead,
        num_simulations=num_simulations,
        random_seed=random_seed,
    )


def run_simulation(
    league: LeagueState,
    projections: Dict[int, PlayerProjection],
    chris_entry_id: int,
    chris_captain_id: int,
    config: SimulationConfig = SimulationConfig(),
    primary_rival_entry_ids: Optional[Iterable[int]] = None,
) -> SimulationResult:
    """Standalone single-scenario entry point: samples fresh player outcomes
    and rival captains, forces Chris's captain to ``chris_captain_id``, and
    returns the aggregated result. For comparing multiple candidate
    captains under identical randomness, use ``captain_battle.py`` instead
    - it reuses the same sampled player outcomes across candidates
    (a "common random numbers" variance-reduction technique), which this
    single-shot function does not attempt.
    """
    rng = np.random.default_rng(config.random_seed)
    required_ids = collect_required_player_ids(league, extra_ids=[chris_captain_id])
    player_points = sample_all_players(required_ids, projections, config.num_simulations, rng, config.min_player_points)

    captain_ids_by_entry = resolve_captain_ids_by_entry(league, config.num_simulations, rng, config.rival_captain_mode)
    captain_ids_by_entry[chris_entry_id] = np.full(config.num_simulations, chris_captain_id)

    return compute_league_outcome(
        league,
        player_points,
        captain_ids_by_entry,
        chris_entry_id,
        config.num_simulations,
        primary_rival_entry_ids,
        random_seed=config.random_seed,
    )
