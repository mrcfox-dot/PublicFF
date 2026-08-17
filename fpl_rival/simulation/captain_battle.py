"""Captain Battle: compare candidate captains on expected points AND on a
configurable mini-league objective, for the same underlying randomness.

Every candidate is run against the SAME sampled player outcomes and the
SAME sampled rival captain choices ("common random numbers") - only
Chris's captain assignment changes between runs. This is deliberate: it
removes simulation noise from the comparison, so any difference between
candidates reflects the captaincy decision itself, not which random seed
happened to favour which player. Without this, the acceptance-test effects
(see the Stage 3 completion report) would be far noisier and harder to
trust at any practical simulation count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .exceptions import InvalidCaptainError
from .models import LeagueState, ManagerState, PlayerProjection, SimulationConfig
from .monte_carlo import SimulationResult, collect_required_player_ids, compute_league_outcome
from .player_outcomes import sample_all_players
from .relevance_adapter import default_primary_rival_ids
from .rival_behaviour import resolve_captain_ids_by_entry

# metric_name -> True if higher is better, False if lower is better
OBJECTIVES = {
    "prob_finish_first": True,
    "prob_top3": True,
    "prob_move_up": True,
    "expected_position": False,
}
DEFAULT_OBJECTIVE = "prob_finish_first"


@dataclass(frozen=True)
class CaptainCandidateResult:
    player_id: int
    player_name: str
    result: SimulationResult


@dataclass(frozen=True)
class CaptainBattleResult:
    chris_entry_id: int
    league_name: str
    starting_position: int
    starting_gap_to_leader: float
    candidates: List[CaptainCandidateResult]
    objective: str
    ranking_by_expected_points: List[int] = field(default_factory=list)  # player_ids, best first
    ranking_by_objective: List[int] = field(default_factory=list)  # player_ids, best first
    num_simulations: int = 0
    random_seed: int = 0

    @property
    def expected_points_winner(self) -> CaptainCandidateResult:
        return self._by_id(self.ranking_by_expected_points[0])

    @property
    def objective_winner(self) -> CaptainCandidateResult:
        return self._by_id(self.ranking_by_objective[0])

    def _by_id(self, player_id: int) -> CaptainCandidateResult:
        return next(c for c in self.candidates if c.player_id == player_id)


def _validate_candidates(chris: ManagerState, candidate_captain_ids: List[int], projections: Dict[int, PlayerProjection]) -> None:
    if not candidate_captain_ids:
        raise InvalidCaptainError("At least one candidate captain is required.")
    for player_id in candidate_captain_ids:
        if player_id not in chris.starting_xi:
            raise InvalidCaptainError(
                f"Candidate captain player_id={player_id} is not in {chris.name}'s starting XI - "
                f"XI is {chris.starting_xi}."
            )
        if player_id not in projections:
            from .exceptions import MissingProjectionError

            raise MissingProjectionError(
                f"Candidate captain player_id={player_id} has no projection supplied."
            )


def run_captain_battle(
    chris: ManagerState,
    league: LeagueState,
    projections: Dict[int, PlayerProjection],
    candidate_captain_ids: List[int],
    config: SimulationConfig = SimulationConfig(),
    objective: str = DEFAULT_OBJECTIVE,
    primary_rival_entry_ids: Optional[List[int]] = None,
) -> CaptainBattleResult:
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective {objective!r}; choose from {sorted(OBJECTIVES)}.")
    if league.manager(chris.entry_id) is None:
        raise InvalidCaptainError(f"{chris.name} (entry {chris.entry_id}) is not a member of league {league.league_name}.")

    _validate_candidates(chris, candidate_captain_ids, projections)

    if primary_rival_entry_ids is None:
        primary_rival_entry_ids = default_primary_rival_ids(chris, league)

    rng = np.random.default_rng(config.random_seed)
    required_ids = collect_required_player_ids(league, extra_ids=candidate_captain_ids)
    player_points = sample_all_players(required_ids, projections, config.num_simulations, rng, config.min_player_points)

    # Sampled ONCE, shared by every candidate below (common random numbers).
    base_captain_ids = resolve_captain_ids_by_entry(league, config.num_simulations, rng, config.rival_captain_mode)

    candidates: List[CaptainCandidateResult] = []
    for player_id in candidate_captain_ids:
        captain_ids_this_run = dict(base_captain_ids)
        captain_ids_this_run[chris.entry_id] = np.full(config.num_simulations, player_id)

        result = compute_league_outcome(
            league,
            player_points,
            captain_ids_this_run,
            chris.entry_id,
            config.num_simulations,
            primary_rival_entry_ids,
            random_seed=config.random_seed,
        )
        candidates.append(
            CaptainCandidateResult(player_id=player_id, player_name=projections[player_id].player_name, result=result)
        )

    higher_is_better = OBJECTIVES[objective]
    ranking_by_points = sorted(candidates, key=lambda c: -c.result.expected_gameweek_points)
    ranking_by_objective = sorted(
        candidates, key=lambda c: -getattr(c.result, objective) if higher_is_better else getattr(c.result, objective)
    )

    leader_total = max(m.current_total_points for m in league.managers)
    starting_gap = max(0.0, leader_total - chris.current_total_points)

    return CaptainBattleResult(
        chris_entry_id=chris.entry_id,
        league_name=league.league_name,
        starting_position=chris.current_league_position,
        starting_gap_to_leader=starting_gap,
        candidates=candidates,
        objective=objective,
        ranking_by_expected_points=[c.player_id for c in ranking_by_points],
        ranking_by_objective=[c.player_id for c in ranking_by_objective],
        num_simulations=config.num_simulations,
        random_seed=config.random_seed,
    )
