"""Data models for the Stage 3 League Win Simulator.

Nothing in this module touches the network. These are the only shapes the
simulation engine understands - they can be populated from synthetic
fixtures (today, since no gameweek exists yet) or from a future live
adapter, without the engine itself changing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

from .exceptions import InvalidManagerStateError


@dataclass(frozen=True)
class PlayerProjection:
    """A player's projected gameweek score, as an (independent) Normal distribution.

    ``projected_mean_points`` and ``projected_standard_deviation`` are
    supplied by the caller - they are NOT derived from any real statistical
    model in Stage 3. See ``player_outcomes.py`` for the sampling
    methodology and its documented limitations.
    """

    player_id: int
    player_name: str
    projected_mean_points: float
    projected_standard_deviation: float

    def __post_init__(self):
        if self.projected_standard_deviation < 0:
            raise InvalidManagerStateError(
                f"{self.player_name}: projected_standard_deviation must be >= 0."
            )


@dataclass(frozen=True)
class ManagerState:
    """One manager's current mini-league standing and gameweek squad.

    ``captain_probabilities``, if set, is a rival-only feature: a mapping
    of {player_id: probability} used when the engine runs in
    "probabilistic" rival-captain mode (see rival_behaviour.py). It is
    ignored for Chris - his captain is the explicit Captain Battle
    candidate being tested in a given run, never sampled.
    """

    entry_id: int
    name: str
    current_total_points: float
    current_league_position: int
    starting_xi: Tuple[int, ...]
    captain_id: int
    vice_captain_id: Optional[int] = None
    bench: Tuple[int, ...] = ()
    captain_probabilities: Optional[Dict[int, float]] = None

    def __post_init__(self):
        if not self.starting_xi:
            raise InvalidManagerStateError(f"{self.name}: starting_xi must not be empty.")
        if len(set(self.starting_xi)) != len(self.starting_xi):
            raise InvalidManagerStateError(f"{self.name}: starting_xi contains duplicate players.")
        if self.captain_id not in self.starting_xi:
            raise InvalidManagerStateError(
                f"{self.name}: captain_id {self.captain_id} is not in starting_xi."
            )

    def with_captain(self, captain_id: int) -> "ManagerState":
        """Returns a copy with a different captain - used by Captain Battle
        to try each candidate without mutating the original state."""
        return replace(self, captain_id=captain_id)


@dataclass(frozen=True)
class LeagueState:
    """A classic mini league's current state - the simulation's starting point."""

    league_id: Optional[int]
    league_name: str
    managers: Tuple[ManagerState, ...]

    def __post_init__(self):
        entry_ids = [m.entry_id for m in self.managers]
        if len(set(entry_ids)) != len(entry_ids):
            raise InvalidManagerStateError(f"League {self.league_name}: duplicate entry_id in managers.")
        if not (2 <= len(self.managers) <= 50):
            raise InvalidManagerStateError(
                f"League {self.league_name}: simulator supports 2-50 managers, got {len(self.managers)}."
            )

    def manager(self, entry_id: int) -> Optional[ManagerState]:
        return next((m for m in self.managers if m.entry_id == entry_id), None)


@dataclass(frozen=True)
class SimulationConfig:
    """Tunable Monte Carlo parameters. All defaults are documented, not magic."""

    num_simulations: int = 50_000
    random_seed: int = 42
    # "fixed": every rival uses their configured captain_id with certainty.
    # "probabilistic": rivals with a captain_probabilities distribution are
    # sampled from it each simulation; rivals without one still use their
    # fixed captain_id.
    rival_captain_mode: str = "fixed"
    # Simplified floor on a single player's simulated gameweek score - see
    # player_outcomes.py for why this exists and what it does NOT model.
    min_player_points: float = -4.0

    VALID_MODES = ("fixed", "probabilistic")

    def __post_init__(self):
        if self.rival_captain_mode not in self.VALID_MODES:
            raise ValueError(
                f"rival_captain_mode must be one of {self.VALID_MODES}, got {self.rival_captain_mode!r}."
            )
        if self.num_simulations < 1:
            raise ValueError("num_simulations must be >= 1.")
