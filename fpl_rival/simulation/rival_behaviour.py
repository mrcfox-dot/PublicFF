"""Rival captain behaviour: fixed vs. probabilistic captain choice.

Two modes (see ``SimulationConfig.rival_captain_mode``):

* ``"fixed"`` - every manager captains their configured ``captain_id`` with
  certainty, every simulation. Simple, and a reasonable approximation for
  "this rival always captains their nailed premium".
* ``"probabilistic"`` - a manager with a ``captain_probabilities``
  distribution has their captain re-sampled independently each simulation,
  from that distribution. A manager without one still falls back to their
  fixed ``captain_id``.

Stage 2's behavioural history (captain concentration, etc.) is a natural
future source for these probabilities, but this module does not attempt to
infer them - they are supplied manually, and that is documented as a
Stage 3 limitation.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .exceptions import InvalidCaptainDistributionError
from .models import LeagueState

FIXED = "fixed"
PROBABILISTIC = "probabilistic"


def validate_captain_distribution(distribution: Dict[int, float], owner_name: str, tolerance: float = 1e-6) -> None:
    if not distribution:
        raise InvalidCaptainDistributionError(f"{owner_name}: captain_probabilities is empty.")
    for player_id, prob in distribution.items():
        if prob < 0:
            raise InvalidCaptainDistributionError(
                f"{owner_name}: negative probability {prob} for player_id={player_id}."
            )
    total = sum(distribution.values())
    if not math.isclose(total, 1.0, abs_tol=tolerance):
        raise InvalidCaptainDistributionError(
            f"{owner_name}: captain_probabilities must sum to 1.0 (within {tolerance}), got {total}."
        )


def sample_captain_ids(
    distribution: Dict[int, float], num_simulations: int, rng: np.random.Generator
) -> np.ndarray:
    """One captain player_id per simulation, drawn i.i.d. from ``distribution``."""
    player_ids = np.array(list(distribution.keys()))
    probs = np.array(list(distribution.values()))
    probs = probs / probs.sum()  # guard against float drift after validation
    return rng.choice(player_ids, size=num_simulations, p=probs)


def resolve_captain_ids_by_entry(
    league: LeagueState,
    num_simulations: int,
    rng: np.random.Generator,
    rival_captain_mode: str,
) -> Dict[int, np.ndarray]:
    """Captain player_id per simulation, for every manager in the league.

    Callers implementing a Captain Battle for Chris should overwrite his
    entry in the returned dict with his candidate captain afterwards - this
    function treats every manager (Chris included) identically, using
    whatever their ManagerState says.
    """
    result: Dict[int, np.ndarray] = {}
    for manager in league.managers:
        if rival_captain_mode == PROBABILISTIC and manager.captain_probabilities is not None:
            validate_captain_distribution(manager.captain_probabilities, manager.name)
            result[manager.entry_id] = sample_captain_ids(manager.captain_probabilities, num_simulations, rng)
        else:
            result[manager.entry_id] = np.full(num_simulations, manager.captain_id)
    return result
