"""Turns shared per-player simulated points into one manager's gameweek score.

Scoring rule (the "minimum required version" from the Stage 3 brief):

    gameweek_score = sum(starting_xi player points) + captain_bonus

where ``captain_bonus`` is one extra copy of the captain's points (so the
captain's total contribution is 2x, everyone else's is 1x), with one small
addition: if the captain's simulated score rounds to exactly 0 - a simple
proxy for "didn't really play" - the vice-captain's score is used for the
bonus instead. This is a deliberate simplification of FPL's real
"captain didn't play -> automatic vice-captain" rule; it does not model
injuries, rotation or fixture postponements directly (see limitations).

Bench scoring (auto-substitutions) is explicitly NOT implemented - bench
players never contribute, regardless of what the starting XI scores. This
keeps the model simple and is documented as a limitation.

CRITICAL correctness requirement this module exists to satisfy: every
manager's score for a given simulation is built from the SAME shared
``player_points`` arrays (produced once by ``player_outcomes.py``). If
Chris and a rival both own Salah, ``player_points[salah_id][i]`` is the one
and only simulated Salah score for simulation ``i``, used by both of them.
This module never re-samples a player - it only reads from what it's given.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .models import ManagerState


def _gather(player_points: Dict[int, np.ndarray], id_array: np.ndarray) -> np.ndarray:
    """Vectorized "pick each simulation's value for whichever player_id was
    chosen that simulation" - one lookup per unique id in id_array."""
    result = np.empty(len(id_array), dtype=float)
    for player_id in np.unique(id_array):
        mask = id_array == player_id
        result[mask] = player_points[player_id][mask]
    return result


def simulate_manager_gameweek(
    manager: ManagerState,
    player_points: Dict[int, np.ndarray],
    captain_id_per_sim: np.ndarray,
    vice_captain_id_per_sim: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Returns an array of shape (num_simulations,): this manager's total
    gameweek score in each simulation.

    ``captain_id_per_sim`` lets the captain vary by simulation (needed for
    probabilistic rival captains); pass a constant array (e.g.
    ``np.full(n, captain_id)``) for a fixed captain.
    """
    num_simulations = len(captain_id_per_sim)

    xi_total = np.zeros(num_simulations, dtype=float)
    for player_id in manager.starting_xi:
        xi_total += player_points[player_id]

    captain_points = _gather(player_points, captain_id_per_sim)

    if vice_captain_id_per_sim is None:
        if manager.vice_captain_id is not None:
            vice_captain_id_per_sim = np.full(num_simulations, manager.vice_captain_id)

    if vice_captain_id_per_sim is not None:
        vice_points = _gather(player_points, vice_captain_id_per_sim)
        captain_bonus = np.where(captain_points == 0, vice_points, captain_points)
    else:
        captain_bonus = captain_points

    return xi_total + captain_bonus
