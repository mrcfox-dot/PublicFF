"""Player gameweek point outcome model.

Methodology (deliberately simple - see Stage 3 completion report for why
this is enough to test the *league strategy mechanism*, not a claim about
being a good football scoring model):

    simulated_points = round(clip(Normal(mean, std), floor=min_points, ceiling=None))

* ``Normal(mean, std)`` - a plain Gaussian draw per player, per simulation.
* ``clip(..., floor=min_points)`` - real FPL scores are very rarely worse
  than about -4 (e.g. a defender's red card + own goal); the Gaussian tail
  below that is clipped off so a single bad simulated point-mass doesn't
  represent something that basically never happens in the real game. There
  is no ceiling clip - big hauls are legitimately possible and the Normal's
  upper tail is left alone.
* ``round(...)`` - real FPL gameweek scores are always integers, so the
  clipped draw is rounded to the nearest whole point.
* Zero is fully supported (no special-casing) - it is just a normal,
  reachable point on the distribution.

This is NOT a specialised football scoring distribution (no separate
appearance/goal/assist/bonus components, no position-specific shape). That
is an explicit, documented simplification - see the Stage 3 completion
report's "known modelling weaknesses" section.

CORRELATION LIMITATION: each player is sampled independently of every
other player. In reality, teammates' scores are positively correlated
(same match, same result) and some players are negatively correlated
(playing-time competition). This version does not model that. The
architecture keeps the door open for it though: this module's only
contract is "return one simulated-points array per player_id, all of the
same length". A future version could replace the body of
``sample_all_players`` with a multivariate/copula draw (e.g. a shared
per-team performance factor added to each of that team's players) without
changing anything downstream - ``manager_outcomes.py`` and
``monte_carlo.py`` only ever consume the resulting
``{player_id: np.ndarray}`` dict, never the sampling internals.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from .exceptions import MissingProjectionError
from .models import PlayerProjection


def sample_all_players(
    player_ids: Iterable[int],
    projections: Dict[int, PlayerProjection],
    num_simulations: int,
    rng: np.random.Generator,
    min_player_points: float = -4.0,
) -> Dict[int, np.ndarray]:
    """Samples ``num_simulations`` gameweek scores for every player in
    ``player_ids``, each as an independent array of shape ``(num_simulations,)``.

    Every player in ``player_ids`` MUST have an entry in ``projections`` -
    this is what makes "missing player projections fail clearly" true: a
    squad referencing an unprojected player raises immediately, rather than
    silently treating them as a zero or being skipped.
    """
    result: Dict[int, np.ndarray] = {}
    for player_id in player_ids:
        if player_id not in projections:
            raise MissingProjectionError(
                f"No projection supplied for player_id={player_id} - cannot simulate their gameweek score."
            )
        proj = projections[player_id]
        raw = rng.normal(proj.projected_mean_points, proj.projected_standard_deviation, size=num_simulations)
        clipped = np.clip(raw, min_player_points, None)
        result[player_id] = np.round(clipped)
    return result
