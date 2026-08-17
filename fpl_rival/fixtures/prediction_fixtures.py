"""Synthetic fixtures for the Stage 4 Rival Prediction Engine - SYNTHETIC DATA ONLY.

Never mixed with live FPL data or presented as a real prediction. Reuses
the same player pool as Stage 3's synthetic fixtures
(``fpl_rival.fixtures.simulation_fixtures`` - Salah/Palmer/Haaland/Watkins
ids, real names as readable labels only) so the Stage 4 -> Stage 3
integration fixtures don't need an id-remapping step.

Five personas, all captaining from the same 4-player pool each gameweek so
their predicted distributions are directly comparable:

* ``template_manager_history``      - usually captains the consensus favourite (Salah).
* ``loyal_captain_manager_history``  - repeatedly captains the same premium player (Haaland), regardless of matchup.
* ``differential_manager_history``   - frequently avoids the consensus favourite.
* ``expected_points_follower_history`` - captains whichever player has the highest EP that week (EP rotates weekly - see ``weekly_expected_points_series``).
* ``erratic_manager_history``        - captain chosen ~uniformly at random each week.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Dict, Tuple

from fpl_rival.fixtures.simulation_fixtures import (
    HAALAND_ID,
    PALMER_ID,
    SALAH_ID,
    TEMPLATE_IDS,
    WATKINS_ID,
    captain_attack,
)
from fpl_rival.prediction.models import CaptainObservation, LeagueHistory

CAPTAIN_POOL = (SALAH_ID, PALMER_ID, HAALAND_ID, WATKINS_ID)
PLAYER_NAMES = {SALAH_ID: "Salah", PALMER_ID: "Palmer", HAALAND_ID: "Haaland", WATKINS_ID: "Watkins"}
CONSENSUS_FAVOURITE_ID = SALAH_ID  # the "template" captain in this synthetic pool


def _observations(num_gameweeks: int, choose_captain, start_gw: int = 1) -> Tuple[CaptainObservation, ...]:
    return tuple(
        CaptainObservation(gameweek=gw, owned_player_ids=CAPTAIN_POOL, captain_id=choose_captain(gw))
        for gw in range(start_gw, start_gw + num_gameweeks)
    )


def weekly_expected_points_series(num_gameweeks: int, seed: int = 1, start_gw: int = 1) -> Dict[int, Dict[int, float]]:
    """A different (synthetic, illustrative) top-EP player most weeks, so
    the "expected points follower" persona has something real to track."""
    rng = random.Random(seed)
    series = {}
    for gw in range(start_gw, start_gw + num_gameweeks):
        base = {SALAH_ID: 7.5, PALMER_ID: 6.5, HAALAND_ID: 7.0, WATKINS_ID: 5.5}
        series[gw] = {pid: v + rng.uniform(-2.5, 2.5) for pid, v in base.items()}
    return series


def template_manager_history(num_gameweeks: int, seed: int = 1, start_gw: int = 1) -> Tuple[CaptainObservation, ...]:
    rng = random.Random(seed)

    def choose(gw: int) -> int:
        return CONSENSUS_FAVOURITE_ID if rng.random() < 0.88 else rng.choice(CAPTAIN_POOL)

    return _observations(num_gameweeks, choose, start_gw)


def loyal_captain_manager_history(num_gameweeks: int, seed: int = 2, start_gw: int = 1) -> Tuple[CaptainObservation, ...]:
    rng = random.Random(seed)

    def choose(gw: int) -> int:
        return HAALAND_ID if rng.random() < 0.95 else rng.choice(CAPTAIN_POOL)

    return _observations(num_gameweeks, choose, start_gw)


def differential_manager_history(num_gameweeks: int, seed: int = 3, start_gw: int = 1) -> Tuple[CaptainObservation, ...]:
    rng = random.Random(seed)
    differentials = (PALMER_ID, WATKINS_ID)

    def choose(gw: int) -> int:
        if rng.random() < 0.1:
            return CONSENSUS_FAVOURITE_ID
        return rng.choice(differentials)

    return _observations(num_gameweeks, choose, start_gw)


def expected_points_follower_history(
    num_gameweeks: int, seed: int = 4, start_gw: int = 1
) -> Tuple[Tuple[CaptainObservation, ...], Dict[int, Dict[int, float]]]:
    ep_series = weekly_expected_points_series(num_gameweeks, seed=seed + 100, start_gw=start_gw)
    rng = random.Random(seed)

    def choose(gw: int) -> int:
        top_ep_player = max(ep_series[gw], key=ep_series[gw].get)
        return top_ep_player if rng.random() < 0.9 else rng.choice(CAPTAIN_POOL)

    return _observations(num_gameweeks, choose, start_gw), ep_series


def erratic_manager_history(num_gameweeks: int, seed: int = 5, start_gw: int = 1) -> Tuple[CaptainObservation, ...]:
    rng = random.Random(seed)

    def choose(gw: int) -> int:
        return rng.choice(CAPTAIN_POOL)

    return _observations(num_gameweeks, choose, start_gw)


# ---------------------------------------------------------------------------
# End-to-end integration scenario (Stage 4 brief items 9 and 18):
# history -> Stage 4 prediction -> Stage 3 captain probabilities -> simulation
# ---------------------------------------------------------------------------
def dave_salah_loyal_history(num_gameweeks: int = 10, seed: int = 10) -> Tuple[CaptainObservation, ...]:
    """Dave historically captains Salah heavily (~90%)."""
    rng = random.Random(seed)

    def choose(gw: int) -> int:
        return SALAH_ID if rng.random() < 0.9 else PALMER_ID

    return _observations(num_gameweeks, choose)


def dave_balanced_history(num_gameweeks: int = 10, seed: int = 11) -> Tuple[CaptainObservation, ...]:
    """Dave's history is changed so Salah/Palmer are roughly a coin flip."""
    rng = random.Random(seed)

    def choose(gw: int) -> int:
        return SALAH_ID if rng.random() < 0.5 else PALMER_ID

    return _observations(num_gameweeks, choose)


def full_pipeline_scenario(dave_history: Tuple[CaptainObservation, ...]):
    """Builds the Stage 3 inputs (Chris behind Dave, both owning Salah AND
    Palmer - so Dave's captain uncertainty is genuinely between two players
    Chris can also choose between) needed to run the full
    history -> prediction -> simulation -> decision pipeline end to end.

    Reuses ``simulation_fixtures.captain_attack()`` for Chris's side of the
    league (he already owns Salah, Palmer and the template fillers there);
    only Dave's XI is adjusted here so he also owns Palmer, which is what
    makes his GW captaincy uncertainty between Salah/Palmer meaningful.
    """
    chris, league, projections, candidates = captain_attack()
    dave = league.manager(900102)  # "League Leader" in captain_attack()
    template_9 = tuple(TEMPLATE_IDS[:9])
    dave_with_palmer = replace(dave, starting_xi=(SALAH_ID, PALMER_ID) + template_9)
    managers = tuple(dave_with_palmer if m.entry_id == dave.entry_id else m for m in league.managers)
    league = replace(league, managers=managers)

    dave_league_history = LeagueHistory({dave.entry_id: dave_history})
    expected_points = {pid: proj.projected_mean_points for pid, proj in projections.items()}

    return chris, league, projections, candidates, dave_with_palmer, dave_league_history, expected_points
