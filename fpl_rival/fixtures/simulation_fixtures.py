"""Synthetic fixtures for the Stage 3 League Win Simulator - SYNTHETIC DATA ONLY.

Never mixed with live FPL data: every entry ID here is >= 900000, and the
projections below are illustrative numbers chosen to exercise the
simulator's mechanics, not real predictions (see ``PlayerProjection``'s
docstring). Real player names (Salah, Palmer, ...) are used purely as
readable labels, matching the Stage 3 brief's own example CLI output - the
NUMBERS attached to them are made up for testing purposes only.

Four scenarios, one shared player pool:

* ``captain_attack()``   - Chris trails the leader; tests whether a
  differential captain beats matching the leader's likely captain.
* ``captain_defend()``   - Chris leads comfortably; tests whether matching
  the chasing rival's likely captain reduces downside risk.
* ``shared_squad_low_diff()`` / ``shared_squad_high_diff()`` - two managers
  with mostly-identical vs. mostly-different squads, to verify shared
  player outcomes drive relative variance correctly.
* ``rival_uncertainty_scenario_a()`` / ``_scenario_b()`` - the same
  attacking situation, with the rival's captain probability shifted, to
  show the optimal captain can flip in response.

The squad design in every scenario deliberately keeps most players
identical between Chris and whichever manager he's being compared to, so
that the handful of players that DO differ are the only source of relative
variance in the comparison. This isolates the effect the acceptance tests
are checking for - it is a controlled synthetic setup, not a realistic
squad.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from fpl_rival.simulation.models import LeagueState, ManagerState, PlayerProjection

# -- Shared synthetic player pool (illustrative numbers only) -------------------

SALAH_ID, PALMER_ID, HAALAND_ID, WATKINS_ID = 1, 2, 3, 4
TEMPLATE_IDS = list(range(5001, 5012))  # 11 generic "template" players

SALAH = PlayerProjection(SALAH_ID, "Salah", projected_mean_points=8.5, projected_standard_deviation=6.0)
PALMER = PlayerProjection(PALMER_ID, "Palmer", projected_mean_points=7.0, projected_standard_deviation=7.5)
HAALAND = PlayerProjection(HAALAND_ID, "Haaland", projected_mean_points=8.0, projected_standard_deviation=6.5)
WATKINS = PlayerProjection(WATKINS_ID, "Watkins", projected_mean_points=6.0, projected_standard_deviation=5.5)

TEMPLATE_MEAN, TEMPLATE_STD = 4.0, 3.0


def _template_projections() -> Dict[int, PlayerProjection]:
    return {
        pid: PlayerProjection(pid, f"Template{pid - 5000}", TEMPLATE_MEAN, TEMPLATE_STD) for pid in TEMPLATE_IDS
    }


def _base_projections() -> Dict[int, PlayerProjection]:
    projections = {SALAH_ID: SALAH, PALMER_ID: PALMER, HAALAND_ID: HAALAND, WATKINS_ID: WATKINS}
    projections.update(_template_projections())
    return projections


def _fillers(start_entry_id: int, count: int, start_points: float, step: float) -> List[ManagerState]:
    """Non-threatening lower-table managers, purely to make the league a
    realistic size. They never come close to 1st place."""
    xi = tuple(TEMPLATE_IDS[:9]) + (TEMPLATE_IDS[9], TEMPLATE_IDS[10])
    fillers = []
    for i in range(count):
        fillers.append(
            ManagerState(
                entry_id=start_entry_id + i,
                name=f"[SYNTHETIC] Filler {i + 1}",
                current_total_points=start_points - i * step,
                current_league_position=3 + i,
                starting_xi=xi,
                captain_id=TEMPLATE_IDS[0],
            )
        )
    return fillers


# ---------------------------------------------------------------------------
# Scenario: captain_attack - Chris trails the leader
# ---------------------------------------------------------------------------
def captain_attack() -> Tuple[ManagerState, LeagueState, Dict[int, PlayerProjection], List[int]]:
    """Chris and the leader both own Salah; the leader captains Salah with
    certainty. Chris ALSO owns Palmer, which the leader does not. Returns
    (chris, league, projections, candidate_captain_ids=[Salah, Palmer])."""
    projections = _base_projections()
    template_9 = tuple(TEMPLATE_IDS[:9])

    chris = ManagerState(
        entry_id=900101,
        name="[SYNTHETIC] Chris Fox",
        current_total_points=985,
        current_league_position=2,
        starting_xi=(SALAH_ID, PALMER_ID) + template_9,
        captain_id=SALAH_ID,  # overridden per candidate by Captain Battle
        vice_captain_id=TEMPLATE_IDS[0],
    )
    leader = ManagerState(
        entry_id=900102,
        name="[SYNTHETIC] League Leader",
        current_total_points=1000,
        current_league_position=1,
        starting_xi=(SALAH_ID,) + template_9 + (TEMPLATE_IDS[9],),
        captain_id=SALAH_ID,
        vice_captain_id=TEMPLATE_IDS[0],
    )
    fillers = _fillers(900110, 5, start_points=900, step=50)

    league = LeagueState(
        league_id=900001, league_name="[SYNTHETIC] Captain Attack League", managers=(chris, leader) + tuple(fillers)
    )
    return chris, league, projections, [SALAH_ID, PALMER_ID]


# ---------------------------------------------------------------------------
# Scenario: captain_defend - Chris leads comfortably
# ---------------------------------------------------------------------------
def captain_defend() -> Tuple[ManagerState, LeagueState, Dict[int, PlayerProjection], List[int]]:
    """Chris leads; the closest rival owns Salah (highest-projected player)
    and captains him with certainty. Chris also owns Palmer, which the
    rival does not. Returns (chris, league, projections, [Salah, Palmer])."""
    projections = _base_projections()
    template_9 = tuple(TEMPLATE_IDS[:9])

    chris = ManagerState(
        entry_id=900201,
        name="[SYNTHETIC] Chris Fox",
        current_total_points=1000,
        current_league_position=1,
        starting_xi=(SALAH_ID, PALMER_ID) + template_9,
        captain_id=SALAH_ID,
        vice_captain_id=TEMPLATE_IDS[0],
    )
    rival = ManagerState(
        entry_id=900202,
        name="[SYNTHETIC] Closest Rival",
        current_total_points=985,
        current_league_position=2,
        starting_xi=(SALAH_ID,) + template_9 + (TEMPLATE_IDS[9],),
        captain_id=SALAH_ID,
        vice_captain_id=TEMPLATE_IDS[0],
    )
    fillers = _fillers(900210, 5, start_points=900, step=50)

    league = LeagueState(
        league_id=900002, league_name="[SYNTHETIC] Captain Defend League", managers=(chris, rival) + tuple(fillers)
    )
    return chris, league, projections, [SALAH_ID, PALMER_ID]


# ---------------------------------------------------------------------------
# Scenario: shared_squad - overlap vs. relative variance
# ---------------------------------------------------------------------------
def shared_squad_scenario(num_shared: int) -> Tuple[ManagerState, ManagerState, LeagueState, Dict[int, PlayerProjection]]:
    """Two managers whose 11-man XIs share ``num_shared`` template players
    and differ in the remaining ``11 - num_shared`` slots (each manager's
    differing slots use entirely distinct player ids from the other's).
    Both captain the same shared player, so captaincy is not a variable
    here - only squad overlap is."""
    if not (0 <= num_shared <= 11):
        raise ValueError("num_shared must be between 0 and 11.")

    projections = dict(_template_projections())
    shared_ids = tuple(TEMPLATE_IDS[:num_shared]) if num_shared > 0 else ()

    num_diff = 11 - num_shared
    next_id = max(TEMPLATE_IDS) + 1
    chris_only_ids = tuple(range(next_id, next_id + num_diff))
    rival_only_ids = tuple(range(next_id + num_diff, next_id + 2 * num_diff))
    for pid in chris_only_ids + rival_only_ids:
        projections[pid] = PlayerProjection(pid, f"Filler{pid}", TEMPLATE_MEAN, TEMPLATE_STD)

    captain_id = shared_ids[0] if shared_ids else TEMPLATE_IDS[0]
    if captain_id not in projections:
        projections[captain_id] = PlayerProjection(captain_id, f"Template{captain_id - 5000}", TEMPLATE_MEAN, TEMPLATE_STD)

    chris_xi = shared_ids + chris_only_ids
    rival_xi = shared_ids + rival_only_ids
    chris_captain = captain_id if captain_id in chris_xi else chris_xi[0]
    rival_captain = captain_id if captain_id in rival_xi else rival_xi[0]

    chris = ManagerState(
        entry_id=900301,
        name="[SYNTHETIC] Chris Fox",
        current_total_points=1000,
        current_league_position=1,
        starting_xi=chris_xi,
        captain_id=chris_captain,
    )
    rival = ManagerState(
        entry_id=900302,
        name="[SYNTHETIC] Comparison Rival",
        current_total_points=1000,
        current_league_position=2,
        starting_xi=rival_xi,
        captain_id=rival_captain,
    )
    league = LeagueState(league_id=900003, league_name="[SYNTHETIC] Shared Squad League", managers=(chris, rival))
    return chris, rival, league, projections


def shared_squad_low_diff():
    """1 of 11 XI slots differ between Chris and the comparison rival."""
    return shared_squad_scenario(num_shared=10)


def shared_squad_high_diff():
    """6 of 11 XI slots differ between Chris and the comparison rival."""
    return shared_squad_scenario(num_shared=5)


# ---------------------------------------------------------------------------
# Scenario: rival_uncertainty - optimal captain shifts with rival probability
# ---------------------------------------------------------------------------
def _rival_uncertainty(leader_captain_probabilities: Dict[int, float]):
    projections = _base_projections()
    template_9 = tuple(TEMPLATE_IDS[:9])
    shared_xi = (SALAH_ID, PALMER_ID) + template_9  # Chris and the leader own an IDENTICAL XI here

    chris = ManagerState(
        entry_id=900401,
        name="[SYNTHETIC] Chris Fox",
        current_total_points=997,
        current_league_position=2,
        starting_xi=shared_xi,
        captain_id=SALAH_ID,
        vice_captain_id=TEMPLATE_IDS[0],
    )
    leader = ManagerState(
        entry_id=900402,
        name="[SYNTHETIC] League Leader",
        current_total_points=1000,
        current_league_position=1,
        starting_xi=shared_xi,
        captain_id=SALAH_ID,  # fallback if probabilistic mode is off
        vice_captain_id=TEMPLATE_IDS[0],
        captain_probabilities=leader_captain_probabilities,
    )
    fillers = _fillers(900410, 3, start_points=900, step=50)

    league = LeagueState(
        league_id=900004, league_name="[SYNTHETIC] Rival Uncertainty League", managers=(chris, leader) + tuple(fillers)
    )
    return chris, league, projections, [SALAH_ID, PALMER_ID]


def rival_uncertainty_scenario_a():
    """Leader captain probability: Salah 90%, Palmer 10%."""
    return _rival_uncertainty({SALAH_ID: 0.9, PALMER_ID: 0.1})


def rival_uncertainty_scenario_b():
    """Leader captain probability: Salah 50%, Palmer 50%."""
    return _rival_uncertainty({SALAH_ID: 0.5, PALMER_ID: 0.5})
