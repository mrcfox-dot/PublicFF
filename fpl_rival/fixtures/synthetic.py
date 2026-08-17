"""Synthetic manager/league builders for Stage 2 development and tests.

SYNTHETIC DATA ONLY - see package docstring. Every entry ID here is >=
900000 and every league ID is >= 900000, specifically so a bug can never
make live and synthetic data collide.

Player pool (fake, deterministic): elements 1-8 are goalkeepers, 10-24 are
defenders, 30-49 are midfielders, 60-69 are forwards. Names are just
"Player <id>" - there is no attempt to resemble real footballers.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fpl_rival.intelligence.models import (
    GameContext,
    GameweekHistoryRow,
    GameweekSquad,
    LeagueData,
    ManagerData,
    Pick,
)

GK, DEF, MID, FWD = 1, 2, 3, 4

GK_POOL = list(range(1, 9))  # 1-8
DEF_POOL = list(range(10, 25))  # 10-24
MID_POOL = list(range(30, 50))  # 30-49
FWD_POOL = list(range(60, 70))  # 60-69

_ANCHOR = datetime(2026, 8, 1, tzinfo=timezone.utc)


def _deadline(event: int) -> str:
    return (_ANCHOR + timedelta(days=7 * event)).isoformat().replace("+00:00", "Z")


def build_context() -> GameContext:
    element_names = {}
    element_types = {}
    for pool, etype, label in ((GK_POOL, GK, "GK"), (DEF_POOL, DEF, "DEF"), (MID_POOL, MID, "MID"), (FWD_POOL, FWD, "FWD")):
        for eid in pool:
            element_names[eid] = f"{label}{eid}"
            element_types[eid] = etype

    chip_windows = {
        "wildcard": [(2, 19), (20, 38)],
        "freehit": [(2, 19), (20, 38)],
        "bboost": [(1, 19), (20, 38)],
        "3xc": [(1, 19), (20, 38)],
    }
    event_deadlines = {e: _deadline(e) for e in range(1, 39)}
    return GameContext(
        element_names=element_names,
        element_types=element_types,
        chip_windows=chip_windows,
        event_deadlines=event_deadlines,
        season_total_gameweeks=38,
    )


def _squad(
    event: int,
    starting_ids: list,
    bench_ids: list,
    captain: int,
    vice_captain: int,
    ctx: GameContext,
    active_chip=None,
    points=None,
    points_on_bench=None,
) -> GameweekSquad:
    picks = []
    for i, eid in enumerate(starting_ids, start=1):
        picks.append(
            Pick(
                element=eid,
                position=i,
                multiplier=2 if eid == captain else 1,
                is_captain=eid == captain,
                is_vice_captain=eid == vice_captain,
                element_type=ctx.element_types.get(eid),
            )
        )
    for i, eid in enumerate(bench_ids, start=len(starting_ids) + 1):
        picks.append(
            Pick(element=eid, position=i, multiplier=0, is_captain=False, is_vice_captain=False, element_type=ctx.element_types.get(eid))
        )
    return GameweekSquad(event=event, picks=tuple(picks), active_chip=active_chip, points=points, points_on_bench=points_on_bench)


def _default_squad_11_4(seed: int, ctx: GameContext) -> tuple:
    """A plausible 1 GK / 4 DEF / 4 MID / 2 FWD XI plus a 4-man bench, offset
    by ``seed`` so different managers can share or differ in personnel."""
    starting = [
        GK_POOL[seed % len(GK_POOL)],
        *[DEF_POOL[(seed + i) % len(DEF_POOL)] for i in range(4)],
        *[MID_POOL[(seed + i) % len(MID_POOL)] for i in range(4)],
        *[FWD_POOL[(seed + i) % len(FWD_POOL)] for i in range(2)],
    ]
    bench = [
        GK_POOL[(seed + 1) % len(GK_POOL)],
        DEF_POOL[(seed + 4) % len(DEF_POOL)],
        MID_POOL[(seed + 4) % len(MID_POOL)],
        FWD_POOL[(seed + 2) % len(FWD_POOL)],
    ]
    return starting, bench


def _history(events: list, points_per_gw: int, transfers_per_gw: int = 1, hit_cost_per_gw: int = 0, bench_points_per_gw: int = 3) -> tuple:
    rows = []
    total = 0
    for ev in events:
        total += points_per_gw
        rows.append(
            GameweekHistoryRow(
                event=ev,
                points=points_per_gw,
                total_points=total,
                rank=None,
                overall_rank=None,
                bank=0,
                value=1000,
                event_transfers=transfers_per_gw,
                event_transfers_cost=hit_cost_per_gw,
                points_on_bench=bench_points_per_gw,
            )
        )
    return tuple(rows)


def make_manager(
    entry_id: int,
    manager_name: str,
    team_name: str,
    league_position: int,
    total_points: int,
    ctx: GameContext,
    events: list,
    seed: int = 0,
    captain_seed_fn=None,
    transfers_per_gw: int = 1,
    hit_cost_per_gw: int = 0,
    chips_used: tuple = (),
    transfers: tuple = (),
    overall_rank=None,
    team_value=1000,
) -> ManagerData:
    squads_by_event = {}
    for i, ev in enumerate(events):
        starting, bench = _default_squad_11_4(seed, ctx)
        captain = captain_seed_fn(ev) if captain_seed_fn else starting[5]
        vice = starting[6] if len(starting) > 6 else starting[0]
        squads_by_event[ev] = _squad(ev, starting, bench, captain, vice, ctx, points_on_bench=3 + (i % 4))

    return ManagerData(
        entry_id=entry_id,
        manager_name=manager_name,
        team_name=team_name,
        league_position=league_position,
        total_points=total_points,
        overall_points=total_points,
        overall_rank=overall_rank,
        team_value=team_value,
        bank=0,
        gameweek_history=_history(events, points_per_gw=max(1, total_points // max(1, len(events))), transfers_per_gw=transfers_per_gw, hit_cost_per_gw=hit_cost_per_gw),
        chips_used=chips_used,
        transfers=transfers,
        squads_by_event=squads_by_event,
        is_synthetic=True,
    )


def _league(league_id: int, name: str, managers: list, latest_completed_event) -> LeagueData:
    return LeagueData(
        league_id=league_id,
        league_name=name,
        managers=tuple(managers),
        latest_completed_event=latest_completed_event,
        is_synthetic=True,
    )


# ---------------------------------------------------------------------------
# Scenario A: Chris leads by 70 points late in the season -> expect DEFEND
# ---------------------------------------------------------------------------
def scenario_a_defend():
    ctx = build_context()
    events = list(range(1, 35))  # 34 gameweeks played, 4 remaining (season=38)
    chris = make_manager(900001, "Chris Fox", "[SYNTHETIC] Great Exhibition", 1, 1800, ctx, events, seed=0)
    rival2 = make_manager(900002, "Rival Two", "[SYNTHETIC] Runner Up", 2, 1730, ctx, events, seed=1)
    rival3 = make_manager(900003, "Rival Three", "[SYNTHETIC] Third Place", 3, 1600, ctx, events, seed=2)
    league = _league(900101, "[SYNTHETIC] Scenario A - Defend", [chris, rival2, rival3], latest_completed_event=34)
    return league, ctx, 900001


# ---------------------------------------------------------------------------
# Scenario B: Chris trails leader by 30 with 6 gameweeks remaining -> ATTACK
# ---------------------------------------------------------------------------
def scenario_b_attack():
    ctx = build_context()
    events = list(range(1, 33))  # 32 played, 6 remaining
    leader = make_manager(900011, "Leader", "[SYNTHETIC] Table Toppers", 1, 1600, ctx, events, seed=0)
    chris = make_manager(900012, "Chris Fox", "[SYNTHETIC] Great Exhibition", 2, 1570, ctx, events, seed=1)
    rival3 = make_manager(900013, "Rival Three", "[SYNTHETIC] Third Place", 3, 1500, ctx, events, seed=2)
    league = _league(900102, "[SYNTHETIC] Scenario B - Attack", [leader, chris, rival3], latest_completed_event=32)
    return league, ctx, 900012


# ---------------------------------------------------------------------------
# Scenario C: Chris trails leader by 80 with 3 gameweeks remaining -> DESPERATE
# ---------------------------------------------------------------------------
def scenario_c_desperate():
    ctx = build_context()
    events = list(range(1, 36))  # 35 played, 3 remaining
    leader = make_manager(900021, "Leader", "[SYNTHETIC] Table Toppers", 1, 1900, ctx, events, seed=0)
    chris = make_manager(900022, "Chris Fox", "[SYNTHETIC] Great Exhibition", 2, 1820, ctx, events, seed=1)
    rival3 = make_manager(900023, "Rival Three", "[SYNTHETIC] Third Place", 3, 1700, ctx, events, seed=2)
    league = _league(900103, "[SYNTHETIC] Scenario C - Desperate", [leader, chris, rival3], latest_completed_event=35)
    return league, ctx, 900022


# ---------------------------------------------------------------------------
# Scenario D: two managers with almost identical squads -> high overlap/exposure
# ---------------------------------------------------------------------------
def scenario_d_identical_squads():
    ctx = build_context()
    events = list(range(1, 11))

    def captain_fn(ev):
        return MID_POOL[0]

    chris = make_manager(900031, "Chris Fox", "[SYNTHETIC] Great Exhibition", 1, 620, ctx, events, seed=0, captain_seed_fn=captain_fn)

    # Twin: identical seed (=> identical XI/bench selection) and identical
    # captain choices every gameweek.
    twin = make_manager(900032, "Twin Rival", "[SYNTHETIC] Copycat FC", 2, 600, ctx, events, seed=0, captain_seed_fn=captain_fn)

    # A genuinely different manager for contrast in the same test.
    different = make_manager(900033, "Different Rival", "[SYNTHETIC] Contrarian United", 3, 480, ctx, events, seed=5)

    league = _league(900104, "[SYNTHETIC] Scenario D - Identical Squads", [chris, twin, different], latest_completed_event=10)
    return league, ctx, 900031


# ---------------------------------------------------------------------------
# Scenario E: a rival owns several high-league-ownership players Chris lacks
# ---------------------------------------------------------------------------
def scenario_e_threats():
    ctx = build_context()
    events = list(range(1, 11))

    # Template players: heavily owned across the league except by Chris.
    template_starting = [GK_POOL[0], DEF_POOL[0], DEF_POOL[1], DEF_POOL[2], DEF_POOL[3], MID_POOL[0], MID_POOL[1], MID_POOL[2], MID_POOL[3], FWD_POOL[0], FWD_POOL[1]]
    template_bench = [GK_POOL[1], DEF_POOL[4], MID_POOL[4], FWD_POOL[2]]

    def templated_manager(entry_id, name, team, position, points, captain_elem):
        squads_by_event = {ev: _squad(ev, template_starting, template_bench, captain_elem, template_starting[1], ctx) for ev in events}
        return ManagerData(
            entry_id=entry_id,
            manager_name=name,
            team_name=team,
            league_position=position,
            total_points=points,
            overall_points=points,
            team_value=1000,
            gameweek_history=_history(events, points_per_gw=max(1, points // len(events))),
            squads_by_event=squads_by_event,
            is_synthetic=True,
        )

    rival_a = templated_manager(900041, "Template Rival A", "[SYNTHETIC] Template XI A", 2, 640, MID_POOL[0])
    rival_b = templated_manager(900042, "Template Rival B", "[SYNTHETIC] Template XI B", 3, 630, MID_POOL[0])
    rival_c = templated_manager(900043, "Template Rival C", "[SYNTHETIC] Template XI C", 4, 600, FWD_POOL[0])

    # Chris deliberately owns none of the template players - a fully
    # differentiated squad, seeded well away from the template pool indices.
    chris = make_manager(900044, "Chris Fox", "[SYNTHETIC] Great Exhibition", 1, 650, ctx, events, seed=6)

    league = _league(
        900105, "[SYNTHETIC] Scenario E - Threats", [chris, rival_a, rival_b, rival_c], latest_completed_event=10
    )
    return league, ctx, 900044


# ---------------------------------------------------------------------------
# Pre-season scenario: no completed gameweeks at all
# ---------------------------------------------------------------------------
def scenario_preseason():
    ctx = build_context()
    chris = ManagerData(entry_id=900051, manager_name="Chris Fox", team_name="[SYNTHETIC] Great Exhibition", is_synthetic=True)
    rival = ManagerData(entry_id=900052, manager_name="Rival One", team_name="[SYNTHETIC] Early Bird", is_synthetic=True)
    league = _league(900106, "[SYNTHETIC] Preseason League", [chris, rival], latest_completed_event=None)
    return league, ctx, 900051
