"""Section 8: pre-season handling.

When no gameweek has finished yet, none of the scoring-dependent analytics
(sections 1-7 beyond bare membership) can produce a real answer. Rather
than fabricate placeholder numbers, this module returns exactly what public
data *is* available right now plus an explicit list of what is blocked and
why.
"""

from __future__ import annotations

from .models import LeagueData, ManagerData

ACTIVATES_AFTER_GW1 = [
    "Manager profiles: squad, XI, bench, captain/vice, chips used, transfers, hits",
    "Chris vs. rival squad overlap, differentials and captain exposure",
    "Behavioural classification (transfer behaviour, risk behaviour)",
    "League consensus: ownership %, template/differentiated managers",
    "Threat analysis: shields, threats, weapons",
    "Relevant rival selection (needs scored points gaps)",
    "Strategy state (DEFEND/BALANCED/ATTACK/DESPERATE)",
]

BLOCKED_REASONS = [
    ("League positions and points", "No gameweek has been scored - standings are all zero/null."),
    ("Squad, XI, bench, captain, vice-captain", "Picks only become public after a gameweek's deadline passes."),
    ("Transfers, hits, chips used", "These accrue once the season is live; nothing has happened yet."),
    ("Squad overlap, differentials, captain exposure", "Requires a squad for both Chris and the rival."),
    ("League consensus / ownership %", "Requires at least one manager's squad to be known."),
    ("Threats, shields, weapons", "Requires league consensus plus Chris's own squad."),
    ("Relevant rivals", "Requires scored points gaps between managers."),
    ("Strategy state", "Explicitly gated - returns INSUFFICIENT_SEASON_DATA until GW1 is scored."),
]


def _member_summary(manager: ManagerData) -> dict:
    return {
        "entry_id": manager.entry_id,
        "manager_name": manager.manager_name,
        "team_name": manager.team_name,
        "past_seasons": list(manager.past_seasons),
    }


def build_preseason_report(league: LeagueData, chris: ManagerData) -> dict:
    return {
        "league_id": league.league_id,
        "league_name": league.league_name,
        "number_of_competitors": max(0, len(league.managers) - 1),
        "total_managers_in_league": len(league.managers),
        "membership": [_member_summary(m) for m in league.managers],
        "chris_public_info": _member_summary(chris) if chris else None,
        "activates_after_gw1": ACTIVATES_AFTER_GW1,
        "currently_blocked": [{"area": area, "reason": reason} for area, reason in BLOCKED_REASONS],
    }
