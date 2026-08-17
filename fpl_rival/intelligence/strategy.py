"""Section 7: early league-position strategy state.

Deliberately simple and explicit - this describes Chris's situation, it
does NOT drive any player recommendation (that's a later stage).

States: DEFEND, BALANCED, ATTACK, DESPERATE, INSUFFICIENT_SEASON_DATA.

Decision logic (all thresholds in EngineConfig):

1. No completed gameweek yet -> INSUFFICIENT_SEASON_DATA. Always.
2. Chris is the leader (league_position == 1):
     - margin over the nearest relevant challenger behind him >= defend_min_margin_points -> DEFEND
     - otherwise -> BALANCED (lead is too thin to "defend")
3. Chris is not the leader:
     - deficit_per_remaining_gw = points_behind_leader / gameweeks_remaining
     - deficit_per_remaining_gw >= desperate_points_per_remaining_gw -> DESPERATE
       (the points swing required per remaining gameweek is no longer plausible)
     - else points_behind_leader >= attack_min_deficit_points -> ATTACK
     - else -> BALANCED (a close race, no clear-cut posture)
"""

from __future__ import annotations

from .config import EngineConfig
from .models import LeagueData, ManagerData
from .utils import safe_div

INSUFFICIENT_SEASON_DATA = "INSUFFICIENT_SEASON_DATA"
DEFEND = "DEFEND"
BALANCED = "BALANCED"
ATTACK = "ATTACK"
DESPERATE = "DESPERATE"


def classify_strategy_state(
    chris: ManagerData,
    league: LeagueData,
    relevant_rivals: list,
    config: EngineConfig,
) -> dict:
    metrics = {
        "latest_completed_event": league.latest_completed_event,
        "season_total_gameweeks": config.season_total_gameweeks,
        "gameweeks_remaining": None,
        "chris_league_position": chris.league_position,
        "chris_total_points": chris.total_points,
        "leader_total_points": None,
        "points_behind_leader": None,
        "points_ahead_of_nearest_below": None,
        "deficit_per_remaining_gw": None,
    }

    if league.latest_completed_event is None:
        return {
            "state": INSUFFICIENT_SEASON_DATA,
            "reasoning": "No gameweek has been completed yet this season.",
            "metrics": metrics,
        }

    if chris.league_position is None or chris.total_points is None:
        return {
            "state": INSUFFICIENT_SEASON_DATA,
            "reasoning": "Chris has no scored league position/points yet.",
            "metrics": metrics,
        }

    scored_managers = [m for m in league.managers if m.league_position is not None and m.total_points is not None]
    leader = min(scored_managers, key=lambda m: m.league_position, default=None)
    if leader is None:
        return {
            "state": INSUFFICIENT_SEASON_DATA,
            "reasoning": "No manager in this league has a scored position yet.",
            "metrics": metrics,
        }

    gws_remaining = max(0, config.season_total_gameweeks - league.latest_completed_event)
    metrics["gameweeks_remaining"] = gws_remaining
    metrics["leader_total_points"] = leader.total_points
    points_behind_leader = leader.total_points - chris.total_points
    metrics["points_behind_leader"] = points_behind_leader

    below = [r for r in relevant_rivals if r["points_gap"] < 0]
    margin_ahead_of_below = None
    if below:
        nearest_below = max(below, key=lambda r: r["points_gap"])  # gap closest to 0 (least negative)
        margin_ahead_of_below = -nearest_below["points_gap"]
        metrics["points_ahead_of_nearest_below"] = margin_ahead_of_below

    if chris.league_position == 1:
        if margin_ahead_of_below is not None and margin_ahead_of_below >= config.defend_min_margin_points:
            return {"state": DEFEND, "reasoning": (
                f"Chris leads by {margin_ahead_of_below} points over the nearest relevant challenger "
                f"(>= {config.defend_min_margin_points} threshold)."
            ), "metrics": metrics}
        return {"state": BALANCED, "reasoning": (
            "Chris leads the league but the margin over the nearest relevant challenger is thin "
            "or unknown - not yet a clear DEFEND posture."
        ), "metrics": metrics}

    deficit_per_gw = safe_div(points_behind_leader, gws_remaining) if gws_remaining else None
    metrics["deficit_per_remaining_gw"] = round(deficit_per_gw, 2) if deficit_per_gw is not None else None

    if deficit_per_gw is not None and deficit_per_gw >= config.desperate_points_per_remaining_gw:
        return {"state": DESPERATE, "reasoning": (
            f"{points_behind_leader} points behind the leader with only {gws_remaining} gameweek(s) "
            f"remaining requires {round(deficit_per_gw, 1)} pts/gw swing, at/above the "
            f"{config.desperate_points_per_remaining_gw} threshold."
        ), "metrics": metrics}

    if points_behind_leader >= config.attack_min_deficit_points:
        return {"state": ATTACK, "reasoning": (
            f"{points_behind_leader} points behind the leader (>= {config.attack_min_deficit_points} "
            "threshold) with a plausible recovery pace remaining."
        ), "metrics": metrics}

    return {"state": BALANCED, "reasoning": (
        f"Only {points_behind_leader} points behind the leader - close enough that no clear "
        "posture is warranted yet."
    ), "metrics": metrics}
