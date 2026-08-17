"""Builds the Stage 1 summary: positions, gaps, squad overlap, captains.

Every figure here is derived directly from retrieved API data. Where the
underlying data does not exist (e.g. pre-season, no scored gameweek yet),
the corresponding field is set to None with an explanatory note instead of
being estimated or invented.
"""

from __future__ import annotations

from typing import Optional


def get_latest_finished_event(bootstrap: dict) -> Optional[int]:
    finished = [e for e in bootstrap.get("events", []) if e.get("finished")]
    if not finished:
        return None
    return max(finished, key=lambda e: e["id"])["id"]


def build_element_name_map(bootstrap: dict) -> dict:
    return {el["id"]: el.get("web_name", f"Player {el['id']}") for el in bootstrap.get("elements", [])}


def build_position_summary(my_entry_id: int, managers: list[dict]) -> dict:
    """My position, gap to leader, gap to the manager below, league size."""
    ranked = [m for m in managers if m.get("league_position") is not None]
    ranked.sort(key=lambda m: m["league_position"])

    summary = {
        "num_managers": len(managers),
        "num_managers_with_scores": len(ranked),
        "my_position": None,
        "my_total_points": None,
        "points_behind_leader": None,
        "points_ahead_of_below": None,
        "note": None,
    }

    if not ranked:
        summary["note"] = (
            "No manager in this league has a scored gameweek yet (pre-season, or "
            "current gameweek not yet finished) - position/points are not available "
            "from the public API."
        )
        return summary

    me = next((m for m in ranked if m["entry_id"] == my_entry_id), None)
    if me is None:
        summary["note"] = "Your entry was not found in this league's scored standings."
        return summary

    leader = ranked[0]
    my_index = ranked.index(me)

    summary["my_position"] = me["league_position"]
    summary["my_total_points"] = me["total_points"]
    summary["points_behind_leader"] = (leader["total_points"] or 0) - (me["total_points"] or 0)

    if my_index + 1 < len(ranked):
        below = ranked[my_index + 1]
        summary["points_ahead_of_below"] = (me["total_points"] or 0) - (below["total_points"] or 0)
    else:
        summary["note"] = "You are last among scored managers - no one below you to compare to."

    return summary


def build_squad_comparisons(
    my_entry_id: int,
    manager_details: dict[int, dict],
    name_map: dict,
) -> dict:
    """Squad overlap + exclusive players + captain choices, latest completed GW."""
    result = {
        "latest_completed_gameweek": None,
        "comparisons": [],
        "note": None,
    }

    me = manager_details.get(my_entry_id)
    if me is None:
        result["note"] = "No data retrieved for your own entry."
        return result

    result["latest_completed_gameweek"] = me.get("latest_gameweek_used_for_picks")

    if not me.get("picks"):
        result["note"] = (
            "Your squad for the latest completed gameweek is not available "
            "(no completed gameweek yet, or picks not public). Squad comparisons skipped."
        )
        return result

    my_elements = {p["element"] for p in me["picks"].get("picks", [])}
    my_captain = me.get("captain_element")

    for entry_id, detail in manager_details.items():
        if entry_id == my_entry_id:
            continue
        entry_summary = {
            "entry_id": entry_id,
            "team_name": (detail.get("entry_summary") or {}).get("name"),
            "overlap_count": None,
            "shared_players": [],
            "players_only_i_own": [],
            "players_only_rival_owns": [],
            "my_captain": name_map.get(my_captain, my_captain) if my_captain else None,
            "rival_captain": None,
            "note": None,
        }

        if not detail.get("picks"):
            entry_summary["note"] = "Rival's squad for this gameweek is not available."
            result["comparisons"].append(entry_summary)
            continue

        rival_elements = {p["element"] for p in detail["picks"].get("picks", [])}
        shared = my_elements & rival_elements
        mine_only = my_elements - rival_elements
        rival_only = rival_elements - my_elements

        entry_summary["overlap_count"] = len(shared)
        entry_summary["shared_players"] = sorted(name_map.get(e, str(e)) for e in shared)
        entry_summary["players_only_i_own"] = sorted(name_map.get(e, str(e)) for e in mine_only)
        entry_summary["players_only_rival_owns"] = sorted(name_map.get(e, str(e)) for e in rival_only)

        rival_captain = detail.get("captain_element")
        entry_summary["rival_captain"] = name_map.get(rival_captain, rival_captain) if rival_captain else None

        result["comparisons"].append(entry_summary)

    return result
