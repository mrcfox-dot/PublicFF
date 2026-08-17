"""Section 4: league-wide consensus - ownership, template, differentiation.

All percentages are computed only over managers who have a latest-gameweek
squad available (``managers_with_squad_data``); if that number is zero the
whole consensus block reports empty results with a clear note rather than
inventing figures.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

from .config import EngineConfig
from .models import GameContext, LeagueData
from .utils import overlap_stats, safe_pct


def build_league_consensus(league: LeagueData, ctx: GameContext, config: EngineConfig) -> dict:
    squads = {m.entry_id: m.latest_squad for m in league.managers if m.latest_squad is not None}
    n = len(squads)

    result = {
        "managers_with_squad_data": n,
        "managers_total": len(league.managers),
        "player_ownership": [],
        "captain_ownership": [],
        "most_common_players": [],
        "most_common_captain": None,
        "most_template_managers": [],
        "most_differentiated_managers": [],
        "average_pairwise_squad_overlap_pct": None,
        "unique_players": [],
        "note": None,
    }

    if n == 0:
        result["note"] = "No manager in this league has retrievable squad data yet - consensus unavailable."
        return result

    owner_counts = Counter()
    owner_map = {}  # element -> [entry_id, ...]
    captain_counts = Counter()
    for entry_id, squad in squads.items():
        for p in squad.picks:
            owner_counts[p.element] += 1
            owner_map.setdefault(p.element, []).append(entry_id)
        cap = squad.captain_element
        if cap is not None:
            captain_counts[cap] += 1

    ownership = [
        {
            "element": element,
            "name": ctx.player_name(element),
            "owners": count,
            "mini_league_ownership_percentage": safe_pct(count, n),
        }
        for element, count in owner_counts.items()
    ]
    ownership.sort(key=lambda x: (-x["owners"], x["name"]))
    result["player_ownership"] = ownership
    result["most_common_players"] = ownership[: config.consensus_top_n]

    captains_with_data = sum(captain_counts.values())
    captain_ownership = [
        {
            "element": element,
            "name": ctx.player_name(element),
            "managers_captaining": count,
            "captain_ownership_percentage": safe_pct(count, captains_with_data),
        }
        for element, count in captain_counts.items()
    ]
    captain_ownership.sort(key=lambda x: (-x["managers_captaining"], x["name"]))
    result["captain_ownership"] = captain_ownership
    if captain_ownership:
        result["most_common_captain"] = captain_ownership[0]

    result["unique_players"] = [
        {"element": e, "name": ctx.player_name(e), "owner_entry_id": owners[0]}
        for e, owners in owner_map.items()
        if len(owners) == 1
    ]

    # Pairwise overlap: also drives "template" (highest avg overlap with the
    # rest of the league) vs. "differentiated" (lowest avg overlap) managers.
    entry_ids = list(squads.keys())
    avg_overlap_by_entry = {eid: [] for eid in entry_ids}
    pairwise_pcts = []
    for a, b in combinations(entry_ids, 2):
        a_ids = {p.element for p in squads[a].picks}
        b_ids = {p.element for p in squads[b].picks}
        stats = overlap_stats(a_ids, b_ids)
        pct = stats["jaccard_pct"]
        if pct is not None:
            pairwise_pcts.append(pct)
            avg_overlap_by_entry[a].append(pct)
            avg_overlap_by_entry[b].append(pct)

    if pairwise_pcts:
        result["average_pairwise_squad_overlap_pct"] = round(sum(pairwise_pcts) / len(pairwise_pcts), 1)

    manager_avg = []
    for eid, pcts in avg_overlap_by_entry.items():
        if not pcts:
            continue
        mgr = league.manager(eid)
        manager_avg.append(
            {
                "entry_id": eid,
                "manager_name": mgr.manager_name if mgr else None,
                "team_name": mgr.team_name if mgr else None,
                "average_overlap_with_league_pct": round(sum(pcts) / len(pcts), 1),
            }
        )

    manager_avg_by_template = sorted(manager_avg, key=lambda x: -x["average_overlap_with_league_pct"])
    result["most_template_managers"] = manager_avg_by_template[: config.consensus_top_n]
    result["most_differentiated_managers"] = list(reversed(manager_avg_by_template))[: config.consensus_top_n]
    # Full (unsliced) list, keyed for O(1) lookup - used by risk classification
    # to find one manager's squad-deviation-from-league figure.
    result["manager_average_overlap_by_entry"] = {m["entry_id"]: m["average_overlap_with_league_pct"] for m in manager_avg}

    return result
