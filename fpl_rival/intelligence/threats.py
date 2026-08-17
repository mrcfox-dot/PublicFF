"""Section 5: shields, threats and weapons - relative league exposure only.

These functions make no claim about a player's underlying FPL quality -
only about how exposed Chris is to that player relative to the managers he
is actually competing with (his "relevant rivals", see relevance.py).

* Shield: a player Chris owns who is also heavily owned by rivals at or
  around his own position - owning them limits relative movement either way.
* Threat: a player heavily owned/captained among Chris's relevant rivals
  that Chris does NOT own - a source of relative risk if they haul.
* Weapon: a player Chris owns that his relevant rivals largely do not - a
  source of relative upside/downside versus that specific group.
"""

from __future__ import annotations

from collections import Counter

from .config import EngineConfig
from .models import GameContext, LeagueData, ManagerData
from .utils import safe_pct


def _ownership_within(managers: list, ctx: GameContext) -> dict:
    """Player + captain ownership % within an arbitrary subset of managers."""
    squads = [m.latest_squad for m in managers if m.latest_squad is not None]
    n = len(squads)
    if n == 0:
        return {"n": 0, "player_pct": {}, "captain_pct": {}}

    owner_counts = Counter()
    captain_counts = Counter()
    for squad in squads:
        for p in squad.picks:
            owner_counts[p.element] += 1
        cap = squad.captain_element
        if cap is not None:
            captain_counts[cap] += 1

    player_pct = {e: safe_pct(c, n) for e, c in owner_counts.items()}
    captain_pct = {e: safe_pct(c, n) for e, c in captain_counts.items()}
    return {"n": n, "player_pct": player_pct, "captain_pct": captain_pct}


def build_threat_analysis(
    chris: ManagerData,
    league: LeagueData,
    relevant_rivals: list,
    ctx: GameContext,
    config: EngineConfig,
) -> dict:
    result = {
        "shields": [],
        "threats": [],
        "weapons": [],
        "relevant_rivals_with_squad_data": 0,
        "shield_rivals_with_squad_data": 0,
        "note": None,
    }

    if chris.latest_squad is None:
        result["note"] = "Chris has no retrievable squad yet - shields/threats/weapons unavailable."
        return result

    rival_entry_ids = {r["entry_id"] for r in relevant_rivals}
    rival_managers = [m for m in league.managers if m.entry_id in rival_entry_ids]
    if not rival_managers:
        result["note"] = "No relevant rivals identified yet - shields/threats/weapons unavailable."
        return result

    chris_ids = {p.element for p in chris.latest_squad.picks}

    # -- Threats & weapons: ownership among ALL relevant rivals -----------------
    relevant_ownership = _ownership_within(rival_managers, ctx)
    result["relevant_rivals_with_squad_data"] = relevant_ownership["n"]

    for element, pct in relevant_ownership["player_pct"].items():
        if element in chris_ids:
            continue
        captain_pct = relevant_ownership["captain_pct"].get(element)
        if pct >= config.high_ownership_threshold_pct or (captain_pct or 0) >= config.high_ownership_threshold_pct:
            result["threats"].append(
                {
                    "element": element,
                    "name": ctx.player_name(element),
                    "ownership_pct_among_relevant_rivals": pct,
                    "captain_pct_among_relevant_rivals": captain_pct,
                }
            )

    for element in chris_ids:
        pct = relevant_ownership["player_pct"].get(element, 0.0)
        if pct <= config.low_ownership_threshold_pct:
            result["weapons"].append(
                {"element": element, "name": ctx.player_name(element), "ownership_pct_among_relevant_rivals": pct}
            )

    # -- Shields: ownership among rivals at/around or above Chris's position ----
    shield_rivals = [
        m for m in rival_managers
        if any(
            r["entry_id"] == m.entry_id
            and (r["position_distance"] <= config.always_relevant_position_distance or r["points_gap"] >= 0)
            for r in relevant_rivals
        )
    ]
    shield_ownership = _ownership_within(shield_rivals, ctx)
    result["shield_rivals_with_squad_data"] = shield_ownership["n"]

    for element in chris_ids:
        pct = shield_ownership["player_pct"].get(element, 0.0)
        if pct >= config.high_ownership_threshold_pct:
            result["shields"].append(
                {"element": element, "name": ctx.player_name(element), "ownership_pct_among_shield_rivals": pct}
            )

    for bucket in ("threats", "weapons", "shields"):
        result[bucket].sort(key=lambda x: -(x.get("ownership_pct_among_relevant_rivals") or x.get("ownership_pct_among_shield_rivals") or 0))

    return result
