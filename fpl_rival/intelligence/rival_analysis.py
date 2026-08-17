"""Section 2: Chris vs. one rival - overlap, differentials, captain exposure.

Methodology (kept deliberately simple and deterministic):

* Squad overlap = shared elements / squad size, computed separately for the
  full 15 and for the starting XI (only when both managers have XI data).
* Captain overlap % = (# comparable gameweeks where both captained the same
  player) / (# comparable gameweeks), where "comparable" means both
  managers have a known captain for that event.
* Effective exposure % = 100 * (w_squad * squad_overlap_fraction_15
                                 + w_captain * captain_overlap_fraction)
  using the weights in ``EngineConfig``. This is a simple linear blend, not
  a statistically fitted model - see the Stage 2 completion notes for the
  documented limitation.
"""

from __future__ import annotations

from typing import Optional

from .config import EngineConfig
from .models import GameContext, ManagerData
from .utils import overlap_stats, safe_pct


def _element_ids(squad, xi_only: bool) -> Optional[set]:
    if squad is None:
        return None
    picks = squad.starting_xi if xi_only else squad.picks
    return {p.element for p in picks}


def _captain_overlap(chris: ManagerData, rival: ManagerData) -> dict:
    common_events = sorted(set(chris.squads_by_event) & set(rival.squads_by_event))
    comparable = []
    for event in common_events:
        c_cap = chris.squads_by_event[event].captain_element
        r_cap = rival.squads_by_event[event].captain_element
        if c_cap is not None and r_cap is not None:
            comparable.append((event, c_cap, r_cap))

    if not comparable:
        return {
            "comparable_gameweeks": 0,
            "matching_gameweeks": 0,
            "captain_overlap_pct": None,
            "note": "No gameweeks with known captain data for both managers.",
        }

    matches = [e for e, c, r in comparable if c == r]
    return {
        "comparable_gameweeks": len(comparable),
        "matching_gameweeks": len(matches),
        "captain_overlap_pct": safe_pct(len(matches), len(comparable)),
        "note": None,
    }


def compare_managers(
    chris: ManagerData, rival: ManagerData, ctx: GameContext, config: EngineConfig
) -> dict:
    chris_15 = _element_ids(chris.latest_squad, xi_only=False)
    rival_15 = _element_ids(rival.latest_squad, xi_only=False)
    chris_xi = _element_ids(chris.latest_squad, xi_only=True)
    rival_xi = _element_ids(rival.latest_squad, xi_only=True)

    result = {
        "entry_id": rival.entry_id,
        "manager_name": rival.manager_name,
        "team_name": rival.team_name,
        "squad_overlap_15": None,
        "squad_overlap_xi": None,
        "players_chris_only": [],
        "players_rival_only": [],
        "captain_exposure": None,
        "same_captain_latest_gw": None,
        "effective_exposure_pct": None,
        "effective_exposure_methodology": (
            f"{config.exposure_weight_squad_overlap:.0%} weight on 15-man squad overlap fraction, "
            f"{config.exposure_weight_captain_overlap:.0%} weight on multi-gameweek captain-overlap "
            "fraction (or latest-gameweek same-captain flag if no multi-gameweek history exists)."
        ),
        "data_gaps": [],
    }

    if chris_15 is None or rival_15 is None:
        result["data_gaps"].append("Latest-gameweek squad missing for one or both managers - overlap unavailable.")
        return result

    ov15 = overlap_stats(chris_15, rival_15)
    result["squad_overlap_15"] = {
        "shared_count": ov15["shared_count"],
        "shared_pct_of_chris_squad": ov15["overlap_pct_of_a"],
        "chris_squad_size": ov15["a_count"],
        "rival_squad_size": ov15["b_count"],
    }
    result["players_chris_only"] = sorted(ctx.player_name(e) for e in ov15["only_a_ids"])
    result["players_rival_only"] = sorted(ctx.player_name(e) for e in ov15["only_b_ids"])

    if chris_xi and rival_xi:
        ovxi = overlap_stats(chris_xi, rival_xi)
        result["squad_overlap_xi"] = {
            "shared_count": ovxi["shared_count"],
            "shared_pct_of_chris_xi": ovxi["overlap_pct_of_a"],
        }
    else:
        result["data_gaps"].append("Starting XI not available for one or both managers.")

    captain_exposure = _captain_overlap(chris, rival)
    result["captain_exposure"] = captain_exposure
    if captain_exposure["comparable_gameweeks"] < config.min_captain_datapoints:
        result["data_gaps"].append(
            f"Only {captain_exposure['comparable_gameweeks']} comparable captain gameweek(s) "
            f"(need {config.min_captain_datapoints}) - captain overlap is low-confidence."
        )

    chris_cap = chris.latest_squad.captain_element if chris.latest_squad else None
    rival_cap = rival.latest_squad.captain_element if rival.latest_squad else None
    if chris_cap is not None and rival_cap is not None:
        result["same_captain_latest_gw"] = chris_cap == rival_cap

    squad_overlap_fraction = ov15["shared_count"] / ov15["a_count"] if ov15["a_count"] else 0.0
    if captain_exposure["captain_overlap_pct"] is not None:
        captain_fraction = captain_exposure["captain_overlap_pct"] / 100.0
    elif result["same_captain_latest_gw"] is not None:
        captain_fraction = 1.0 if result["same_captain_latest_gw"] else 0.0
    else:
        captain_fraction = None

    if captain_fraction is not None:
        exposure = (
            config.exposure_weight_squad_overlap * squad_overlap_fraction
            + config.exposure_weight_captain_overlap * captain_fraction
        )
        result["effective_exposure_pct"] = round(100 * exposure, 1)
    else:
        result["data_gaps"].append("No captain data at all - effective exposure could not be computed.")

    return result
