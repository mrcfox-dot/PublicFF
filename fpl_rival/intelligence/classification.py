"""Section 3: behavioural classifications.

Every classification is computed from named, inspectable metrics - no
opaque scoring. When there isn't enough history for a fair read, the
functions here return ``"Insufficient data"`` instead of guessing.
"""

from __future__ import annotations

from .config import EngineConfig
from .models import ManagerData
from .utils import clamp, safe_pct


def classify_transfer_behaviour(profile: dict, config: EngineConfig) -> dict:
    gws_played = profile.get("gameweeks_played") or 0
    avg = profile.get("average_transfers_per_gameweek")

    result = {
        "classification": "Insufficient data",
        "metrics": {
            "gameweeks_played": gws_played,
            "min_gameweeks_required": config.min_gameweeks_for_classification,
            "average_transfers_per_gameweek": avg,
        },
    }

    if gws_played < config.min_gameweeks_for_classification or avg is None:
        return result

    for label, min_avg in config.transfer_bands:
        if avg >= min_avg:
            result["classification"] = label
            break
    else:
        result["classification"] = config.transfer_bands[-1][0]

    return result


def classify_risk_behaviour(
    manager: ManagerData,
    profile: dict,
    consensus: dict,
    config: EngineConfig,
) -> dict:
    gws_played = profile.get("gameweeks_played") or 0

    metrics = {
        "gameweeks_played": gws_played,
        "min_gameweeks_required": config.min_gameweeks_for_classification,
        "hits_taken": profile.get("hits_taken"),
        "average_transfers_per_gameweek": profile.get("average_transfers_per_gameweek"),
        "unique_captains_used": None,
        "differential_ownership_pct": None,
        "squad_deviation_from_league_pct": None,
    }
    result = {"classification": "Insufficient data", "risk_score": None, "components": {}, "metrics": metrics}

    if gws_played < config.min_gameweeks_for_classification:
        return result

    squad = manager.latest_squad
    if squad is None or consensus.get("managers_with_squad_data", 0) == 0:
        return result

    captain_concentration = profile.get("captain_concentration")
    unique_captains = captain_concentration["unique_captains_used"] if captain_concentration else None
    captain_datapoints = captain_concentration["total_captain_data_points"] if captain_concentration else 0
    metrics["unique_captains_used"] = unique_captains

    ownership_by_element = {row["element"]: row["mini_league_ownership_percentage"] for row in consensus["player_ownership"]}
    squad_elements = [p.element for p in squad.picks]
    low_owned = [
        e for e in squad_elements
        if ownership_by_element.get(e, 0.0) is not None and ownership_by_element.get(e, 0.0) <= config.low_ownership_threshold_pct
    ]
    differential_pct = safe_pct(len(low_owned), len(squad_elements)) if squad_elements else None
    metrics["differential_ownership_pct"] = differential_pct

    avg_overlap_pct = consensus.get("manager_average_overlap_by_entry", {}).get(manager.entry_id)
    squad_deviation_pct = round(100 - avg_overlap_pct, 1) if avg_overlap_pct is not None else None
    metrics["squad_deviation_from_league_pct"] = squad_deviation_pct

    # -- normalize each component to [0, 1], 1 = more "aggressive" --------------
    hits = profile.get("hits_taken") or 0
    hits_component = clamp(hits / config.risk_hits_saturation) if config.risk_hits_saturation else 0.0

    differential_component = clamp(differential_pct / 100.0) if differential_pct is not None else None

    if unique_captains is not None and captain_datapoints >= config.min_captain_datapoints:
        variety_ratio = unique_captains / captain_datapoints
        captain_variety_component = clamp(variety_ratio / config.risk_captain_variety_saturation)
    else:
        captain_variety_component = None

    avg_transfers = profile.get("average_transfers_per_gameweek")
    transfer_freq_component = (
        clamp(avg_transfers / config.risk_transfer_freq_saturation) if avg_transfers is not None else None
    )

    deviation_component = clamp(squad_deviation_pct / 100.0) if squad_deviation_pct is not None else None

    components = {
        "hits": {"value": hits_component, "weight": config.risk_weight_hits, "raw": hits},
        "differential_ownership": {
            "value": differential_component,
            "weight": config.risk_weight_differential,
            "raw": differential_pct,
        },
        "captain_variety": {
            "value": captain_variety_component,
            "weight": config.risk_weight_captain_variety,
            "raw": unique_captains,
        },
        "transfer_frequency": {
            "value": transfer_freq_component,
            "weight": config.risk_weight_transfer_frequency,
            "raw": avg_transfers,
        },
        "squad_deviation": {
            "value": deviation_component,
            "weight": config.risk_weight_squad_deviation,
            "raw": squad_deviation_pct,
        },
    }
    result["components"] = components

    available = [(c["value"], c["weight"]) for c in components.values() if c["value"] is not None]
    if not available:
        return result

    weight_sum = sum(w for _, w in available)
    if weight_sum == 0:
        return result
    risk_score = sum(v * w for v, w in available) / weight_sum
    result["risk_score"] = round(risk_score, 3)

    if risk_score >= config.risk_aggressive_threshold:
        result["classification"] = "Aggressive"
    elif risk_score <= config.risk_conservative_threshold:
        result["classification"] = "Conservative"
    else:
        result["classification"] = "Balanced"

    return result
