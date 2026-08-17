"""Section 6: relevant rival selection.

Deterministic, points-gap-and-position based - not a fixed "top 5". Each
rival gets a relevance_score in [0, 1]:

    relevance_score = 1 - (gap_weight * normalized_points_gap
                            + position_weight * normalized_position_distance)

normalized_points_gap = |rival_points - chris_points| / max_abs_gap_in_league
normalized_position_distance = |rival_position - chris_position| / (managers - 1)

A rival is "relevant" if relevance_score >= config.relevance_score_threshold,
OR they sit within config.always_relevant_position_distance places of Chris
in the table (the people he can see are always relevant, however the score
formula happens to land). The resulting group is capped between
min_relevant_rivals and max_relevant_rivals for practicality in very large
or very tightly-packed leagues - both bounds are configurable, not the
selection criterion itself.
"""

from __future__ import annotations

from .config import EngineConfig
from .models import LeagueData, ManagerData
from .utils import clamp


def compute_relevant_rivals(chris: ManagerData, league: LeagueData, config: EngineConfig) -> dict:
    result = {"relevant_rivals": [], "primary_rivals": [], "note": None}

    if chris.total_points is None or chris.league_position is None:
        result["note"] = "Chris has no scored league position/points yet - relevance cannot be computed."
        return result

    scored_rivals = [
        m for m in league.managers
        if m.entry_id != chris.entry_id and m.total_points is not None and m.league_position is not None
    ]
    if not scored_rivals:
        result["note"] = "No rival has scored league points yet - relevance cannot be computed."
        return result

    gaps = [abs(m.total_points - chris.total_points) for m in scored_rivals]
    max_abs_gap = max(gaps) or 1  # avoid /0 when everyone is tied
    max_position_distance = max(1, len(league.managers) - 1)

    scored = []
    for m in scored_rivals:
        gap = m.total_points - chris.total_points
        position_distance = abs(m.league_position - chris.league_position)
        normalized_gap = abs(gap) / max_abs_gap
        normalized_position = position_distance / max_position_distance
        relevance_score = 1 - clamp(
            config.relevance_gap_weight * normalized_gap
            + config.relevance_position_weight * normalized_position
        )
        always_relevant = position_distance <= config.always_relevant_position_distance
        scored.append(
            {
                "entry_id": m.entry_id,
                "manager_name": m.manager_name,
                "team_name": m.team_name,
                "league_position": m.league_position,
                "points_gap": gap,  # positive: rival ahead of Chris; negative: rival behind
                "position_distance": position_distance,
                "relevance_score": round(relevance_score, 3),
                "always_relevant": always_relevant,
            }
        )

    scored.sort(key=lambda r: -r["relevance_score"])

    qualifying = [r for r in scored if r["always_relevant"] or r["relevance_score"] >= config.relevance_score_threshold]
    if len(qualifying) < config.min_relevant_rivals:
        qualifying = scored[: config.min_relevant_rivals]
    if len(qualifying) > config.max_relevant_rivals:
        qualifying = qualifying[: config.max_relevant_rivals]

    result["relevant_rivals"] = qualifying

    if qualifying:
        top_score = qualifying[0]["relevance_score"]
        result["primary_rivals"] = [r for r in qualifying if r["relevance_score"] == top_score]

    return result
