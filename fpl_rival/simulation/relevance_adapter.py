"""Thin adapter reusing Stage 2's relevant-rival selection, without coupling
the simulation layer's data model to the intelligence layer's.

``fpl_rival.intelligence.relevance.compute_relevant_rivals`` already
implements a documented, configurable, points-gap-and-position relevance
score (see Stage 2). Rather than reimplement or duplicate that logic, this
module translates a simulation ``LeagueState`` into the small subset of
``fpl_rival.intelligence.models`` shapes that function actually needs
(entry_id, total_points, league_position - nothing else), calls it, and
translates the result back into a plain list of entry ids. If Stage 2's
relevance module is ever unavailable or the translation fails for any
reason, this falls back to treating every other manager as relevant -
Captain Battle still works, just without the narrower rival focus.
"""

from __future__ import annotations

from typing import List

from .models import LeagueState, ManagerState


def default_primary_rival_ids(chris: ManagerState, league: LeagueState) -> List[int]:
    all_other_ids = [m.entry_id for m in league.managers if m.entry_id != chris.entry_id]
    try:
        from fpl_rival.intelligence.config import EngineConfig
        from fpl_rival.intelligence.models import LeagueData, ManagerData
        from fpl_rival.intelligence.relevance import compute_relevant_rivals
    except ImportError:
        return all_other_ids

    def _to_manager_data(m: ManagerState) -> ManagerData:
        return ManagerData(
            entry_id=m.entry_id,
            manager_name=m.name,
            team_name=m.name,
            league_position=m.current_league_position,
            total_points=m.current_total_points,
        )

    intel_chris = _to_manager_data(chris)
    intel_league = LeagueData(
        league_id=league.league_id or 0,
        league_name=league.league_name,
        managers=tuple(_to_manager_data(m) for m in league.managers),
    )

    try:
        result = compute_relevant_rivals(intel_chris, intel_league, EngineConfig())
    except Exception:
        return all_other_ids

    relevant_ids = [r["entry_id"] for r in result.get("relevant_rivals", [])]
    return relevant_ids or all_other_ids
