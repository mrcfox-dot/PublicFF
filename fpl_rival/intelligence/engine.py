"""Orchestrates every analytics module into one deterministic report.

This is the only entry point analytics callers should need. It never
touches the network - it operates purely on ``LeagueData``/``GameContext``,
which can come from live collection (``collect.py``) or from synthetic
fixtures (``fpl_rival/fixtures``), making it fully unit-testable.
"""

from __future__ import annotations

from .classification import classify_risk_behaviour, classify_transfer_behaviour
from .config import EngineConfig
from .consensus import build_league_consensus
from .models import GameContext, LeagueData
from .preseason import build_preseason_report
from .profiles import build_manager_profile
from .relevance import compute_relevant_rivals
from .rival_analysis import compare_managers
from .strategy import classify_strategy_state
from .threats import build_threat_analysis


class UnknownManagerError(ValueError):
    pass


def run(
    league: LeagueData,
    chris_entry_id: int,
    ctx: GameContext,
    config: EngineConfig = None,
) -> dict:
    config = config or EngineConfig()
    chris = league.manager(chris_entry_id)
    if chris is None:
        raise UnknownManagerError(
            f"Entry {chris_entry_id} is not a member of league {league.league_id} ({league.league_name})."
        )

    if league.latest_completed_event is None:
        return {
            "mode": "preseason",
            "league_id": league.league_id,
            "league_name": league.league_name,
            "is_synthetic": league.is_synthetic,
            "preseason": build_preseason_report(league, chris),
        }

    profiles = {m.entry_id: build_manager_profile(m, ctx, config) for m in league.managers}
    consensus = build_league_consensus(league, ctx, config)
    relevance = compute_relevant_rivals(chris, league, config)
    strategy = classify_strategy_state(chris, league, relevance["relevant_rivals"], config)
    threats = build_threat_analysis(chris, league, relevance["relevant_rivals"], ctx, config)

    comparisons = {}
    classifications = {}
    for m in league.managers:
        if m.entry_id == chris.entry_id:
            continue
        comparisons[m.entry_id] = compare_managers(chris, m, ctx, config)
        classifications[m.entry_id] = {
            "transfer_behaviour": classify_transfer_behaviour(profiles[m.entry_id], config),
            "risk_behaviour": classify_risk_behaviour(m, profiles[m.entry_id], consensus, config),
        }

    chris_classification = {
        "transfer_behaviour": classify_transfer_behaviour(profiles[chris.entry_id], config),
        "risk_behaviour": classify_risk_behaviour(chris, profiles[chris.entry_id], consensus, config),
    }

    primary_rival_entry_id = (
        relevance["primary_rivals"][0]["entry_id"] if relevance["primary_rivals"] else None
    )

    return {
        "mode": "active",
        "league_id": league.league_id,
        "league_name": league.league_name,
        "is_synthetic": league.is_synthetic,
        "chris_entry_id": chris.entry_id,
        "latest_completed_event": league.latest_completed_event,
        "profiles": profiles,
        "chris_classification": chris_classification,
        "consensus": consensus,
        "relevance": relevance,
        "strategy": strategy,
        "threats": threats,
        "comparisons": comparisons,
        "classifications": classifications,
        "primary_rival_entry_id": primary_rival_entry_id,
    }
