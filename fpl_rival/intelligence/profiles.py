"""Section 1: per-manager profile metrics.

Every metric here is computed directly from ``ManagerData`` / ``GameContext``.
Anything that needs data we don't have (e.g. a full picks history when only
the latest gameweek's picks were retrieved) is reported as ``None`` with a
matching entry in ``data_gaps`` - never guessed.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Optional

from .config import EngineConfig
from .models import GameContext, ManagerData
from .utils import safe_div, safe_pct


def _squad_names(squad, ctx: GameContext) -> list:
    if squad is None:
        return []
    return [
        {
            "element": p.element,
            "name": ctx.player_name(p.element),
            "position_slot": p.position,
            "multiplier": p.multiplier,
            "is_captain": p.is_captain,
            "is_vice_captain": p.is_vice_captain,
        }
        for p in sorted(squad.picks, key=lambda x: x.position)
    ]


def _chips_remaining(manager: ManagerData, ctx: GameContext) -> Optional[dict]:
    """For each chip name, how many of its seasonal windows are unused.

    Uses FPL's own chip windows (bootstrap-static 'chips': each chip name
    has one usable window per half-season). A window counts as used if any
    of the manager's recorded chip events falls inside it.
    """
    if not ctx.chip_windows:
        return None
    remaining = {}
    for name, windows in ctx.chip_windows.items():
        used_events = [c.event for c in manager.chips_used if c.name == name]
        unused_windows = 0
        for start, stop in windows:
            if not any(start <= ev <= stop for ev in used_events):
                unused_windows += 1
        remaining[name] = unused_windows
    return remaining


def _transfers_by_event(manager: ManagerData) -> dict:
    by_event = {}
    for row in manager.gameweek_history:
        if row.event_transfers is not None:
            by_event[row.event] = row.event_transfers
    return by_event


def _captain_history(manager: ManagerData, ctx: GameContext) -> list:
    history = []
    for event in sorted(manager.squads_by_event):
        squad = manager.squads_by_event[event]
        cap = squad.captain_element
        if cap is not None:
            history.append({"event": event, "captain": ctx.player_name(cap), "element": cap})
    return history


def _captain_concentration(captain_history: list) -> Optional[dict]:
    if not captain_history:
        return None
    counts = Counter(c["element"] for c in captain_history)
    most_common_element, most_common_count = counts.most_common(1)[0]
    total = len(captain_history)
    return {
        "unique_captains_used": len(counts),
        "total_captain_data_points": total,
        "most_captained_element": most_common_element,
        "most_captained_count": most_common_count,
        "concentration_pct": safe_pct(most_common_count, total),
    }


def _formation_history(manager: ManagerData) -> list:
    history = []
    for event in sorted(manager.squads_by_event):
        formation = manager.squads_by_event[event].formation
        if formation is not None:
            history.append({"event": event, "formation": formation})
    return history


def _bench_points(manager: ManagerData) -> dict:
    """Sourced from gameweek_history rows (always available once history is
    fetched - does not require a picks call for every gameweek)."""
    by_event = {
        row.event: row.points_on_bench
        for row in manager.gameweek_history
        if row.points_on_bench is not None
    }
    total = sum(by_event.values()) if by_event else None
    return {"by_event": by_event, "total_season": total}


def _transfer_timing(manager: ManagerData, ctx: GameContext, config: EngineConfig) -> Optional[dict]:
    samples = []
    for t in manager.transfers:
        deadline_str = ctx.event_deadlines.get(t.event)
        if not t.time or not deadline_str:
            continue
        try:
            transfer_dt = datetime.fromisoformat(t.time.replace("Z", "+00:00"))
            deadline_dt = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        hours_before = (deadline_dt - transfer_dt).total_seconds() / 3600.0
        samples.append(hours_before)

    if not samples:
        return None

    samples.sort()
    mid = len(samples) // 2
    median = samples[mid] if len(samples) % 2 else (samples[mid - 1] + samples[mid]) / 2
    late_count = sum(1 for h in samples if h < config.late_transfer_threshold_hours)
    return {
        "sample_count": len(samples),
        "median_hours_before_deadline": round(median, 1),
        "late_transfer_count": late_count,
        "late_transfer_pct": safe_pct(late_count, len(samples)),
    }


def build_manager_profile(manager: ManagerData, ctx: GameContext, config: EngineConfig) -> dict:
    data_gaps = []

    squad = manager.latest_squad
    if squad is None:
        data_gaps.append("No squad/picks data available - current squad, XI, bench, captain unknown.")

    transfers_by_event = _transfers_by_event(manager)
    total_transfers = sum(transfers_by_event.values()) if transfers_by_event else (
        len(manager.transfers) if manager.transfers else None
    )
    if not transfers_by_event and not manager.transfers:
        data_gaps.append("No transfer/history data available - transfer counts unknown.")

    gws_played = len(manager.gameweek_history)
    avg_transfers = safe_div(total_transfers, gws_played) if total_transfers is not None and gws_played else None

    total_hit_cost = None
    hits_count = None
    if manager.gameweek_history:
        costs = [r.event_transfers_cost for r in manager.gameweek_history if r.event_transfers_cost is not None]
        if costs:
            total_hit_cost = sum(costs)
            hits_count = total_hit_cost // 4 if total_hit_cost else 0

    captain_history = _captain_history(manager, ctx)
    if not captain_history:
        data_gaps.append("No per-gameweek picks retrieved - captain history/concentration unavailable.")

    formation_history = _formation_history(manager)
    if manager.squads_by_event and not formation_history:
        data_gaps.append("Player element types unknown - formation history could not be computed.")

    chips_remaining = _chips_remaining(manager, ctx)
    if chips_remaining is None:
        data_gaps.append("Chip window rules unavailable - chips remaining unknown.")

    transfer_timing = _transfer_timing(manager, ctx, config)
    if transfer_timing is None and manager.transfers:
        data_gaps.append("Transfer timestamps or gameweek deadlines unavailable - transfer timing unknown.")

    team_value = manager.team_value / 10 if manager.team_value is not None else None

    return {
        "entry_id": manager.entry_id,
        "manager_name": manager.manager_name,
        "team_name": manager.team_name,
        "league_position": manager.league_position,
        "overall_rank": manager.overall_rank,
        "total_points": manager.total_points,
        "overall_points": manager.overall_points,
        "current_squad": _squad_names(squad, ctx),
        "starting_xi": _squad_names(
            None if squad is None else type(squad)(squad.event, squad.starting_xi), ctx
        ),
        "bench": _squad_names(
            None if squad is None else type(squad)(squad.event, squad.bench), ctx
        ),
        "captain": ctx.player_name(squad.captain_element) if squad and squad.captain_element else None,
        "vice_captain": ctx.player_name(squad.vice_captain_element) if squad and squad.vice_captain_element else None,
        "chips_used": [{"name": c.name, "event": c.event} for c in manager.chips_used],
        "chips_remaining": chips_remaining,
        "transfers_by_event": transfers_by_event,
        "total_transfers": total_transfers,
        "hits_taken": hits_count,
        "points_lost_to_hits": total_hit_cost,
        "average_transfers_per_gameweek": round(avg_transfers, 2) if avg_transfers is not None else None,
        "gameweeks_played": gws_played,
        "captain_history": captain_history,
        "captain_concentration": _captain_concentration(captain_history),
        "formation_history": formation_history,
        "bench_points": _bench_points(manager),
        "transfer_timing": transfer_timing,
        "team_value_millions": team_value,
        "bank_millions": manager.bank / 10 if manager.bank is not None else None,
        "data_gaps": data_gaps,
    }
