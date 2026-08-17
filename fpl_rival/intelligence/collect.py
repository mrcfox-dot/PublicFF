"""Live data collection for the Rival Intelligence Engine (Stage 2).

Builds on Stage 1's retrieval layer (``fpl_rival.api``, ``fpl_rival.league``)
WITHOUT modifying it, and adds the one extra thing Stage 2 needs: picks for
every *finished* gameweek per manager (Stage 1 only fetches the latest
one), so that captain history, formation history, and bench-points-by-week
have something to work with once the season is live. Right now the season
has zero finished gameweeks, so this loop is a no-op - that's expected,
not a bug.

This module is retrieval, not analytics: it only builds ``ManagerData`` /
``LeagueData`` / ``GameContext`` objects. It never computes a metric.
"""

from __future__ import annotations

from fpl_rival.api import FPLAPIError, FPLClient
from fpl_rival.league import build_manager_list, fetch_league

from .models import (
    ChipEvent,
    GameContext,
    GameweekHistoryRow,
    GameweekSquad,
    LeagueData,
    ManagerData,
    Pick,
    TransferEvent,
)


def build_game_context(bootstrap: dict) -> GameContext:
    element_names = {el["id"]: el.get("web_name", f"Player {el['id']}") for el in bootstrap.get("elements", [])}
    element_types = {el["id"]: el.get("element_type") for el in bootstrap.get("elements", [])}

    chip_windows: dict = {}
    for chip in bootstrap.get("chips", []):
        chip_windows.setdefault(chip["name"], []).append((chip["start_event"], chip["stop_event"]))

    event_deadlines = {e["id"]: e.get("deadline_time") for e in bootstrap.get("events", [])}

    return GameContext(
        element_names=element_names,
        element_types=element_types,
        chip_windows=chip_windows,
        event_deadlines=event_deadlines,
        season_total_gameweeks=len(bootstrap.get("events", [])) or 38,
    )


def get_finished_events(bootstrap: dict) -> list:
    return sorted(e["id"] for e in bootstrap.get("events", []) if e.get("finished"))


def _convert_picks(event: int, raw_picks: dict, ctx: GameContext) -> GameweekSquad:
    picks = tuple(
        Pick(
            element=p["element"],
            position=p["position"],
            multiplier=p.get("multiplier", 0),
            is_captain=bool(p.get("is_captain")),
            is_vice_captain=bool(p.get("is_vice_captain")),
            element_type=ctx.element_types.get(p["element"]),
        )
        for p in raw_picks.get("picks", [])
    )
    entry_history = raw_picks.get("entry_history", {}) or {}
    return GameweekSquad(
        event=event,
        picks=picks,
        active_chip=raw_picks.get("active_chip"),
        points=entry_history.get("points"),
        points_on_bench=entry_history.get("points_on_bench"),
    )


def collect_manager_data(
    client: FPLClient,
    entry_id: int,
    manager_row: dict,
    finished_events: list,
    ctx: GameContext,
    errors: list,
    fetch_full_picks_history: bool = True,
) -> ManagerData:
    """Retrieves everything Stage 2 needs for one manager.

    ``manager_row`` is the merged standings/new_entries dict Stage 1's
    ``league.build_manager_list`` already produces - it supplies name,
    team name, league position and total points so we don't refetch them.
    """
    try:
        entry = client.get_entry(entry_id)
    except FPLAPIError as exc:
        errors.append(f"entry/{entry_id}: {exc}")
        entry = None

    try:
        history = client.get_entry_history(entry_id)
    except FPLAPIError as exc:
        errors.append(f"entry/{entry_id}/history: {exc}")
        history = None

    try:
        raw_transfers = client.get_entry_transfers(entry_id)
    except FPLAPIError as exc:
        errors.append(f"entry/{entry_id}/transfers: {exc}")
        raw_transfers = None

    gameweek_history: tuple = ()
    chips_used: tuple = ()
    past_seasons: tuple = ()
    if history is not None:
        gameweek_history = tuple(
            GameweekHistoryRow(
                event=row["event"],
                points=row.get("points"),
                total_points=row.get("total_points"),
                rank=row.get("rank"),
                overall_rank=row.get("overall_rank"),
                bank=row.get("bank"),
                value=row.get("value"),
                event_transfers=row.get("event_transfers"),
                event_transfers_cost=row.get("event_transfers_cost"),
                points_on_bench=row.get("points_on_bench"),
            )
            for row in history.get("current", [])
        )
        chips_used = tuple(ChipEvent(name=c["name"], event=c["event"]) for c in history.get("chips", []))
        past_seasons = tuple(
            {"season_name": s.get("season_name"), "total_points": s.get("total_points"), "rank": s.get("rank")}
            for s in history.get("past", [])
        )
    else:
        errors.append(f"entry/{entry_id}/history: not available.")

    transfers: tuple = ()
    if raw_transfers is not None:
        transfers = tuple(
            TransferEvent(
                element_in=t["element_in"], element_out=t["element_out"], event=t["event"], time=t.get("time")
            )
            for t in raw_transfers
        )
    else:
        errors.append(f"entry/{entry_id}/transfers: not available.")

    squads_by_event = {}
    events_to_fetch = finished_events if fetch_full_picks_history else finished_events[-1:]
    for event in events_to_fetch:
        try:
            raw_picks = client.get_entry_picks(entry_id, event)
        except FPLAPIError as exc:
            errors.append(f"entry/{entry_id}/event/{event}/picks: {exc}")
            continue
        if raw_picks is None:
            errors.append(f"entry/{entry_id}/event/{event}/picks: not available.")
            continue
        squads_by_event[event] = _convert_picks(event, raw_picks, ctx)

    return ManagerData(
        entry_id=entry_id,
        manager_name=manager_row.get("manager_name"),
        team_name=manager_row.get("team_name"),
        league_position=manager_row.get("league_position"),
        total_points=manager_row.get("total_points"),
        overall_points=entry.get("summary_overall_points") if entry else None,
        overall_rank=entry.get("summary_overall_rank") if entry else None,
        team_value=entry.get("last_deadline_value") if entry else None,
        bank=entry.get("last_deadline_bank") if entry else None,
        gameweek_history=gameweek_history,
        chips_used=chips_used,
        transfers=transfers,
        squads_by_event=squads_by_event,
        past_seasons=past_seasons,
        is_synthetic=False,
    )


def collect_league_data(
    client: FPLClient,
    league_id: int,
    bootstrap: dict,
    chris_entry_id: int = None,
    max_managers: int = 50,
    fetch_full_picks_history: bool = True,
) -> tuple:
    """Returns ``(LeagueData, GameContext, errors)`` for a live league.

    Reuses Stage 1's ``fetch_league``/``build_manager_list`` for standings
    and membership (including their pagination handling), then layers on
    the deeper per-manager retrieval Stage 2 needs. If ``chris_entry_id``
    would be excluded by ``max_managers``, he is always fetched anyway -
    the engine cannot run without him.
    """
    errors: list = []
    ctx = build_game_context(bootstrap)
    finished_events = get_finished_events(bootstrap)
    latest = finished_events[-1] if finished_events else None

    league_raw = fetch_league(client, league_id)
    errors.extend(league_raw.errors)
    manager_rows = build_manager_list(league_raw)

    if len(manager_rows) > max_managers:
        kept = manager_rows[:max_managers]
        if chris_entry_id is not None and chris_entry_id not in {r["entry_id"] for r in kept}:
            chris_row = next((r for r in manager_rows if r["entry_id"] == chris_entry_id), None)
            if chris_row is not None:
                kept.append(chris_row)
        errors.append(
            f"League has {len(manager_rows)} managers; limited deep fetch to {len(kept)} "
            f"(first {max_managers} by rank, plus Chris if he'd have been cut)."
        )
        manager_rows = kept

    managers = [
        collect_manager_data(client, row["entry_id"], row, finished_events, ctx, errors, fetch_full_picks_history)
        for row in manager_rows
    ]

    league_name = (league_raw.meta or {}).get("name") or f"League {league_id}"
    league_data = LeagueData(
        league_id=league_id,
        league_name=league_name,
        managers=tuple(managers),
        latest_completed_event=latest,
        is_synthetic=False,
    )
    return league_data, ctx, errors
