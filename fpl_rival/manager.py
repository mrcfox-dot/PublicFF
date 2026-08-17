"""Per-manager data retrieval: history, latest picks, transfers, chips."""

from __future__ import annotations

from typing import Optional

from .api import FPLAPIError, FPLClient


def fetch_manager_detail(
    client: FPLClient,
    entry_id: int,
    latest_finished_event: Optional[int],
    errors: list,
) -> dict:
    """Gathers as much public data as is available for one manager.

    Any single failed call is recorded in ``errors`` and the corresponding
    field is left as None / empty rather than guessed at.
    """
    detail: dict = {
        "entry_id": entry_id,
        "entry_summary": None,
        "overall_points": None,
        "overall_rank": None,
        "gameweek_history": [],
        "chips_used": [],
        "transfers": [],
        "transfer_count": None,
        "latest_gameweek_used_for_picks": latest_finished_event,
        "picks": None,
        "starting_xi": [],
        "bench": [],
        "captain_element": None,
        "vice_captain_element": None,
        "active_chip_latest_gw": None,
    }

    try:
        entry = client.get_entry(entry_id)
    except FPLAPIError as exc:
        errors.append(f"entry/{entry_id}: {exc}")
        entry = None
    if entry is not None:
        detail["entry_summary"] = entry
        detail["overall_points"] = entry.get("summary_overall_points")
        detail["overall_rank"] = entry.get("summary_overall_rank")
    else:
        errors.append(f"entry/{entry_id}: no data returned (private or removed account).")

    try:
        history = client.get_entry_history(entry_id)
    except FPLAPIError as exc:
        errors.append(f"entry/{entry_id}/history: {exc}")
        history = None
    if history is not None:
        detail["gameweek_history"] = history.get("current", [])
        detail["chips_used"] = history.get("chips", [])
    else:
        errors.append(f"entry/{entry_id}/history: not available.")

    try:
        transfers = client.get_entry_transfers(entry_id)
    except FPLAPIError as exc:
        errors.append(f"entry/{entry_id}/transfers: {exc}")
        transfers = None
    if transfers is not None:
        detail["transfers"] = transfers
        detail["transfer_count"] = len(transfers)
    else:
        errors.append(f"entry/{entry_id}/transfers: not available.")

    if latest_finished_event is not None:
        try:
            picks = client.get_entry_picks(entry_id, latest_finished_event)
        except FPLAPIError as exc:
            errors.append(f"entry/{entry_id}/event/{latest_finished_event}/picks: {exc}")
            picks = None
        if picks is not None:
            detail["picks"] = picks
            picks_list = picks.get("picks", [])
            detail["starting_xi"] = [p for p in picks_list if p.get("position", 99) <= 11]
            detail["bench"] = [p for p in picks_list if p.get("position", 0) > 11]
            for p in picks_list:
                if p.get("is_captain"):
                    detail["captain_element"] = p.get("element")
                if p.get("is_vice_captain"):
                    detail["vice_captain_element"] = p.get("element")
            entry_history = picks.get("entry_history", {})
            detail["active_chip_latest_gw"] = entry_history.get("active_chip")
        else:
            errors.append(
                f"entry/{entry_id}/event/{latest_finished_event}/picks: not available "
                "(manager may not have played that gameweek, or picks are not public yet)."
            )
    else:
        errors.append(f"entry/{entry_id}: no completed gameweek exists yet this season - picks skipped.")

    return detail
