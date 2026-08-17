"""League standings retrieval, including pagination handling.

The FPL standings endpoint paginates two independent lists on the same
resource: ``standings`` (ranked, scored managers - only populated once at
least one gameweek has been scored) and ``new_entries`` (everyone who has
joined the league, always populated). A league can have many pages of
either list; this module walks both to completion and records any page
that failed so the caller can report a partial result instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .api import FPLAPIError, FPLClient

MAX_PAGES_SAFETY = 200  # hard stop so a malformed has_next loop can't run forever


@dataclass
class LeagueData:
    league_id: int
    meta: Optional[dict] = None
    standings: list = field(default_factory=list)
    new_entries: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def has_scored_standings(self) -> bool:
        return len(self.standings) > 0


def fetch_league(client: FPLClient, league_id: int) -> LeagueData:
    data = LeagueData(league_id=league_id)

    # First page gives us league metadata plus page 1 of both lists.
    try:
        first = client.get_league_standings_page(league_id, 1, 1)
    except FPLAPIError as exc:
        data.errors.append(f"Could not load league {league_id}: {exc}")
        return data

    if first is None:
        data.errors.append(f"League {league_id} not found (404).")
        return data

    data.meta = first.get("league")
    data.standings.extend(first.get("standings", {}).get("results", []))
    data.new_entries.extend(first.get("new_entries", {}).get("results", []))

    _paginate_standings(client, league_id, first, data)
    _paginate_new_entries(client, league_id, first, data)

    return data


def _paginate_standings(client: FPLClient, league_id: int, first_page: dict, data: LeagueData) -> None:
    has_next = first_page.get("standings", {}).get("has_next", False)
    page = 2
    while has_next and page <= MAX_PAGES_SAFETY:
        try:
            result = client.get_league_standings_page(league_id, page_standings=page, page_new_entries=1)
        except FPLAPIError as exc:
            data.errors.append(f"Standings page {page} failed for league {league_id}: {exc}")
            break
        if result is None:
            data.errors.append(f"Standings page {page} returned nothing for league {league_id}.")
            break
        block = result.get("standings", {})
        data.standings.extend(block.get("results", []))
        has_next = block.get("has_next", False)
        page += 1


def _paginate_new_entries(client: FPLClient, league_id: int, first_page: dict, data: LeagueData) -> None:
    has_next = first_page.get("new_entries", {}).get("has_next", False)
    page = 2
    while has_next and page <= MAX_PAGES_SAFETY:
        try:
            result = client.get_league_standings_page(league_id, page_standings=1, page_new_entries=page)
        except FPLAPIError as exc:
            data.errors.append(f"new_entries page {page} failed for league {league_id}: {exc}")
            break
        if result is None:
            data.errors.append(f"new_entries page {page} returned nothing for league {league_id}.")
            break
        block = result.get("new_entries", {})
        data.new_entries.extend(block.get("results", []))
        has_next = block.get("has_next", False)
        page += 1


def build_manager_list(data: LeagueData) -> list[dict]:
    """Merge standings + new_entries into one manager list.

    Prefers scored standings rows (has rank/points). Falls back to
    new_entries (membership only, no score yet - e.g. pre-season) for any
    manager not present in standings, clearly flagged as such.
    """
    by_entry: dict[int, dict] = {}

    for row in data.standings:
        entry_id = row.get("entry")
        by_entry[entry_id] = {
            "entry_id": entry_id,
            "manager_name": row.get("player_name"),
            "team_name": row.get("entry_name"),
            "league_position": row.get("rank"),
            "last_rank": row.get("last_rank"),
            "total_points": row.get("total"),
            "event_total": row.get("event_total"),
            "source": "standings",
        }

    for row in data.new_entries:
        entry_id = row.get("entry")
        if entry_id in by_entry:
            continue
        first = row.get("player_first_name") or ""
        last = row.get("player_last_name") or ""
        by_entry[entry_id] = {
            "entry_id": entry_id,
            "manager_name": (first + " " + last).strip() or None,
            "team_name": row.get("entry_name"),
            "league_position": None,
            "last_rank": None,
            "total_points": None,
            "event_total": None,
            "source": "new_entries_only (no scored gameweek yet)",
        }

    managers = list(by_entry.values())
    managers.sort(key=lambda m: (m["league_position"] is None, m["league_position"] or 0, m["entry_id"]))
    return managers
