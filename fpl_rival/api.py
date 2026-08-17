"""Thin, defensive client for the public Fantasy Premier League (FPL) API.

Every method returns plain dict/list data (already JSON-decoded) or ``None``
when the resource genuinely does not exist (e.g. a 404 for picks before a
manager entered the game). Network / server errors raise ``FPLAPIError`` so
callers can decide how to record the gap rather than silently fabricating
data.

No authentication is used anywhere in this module - only endpoints that are
readable without logging in to fantasy.premierleague.com are called.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://fantasy.premierleague.com/api"
REQUEST_TIMEOUT = 15  # seconds
REQUEST_DELAY = 0.15  # be polite between calls; not an official rate limit


class FPLAPIError(Exception):
    """Raised when a request fails after retries (network/server problem)."""


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "fpl-rival-mvp-stage1/0.1 (local prototype, public-data-only)",
            "Accept": "application/json",
        }
    )
    return session


class FPLClient:
    """Wraps the handful of public FPL endpoints this prototype needs."""

    def __init__(self) -> None:
        self._session = _build_session()

    def _get(self, path: str, params: Optional[dict] = None) -> Optional[Any]:
        url = f"{BASE_URL}{path}"
        time.sleep(REQUEST_DELAY)
        try:
            resp = self._session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            raise FPLAPIError(f"Network error calling {url}: {exc}") from exc

        if resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            raise FPLAPIError(f"HTTP {resp.status_code} calling {url}: {resp.text[:200]}")

        try:
            return resp.json()
        except ValueError as exc:
            raise FPLAPIError(f"Invalid JSON from {url}: {exc}") from exc

    # -- Bootstrap / global data -------------------------------------------------

    def get_bootstrap_static(self) -> dict:
        data = self._get("/bootstrap-static/")
        if data is None:
            raise FPLAPIError("bootstrap-static returned nothing (unexpected 404)")
        return data

    # -- Manager (entry) data -----------------------------------------------------

    def get_entry(self, entry_id: int) -> Optional[dict]:
        return self._get(f"/entry/{entry_id}/")

    def get_entry_history(self, entry_id: int) -> Optional[dict]:
        return self._get(f"/entry/{entry_id}/history/")

    def get_entry_transfers(self, entry_id: int) -> Optional[list]:
        return self._get(f"/entry/{entry_id}/transfers/")

    def get_entry_picks(self, entry_id: int, event: int) -> Optional[dict]:
        """Picks for one gameweek. Returns None if unavailable (404).

        Note: FPL only exposes another manager's picks for a gameweek after
        that gameweek's transfer deadline has passed - this is a platform
        restriction, not something this client can work around.
        """
        return self._get(f"/entry/{entry_id}/event/{event}/picks/")

    # -- Classic league data -------------------------------------------------------

    def get_league_standings_page(
        self, league_id: int, page_standings: int = 1, page_new_entries: int = 1
    ) -> Optional[dict]:
        return self._get(
            f"/leagues-classic/{league_id}/standings/",
            params={"page_standings": page_standings, "page_new_entries": page_new_entries},
        )
