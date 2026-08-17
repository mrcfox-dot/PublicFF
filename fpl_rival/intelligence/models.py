"""Data models shared by live collection, synthetic fixtures and analytics.

These are the *only* shapes the analytics engine understands. Live FPL
retrieval and synthetic test fixtures both normalize into these models, so
the engine itself never has to know or care where the data came from.

Every optional field defaults to ``None`` / empty and MUST stay that way
when the underlying data was not retrieved - the engine never invents a
value to fill a gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

GK, DEF, MID, FWD = 1, 2, 3, 4
ELEMENT_TYPE_LABEL = {GK: "G", DEF: "D", MID: "M", FWD: "F"}


@dataclass(frozen=True)
class Pick:
    element: int
    position: int  # 1-15: squad slot order (1-11 starting XI, 12-15 bench)
    multiplier: int
    is_captain: bool
    is_vice_captain: bool
    element_type: Optional[int] = None  # 1 GK, 2 DEF, 3 MID, 4 FWD (when known)


@dataclass(frozen=True)
class GameweekSquad:
    """One manager's picks for a single gameweek."""

    event: int
    picks: tuple = ()  # tuple[Pick, ...]
    active_chip: Optional[str] = None
    points: Optional[int] = None
    points_on_bench: Optional[int] = None

    @property
    def starting_xi(self) -> tuple:
        return tuple(p for p in self.picks if p.position <= 11)

    @property
    def bench(self) -> tuple:
        return tuple(p for p in self.picks if p.position > 11)

    @property
    def captain_element(self) -> Optional[int]:
        return next((p.element for p in self.picks if p.is_captain), None)

    @property
    def vice_captain_element(self) -> Optional[int]:
        return next((p.element for p in self.picks if p.is_vice_captain), None)

    @property
    def formation(self) -> Optional[str]:
        """"D-M-F" shape of the starting XI. None if element types are unknown."""
        xi = self.starting_xi
        if not xi or any(p.element_type is None for p in xi):
            return None
        counts = {DEF: 0, MID: 0, FWD: 0}
        for p in xi:
            if p.element_type in counts:
                counts[p.element_type] += 1
        return f"{counts[DEF]}-{counts[MID]}-{counts[FWD]}"


@dataclass(frozen=True)
class GameweekHistoryRow:
    """One row of an entry's /history/ 'current' list."""

    event: int
    points: Optional[int] = None
    total_points: Optional[int] = None
    rank: Optional[int] = None
    overall_rank: Optional[int] = None
    bank: Optional[int] = None
    value: Optional[int] = None
    event_transfers: Optional[int] = None
    event_transfers_cost: Optional[int] = None
    points_on_bench: Optional[int] = None


@dataclass(frozen=True)
class ChipEvent:
    name: str
    event: int


@dataclass(frozen=True)
class TransferEvent:
    element_in: int
    element_out: int
    event: int
    time: Optional[str] = None  # ISO8601 timestamp, when known


@dataclass
class ManagerData:
    entry_id: int
    manager_name: Optional[str] = None
    team_name: Optional[str] = None
    league_position: Optional[int] = None
    total_points: Optional[int] = None
    overall_points: Optional[int] = None
    overall_rank: Optional[int] = None
    team_value: Optional[int] = None  # tenths of a million, e.g. 1005 = GBP 100.5m
    bank: Optional[int] = None
    gameweek_history: tuple = ()  # tuple[GameweekHistoryRow, ...]
    chips_used: tuple = ()  # tuple[ChipEvent, ...]
    transfers: tuple = ()  # tuple[TransferEvent, ...]
    squads_by_event: dict = field(default_factory=dict)  # {event: GameweekSquad}
    past_seasons: tuple = ()  # tuple[dict] - {season_name, total_points, rank}, public even pre-season
    is_synthetic: bool = False

    @property
    def latest_event_with_squad(self) -> Optional[int]:
        return max(self.squads_by_event) if self.squads_by_event else None

    @property
    def latest_squad(self) -> Optional[GameweekSquad]:
        e = self.latest_event_with_squad
        return self.squads_by_event.get(e) if e is not None else None

    @property
    def events_played(self) -> tuple:
        """Events for which we have a scored history row (proxy for 'played')."""
        return tuple(sorted(row.event for row in self.gameweek_history))


@dataclass
class GameContext:
    """Season-wide reference data needed to interpret manager data.

    Populated once per run from FPL's bootstrap-static (live) or supplied
    directly by a synthetic fixture. Never per-manager.
    """

    element_names: dict = field(default_factory=dict)  # element_id -> web_name
    element_types: dict = field(default_factory=dict)  # element_id -> 1/2/3/4
    chip_windows: dict = field(default_factory=dict)  # chip_name -> [(start_event, stop_event), ...]
    event_deadlines: dict = field(default_factory=dict)  # event -> ISO8601 deadline string
    season_total_gameweeks: int = 38

    def player_name(self, element_id: int) -> str:
        return self.element_names.get(element_id, f"Player {element_id}")


@dataclass
class LeagueData:
    league_id: int
    league_name: str
    managers: tuple = ()  # tuple[ManagerData, ...]
    latest_completed_event: Optional[int] = None
    is_synthetic: bool = False

    def manager(self, entry_id: int) -> Optional[ManagerData]:
        return next((m for m in self.managers if m.entry_id == entry_id), None)
