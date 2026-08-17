"""Normalized historical dataset shape + chronological split model.

``HistoricalRecord`` is the "generic historical dataset interface" the
Stage 4 brief asked for (its fields mirror Stage 4's ``CaptainObservation``
plus the extra columns a real dataset naturally has: season, full squad
vs. starting XI, chip, cumulative/gameweek points, rank). Every importer
adaptor in ``importer.py`` produces a list of these and nothing else -
downstream code (baselines, fitting, ablation, ...) never sees a raw CSV
row or a dataset-specific column name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from fpl_rival.prediction.models import CaptainObservation


class DatasetError(Exception):
    """Raised for structurally invalid or unusable dataset input."""


@dataclass(frozen=True)
class HistoricalRecord:
    """One manager's decision for one gameweek, normalized.

    Required for the record to be usable at all: ``season``,
    ``manager_id``, ``gameweek``, ``starting_xi``, ``captain``. Everything
    else is optional - a real dataset will not always have every column,
    and Stage 4.5 must degrade gracefully rather than fabricate missing
    values (see ``importer.py``'s validation).
    """

    season: str
    manager_id: int
    gameweek: int
    starting_xi: Tuple[int, ...]
    captain: int
    manager_name: Optional[str] = None
    squad: Optional[Tuple[int, ...]] = None  # full 15, if available (starting_xi is what matters for captaincy)
    vice_captain: Optional[int] = None
    transfers: int = 0
    chip: Optional[str] = None
    total_points: Optional[int] = None
    gameweek_points: Optional[int] = None
    rank: Optional[int] = None

    def __post_init__(self):
        if self.captain not in self.starting_xi:
            raise DatasetError(
                f"manager {self.manager_id}, GW{self.gameweek}: captain {self.captain} "
                f"is not in starting_xi {self.starting_xi} - unusable record."
            )

    def to_captain_observation(self) -> CaptainObservation:
        """Candidates = starting_xi, matching Stage 3/4's own semantics
        (a captain must be a starting-XI player)."""
        return CaptainObservation(
            gameweek=self.gameweek,
            owned_player_ids=self.starting_xi,
            captain_id=self.captain,
            vice_captain_id=self.vice_captain,
            transfers=self.transfers,
            points=self.gameweek_points,
            rank=self.rank,
        )


@dataclass(frozen=True)
class DatasetSummary:
    num_records: int
    num_managers: int
    num_gameweeks: int
    seasons: Tuple[str, ...]
    min_gameweek: Optional[int]
    max_gameweek: Optional[int]
    records_missing_rank: int
    records_missing_total_points: int
    records_missing_chip: int
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    is_synthetic: bool = False


@dataclass(frozen=True)
class ChronologicalPoint:
    """A (season, gameweek) point in a chronological ordering.

    Comparisons between points are only meaningful relative to a
    ``season_order`` (see ``DatasetSplit``) - a bare ``ChronologicalPoint``
    doesn't know which season came first.
    """

    season: str
    gameweek: int


TRAIN, VALIDATION, TEST, EXCLUDED = "train", "validation", "test", "excluded"


@dataclass(frozen=True)
class DatasetSplit:
    """Chronological train/validation/test boundaries.

    Every (season, gameweek) up to and including ``train_through`` is
    TRAIN (predictions are still generated there - a manager's history has
    to start somewhere - but TRAIN-period metrics are diagnostic only,
    never used to pick a model or report headline numbers). Everything
    after ``train_through`` up to and including ``validation_through`` is
    VALIDATION (used for parameter search / calibration fitting).
    Everything after that, up to and including ``test_through``, is TEST
    (evaluated exactly once, after every choice has already been made on
    validation - this is the number that gets reported as "held-out
    performance"). Anything after ``test_through`` is EXCLUDED.

    ``season_order`` gives the chronological ordering of season labels
    (oldest first) - required whenever a dataset spans more than one
    season, since season labels don't sort chronologically on their own
    (e.g. "2025-26" before "2026-27" happens to sort correctly as a
    string, but nothing here relies on that - it's explicit).
    """

    train_through: ChronologicalPoint
    validation_through: ChronologicalPoint
    test_through: ChronologicalPoint
    season_order: Tuple[str, ...]

    def _rank(self, season: str, gameweek: int) -> Tuple[int, int]:
        if season not in self.season_order:
            raise DatasetError(f"Season {season!r} is not in season_order {self.season_order}.")
        return (self.season_order.index(season), gameweek)

    def phase_of(self, season: str, gameweek: int) -> str:
        point_rank = self._rank(season, gameweek)
        if point_rank <= self._rank(self.train_through.season, self.train_through.gameweek):
            return TRAIN
        if point_rank <= self._rank(self.validation_through.season, self.validation_through.gameweek):
            return VALIDATION
        if point_rank <= self._rank(self.test_through.season, self.test_through.gameweek):
            return TEST
        return EXCLUDED

    def describe(self) -> str:
        return (
            f"train: through {self.train_through.season} GW{self.train_through.gameweek} | "
            f"validation: through {self.validation_through.season} GW{self.validation_through.gameweek} | "
            f"test: through {self.test_through.season} GW{self.test_through.gameweek}"
        )
