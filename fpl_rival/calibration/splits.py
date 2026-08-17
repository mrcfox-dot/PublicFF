"""Chronological train/validation/test split construction.

Prefers explicit boundaries (season + gameweek) when the caller knows
them; ``infer_default_split`` picks reasonable ones automatically from
whatever gameweek range is actually in the dataset, using
season-then-gameweek order (earlier seasons = training, most recent
season = test, matching Stage 4.5 brief item 5's guidance) - a fallback
for the CLI when the user doesn't specify cutoffs explicitly.
"""

from __future__ import annotations

from typing import List

from .models import ChronologicalPoint, DatasetError, DatasetSplit, HistoricalRecord


def infer_default_split(
    records: List[HistoricalRecord],
    train_fraction: float = 0.4,
    validation_fraction: float = 0.3,
) -> DatasetSplit:
    """Splits by fraction of (season, gameweek) points, chronologically -
    not by row count (a manager with more rows shouldn't skew the cutoff).
    """
    if not records:
        raise DatasetError("Cannot infer a split from an empty dataset.")
    if not (0 < train_fraction < 1) or not (0 < validation_fraction < 1) or train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction and validation_fraction must be in (0,1) and sum to < 1.")

    season_order = tuple(sorted({r.season for r in records}))
    points = sorted({(r.season, r.gameweek) for r in records}, key=lambda sg: (season_order.index(sg[0]), sg[1]))

    n = len(points)
    train_cut = max(0, min(n - 1, int(n * train_fraction) - 1))
    validation_cut = max(train_cut, min(n - 1, int(n * (train_fraction + validation_fraction)) - 1))
    test_cut = n - 1

    return DatasetSplit(
        train_through=ChronologicalPoint(*points[train_cut]),
        validation_through=ChronologicalPoint(*points[validation_cut]),
        test_through=ChronologicalPoint(*points[test_cut]),
        season_order=season_order,
    )


def explicit_split(
    train_through: str,
    validation_through: str,
    test_through: str,
    season_order: List[str],
) -> DatasetSplit:
    """Parses "season:gw" strings (e.g. "2026-27:15") into a DatasetSplit."""

    def _parse(point_str: str) -> ChronologicalPoint:
        season, _, gw = point_str.rpartition(":")
        if not season or not gw:
            raise ValueError(f"Expected 'season:gameweek', got {point_str!r}.")
        return ChronologicalPoint(season=season, gameweek=int(gw))

    return DatasetSplit(
        train_through=_parse(train_through),
        validation_through=_parse(validation_through),
        test_through=_parse(test_through),
        season_order=tuple(season_order),
    )
