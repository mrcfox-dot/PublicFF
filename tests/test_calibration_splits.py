"""Unit tests for fpl_rival.calibration.splits."""

import unittest

from fpl_rival.calibration.models import DatasetError, HistoricalRecord, TEST, TRAIN, VALIDATION
from fpl_rival.calibration.splits import explicit_split, infer_default_split

SQUAD = tuple(range(1, 12))


def _records(gws, season="2026-27"):
    return [HistoricalRecord(season=season, manager_id=1, gameweek=gw, starting_xi=SQUAD, captain=1) for gw in gws]


class TestInferDefaultSplit(unittest.TestCase):
    def test_partitions_into_three_nonoverlapping_ranges(self):
        records = _records(range(1, 31))
        split = infer_default_split(records, train_fraction=0.4, validation_fraction=0.3)
        phases = [split.phase_of("2026-27", gw) for gw in range(1, 31)]
        self.assertIn(TRAIN, phases)
        self.assertIn(VALIDATION, phases)
        self.assertIn(TEST, phases)
        # train gameweeks all come before validation gameweeks which all come before test
        train_gws = [gw for gw in range(1, 31) if split.phase_of("2026-27", gw) == TRAIN]
        val_gws = [gw for gw in range(1, 31) if split.phase_of("2026-27", gw) == VALIDATION]
        test_gws = [gw for gw in range(1, 31) if split.phase_of("2026-27", gw) == TEST]
        self.assertLess(max(train_gws), min(val_gws))
        self.assertLess(max(val_gws), min(test_gws))

    def test_empty_dataset_raises(self):
        with self.assertRaises(DatasetError):
            infer_default_split([])

    def test_invalid_fractions_rejected(self):
        records = _records(range(1, 10))
        with self.assertRaises(ValueError):
            infer_default_split(records, train_fraction=0.7, validation_fraction=0.5)


class TestExplicitSplit(unittest.TestCase):
    def test_parses_season_colon_gameweek(self):
        split = explicit_split("2026-27:10", "2026-27:20", "2026-27:30", ["2026-27"])
        self.assertEqual(split.train_through.season, "2026-27")
        self.assertEqual(split.train_through.gameweek, 10)

    def test_malformed_string_rejected(self):
        with self.assertRaises(ValueError):
            explicit_split("not-a-valid-point", "2026-27:20", "2026-27:30", ["2026-27"])


if __name__ == "__main__":
    unittest.main()
