"""Unit tests for fpl_rival.calibration.models."""

import unittest

from fpl_rival.calibration.models import ChronologicalPoint, DatasetError, DatasetSplit, HistoricalRecord, EXCLUDED, TEST, TRAIN, VALIDATION


class TestHistoricalRecord(unittest.TestCase):
    def test_captain_must_be_in_starting_xi(self):
        with self.assertRaises(DatasetError):
            HistoricalRecord(season="2026-27", manager_id=1, gameweek=1, starting_xi=(1, 2, 3), captain=99)

    def test_to_captain_observation_uses_starting_xi_as_candidates(self):
        record = HistoricalRecord(season="2026-27", manager_id=1, gameweek=1, starting_xi=(1, 2, 3), captain=2, vice_captain=1, transfers=2, gameweek_points=55, rank=3)
        obs = record.to_captain_observation()
        self.assertEqual(obs.owned_player_ids, (1, 2, 3))
        self.assertEqual(obs.captain_id, 2)
        self.assertEqual(obs.vice_captain_id, 1)
        self.assertEqual(obs.transfers, 2)
        self.assertEqual(obs.points, 55)
        self.assertEqual(obs.rank, 3)


class TestDatasetSplitSingleSeason(unittest.TestCase):
    def setUp(self):
        self.split = DatasetSplit(
            train_through=ChronologicalPoint("2026-27", 10),
            validation_through=ChronologicalPoint("2026-27", 20),
            test_through=ChronologicalPoint("2026-27", 30),
            season_order=("2026-27",),
        )

    def test_phases_assigned_correctly(self):
        self.assertEqual(self.split.phase_of("2026-27", 1), TRAIN)
        self.assertEqual(self.split.phase_of("2026-27", 10), TRAIN)
        self.assertEqual(self.split.phase_of("2026-27", 11), VALIDATION)
        self.assertEqual(self.split.phase_of("2026-27", 20), VALIDATION)
        self.assertEqual(self.split.phase_of("2026-27", 21), TEST)
        self.assertEqual(self.split.phase_of("2026-27", 30), TEST)
        self.assertEqual(self.split.phase_of("2026-27", 31), EXCLUDED)

    def test_unknown_season_raises(self):
        with self.assertRaises(DatasetError):
            self.split.phase_of("1999-00", 1)


class TestDatasetSplitMultiSeason(unittest.TestCase):
    def setUp(self):
        # season_order says 2025-26 comes before 2026-27, regardless of string sort.
        self.split = DatasetSplit(
            train_through=ChronologicalPoint("2025-26", 38),
            validation_through=ChronologicalPoint("2026-27", 19),
            test_through=ChronologicalPoint("2026-27", 38),
            season_order=("2025-26", "2026-27"),
        )

    def test_earlier_season_is_train_even_late_gameweeks(self):
        self.assertEqual(self.split.phase_of("2025-26", 38), TRAIN)

    def test_later_season_early_gameweek_is_validation_not_train(self):
        # GW1 of the LATER season must not be mistaken for "early" (train) -
        # season ordering, not raw gameweek number, must drive the comparison.
        self.assertEqual(self.split.phase_of("2026-27", 1), VALIDATION)

    def test_later_season_late_gameweek_is_test(self):
        self.assertEqual(self.split.phase_of("2026-27", 38), TEST)


if __name__ == "__main__":
    unittest.main()
