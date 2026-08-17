"""Unit tests for fpl_rival.prediction.models."""

import unittest

from fpl_rival.prediction.models import CaptainObservation, LeagueHistory


class TestCaptainObservation(unittest.TestCase):
    def test_captain_must_be_owned(self):
        with self.assertRaises(ValueError):
            CaptainObservation(gameweek=1, owned_player_ids=(1, 2, 3), captain_id=99)

    def test_valid_observation_constructs(self):
        obs = CaptainObservation(gameweek=1, owned_player_ids=(1, 2, 3), captain_id=2)
        self.assertEqual(obs.captain_id, 2)


class TestLeagueHistoryBefore(unittest.TestCase):
    def setUp(self):
        self.history = LeagueHistory(
            {
                1: tuple(CaptainObservation(gameweek=gw, owned_player_ids=(1, 2), captain_id=1) for gw in range(1, 11)),
                2: tuple(CaptainObservation(gameweek=gw, owned_player_ids=(1, 2), captain_id=2) for gw in range(1, 11)),
            }
        )

    def test_before_excludes_target_gameweek_itself(self):
        sliced = self.history.before(5)
        self.assertTrue(all(o.gameweek < 5 for o in sliced.for_entry(1)))
        self.assertEqual(max(o.gameweek for o in sliced.for_entry(1)), 4)

    def test_before_zero_returns_empty(self):
        sliced = self.history.before(1)
        self.assertEqual(sliced.for_entry(1), ())

    def test_for_entry_unknown_returns_empty_tuple(self):
        self.assertEqual(self.history.for_entry(999), ())

    def test_other_entries_excludes_given_entry(self):
        others = self.history.other_entries(1)
        self.assertNotIn(1, others)
        self.assertIn(2, others)


if __name__ == "__main__":
    unittest.main()
