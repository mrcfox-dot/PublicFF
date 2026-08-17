"""Unit tests for fpl_rival.calibration.baselines."""

import unittest

from fpl_rival.calibration.baselines import (
    predict_consensus,
    predict_expected_points_favourite,
    predict_last_captain,
    predict_personal_frequency,
    predict_uniform,
)
from fpl_rival.prediction.models import CaptainObservation

CANDIDATES = (1, 2, 3)


def _hist(captains):
    return tuple(CaptainObservation(gameweek=gw, owned_player_ids=CANDIDATES, captain_id=c) for gw, c in captains)


class TestBaselineA_LastCaptain(unittest.TestCase):
    def test_repeats_previous_captain_when_still_owned(self):
        probs = predict_last_captain(CANDIDATES, _hist([(1, 1), (2, 2)]))
        self.assertEqual(probs[2], 1.0)
        self.assertEqual(probs[1], 0.0)

    def test_falls_back_to_uniform_when_no_history(self):
        probs = predict_last_captain(CANDIDATES, ())
        self.assertTrue(all(abs(p - 1 / 3) < 1e-9 for p in probs.values()))

    def test_falls_back_to_uniform_when_previous_captain_no_longer_owned(self):
        # Manager captained player 1 last time, but has since sold him -
        # today's candidate set doesn't include player 1 at all.
        history = _hist([(1, 1)])
        probs = predict_last_captain((2, 3, 4), history)
        self.assertTrue(all(abs(p - 1 / 3) < 1e-9 for p in probs.values()))


class TestBaselineB_PersonalFrequency(unittest.TestCase):
    def test_proportional_to_historical_frequency(self):
        probs = predict_personal_frequency(CANDIDATES, _hist([(1, 1), (2, 1), (3, 2)]))
        self.assertAlmostEqual(probs[1], 2 / 3)
        self.assertAlmostEqual(probs[2], 1 / 3)
        self.assertEqual(probs[3], 0.0)

    def test_uniform_when_no_history(self):
        probs = predict_personal_frequency(CANDIDATES, ())
        self.assertTrue(all(abs(p - 1 / 3) < 1e-9 for p in probs.values()))


class TestBaselineC_Consensus(unittest.TestCase):
    def test_none_when_no_other_managers(self):
        self.assertIsNone(predict_consensus(CANDIDATES, {}))

    def test_pooled_frequency_across_others(self):
        others = {10: _hist([(1, 1), (2, 1)]), 11: _hist([(1, 2)])}
        probs = predict_consensus(CANDIDATES, others)
        self.assertAlmostEqual(probs[1], 2 / 3)
        self.assertAlmostEqual(probs[2], 1 / 3)


class TestBaselineD_ExpectedPoints(unittest.TestCase):
    def test_none_when_no_ep_data(self):
        self.assertIsNone(predict_expected_points_favourite(CANDIDATES, None))
        self.assertIsNone(predict_expected_points_favourite(CANDIDATES, {}))

    def test_picks_highest_owned_projection(self):
        probs = predict_expected_points_favourite(CANDIDATES, {1: 5.0, 2: 9.0, 3: 4.0})
        self.assertEqual(probs[2], 1.0)
        self.assertEqual(probs[1], 0.0)

    def test_ignores_projections_for_unowned_players(self):
        probs = predict_expected_points_favourite(CANDIDATES, {1: 5.0, 99: 100.0})
        self.assertEqual(probs[1], 1.0)  # 99 isn't a candidate, must be ignored


class TestBaselineE_Uniform(unittest.TestCase):
    def test_equal_across_candidates(self):
        probs = predict_uniform((1, 2, 3, 4))
        self.assertTrue(all(abs(p - 0.25) < 1e-9 for p in probs.values()))
        self.assertAlmostEqual(sum(probs.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
