"""Unit tests for fpl_rival.simulation.player_outcomes."""

import unittest

import numpy as np

from fpl_rival.simulation.exceptions import MissingProjectionError
from fpl_rival.simulation.models import PlayerProjection
from fpl_rival.simulation.player_outcomes import sample_all_players

PROJECTIONS = {
    1: PlayerProjection(1, "Alpha", projected_mean_points=6.0, projected_standard_deviation=3.0),
    2: PlayerProjection(2, "Beta", projected_mean_points=2.0, projected_standard_deviation=1.0),
}


class TestSampleAllPlayers(unittest.TestCase):
    def test_reproducible_with_fixed_seed(self):
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        out1 = sample_all_players([1, 2], PROJECTIONS, 1000, rng1)
        out2 = sample_all_players([1, 2], PROJECTIONS, 1000, rng2)
        np.testing.assert_array_equal(out1[1], out2[1])
        np.testing.assert_array_equal(out1[2], out2[2])

    def test_different_seeds_differ(self):
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        out1 = sample_all_players([1], PROJECTIONS, 1000, rng1)
        out2 = sample_all_players([1], PROJECTIONS, 1000, rng2)
        self.assertFalse(np.array_equal(out1[1], out2[1]))

    def test_configured_mean_is_approximated(self):
        rng = np.random.default_rng(42)
        out = sample_all_players([1], PROJECTIONS, 200_000, rng)
        self.assertAlmostEqual(float(np.mean(out[1])), 6.0, delta=0.1)

    def test_configured_uncertainty_is_approximated(self):
        rng = np.random.default_rng(42)
        out = sample_all_players([1], PROJECTIONS, 200_000, rng, min_player_points=-100)  # disable clipping for this check
        self.assertAlmostEqual(float(np.std(out[1])), 3.0, delta=0.15)

    def test_floor_prevents_impossible_extreme_negatives(self):
        rng = np.random.default_rng(0)
        out = sample_all_players([2], PROJECTIONS, 200_000, rng, min_player_points=-4.0)
        self.assertGreaterEqual(float(np.min(out[2])), -4.0)

    def test_zero_points_is_reachable(self):
        rng = np.random.default_rng(0)
        out = sample_all_players([2], PROJECTIONS, 50_000, rng)  # Beta mean=2, std=1 -> 0 is ~2 sd away, reachable
        self.assertIn(0.0, set(np.unique(out[2]).tolist()))

    def test_scores_are_integers(self):
        rng = np.random.default_rng(0)
        out = sample_all_players([1], PROJECTIONS, 1000, rng)
        self.assertTrue(np.all(out[1] == np.round(out[1])))

    def test_missing_projection_fails_clearly(self):
        rng = np.random.default_rng(0)
        with self.assertRaises(MissingProjectionError):
            sample_all_players([1, 999], PROJECTIONS, 100, rng)


if __name__ == "__main__":
    unittest.main()
