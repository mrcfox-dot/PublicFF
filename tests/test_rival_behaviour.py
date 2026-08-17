"""Unit tests for fpl_rival.simulation.rival_behaviour."""

import unittest

import numpy as np

from fpl_rival.simulation.exceptions import InvalidCaptainDistributionError
from fpl_rival.simulation.models import LeagueState, ManagerState
from fpl_rival.simulation.rival_behaviour import (
    resolve_captain_ids_by_entry,
    sample_captain_ids,
    validate_captain_distribution,
)


class TestValidateCaptainDistribution(unittest.TestCase):
    def test_valid_distribution_passes(self):
        validate_captain_distribution({1: 0.75, 2: 0.2, 3: 0.05}, "Rival")  # no raise

    def test_sum_not_one_fails_cleanly(self):
        with self.assertRaises(InvalidCaptainDistributionError):
            validate_captain_distribution({1: 0.5, 2: 0.2}, "Rival")

    def test_negative_probability_fails_cleanly(self):
        with self.assertRaises(InvalidCaptainDistributionError):
            validate_captain_distribution({1: 1.2, 2: -0.2}, "Rival")

    def test_empty_distribution_fails_cleanly(self):
        with self.assertRaises(InvalidCaptainDistributionError):
            validate_captain_distribution({}, "Rival")

    def test_within_tolerance_passes(self):
        validate_captain_distribution({1: 0.6666667, 2: 0.3333332}, "Rival")  # no raise


class TestSampleCaptainIds(unittest.TestCase):
    def test_respects_distribution_roughly(self):
        rng = np.random.default_rng(1)
        samples = sample_captain_ids({1: 0.9, 2: 0.1}, 50_000, rng)
        frac_1 = float(np.mean(samples == 1))
        self.assertAlmostEqual(frac_1, 0.9, delta=0.02)


class TestResolveCaptainIdsByEntry(unittest.TestCase):
    def _league(self, mode_manager_dist):
        chris = ManagerState(entry_id=1, name="Chris", current_total_points=0, current_league_position=1, starting_xi=(1, 2), captain_id=1)
        rival = ManagerState(
            entry_id=2, name="Rival", current_total_points=0, current_league_position=2,
            starting_xi=(1, 2), captain_id=1, captain_probabilities=mode_manager_dist,
        )
        return LeagueState(league_id=1, league_name="L", managers=(chris, rival))

    def test_fixed_mode_ignores_distribution(self):
        league = self._league({1: 0.5, 2: 0.5})
        rng = np.random.default_rng(1)
        result = resolve_captain_ids_by_entry(league, 100, rng, rival_captain_mode="fixed")
        self.assertTrue(np.all(result[2] == 1))

    def test_probabilistic_mode_samples_distribution(self):
        league = self._league({1: 0.5, 2: 0.5})
        rng = np.random.default_rng(1)
        result = resolve_captain_ids_by_entry(league, 50_000, rng, rival_captain_mode="probabilistic")
        frac_1 = float(np.mean(result[2] == 1))
        self.assertAlmostEqual(frac_1, 0.5, delta=0.02)

    def test_probabilistic_mode_falls_back_to_fixed_when_no_distribution(self):
        league = self._league(None)
        rng = np.random.default_rng(1)
        result = resolve_captain_ids_by_entry(league, 100, rng, rival_captain_mode="probabilistic")
        self.assertTrue(np.all(result[2] == 1))

    def test_invalid_distribution_fails_cleanly(self):
        league = self._league({1: 0.5, 2: 0.6})
        rng = np.random.default_rng(1)
        with self.assertRaises(InvalidCaptainDistributionError):
            resolve_captain_ids_by_entry(league, 100, rng, rival_captain_mode="probabilistic")


if __name__ == "__main__":
    unittest.main()
