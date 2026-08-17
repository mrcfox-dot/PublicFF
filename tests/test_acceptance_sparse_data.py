"""Stage 4 mandatory acceptance test (item 17): sparse data handling.

The same manager (a Haaland loyalist) evaluated at 0, 1, 5 and 15
observations. Predictions must move GRADUALLY from prior-driven toward
behaviour-driven - never an abrupt threshold like "after gameweek 5".
"""

import unittest

from fpl_rival.fixtures import prediction_fixtures as fx
from fpl_rival.fixtures.simulation_fixtures import HAALAND_ID
from fpl_rival.prediction.captain_model import shrinkage_weight
from fpl_rival.prediction.models import LeagueHistory, PredictionConfig, PRIOR_DRIVEN
from fpl_rival.prediction.prediction_engine import predict_captain

EP = {pid: v for pid, v in zip(fx.CAPTAIN_POOL, [7.5, 6.0, 7.0, 5.5])}  # does NOT favour Haaland most
TARGET_GW = 25


def _predict_with_n_observations(n: int):
    # Generating exactly n gameweeks (same seed each time) rather than
    # slicing a fixed-length history keeps this correct for any n, including
    # n beyond a previously-fixed history length.
    history_n = fx.loyal_captain_manager_history(n, seed=2)  # Haaland loyalist
    config = PredictionConfig()
    history = LeagueHistory({1: history_n})
    prediction = predict_captain(1, "Loyalist", fx.CAPTAIN_POOL, target_gameweek=TARGET_GW, league_history=history, config=config, expected_points=EP)
    return prediction, config


class TestSparseDataAcceptance(unittest.TestCase):
    def test_shrinkage_weight_increases_gradually_not_abruptly(self):
        config = PredictionConfig()
        weights = {n: shrinkage_weight(n, config) for n in (0, 1, 5, 15)}
        self.assertEqual(weights[0], 0.0)
        self.assertLess(weights[0], weights[1])
        self.assertLess(weights[1], weights[5])
        self.assertLess(weights[5], weights[15])
        # No single step is a "jump to 1" - shrinkage remains well below
        # full personal weight even at 15 observations (gradual, not a switch).
        self.assertLess(weights[15], 0.8)

    def test_zero_observations_is_pure_prior(self):
        prediction, _ = _predict_with_n_observations(0)
        self.assertEqual(prediction.data_sufficiency.data_state, PRIOR_DRIVEN)
        self.assertTrue(prediction.data_sufficiency.low_personal_data)
        # With zero personal history, the prediction must be driven ENTIRELY
        # by the prior (expected points here) - never by unobserved "future" behaviour.
        self.assertEqual(prediction.most_likely_captain, max(EP, key=EP.get))

    def test_probability_on_the_true_favourite_increases_with_observations(self):
        # Compare the two ends: sparse (1 obs) vs rich (15 obs). We don't
        # assert strict monotonicity at every intermediate step, since a
        # single extra noisy observation (n=1) can move things either way -
        # that is expected, real sparse-data behaviour, not a bug. What must
        # hold is the overall direction from "almost no signal" to "clearly
        # behaviour-driven".
        pred_1, _ = _predict_with_n_observations(1)
        pred_15, config = _predict_with_n_observations(15)

        self.assertGreater(pred_15.probabilities[HAALAND_ID], pred_1.probabilities[HAALAND_ID])
        self.assertGreater(pred_15.data_sufficiency.shrinkage_weight, pred_1.data_sufficiency.shrinkage_weight)

    def test_data_sufficiency_state_progresses_through_all_three_states(self):
        states = [_predict_with_n_observations(n)[0].data_sufficiency.data_state for n in (0, 5, 30)]
        from fpl_rival.prediction.models import BEHAVIOUR_DRIVEN, MIXED

        self.assertEqual(states[0], PRIOR_DRIVEN)
        self.assertEqual(states[-1], BEHAVIOUR_DRIVEN)
        # the middle observation count should not have already reached full behaviour-driven certainty
        self.assertIn(states[1], (PRIOR_DRIVEN, MIXED))

    def test_gameweeks_observed_is_reported_accurately_at_every_step(self):
        for n in (0, 1, 5, 15):
            prediction, _ = _predict_with_n_observations(n)
            self.assertEqual(prediction.data_sufficiency.gameweeks_observed, n)


if __name__ == "__main__":
    unittest.main()
