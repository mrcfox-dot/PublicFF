"""Unit tests for fpl_rival.prediction.captain_model."""

import unittest

from fpl_rival.prediction.captain_model import (
    _minmax_normalize,
    classify_data_sufficiency,
    score_candidates,
    shrinkage_weight,
    softmax_probabilities,
)
from fpl_rival.prediction.models import BEHAVIOUR_DRIVEN, MIXED, PRIOR_DRIVEN, PredictionConfig


class TestShrinkageWeight(unittest.TestCase):
    def test_zero_decisions_is_zero(self):
        self.assertEqual(shrinkage_weight(0, PredictionConfig()), 0.0)

    def test_equals_half_at_k(self):
        config = PredictionConfig(personal_shrinkage_k=6.0)
        self.assertAlmostEqual(shrinkage_weight(6, config), 0.5)

    def test_strictly_increasing_and_gradual(self):
        config = PredictionConfig()
        weights = [shrinkage_weight(n, config) for n in (0, 1, 5, 15, 50)]
        for a, b in zip(weights, weights[1:]):
            self.assertLess(a, b)
        self.assertLess(weights[-1], 1.0)  # approaches but never reaches 1


class TestMinMaxNormalize(unittest.TestCase):
    def test_bounds_to_zero_one(self):
        result = _minmax_normalize({1: 3.0, 2: 7.0, 3: 5.0})
        self.assertEqual(result[1], 0.0)
        self.assertEqual(result[2], 1.0)
        self.assertAlmostEqual(result[3], 0.5)

    def test_equal_values_fall_back_to_flat_half(self):
        result = _minmax_normalize({1: 4.0, 2: 4.0})
        self.assertEqual(result, {1: 0.5, 2: 0.5})

    def test_empty_input(self):
        self.assertEqual(_minmax_normalize({}), {})


class TestScoreCandidates(unittest.TestCase):
    def test_only_prior_components_contribute_at_zero_shrinkage(self):
        raw = {
            1: {"personal_loyalty_rate": 1.0, "recency_weighted_rate": 1.0, "concentration_share": 1.0, "league_consensus_rate": 0.2},
            2: {"personal_loyalty_rate": 0.0, "recency_weighted_rate": 0.0, "concentration_share": 0.0, "league_consensus_rate": 0.2},
        }
        result = score_candidates(raw, None, None, shrinkage=0.0, config=PredictionConfig())
        # personal components zeroed out by shrinkage=0 -> both candidates score equal (only consensus, which is equal)
        self.assertAlmostEqual(result[1]["score"], result[2]["score"])

    def test_personal_components_contribute_at_full_shrinkage(self):
        raw = {
            1: {"personal_loyalty_rate": 1.0, "recency_weighted_rate": 1.0, "concentration_share": 1.0, "league_consensus_rate": 0.0},
            2: {"personal_loyalty_rate": 0.0, "recency_weighted_rate": 0.0, "concentration_share": 0.0, "league_consensus_rate": 0.0},
        }
        result = score_candidates(raw, None, None, shrinkage=1.0, config=PredictionConfig())
        self.assertGreater(result[1]["score"], result[2]["score"])

    def test_contributions_are_exposed_per_component(self):
        raw = {1: {"personal_loyalty_rate": 0.5, "recency_weighted_rate": 0.5, "concentration_share": 0.5, "league_consensus_rate": 0.5}}
        result = score_candidates(raw, {1: 0.8}, {1: 9.0}, shrinkage=0.5, config=PredictionConfig())
        contributions = result[1]["contributions"]
        for key in ("personal_history", "recent_captaincy", "concentration", "league_consensus", "global_consensus", "expected_points"):
            self.assertIn(key, contributions)


class TestSoftmaxProbabilities(unittest.TestCase):
    def test_sums_to_one(self):
        probs = softmax_probabilities({1: 2.0, 2: 0.5, 3: -1.0}, temperature=1.0)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=9)

    def test_no_negative_probabilities(self):
        probs = softmax_probabilities({1: -5.0, 2: 5.0, 3: 0.0}, temperature=1.0)
        self.assertTrue(all(p >= 0 for p in probs.values()))

    def test_higher_score_gets_higher_probability(self):
        probs = softmax_probabilities({1: 3.0, 2: 1.0}, temperature=1.0)
        self.assertGreater(probs[1], probs[2])

    def test_equal_scores_split_evenly(self):
        probs = softmax_probabilities({1: 1.0, 2: 1.0, 3: 1.0}, temperature=1.0)
        for p in probs.values():
            self.assertAlmostEqual(p, 1 / 3)

    def test_empty_input_returns_empty(self):
        self.assertEqual(softmax_probabilities({}, temperature=1.0), {})


class TestClassifyDataSufficiency(unittest.TestCase):
    def test_zero_history_is_prior_driven_and_low_data(self):
        result = classify_data_sufficiency(0, 0, concentration_idx=0.0, config=PredictionConfig())
        self.assertEqual(result.data_state, PRIOR_DRIVEN)
        self.assertTrue(result.low_personal_data)

    def test_large_history_is_behaviour_driven(self):
        config = PredictionConfig()
        result = classify_data_sufficiency(30, 30, concentration_idx=0.5, config=config)
        self.assertEqual(result.data_state, BEHAVIOUR_DRIVEN)
        self.assertFalse(result.low_personal_data)

    def test_mid_range_is_mixed(self):
        config = PredictionConfig(personal_shrinkage_k=6.0)
        result = classify_data_sufficiency(6, 6, concentration_idx=0.3, config=config)  # shrinkage=0.5
        self.assertEqual(result.data_state, MIXED)

    def test_thresholds_are_configurable(self):
        config = PredictionConfig(prior_driven_max_shrinkage=0.9)  # very permissive
        result = classify_data_sufficiency(6, 6, concentration_idx=0.0, config=config)
        self.assertEqual(result.data_state, PRIOR_DRIVEN)


if __name__ == "__main__":
    unittest.main()
