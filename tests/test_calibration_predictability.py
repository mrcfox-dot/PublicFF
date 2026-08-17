"""Unit tests for fpl_rival.calibration.predictability."""

import math
import unittest

from fpl_rival.calibration.predictability import (
    HIGH,
    INSUFFICIENT_DATA,
    LOW,
    PredictabilityConfig,
    compute_manager_predictability,
)
from fpl_rival.prediction.models import CaptainObservation

SALAH, PALMER = 1, 2
SQUAD = (SALAH, PALMER, 3)


def _hist(captains):
    return tuple(CaptainObservation(gameweek=gw, owned_player_ids=SQUAD, captain_id=c) for gw, c in captains)


def _result(probs, actual):
    return {"candidate_probabilities": probs, "actual_captain_id": actual}


class TestPredictability(unittest.TestCase):
    def test_insufficient_data_below_minimum(self):
        config = PredictabilityConfig(min_predictions=5)
        result = compute_manager_predictability(1, "M", _hist([(1, SALAH)]), [_result({SALAH: 1.0, PALMER: 0.0}, SALAH)], config)
        self.assertEqual(result.label, INSUFFICIENT_DATA)
        self.assertTrue(math.isnan(result.predictability_score))

    def test_highly_predictable_manager_scores_high(self):
        config = PredictabilityConfig(min_predictions=3)
        history = _hist([(gw, SALAH) for gw in range(1, 11)])
        results = [_result({SALAH: 0.95, PALMER: 0.05}, SALAH) for _ in range(10)]
        result = compute_manager_predictability(1, "Loyal", history, results, config)
        self.assertEqual(result.label, HIGH)
        self.assertGreater(result.predictability_score, 0.7)

    def test_unpredictable_manager_scores_low(self):
        config = PredictabilityConfig(min_predictions=3)
        history = _hist([(1, SALAH), (2, PALMER), (3, SALAH), (4, PALMER), (5, SALAH), (6, PALMER)])
        # model was uncertain (near-uniform) AND frequently wrong
        results = [_result({SALAH: 0.5, PALMER: 0.5}, SALAH if i % 2 else PALMER) for i in range(6)]
        result = compute_manager_predictability(1, "Erratic", history, results, config)
        self.assertEqual(result.label, LOW)

    def test_score_is_bounded_between_0_and_1(self):
        config = PredictabilityConfig(min_predictions=1)
        history = _hist([(1, SALAH)])
        results = [_result({SALAH: 1.0, PALMER: 0.0}, SALAH)]
        result = compute_manager_predictability(1, "M", history, results, config)
        self.assertGreaterEqual(result.predictability_score, 0.0)
        self.assertLessEqual(result.predictability_score, 1.0)

    def test_all_components_exposed_not_hidden(self):
        config = PredictabilityConfig(min_predictions=1)
        history = _hist([(1, SALAH)])
        results = [_result({SALAH: 1.0, PALMER: 0.0}, SALAH)]
        result = compute_manager_predictability(1, "M", history, results, config)
        self.assertIsNotNone(result.concentration_index)
        self.assertIsNotNone(result.mean_normalized_entropy)
        self.assertIsNotNone(result.top1_accuracy)


if __name__ == "__main__":
    unittest.main()
