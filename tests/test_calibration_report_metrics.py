"""Unit tests for fpl_rival.calibration.calibration_report (ECE + temperature scaling)."""

import unittest

from fpl_rival.calibration.calibration_report import apply_temperature, expected_calibration_error, fit_temperature
from fpl_rival.prediction.calibration import compute_calibration_buckets
from fpl_rival.prediction.captain_model import softmax_probabilities
from fpl_rival.prediction.models import CandidateFeatures, CaptainPrediction, DataSufficiency, PRIOR_DRIVEN

SALAH, PALMER = 1, 2


def _prediction(raw_scores, actual_captain_id_unused=None):
    probabilities = softmax_probabilities(raw_scores, temperature=1.0)
    features = {
        cid: CandidateFeatures(
            player_id=cid, times_owned=1, times_captained=1, personal_loyalty_rate=0.5, recency_weighted_rate=0.5,
            concentration_share=0.5, league_consensus_rate=0.0, global_consensus_value=None, expected_points_value=None,
            raw_score=score,
        )
        for cid, score in raw_scores.items()
    }
    data_sufficiency = DataSufficiency(0, 0, 0.0, PRIOR_DRIVEN, True, "Low", 0.0)
    return CaptainPrediction(1, "M", 1, probabilities, features, data_sufficiency, "test")


class TestExpectedCalibrationError(unittest.TestCase):
    def test_zero_for_perfectly_calibrated_buckets(self):
        pairs = [{"candidate_probabilities": {SALAH: 0.5, PALMER: 0.5}, "actual_captain_id": SALAH if i % 2 else PALMER} for i in range(10)]
        buckets = compute_calibration_buckets(pairs, num_buckets=10)
        self.assertAlmostEqual(expected_calibration_error(buckets), 0.0, places=6)

    def test_positive_for_overconfident_predictions(self):
        # Always predicts 95% for the winner but is right only half the time.
        pairs = [{"candidate_probabilities": {SALAH: 0.95, PALMER: 0.05}, "actual_captain_id": SALAH if i % 2 else PALMER} for i in range(10)]
        buckets = compute_calibration_buckets(pairs, num_buckets=10)
        self.assertGreater(expected_calibration_error(buckets), 0.2)

    def test_nan_for_empty_input(self):
        import math

        buckets = compute_calibration_buckets([], num_buckets=10)
        self.assertTrue(math.isnan(expected_calibration_error(buckets)))


class TestTemperatureScaling(unittest.TestCase):
    def test_temperature_one_reproduces_original_probabilities(self):
        prediction = _prediction({SALAH: 2.0, PALMER: 0.5})
        from fpl_rival.calibration.calibration_report import _rescale_with_temperature

        rescaled = _rescale_with_temperature(prediction, 1.0)
        for cid in prediction.probabilities:
            self.assertAlmostEqual(rescaled[cid], prediction.probabilities[cid], places=9)

    def test_low_temperature_sharpens_the_distribution(self):
        prediction = _prediction({SALAH: 2.0, PALMER: 0.5})
        from fpl_rival.calibration.calibration_report import _rescale_with_temperature

        sharp = _rescale_with_temperature(prediction, 0.3)
        original = prediction.probabilities
        self.assertGreater(sharp[SALAH], original[SALAH])

    def test_fit_temperature_improves_or_matches_baseline(self):
        results = [{"prediction": _prediction({SALAH: 0.3, PALMER: 0.25}), "actual_captain_id": SALAH} for _ in range(20)]
        result = fit_temperature(results)
        self.assertLessEqual(result.fitted_validation_log_loss, result.baseline_validation_log_loss + 1e-9)

    def test_apply_temperature_returns_valid_eval_pairs(self):
        results = [{"prediction": _prediction({SALAH: 0.3, PALMER: 0.25}), "actual_captain_id": SALAH}]
        pairs = apply_temperature(results, 0.5)
        self.assertAlmostEqual(sum(pairs[0]["candidate_probabilities"].values()), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
