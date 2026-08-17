"""Unit tests for fpl_rival.prediction.evaluation."""

import math
import unittest

from fpl_rival.prediction.evaluation import brier_score, evaluate, log_loss, top1_accuracy, top2_accuracy

SALAH, PALMER, HAALAND = 1, 2, 3


class TestTop1Top2Accuracy(unittest.TestCase):
    def test_top1_accuracy(self):
        pairs = [
            {"candidate_probabilities": {SALAH: 0.7, PALMER: 0.3}, "actual_captain_id": SALAH},  # correct
            {"candidate_probabilities": {SALAH: 0.7, PALMER: 0.3}, "actual_captain_id": PALMER},  # wrong
        ]
        self.assertAlmostEqual(top1_accuracy(pairs), 0.5)

    def test_top2_accuracy_more_forgiving_than_top1(self):
        pairs = [{"candidate_probabilities": {SALAH: 0.5, PALMER: 0.3, HAALAND: 0.2}, "actual_captain_id": PALMER}]
        self.assertEqual(top1_accuracy(pairs), 0.0)
        self.assertEqual(top2_accuracy(pairs), 1.0)  # Palmer is 2nd highest

    def test_empty_pairs_is_nan(self):
        self.assertTrue(math.isnan(top1_accuracy([])))


class TestLogLoss(unittest.TestCase):
    def test_distinguishes_confident_from_uncertain_correct_predictions(self):
        # Both predict Salah correctly, but with very different confidence.
        confident = [{"candidate_probabilities": {SALAH: 0.99, PALMER: 0.01}, "actual_captain_id": SALAH}]
        uncertain = [{"candidate_probabilities": {SALAH: 0.51, PALMER: 0.49}, "actual_captain_id": SALAH}]
        self.assertLess(log_loss(confident), log_loss(uncertain))

    def test_perfect_prediction_has_near_zero_loss(self):
        pairs = [{"candidate_probabilities": {SALAH: 1.0}, "actual_captain_id": SALAH}]
        self.assertAlmostEqual(log_loss(pairs), 0.0, places=6)

    def test_wrong_confident_prediction_is_penalised_heavily(self):
        wrong_confident = [{"candidate_probabilities": {SALAH: 0.99, PALMER: 0.01}, "actual_captain_id": PALMER}]
        wrong_uncertain = [{"candidate_probabilities": {SALAH: 0.51, PALMER: 0.49}, "actual_captain_id": PALMER}]
        self.assertGreater(log_loss(wrong_confident), log_loss(wrong_uncertain))


class TestBrierScore(unittest.TestCase):
    def test_distinguishes_confidence_levels_even_with_same_winner(self):
        confident = [{"candidate_probabilities": {SALAH: 0.99, PALMER: 0.01}, "actual_captain_id": SALAH}]
        uncertain = [{"candidate_probabilities": {SALAH: 0.51, PALMER: 0.49}, "actual_captain_id": SALAH}]
        self.assertLess(brier_score(confident), brier_score(uncertain))

    def test_perfect_prediction_is_zero(self):
        pairs = [{"candidate_probabilities": {SALAH: 1.0, PALMER: 0.0}, "actual_captain_id": SALAH}]
        self.assertAlmostEqual(brier_score(pairs), 0.0)


class TestEvaluateBundle(unittest.TestCase):
    def test_returns_all_metrics(self):
        pairs = [{"candidate_probabilities": {SALAH: 0.6, PALMER: 0.4}, "actual_captain_id": SALAH}]
        result = evaluate(pairs)
        for key in ("num_predictions", "top1_accuracy", "top2_accuracy", "log_loss", "brier_score"):
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()
