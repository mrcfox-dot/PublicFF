"""Unit tests for fpl_rival.calibration.history_depth."""

import unittest

from fpl_rival.calibration.history_depth import bucket_by_history_depth


def _result(gameweeks_observed, probs, actual):
    return {"gameweeks_observed": gameweeks_observed, "candidate_probabilities": probs, "actual_captain_id": actual}


class TestBucketByHistoryDepth(unittest.TestCase):
    def test_each_result_lands_in_exactly_one_bucket(self):
        results = [
            _result(0, {1: 1.0}, 1),
            _result(4, {1: 1.0}, 1),
            _result(8, {1: 1.0}, 1),
            _result(15, {1: 1.0}, 1),
            _result(25, {1: 1.0}, 1),
        ]
        rows = bucket_by_history_depth(results)
        sizes = {r.bucket_label: r.sample_size for r in rows}
        self.assertEqual(sizes["0-2"], 1)
        self.assertEqual(sizes["3-5"], 1)
        self.assertEqual(sizes["6-10"], 1)
        self.assertEqual(sizes["11-20"], 1)
        self.assertEqual(sizes["21+"], 1)

    def test_empty_bucket_reports_nan_metrics_not_zero(self):
        import math

        rows = bucket_by_history_depth([_result(0, {1: 1.0}, 1)])
        empty_bucket = next(r for r in rows if r.bucket_label == "21+")
        self.assertEqual(empty_bucket.sample_size, 0)
        self.assertTrue(math.isnan(empty_bucket.top1_accuracy))

    def test_boundary_values_included_in_correct_bucket(self):
        rows = bucket_by_history_depth([_result(2, {1: 1.0}, 1), _result(3, {1: 1.0}, 1)])
        sizes = {r.bucket_label: r.sample_size for r in rows}
        self.assertEqual(sizes["0-2"], 1)
        self.assertEqual(sizes["3-5"], 1)


if __name__ == "__main__":
    unittest.main()
