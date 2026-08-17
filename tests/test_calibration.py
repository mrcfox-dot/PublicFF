"""Unit tests for fpl_rival.prediction.calibration."""

import unittest

from fpl_rival.prediction.calibration import compute_calibration_buckets

SALAH, PALMER = 1, 2


class TestCalibrationBuckets(unittest.TestCase):
    def test_bucket_count_matches_num_buckets(self):
        buckets = compute_calibration_buckets([], num_buckets=10)
        self.assertEqual(len(buckets), 10)

    def test_every_candidate_probability_is_counted_not_just_the_winner(self):
        # One prediction with two candidates -> should contribute TWO
        # (candidate, probability) observations to the buckets, not one.
        pairs = [{"candidate_probabilities": {SALAH: 0.7, PALMER: 0.3}, "actual_captain_id": SALAH}]
        buckets = compute_calibration_buckets(pairs, num_buckets=10)
        total_count = sum(b["count"] for b in buckets)
        self.assertEqual(total_count, 2)

    def test_well_calibrated_bucket_shows_matching_actual_rate(self):
        # 10 predictions all assigning ~70-80% to the eventual actual captain.
        pairs = [{"candidate_probabilities": {SALAH: 0.75, PALMER: 0.25}, "actual_captain_id": SALAH} for _ in range(7)]
        pairs += [{"candidate_probabilities": {SALAH: 0.75, PALMER: 0.25}, "actual_captain_id": PALMER} for _ in range(3)]
        buckets = compute_calibration_buckets(pairs, num_buckets=10)
        bucket_70_80 = next(b for b in buckets if b["range_low"] == 0.7)
        self.assertEqual(bucket_70_80["count"], 10)  # Salah's 0.75 appears 10 times
        self.assertAlmostEqual(bucket_70_80["actual_occurrence_rate"], 0.7)

        bucket_20_30 = next(b for b in buckets if b["range_low"] == 0.2)
        self.assertEqual(bucket_20_30["count"], 10)  # Palmer's 0.25 appears 10 times
        self.assertAlmostEqual(bucket_20_30["actual_occurrence_rate"], 0.3)

    def test_empty_bucket_has_none_rate_not_zero(self):
        buckets = compute_calibration_buckets([], num_buckets=10)
        for b in buckets:
            self.assertIsNone(b["actual_occurrence_rate"])

    def test_probability_of_exactly_one_goes_in_last_bucket(self):
        pairs = [{"candidate_probabilities": {SALAH: 1.0}, "actual_captain_id": SALAH}]
        buckets = compute_calibration_buckets(pairs, num_buckets=10)
        self.assertEqual(buckets[-1]["count"], 1)

    def test_invalid_num_buckets_rejected(self):
        with self.assertRaises(ValueError):
            compute_calibration_buckets([], num_buckets=0)


if __name__ == "__main__":
    unittest.main()
