"""Unit tests for fpl_rival.calibration.fitting."""

import unittest

from fpl_rival.calibration.fitting import DEFAULT_GRIDS, coordinate_search, fittable_parameters
from fpl_rival.calibration.models import ChronologicalPoint, DatasetSplit, HistoricalRecord
from fpl_rival.prediction.models import PredictionConfig

SALAH, PALMER = 1, 2
SQUAD = (SALAH, PALMER, 3, 4, 5, 6, 7, 8, 9, 10, 11)


def _dataset(num_managers=2, num_gameweeks=20):
    records = []
    for manager_id in range(1, num_managers + 1):
        for gw in range(1, num_gameweeks + 1):
            captain = SALAH if (gw + manager_id) % 3 != 0 else PALMER
            records.append(HistoricalRecord(season="2026-27", manager_id=manager_id, gameweek=gw, starting_xi=SQUAD, captain=captain))
    return records


class TestFittableParameters(unittest.TestCase):
    def test_single_manager_excludes_consensus_weight(self):
        records = _dataset(num_managers=1)
        params = fittable_parameters(records, expected_points=None)
        self.assertNotIn("weight_consensus", params)

    def test_multiple_managers_includes_consensus_weight(self):
        records = _dataset(num_managers=3)
        params = fittable_parameters(records, expected_points=None)
        self.assertIn("weight_consensus", params)

    def test_no_expected_points_excludes_that_weight(self):
        records = _dataset()
        self.assertNotIn("weight_expected_points", fittable_parameters(records, expected_points=None))

    def test_expected_points_supplied_includes_that_weight(self):
        records = _dataset()
        ep = {("2026-27", 5): {SALAH: 8.0, PALMER: 6.0}}
        self.assertIn("weight_expected_points", fittable_parameters(records, expected_points=ep))

    def test_data_sufficiency_thresholds_never_fitted(self):
        records = _dataset()
        params = fittable_parameters(records, expected_points=None)
        for threshold_param in ("prior_driven_max_shrinkage", "behaviour_driven_min_shrinkage", "low_personal_data_gw_threshold", "high_confidence_min_gw"):
            self.assertNotIn(threshold_param, params)


class TestCoordinateSearchReproducibility(unittest.TestCase):
    def test_same_inputs_give_identical_fitted_config(self):
        records = _dataset()
        split = DatasetSplit(ChronologicalPoint("2026-27", 8), ChronologicalPoint("2026-27", 14), ChronologicalPoint("2026-27", 20), ("2026-27",))
        result1 = coordinate_search(records, split, max_passes=2)
        result2 = coordinate_search(records, split, max_passes=2)
        self.assertEqual(result1.fitted_config, result2.fitted_config)
        self.assertEqual(result1.final_validation_log_loss, result2.final_validation_log_loss)


class TestCoordinateSearchImprovement(unittest.TestCase):
    def test_never_makes_validation_log_loss_worse_than_the_base_config(self):
        records = _dataset()
        split = DatasetSplit(ChronologicalPoint("2026-27", 8), ChronologicalPoint("2026-27", 14), ChronologicalPoint("2026-27", 20), ("2026-27",))
        result = coordinate_search(records, split, base_config=PredictionConfig(), max_passes=2)
        self.assertLessEqual(result.final_validation_log_loss, result.base_validation_log_loss + 1e-9)

    def test_search_log_records_every_evaluation(self):
        records = _dataset()
        split = DatasetSplit(ChronologicalPoint("2026-27", 8), ChronologicalPoint("2026-27", 14), ChronologicalPoint("2026-27", 20), ("2026-27",))
        result = coordinate_search(records, split, max_passes=1)
        self.assertGreater(len(result.search_log), 1)


class TestGridsCoverEveryAlwaysFittableParameter(unittest.TestCase):
    def test_every_default_fittable_parameter_has_a_grid(self):
        records = _dataset(num_managers=3)
        ep = {("2026-27", 1): {SALAH: 8.0}}
        for param in fittable_parameters(records, ep):
            self.assertIn(param, DEFAULT_GRIDS)


if __name__ == "__main__":
    unittest.main()
