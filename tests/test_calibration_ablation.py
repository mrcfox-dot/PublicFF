"""Unit tests for fpl_rival.calibration.ablation."""

import unittest

from fpl_rival.calibration.ablation import FEATURE_FAMILIES, run_ablation
from fpl_rival.calibration.models import ChronologicalPoint, DatasetSplit, HistoricalRecord
from fpl_rival.prediction.models import PredictionConfig

SALAH, PALMER = 1, 2
SQUAD = (SALAH, PALMER, 3, 4, 5, 6, 7, 8, 9, 10, 11)


def _dataset():
    records = []
    for manager_id in (1, 2, 3):
        for gw in range(1, 21):
            captain = SALAH if (gw + manager_id) % 3 != 0 else PALMER
            records.append(HistoricalRecord(season="2026-27", manager_id=manager_id, gameweek=gw, starting_xi=SQUAD, captain=captain))
    return records


class TestRunAblation(unittest.TestCase):
    def setUp(self):
        self.records = _dataset()
        self.split = DatasetSplit(ChronologicalPoint("2026-27", 8), ChronologicalPoint("2026-27", 14), ChronologicalPoint("2026-27", 20), ("2026-27",))

    def test_every_family_with_nonzero_weight_is_ablated(self):
        config = PredictionConfig()  # every weight is nonzero by default
        rows = run_ablation(self.records, self.split, config)
        self.assertEqual({r.feature_family for r in rows}, set(FEATURE_FAMILIES))

    def test_ablating_an_already_zero_weight_is_skipped(self):
        from dataclasses import replace

        config = replace(PredictionConfig(), weight_expected_points=0.0)
        rows = run_ablation(self.records, self.split, config)
        self.assertNotIn("no_expected_points", {r.feature_family for r in rows})

    def test_only_families_filter_works(self):
        config = PredictionConfig()
        rows = run_ablation(self.records, self.split, config, only_families=["no_personal_history"])
        self.assertEqual({r.feature_family for r in rows}, {"no_personal_history"})

    def test_removed_parameters_match_the_family(self):
        config = PredictionConfig()
        rows = run_ablation(self.records, self.split, config, only_families=["no_recency"])
        self.assertEqual(rows[0].removed_parameters, ("weight_recency",))


if __name__ == "__main__":
    unittest.main()
