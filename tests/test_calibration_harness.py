"""Unit tests for fpl_rival.calibration.harness - no-leakage, phase tagging."""

import unittest

from fpl_rival.calibration.harness import filter_phase, run_baseline_over_dataset, run_stage4_over_dataset, to_eval_pairs
from fpl_rival.calibration.models import ChronologicalPoint, DatasetSplit, HistoricalRecord, TEST, TRAIN, VALIDATION
from fpl_rival.prediction.models import PredictionConfig

SALAH, PALMER = 1, 2
SQUAD = (SALAH, PALMER, 3, 4, 5, 6, 7, 8, 9, 10, 11)


def _records(manager_id, captains_by_gw, season="2026-27"):
    return [HistoricalRecord(season=season, manager_id=manager_id, gameweek=gw, starting_xi=SQUAD, captain=c) for gw, c in captains_by_gw]


class TestPhaseTagging(unittest.TestCase):
    def test_every_prediction_tagged_with_correct_phase(self):
        records = _records(1, [(gw, SALAH) for gw in range(1, 11)])
        split = DatasetSplit(ChronologicalPoint("2026-27", 3), ChronologicalPoint("2026-27", 6), ChronologicalPoint("2026-27", 10), ("2026-27",))
        results = run_stage4_over_dataset(records, PredictionConfig(), split=split)
        for r in results:
            expected = split.phase_of(r["season"], r["gameweek"])
            self.assertEqual(r["phase"], expected)
        self.assertEqual(len(filter_phase(results, TRAIN)), 3)
        self.assertEqual(len(filter_phase(results, VALIDATION)), 3)
        self.assertEqual(len(filter_phase(results, TEST)), 4)


class TestNoFutureLeakageInHarness(unittest.TestCase):
    """Deliberately feeds a dataset containing gameweeks AFTER the point
    being predicted and confirms the harness (via Stage 4's own leakage
    boundary) ignores them - for both the model and a baseline."""

    def test_stage4_model_ignores_later_gameweeks(self):
        early = [(gw, SALAH) for gw in range(1, 6)]
        gw6 = [(6, SALAH)]  # present identically in both datasets - what we compare
        leaked_future = [(7, PALMER), (8, PALMER)]  # must not influence GW6's prediction
        full_records = _records(1, early + gw6 + leaked_future)
        clean_records = _records(1, early + gw6)

        full_results = run_stage4_over_dataset(full_records, PredictionConfig())
        clean_results = run_stage4_over_dataset(clean_records, PredictionConfig())

        gw6_full = next(r for r in full_results if r["gameweek"] == 6)
        gw6_clean = next(r for r in clean_results if r["gameweek"] == 6)
        self.assertEqual(gw6_full["candidate_probabilities"], gw6_clean["candidate_probabilities"])
        self.assertEqual(gw6_full["gameweeks_observed"], 5)

    def test_baseline_ignores_later_gameweeks(self):
        early = [(gw, SALAH) for gw in range(1, 6)]
        gw6 = [(6, SALAH)]
        leaked_future = [(7, PALMER)]
        full_records = _records(1, early + gw6 + leaked_future)
        clean_records = _records(1, early + gw6)

        full_results = run_baseline_over_dataset(full_records, "last_captain")
        clean_results = run_baseline_over_dataset(clean_records, "last_captain")

        gw6_full = next(r for r in full_results if r["gameweek"] == 6)
        gw6_clean = next(r for r in clean_results if r["gameweek"] == 6)
        self.assertEqual(gw6_full["candidate_probabilities"], gw6_clean["candidate_probabilities"])

    def test_consensus_baseline_ignores_other_managers_future_gameweeks(self):
        # Manager 2's LATER captaincy must not leak into manager 1's GW6 consensus feature.
        manager1 = _records(1, [(gw, SALAH) for gw in range(1, 11)])
        manager2_early = _records(2, [(gw, SALAH) for gw in range(1, 6)])
        manager2_late_leak = _records(2, [(gw, PALMER) for gw in range(6, 11)])

        full_records = manager1 + manager2_early + manager2_late_leak
        clean_records = manager1 + manager2_early

        full_results = run_baseline_over_dataset(full_records, "consensus")
        clean_results = run_baseline_over_dataset(clean_records, "consensus")

        gw6_full = next(r for r in full_results if r["manager_id"] == 1 and r["gameweek"] == 6)
        gw6_clean = next(r for r in clean_results if r["manager_id"] == 1 and r["gameweek"] == 6)
        self.assertEqual(gw6_full["candidate_probabilities"], gw6_clean["candidate_probabilities"])


class TestMaxManagers(unittest.TestCase):
    def test_limits_number_of_managers_processed(self):
        records = []
        for manager_id in range(1, 6):
            records += _records(manager_id, [(1, SALAH)])
        results = run_stage4_over_dataset(records, PredictionConfig(), max_managers=2)
        self.assertEqual(len({r["manager_id"] for r in results}), 2)


class TestToEvalPairs(unittest.TestCase):
    def test_extracts_only_evaluation_relevant_fields(self):
        records = _records(1, [(1, SALAH)])
        results = run_stage4_over_dataset(records, PredictionConfig())
        pairs = to_eval_pairs(results)
        self.assertEqual(set(pairs[0].keys()), {"candidate_probabilities", "actual_captain_id"})


if __name__ == "__main__":
    unittest.main()
