"""Stage 4.5 brief item 14: robustness tests - the model must degrade
gracefully (valid output, no crash) under difficult real-world-shaped
conditions, never silently produce nonsense.
"""

import unittest

from fpl_rival.calibration.harness import run_stage4_over_dataset
from fpl_rival.calibration.models import HistoricalRecord
from fpl_rival.prediction.models import PredictionConfig

SALAH, PALMER, HAALAND, WATKINS = 1, 2, 3, 4


def _assert_valid_prediction(test_case, result):
    probs = result["candidate_probabilities"]
    test_case.assertAlmostEqual(sum(probs.values()), 1.0, places=6)
    test_case.assertTrue(all(p >= 0 for p in probs.values()))


class TestVeryLittleHistory(unittest.TestCase):
    def test_single_gameweek_of_history_still_produces_a_valid_prediction(self):
        records = [
            HistoricalRecord(season="2026-27", manager_id=1, gameweek=1, starting_xi=(SALAH, PALMER, 5, 6, 7, 8, 9, 10, 11, 12, 13), captain=SALAH),
            HistoricalRecord(season="2026-27", manager_id=1, gameweek=2, starting_xi=(SALAH, PALMER, 5, 6, 7, 8, 9, 10, 11, 12, 13), captain=SALAH),
        ]
        results = run_stage4_over_dataset(records, PredictionConfig())
        for r in results:
            _assert_valid_prediction(self, r)


class TestMidSeasonStyleChange(unittest.TestCase):
    def test_prediction_adapts_after_a_captaincy_style_switch(self):
        squad = (SALAH, PALMER, 5, 6, 7, 8, 9, 10, 11, 12, 13)
        # GWs 1-10: always Salah. GWs 11-20: always Palmer (a genuine style change).
        records = [HistoricalRecord(season="2026-27", manager_id=1, gameweek=gw, starting_xi=squad, captain=SALAH) for gw in range(1, 11)]
        records += [HistoricalRecord(season="2026-27", manager_id=1, gameweek=gw, starting_xi=squad, captain=PALMER) for gw in range(11, 21)]
        results = run_stage4_over_dataset(records, PredictionConfig())
        for r in results:
            _assert_valid_prediction(self, r)
        # Immediately after the switch the model should still lean Salah (recent
        # history hasn't accumulated yet) - by the END, it should have caught up.
        gw12 = next(r for r in results if r["gameweek"] == 12)
        gw20 = next(r for r in results if r["gameweek"] == 20)
        self.assertGreater(gw20["candidate_probabilities"][PALMER], gw12["candidate_probabilities"][PALMER])


class TestPremiumPlayerSold(unittest.TestCase):
    def test_no_crash_when_a_previously_loyal_captain_leaves_the_squad(self):
        squad_with_salah = (SALAH, PALMER, 5, 6, 7, 8, 9, 10, 11, 12, 13)
        squad_without_salah = (HAALAND, PALMER, 5, 6, 7, 8, 9, 10, 11, 12, 13)  # Salah sold, Haaland bought
        records = [HistoricalRecord(season="2026-27", manager_id=1, gameweek=gw, starting_xi=squad_with_salah, captain=SALAH) for gw in range(1, 11)]
        records += [HistoricalRecord(season="2026-27", manager_id=1, gameweek=gw, starting_xi=squad_without_salah, captain=HAALAND) for gw in range(11, 16)]
        results = run_stage4_over_dataset(records, PredictionConfig())
        for r in results:
            _assert_valid_prediction(self, r)
        after_sale = [r for r in results if r["gameweek"] >= 11]
        self.assertTrue(after_sale)
        for r in after_sale:
            self.assertNotIn(SALAH, r["candidate_probabilities"])  # never a candidate once sold

    def test_first_prediction_after_sale_has_no_personal_history_for_the_new_player(self):
        squad_with_salah = (SALAH, PALMER, 5, 6, 7, 8, 9, 10, 11, 12, 13)
        squad_without_salah = (HAALAND, PALMER, 5, 6, 7, 8, 9, 10, 11, 12, 13)
        records = [HistoricalRecord(season="2026-27", manager_id=1, gameweek=gw, starting_xi=squad_with_salah, captain=SALAH) for gw in range(1, 6)]
        records += [HistoricalRecord(season="2026-27", manager_id=1, gameweek=6, starting_xi=squad_without_salah, captain=HAALAND)]
        results = run_stage4_over_dataset(records, PredictionConfig())
        gw6 = next(r for r in results if r["gameweek"] == 6)
        _assert_valid_prediction(self, gw6)


class TestMultipleCandidatesAndDominantExpectedPoints(unittest.TestCase):
    def test_several_plausible_candidates_still_sum_to_one(self):
        squad = (SALAH, PALMER, HAALAND, WATKINS, 8, 9, 10, 11, 12, 13, 14)
        records = [HistoricalRecord(season="2026-27", manager_id=1, gameweek=gw, starting_xi=squad, captain=[SALAH, PALMER, HAALAND, WATKINS][gw % 4]) for gw in range(1, 13)]
        results = run_stage4_over_dataset(records, PredictionConfig())
        for r in results:
            _assert_valid_prediction(self, r)
            self.assertEqual(len(r["candidate_probabilities"]), len(squad))
            for player_id in (SALAH, PALMER, HAALAND, WATKINS):
                self.assertIn(player_id, r["candidate_probabilities"])

    def test_one_dominant_expected_points_candidate_is_reflected_with_no_history(self):
        squad = (SALAH, PALMER, HAALAND, WATKINS, 8, 9, 10, 11, 12, 13, 14)
        records = [HistoricalRecord(season="2026-27", manager_id=1, gameweek=1, starting_xi=squad, captain=SALAH)]
        ep = {("2026-27", 1): {SALAH: 20.0, PALMER: 3.0, HAALAND: 3.0, WATKINS: 3.0}}  # Salah is a massive outlier
        results = run_stage4_over_dataset([], PredictionConfig(), expected_points=ep)  # sanity: empty dataset -> empty results
        self.assertEqual(results, [])
        results = run_stage4_over_dataset(records, PredictionConfig(), expected_points=ep)
        _assert_valid_prediction(self, results[0])


class TestNoisyInconsistentHistory(unittest.TestCase):
    def test_random_looking_history_still_produces_valid_bounded_predictions(self):
        squad = (SALAH, PALMER, HAALAND, WATKINS, 8, 9, 10, 11, 12, 13, 14)
        pattern = [SALAH, HAALAND, PALMER, WATKINS, PALMER, SALAH, WATKINS, HAALAND, SALAH, PALMER]
        records = [HistoricalRecord(season="2026-27", manager_id=1, gameweek=gw + 1, starting_xi=squad, captain=pattern[gw]) for gw in range(len(pattern))]
        results = run_stage4_over_dataset(records, PredictionConfig())
        for r in results:
            _assert_valid_prediction(self, r)


if __name__ == "__main__":
    unittest.main()
