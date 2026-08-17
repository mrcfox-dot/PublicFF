"""Unit tests for fpl_rival.prediction.backtest."""

import unittest

from fpl_rival.prediction.backtest import run_backtest
from fpl_rival.prediction.evaluation import evaluate
from fpl_rival.prediction.models import CaptainObservation, LeagueHistory, PredictionConfig

SALAH, PALMER = 1, 2
SQUAD = (SALAH, PALMER)


class TestBacktest(unittest.TestCase):
    def setUp(self):
        # A loyal Salah captainer for 10 gameweeks.
        self.observations = tuple(
            CaptainObservation(gameweek=gw, owned_player_ids=SQUAD, captain_id=SALAH) for gw in range(1, 11)
        )
        self.history = LeagueHistory({1: self.observations})

    def test_returns_one_result_per_observed_gameweek(self):
        results = run_backtest(1, "Dave", self.history, config=PredictionConfig())
        self.assertEqual(len(results), 10)
        self.assertEqual([r["gameweek"] for r in results], list(range(1, 11)))

    def test_first_prediction_uses_zero_prior_history(self):
        results = run_backtest(1, "Dave", self.history, config=PredictionConfig())
        first = results[0]["prediction"]
        self.assertEqual(first.data_sufficiency.gameweeks_observed, 0)

    def test_predictions_only_use_information_before_that_gameweek(self):
        results = run_backtest(1, "Dave", self.history, config=PredictionConfig())
        for i, result in enumerate(results):
            # gameweek i+1's prediction should have seen exactly i prior gameweeks.
            self.assertEqual(result["prediction"].data_sufficiency.gameweeks_observed, i)

    def test_model_improves_as_backtest_progresses(self):
        # A consistent loyalist should become increasingly predictable -
        # log loss on later gameweeks should generally be lower (better)
        # than on the earliest ones, once there's real history to learn from.
        results = run_backtest(1, "Dave", self.history, config=PredictionConfig())
        early_pairs = [{"candidate_probabilities": r["candidate_probabilities"], "actual_captain_id": r["actual_captain_id"]} for r in results[:3]]
        late_pairs = [{"candidate_probabilities": r["candidate_probabilities"], "actual_captain_id": r["actual_captain_id"]} for r in results[-3:]]
        self.assertLess(evaluate(late_pairs)["log_loss"], evaluate(early_pairs)["log_loss"])

    def test_ready_for_evaluation_metrics(self):
        results = run_backtest(1, "Dave", self.history, config=PredictionConfig())
        pairs = [{"candidate_probabilities": r["candidate_probabilities"], "actual_captain_id": r["actual_captain_id"]} for r in results]
        metrics = evaluate(pairs)
        self.assertEqual(metrics["num_predictions"], 10)
        self.assertGreaterEqual(metrics["top1_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
