"""Unit tests for fpl_rival.prediction.stage3_adapter."""

import unittest

from fpl_rival.prediction.models import CaptainObservation, LeagueHistory, PredictionConfig
from fpl_rival.prediction.prediction_engine import predict_captain
from fpl_rival.prediction.stage3_adapter import apply_prediction_to_manager_state, to_captain_probabilities
from fpl_rival.simulation.models import ManagerState

SALAH, PALMER = 1, 2
SQUAD = (SALAH, PALMER)


def _prediction(target_gw=6):
    history = LeagueHistory({1: tuple(CaptainObservation(gameweek=gw, owned_player_ids=SQUAD, captain_id=SALAH) for gw in range(1, target_gw))})
    return predict_captain(1, "Dave", SQUAD, target_gameweek=target_gw, league_history=history, config=PredictionConfig())


class TestToStage3Probabilities(unittest.TestCase):
    def test_produces_a_valid_distribution(self):
        prediction = _prediction()
        distribution = to_captain_probabilities(prediction)
        self.assertAlmostEqual(sum(distribution.values()), 1.0, places=9)
        self.assertEqual(set(distribution.keys()), {SALAH, PALMER})

    def test_directly_usable_by_stage3_validation(self):
        from fpl_rival.simulation.rival_behaviour import validate_captain_distribution

        prediction = _prediction()
        distribution = to_captain_probabilities(prediction)
        validate_captain_distribution(distribution, "Dave")  # must not raise


class TestApplyPredictionToManagerState(unittest.TestCase):
    def test_sets_captain_probabilities_and_most_likely_captain(self):
        prediction = _prediction()
        manager = ManagerState(entry_id=1, name="Dave", current_total_points=100, current_league_position=1, starting_xi=SQUAD, captain_id=SALAH)
        updated = apply_prediction_to_manager_state(manager, prediction)
        self.assertEqual(updated.captain_probabilities, dict(prediction.probabilities))
        self.assertEqual(updated.captain_id, prediction.most_likely_captain)

    def test_does_not_mutate_original(self):
        prediction = _prediction()
        manager = ManagerState(entry_id=1, name="Dave", current_total_points=100, current_league_position=1, starting_xi=SQUAD, captain_id=SALAH)
        apply_prediction_to_manager_state(manager, prediction)
        self.assertIsNone(manager.captain_probabilities)


if __name__ == "__main__":
    unittest.main()
