"""Stage 4 mandatory acceptance tests (items 9 and 18): the full
history -> prediction -> simulation -> decision pipeline.

Item 9: Chris trails Dave. Dave historically captains Salah heavily.
Stage 4's prediction is fed automatically into Stage 3 (via
stage3_adapter.py, no manual re-specification of probabilities), and the
full pipeline must run end to end and produce a coherent decision.

Item 18: change ONLY Dave's behavioural history (not any Stage 3 input by
hand) and rerun the identical pipeline. Chris's mini-league-optimal
captain must be free to change as a result - proving Stage 4's rival
modelling genuinely influences Stage 3's optimisation, not just a labelled
pass-through.
"""

import unittest
from dataclasses import replace

from fpl_rival.fixtures import prediction_fixtures as fx
from fpl_rival.prediction.models import PredictionConfig
from fpl_rival.prediction.prediction_engine import predict_captain
from fpl_rival.prediction.stage3_adapter import apply_prediction_to_manager_state
from fpl_rival.simulation.captain_battle import run_captain_battle
from fpl_rival.simulation.models import SimulationConfig


def _run_pipeline(dave_history):
    chris, league, projections, candidates, dave, dave_league_history, expected_points = fx.full_pipeline_scenario(dave_history)

    prediction = predict_captain(
        rival_entry_id=dave.entry_id,
        rival_name=dave.name,
        current_squad=dave.starting_xi,
        target_gameweek=11,
        league_history=dave_league_history,
        config=PredictionConfig(),
        expected_points=expected_points,
    )

    dave_updated = apply_prediction_to_manager_state(dave, prediction)
    managers = tuple(dave_updated if m.entry_id == dave.entry_id else m for m in league.managers)
    league = replace(league, managers=managers)

    sim_config = SimulationConfig(num_simulations=30_000, random_seed=42, rival_captain_mode="probabilistic")
    battle_result = run_captain_battle(chris, league, projections, candidates, sim_config)
    return prediction, battle_result


class TestEndToEndPipeline(unittest.TestCase):
    """Item 9: the full pipeline runs and Stage 4's prediction genuinely
    reaches Stage 3 (not just a label - the actual sampled captain rate)."""

    def test_stage4_prediction_feeds_stage3_without_manual_respecification(self):
        dave_history = fx.dave_salah_loyal_history()  # ~90% Salah
        prediction, battle_result = _run_pipeline(dave_history)

        self.assertGreater(prediction.probabilities[fx.SALAH_ID], 0.7)
        # Chris's candidates and result must exist and be well-formed - the
        # pipeline produced a coherent decision, not a crash or NaNs.
        self.assertEqual(len(battle_result.candidates), 2)
        for candidate in battle_result.candidates:
            self.assertGreaterEqual(candidate.result.prob_finish_first, 0.0)
            self.assertLessEqual(candidate.result.prob_finish_first, 1.0)

    def test_chris_trails_dave_in_this_scenario(self):
        dave_history = fx.dave_salah_loyal_history()
        _, battle_result = _run_pipeline(dave_history)
        self.assertGreater(battle_result.starting_gap_to_leader, 0)


class TestStrategyFlipAcceptance(unittest.TestCase):
    """Item 18: changing ONLY Dave's history flips Chris's optimal captain,
    with no manual change to any Stage 3 input."""

    @classmethod
    def setUpClass(cls):
        cls.prediction_loyal, cls.battle_loyal = _run_pipeline(fx.dave_salah_loyal_history())
        cls.prediction_balanced, cls.battle_balanced = _run_pipeline(fx.dave_balanced_history())

    def test_stage4_predictions_genuinely_differ_between_histories(self):
        self.assertGreater(
            self.prediction_loyal.probabilities[fx.SALAH_ID], self.prediction_balanced.probabilities[fx.SALAH_ID]
        )

    def test_objective_winner_changes_between_the_two_histories(self):
        self.assertNotEqual(
            self.battle_loyal.objective_winner.player_name, self.battle_balanced.objective_winner.player_name
        )

    def test_loyal_history_favours_differentiating_from_dave(self):
        # When Dave is (near-)certain to captain Salah, matching him should
        # be the WORSE mini-league choice for a trailing Chris (see Stage 3's
        # own captain_attack acceptance test for the underlying mechanism).
        by_name = {c.player_name: c.result for c in self.battle_loyal.candidates}
        self.assertGreater(by_name["Palmer"].prob_finish_first, by_name["Salah"].prob_finish_first)

    def test_balanced_history_favours_matching_the_higher_mean_captain(self):
        by_name = {c.player_name: c.result for c in self.battle_balanced.candidates}
        self.assertGreater(by_name["Salah"].prob_finish_first, by_name["Palmer"].prob_finish_first)


if __name__ == "__main__":
    unittest.main()
