"""Stage 4 mandatory acceptance test (item 16): behavioural differentiation.

Two managers with IDENTICAL current squads and IDENTICAL expected-points
input. Manager A historically follows consensus captaincy; Manager B
historically chooses differential captains. Once sufficient behavioural
history exists, the model MUST produce meaningfully different probability
distributions for A and B - and that difference must come purely from
their different histories, not from any manager-specific rule (there is
no "if manager == B" anywhere in the engine).
"""

import unittest

from fpl_rival.fixtures import prediction_fixtures as fx
from fpl_rival.prediction.models import LeagueHistory, PredictionConfig
from fpl_rival.prediction.prediction_engine import predict_captain

EP = {pid: v for pid, v in zip(fx.CAPTAIN_POOL, [7.5, 6.0, 7.0, 5.5])}  # Salah highest - same for both managers
TARGET_GW = 21


class TestBehaviouralAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        history_a = fx.template_manager_history(20, seed=101)  # "Manager A": consensus follower
        history_b = fx.differential_manager_history(20, seed=102)  # "Manager B": differential

        cls.pred_a = predict_captain(
            1, "Manager A (consensus follower)", fx.CAPTAIN_POOL, target_gameweek=TARGET_GW,
            league_history=LeagueHistory({1: history_a}), config=PredictionConfig(), expected_points=EP,
        )
        cls.pred_b = predict_captain(
            2, "Manager B (differential)", fx.CAPTAIN_POOL, target_gameweek=TARGET_GW,
            league_history=LeagueHistory({2: history_b}), config=PredictionConfig(), expected_points=EP,
        )

    def test_both_are_behaviour_driven_by_gw21(self):
        # Precondition: this is a genuine "sufficient history" comparison,
        # not an artifact of both still being prior-driven.
        from fpl_rival.prediction.models import BEHAVIOUR_DRIVEN

        self.assertEqual(self.pred_a.data_sufficiency.data_state, BEHAVIOUR_DRIVEN)
        self.assertEqual(self.pred_b.data_sufficiency.data_state, BEHAVIOUR_DRIVEN)

    def test_probability_on_the_consensus_favourite_differs_substantially(self):
        salah_a = self.pred_a.probabilities[fx.SALAH_ID]
        salah_b = self.pred_b.probabilities[fx.SALAH_ID]
        self.assertGreater(salah_a, salah_b)
        self.assertGreater(salah_a - salah_b, 0.3)  # not a marginal difference

    def test_most_likely_captain_differs(self):
        self.assertNotEqual(self.pred_a.most_likely_captain, self.pred_b.most_likely_captain)
        self.assertEqual(self.pred_a.most_likely_captain, fx.CONSENSUS_FAVOURITE_ID)

    def test_difference_emerges_purely_from_history_not_from_inputs(self):
        # Sanity check on the test's own construction, not the model: both
        # managers were given the identical candidate pool and EP input.
        self.assertEqual(set(self.pred_a.probabilities), set(self.pred_b.probabilities))
        for cid in self.pred_a.candidate_features:
            self.assertEqual(
                self.pred_a.candidate_features[cid].expected_points_value,
                self.pred_b.candidate_features[cid].expected_points_value,
            )


if __name__ == "__main__":
    unittest.main()
