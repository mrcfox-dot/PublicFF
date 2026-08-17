"""Stage 4 brief item 15: five synthetic behavioural personas produce
different, sensible probability distributions given the same candidate
pool and expected-points input."""

import unittest

from fpl_rival.fixtures import prediction_fixtures as fx
from fpl_rival.prediction.models import LeagueHistory, PredictionConfig
from fpl_rival.prediction.prediction_engine import predict_captain

EP = {pid: v for pid, v in zip(fx.CAPTAIN_POOL, [7.5, 6.0, 7.0, 5.5])}  # Salah highest
TARGET_GW = 21
NUM_GAMEWEEKS = 20


def _predict(history, expected_points=EP):
    return predict_captain(
        1, "Persona", fx.CAPTAIN_POOL, target_gameweek=TARGET_GW,
        league_history=LeagueHistory({1: history}), config=PredictionConfig(), expected_points=expected_points,
    )


class TestFivePersonas(unittest.TestCase):
    def test_template_manager_favours_the_consensus_player(self):
        pred = _predict(fx.template_manager_history(NUM_GAMEWEEKS))
        self.assertEqual(pred.most_likely_captain, fx.CONSENSUS_FAVOURITE_ID)

    def test_loyal_manager_favours_their_one_premium_player(self):
        # Haaland is the persona's target - see prediction_fixtures.loyal_captain_manager_history.
        from fpl_rival.fixtures.simulation_fixtures import HAALAND_ID

        pred = _predict(fx.loyal_captain_manager_history(NUM_GAMEWEEKS))
        self.assertEqual(pred.most_likely_captain, HAALAND_ID)
        self.assertGreater(pred.probabilities[HAALAND_ID], 0.7)

    def test_differential_manager_does_not_favour_the_consensus_player(self):
        pred = _predict(fx.differential_manager_history(NUM_GAMEWEEKS))
        self.assertNotEqual(pred.most_likely_captain, fx.CONSENSUS_FAVOURITE_ID)

    def test_expected_points_follower_tracks_current_top_ep_player(self):
        history, ep_series = fx.expected_points_follower_history(NUM_GAMEWEEKS)
        pred = _predict(history, expected_points=ep_series.get(TARGET_GW, EP))
        top_ep_player = max(ep_series.get(TARGET_GW, EP), key=ep_series.get(TARGET_GW, EP).get)
        self.assertEqual(pred.most_likely_captain, top_ep_player)

    def test_erratic_manager_has_lower_concentration_than_loyal_manager(self):
        from fpl_rival.prediction.features import concentration_index

        loyal_idx = concentration_index(fx.loyal_captain_manager_history(NUM_GAMEWEEKS))
        erratic_idx = concentration_index(fx.erratic_manager_history(NUM_GAMEWEEKS))
        self.assertLess(erratic_idx, loyal_idx)

    def test_all_five_personas_produce_valid_distributions(self):
        histories = [
            fx.template_manager_history(NUM_GAMEWEEKS),
            fx.loyal_captain_manager_history(NUM_GAMEWEEKS),
            fx.differential_manager_history(NUM_GAMEWEEKS),
            fx.expected_points_follower_history(NUM_GAMEWEEKS)[0],
            fx.erratic_manager_history(NUM_GAMEWEEKS),
        ]
        for history in histories:
            pred = _predict(history)
            self.assertAlmostEqual(sum(pred.probabilities.values()), 1.0, places=9)
            self.assertTrue(all(p >= 0 for p in pred.probabilities.values()))

    def test_personas_are_not_all_identical(self):
        # The whole point: same candidate pool, same EP input, different
        # history -> different predictions.
        preds = {
            "template": _predict(fx.template_manager_history(NUM_GAMEWEEKS)).probabilities,
            "loyal": _predict(fx.loyal_captain_manager_history(NUM_GAMEWEEKS)).probabilities,
            "differential": _predict(fx.differential_manager_history(NUM_GAMEWEEKS)).probabilities,
            "erratic": _predict(fx.erratic_manager_history(NUM_GAMEWEEKS)).probabilities,
        }
        distributions = list(preds.values())
        for i in range(len(distributions)):
            for j in range(i + 1, len(distributions)):
                self.assertNotEqual(distributions[i], distributions[j])


if __name__ == "__main__":
    unittest.main()
