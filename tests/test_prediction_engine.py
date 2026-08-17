"""Unit tests for fpl_rival.prediction.prediction_engine."""

import unittest

from fpl_rival.prediction.exceptions import EmptyCandidateSetError
from fpl_rival.prediction.models import CaptainObservation, LeagueHistory, PredictionConfig
from fpl_rival.prediction.prediction_engine import predict_captain

SALAH, PALMER, HAALAND = 1, 2, 3
SQUAD = (SALAH, PALMER, HAALAND)


def _hist(entry_id, captains):
    return LeagueHistory({entry_id: tuple(CaptainObservation(gameweek=gw, owned_player_ids=SQUAD, captain_id=cap) for gw, cap in captains)})


class TestCandidateRestriction(unittest.TestCase):
    def test_candidates_restricted_to_currently_owned_players(self):
        history = _hist(1, [(1, SALAH)])
        pred = predict_captain(1, "M", (SALAH, PALMER), target_gameweek=2, league_history=history)
        self.assertEqual(set(pred.probabilities.keys()), {SALAH, PALMER})
        self.assertNotIn(HAALAND, pred.probabilities)  # never owned in current_squad -> never a candidate

    def test_empty_squad_fails_clearly(self):
        with self.assertRaises(EmptyCandidateSetError):
            predict_captain(1, "M", (), target_gameweek=1, league_history=LeagueHistory({}))


class TestProbabilityValidity(unittest.TestCase):
    def test_sums_to_one_and_no_negatives(self):
        history = _hist(1, [(gw, SALAH if gw % 2 else PALMER) for gw in range(1, 11)])
        pred = predict_captain(1, "M", SQUAD, target_gameweek=11, league_history=history)
        self.assertAlmostEqual(sum(pred.probabilities.values()), 1.0, places=9)
        self.assertTrue(all(p >= 0 for p in pred.probabilities.values()))


class TestReproducibility(unittest.TestCase):
    def test_same_inputs_give_identical_prediction(self):
        history = _hist(1, [(gw, SALAH) for gw in range(1, 6)])
        pred1 = predict_captain(1, "M", SQUAD, target_gameweek=6, league_history=history, config=PredictionConfig())
        pred2 = predict_captain(1, "M", SQUAD, target_gameweek=6, league_history=history, config=PredictionConfig())
        self.assertEqual(pred1.probabilities, pred2.probabilities)


class TestNoFutureLeakage(unittest.TestCase):
    def test_observations_at_or_after_target_gameweek_are_ignored(self):
        # Deliberately feed a history that INCLUDES the target gameweek and
        # gameweeks after it, captaining a DIFFERENT player than everything
        # before the target - if leakage occurred, this would show up.
        captains_before = [(gw, SALAH) for gw in range(1, 6)]  # gw1-5: always Salah
        captains_leaked = [(6, PALMER), (7, PALMER), (8, PALMER)]  # gw6-8: Palmer (must NOT be seen)
        history = _hist(1, captains_before + captains_leaked)

        pred_with_leak_attempt = predict_captain(1, "M", SQUAD, target_gameweek=6, league_history=history)

        clean_history = _hist(1, captains_before)
        pred_clean = predict_captain(1, "M", SQUAD, target_gameweek=6, league_history=clean_history)

        self.assertEqual(pred_with_leak_attempt.probabilities, pred_clean.probabilities)
        self.assertEqual(pred_with_leak_attempt.data_sufficiency.gameweeks_observed, 5)


if __name__ == "__main__":
    unittest.main()
