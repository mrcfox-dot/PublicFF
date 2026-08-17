"""Unit tests for fpl_rival.simulation.captain_battle - validation and ranking."""

import unittest

from fpl_rival.simulation.captain_battle import run_captain_battle
from fpl_rival.simulation.exceptions import InvalidCaptainError, MissingProjectionError
from fpl_rival.simulation.models import LeagueState, ManagerState, PlayerProjection, SimulationConfig


def _basic_setup():
    projections = {
        1: PlayerProjection(1, "Salah", 8.0, 5.0),
        2: PlayerProjection(2, "Palmer", 6.0, 5.0),
        3: PlayerProjection(3, "Filler", 3.0, 2.0),
    }
    chris = ManagerState(entry_id=1, name="Chris", current_total_points=100, current_league_position=2, starting_xi=(1, 2, 3), captain_id=1)
    rival = ManagerState(entry_id=2, name="Rival", current_total_points=105, current_league_position=1, starting_xi=(1, 3), captain_id=1)
    league = LeagueState(league_id=1, league_name="L", managers=(chris, rival))
    return chris, league, projections


class TestValidation(unittest.TestCase):
    def test_candidate_not_in_xi_fails_clearly(self):
        chris, league, projections = _basic_setup()
        with self.assertRaises(InvalidCaptainError):
            run_captain_battle(chris, league, projections, candidate_captain_ids=[999], config=SimulationConfig(num_simulations=100))

    def test_missing_projection_for_candidate_fails_clearly(self):
        chris, league, projections = _basic_setup()
        del projections[2]
        with self.assertRaises(MissingProjectionError):
            run_captain_battle(chris, league, projections, candidate_captain_ids=[1, 2], config=SimulationConfig(num_simulations=100))

    def test_missing_projection_for_squad_player_fails_clearly(self):
        chris, league, projections = _basic_setup()
        del projections[3]  # a non-candidate but squad-required player
        with self.assertRaises(MissingProjectionError):
            run_captain_battle(chris, league, projections, candidate_captain_ids=[1], config=SimulationConfig(num_simulations=100))

    def test_empty_candidate_list_fails_clearly(self):
        chris, league, projections = _basic_setup()
        with self.assertRaises(InvalidCaptainError):
            run_captain_battle(chris, league, projections, candidate_captain_ids=[], config=SimulationConfig(num_simulations=100))

    def test_unknown_objective_rejected(self):
        chris, league, projections = _basic_setup()
        with self.assertRaises(ValueError):
            run_captain_battle(chris, league, projections, [1, 2], SimulationConfig(num_simulations=100), objective="not_a_real_objective")


class TestRanking(unittest.TestCase):
    def test_rankings_contain_every_candidate_exactly_once(self):
        chris, league, projections = _basic_setup()
        result = run_captain_battle(chris, league, projections, [1, 2], SimulationConfig(num_simulations=2000, random_seed=1))
        self.assertEqual(set(result.ranking_by_expected_points), {1, 2})
        self.assertEqual(set(result.ranking_by_objective), {1, 2})
        self.assertEqual(len(result.candidates), 2)

    def test_expected_points_ranking_matches_actual_means(self):
        chris, league, projections = _basic_setup()
        result = run_captain_battle(chris, league, projections, [1, 2], SimulationConfig(num_simulations=20_000, random_seed=1))
        by_id = {c.player_id: c.result.expected_gameweek_points for c in result.candidates}
        winner_id = result.ranking_by_expected_points[0]
        self.assertEqual(by_id[winner_id], max(by_id.values()))

    def test_objective_ranking_respects_direction_for_expected_position(self):
        chris, league, projections = _basic_setup()
        result = run_captain_battle(
            chris, league, projections, [1, 2], SimulationConfig(num_simulations=20_000, random_seed=1), objective="expected_position"
        )
        by_id = {c.player_id: c.result.expected_position for c in result.candidates}
        winner_id = result.ranking_by_objective[0]
        # lower expected position is better
        self.assertEqual(by_id[winner_id], min(by_id.values()))


if __name__ == "__main__":
    unittest.main()
