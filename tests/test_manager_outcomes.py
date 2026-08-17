"""Unit tests for fpl_rival.simulation.manager_outcomes.

These specifically verify the "shared player outcome" requirement: two
managers who own the same player must see the IDENTICAL simulated score
for that player in a given simulation, because both managers' totals are
built from the same ``player_points`` dict.
"""

import unittest

import numpy as np

from fpl_rival.simulation.manager_outcomes import simulate_manager_gameweek
from fpl_rival.simulation.models import ManagerState


class TestSharedPlayerOutcome(unittest.TestCase):
    def setUp(self):
        n = 10
        # deterministic "player_points" - no RNG needed to prove the mechanism.
        self.player_points = {
            1: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=float),  # shared player (e.g. Salah)
            2: np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float),
            3: np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=float),
        }
        self.n = n

    def test_captain_points_doubled_correctly(self):
        manager = ManagerState(entry_id=1, name="M", current_total_points=0, current_league_position=1, starting_xi=(1, 2, 3), captain_id=1)
        captain_ids = np.full(self.n, 1)
        scores = simulate_manager_gameweek(manager, self.player_points, captain_ids)
        # XI sum = player1 + player2 + player3, captain bonus = player1 again.
        expected = self.player_points[1] * 2 + self.player_points[2] + self.player_points[3]
        np.testing.assert_array_equal(scores, expected)

    def test_xi_points_count_once_each(self):
        manager = ManagerState(entry_id=1, name="M", current_total_points=0, current_league_position=1, starting_xi=(1, 2, 3), captain_id=2)
        captain_ids = np.full(self.n, 2)
        scores = simulate_manager_gameweek(manager, self.player_points, captain_ids)
        expected = self.player_points[1] + self.player_points[2] * 2 + self.player_points[3]
        np.testing.assert_array_equal(scores, expected)

    def test_same_shared_player_produces_identical_contribution_for_two_managers(self):
        manager_a = ManagerState(entry_id=1, name="A", current_total_points=0, current_league_position=1, starting_xi=(1, 2), captain_id=2)
        manager_b = ManagerState(entry_id=2, name="B", current_total_points=0, current_league_position=1, starting_xi=(1, 3), captain_id=3)
        # Both own player 1 but do NOT captain it - isolate player 1's raw
        # (non-doubled) contribution and confirm it's identical for both.
        scores_a = simulate_manager_gameweek(manager_a, self.player_points, np.full(self.n, 2))
        scores_b = simulate_manager_gameweek(manager_b, self.player_points, np.full(self.n, 3))
        contribution_a = scores_a - self.player_points[2] * 2
        contribution_b = scores_b - self.player_points[3] * 2
        np.testing.assert_array_equal(contribution_a, self.player_points[1])
        np.testing.assert_array_equal(contribution_b, self.player_points[1])
        np.testing.assert_array_equal(contribution_a, contribution_b)

    def test_vice_captain_fallback_when_captain_scores_zero(self):
        player_points = {
            1: np.array([0, 0, 5], dtype=float),  # captain: blanks in sims 0,1
            2: np.array([3, 4, 9], dtype=float),  # vice
        }
        manager = ManagerState(entry_id=1, name="M", current_total_points=0, current_league_position=1, starting_xi=(1, 2), captain_id=1, vice_captain_id=2)
        scores = simulate_manager_gameweek(manager, player_points, np.full(3, 1))
        # sim0: captain(1)=0 -> use vice(2)=3 as bonus. xi=1+2=0+3=3, bonus=3 -> total 6
        # sim1: captain=0 -> vice=4. xi=0+4=4, bonus=4 -> total 8
        # sim2: captain=5 (nonzero) -> bonus=5. xi=5+9=14, bonus=5 -> total 19
        np.testing.assert_array_equal(scores, np.array([6.0, 8.0, 19.0]))

    def test_no_vice_captain_means_no_fallback(self):
        player_points = {1: np.array([0.0]), 2: np.array([3.0])}
        manager = ManagerState(entry_id=1, name="M", current_total_points=0, current_league_position=1, starting_xi=(1, 2), captain_id=1, vice_captain_id=None)
        scores = simulate_manager_gameweek(manager, player_points, np.full(1, 1))
        # xi = 0+3=3, bonus = captain(1)=0 (no fallback) -> total 3
        np.testing.assert_array_equal(scores, np.array([3.0]))


if __name__ == "__main__":
    unittest.main()
