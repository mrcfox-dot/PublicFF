"""Stage 3 mandatory acceptance test (item 10): "attack" scenario.

Chris trails the leader. Chris and the leader both own the highest-
projected captain (Salah). The leader captains Salah with certainty.
Chris also owns Palmer, a slightly-lower-projected player the leader does
NOT own.

Required demonstration: "highest projected points captain" (Salah) is NOT
the captain that maximises Chris's mini-league win probability. No numbers
are hardcoded here - every assertion reads a value the simulation actually
produced (see fpl_rival/fixtures/simulation_fixtures.py:captain_attack).
"""

import unittest

from fpl_rival.fixtures import simulation_fixtures as fx
from fpl_rival.simulation.captain_battle import run_captain_battle
from fpl_rival.simulation.models import SimulationConfig


class TestAttackAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        chris, league, projections, candidates = fx.captain_attack()
        config = SimulationConfig(num_simulations=30_000, random_seed=42)
        cls.result = run_captain_battle(chris, league, projections, candidates, config)
        cls.by_name = {c.player_name: c.result for c in cls.result.candidates}

    def test_salah_has_higher_expected_points_than_palmer(self):
        # This is the setup precondition, not the finding - Salah really is
        # the higher-projected captain in this fixture.
        self.assertGreater(self.by_name["Salah"].expected_gameweek_points, self.by_name["Palmer"].expected_gameweek_points)

    def test_palmer_has_higher_probability_of_finishing_first(self):
        # THE required finding: the differential captain (lower mean, but
        # not shared with the leader) beats the higher-mean shared captain
        # on the mini-league objective.
        self.assertGreater(self.by_name["Palmer"].prob_finish_first, self.by_name["Salah"].prob_finish_first)

    def test_expected_points_winner_and_objective_winner_diverge(self):
        self.assertEqual(self.result.expected_points_winner.player_name, "Salah")
        self.assertEqual(self.result.objective_winner.player_name, "Palmer")
        self.assertNotEqual(self.result.expected_points_winner.player_id, self.result.objective_winner.player_id)

    def test_effect_size_is_not_trivial(self):
        # Guards against a degenerate pass (e.g. both ~equal due to a bug) -
        # the gap should be clearly resolvable, not a coin-flip-sized wobble.
        gap = self.by_name["Palmer"].prob_finish_first - self.by_name["Salah"].prob_finish_first
        self.assertGreater(gap, 0.03)  # at least 3 percentage points


if __name__ == "__main__":
    unittest.main()
