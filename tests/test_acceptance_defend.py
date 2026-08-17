"""Stage 3 mandatory acceptance test (item 11): "defend" scenario.

Chris leads comfortably. The closest rival owns Salah (the highest-
projected player) and captains him with certainty. Chris also owns Palmer,
which the rival does not.

Required demonstration: matching the rival's captain (Salah) reduces
Chris's downside risk and maximises his probability of remaining 1st,
relative to captaining the differential option (Palmer). No numbers are
hardcoded - see fpl_rival/fixtures/simulation_fixtures.py:captain_defend.
"""

import unittest

from fpl_rival.fixtures import simulation_fixtures as fx
from fpl_rival.simulation.captain_battle import run_captain_battle
from fpl_rival.simulation.models import SimulationConfig


class TestDefendAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        chris, league, projections, candidates = fx.captain_defend()
        config = SimulationConfig(num_simulations=30_000, random_seed=42)
        cls.result = run_captain_battle(chris, league, projections, candidates, config)
        cls.by_name = {c.player_name: c.result for c in cls.result.candidates}

    def test_chris_starts_in_first(self):
        self.assertEqual(self.result.starting_position, 1)

    def test_matching_captain_reduces_downside_risk(self):
        # THE required finding: captaining the shared/matched player (Salah)
        # gives a materially lower probability of dropping out of 1st than
        # captaining the differential option (Palmer).
        self.assertLess(self.by_name["Salah"].prob_move_down, self.by_name["Palmer"].prob_move_down)

    def test_matching_captain_maximises_probability_of_remaining_first(self):
        self.assertGreater(self.by_name["Salah"].prob_finish_first, self.by_name["Palmer"].prob_finish_first)

    def test_effect_size_is_not_trivial(self):
        gap = self.by_name["Palmer"].prob_move_down - self.by_name["Salah"].prob_move_down
        self.assertGreater(gap, 0.03)


if __name__ == "__main__":
    unittest.main()
