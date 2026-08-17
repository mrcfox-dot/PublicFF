"""Stage 3 mandatory acceptance test (item 13): rival captain uncertainty.

Same underlying attacking situation, but the leader's captain probability
distribution shifts between two scenarios:

  Scenario A: leader captains Salah 90% / Palmer 10% of the time.
  Scenario B: leader captains Salah 50% / Palmer 50% of the time.

Required demonstration: Chris's mini-league-optimal captain choice can
flip depending on this assumption, when mathematically justified. No
numbers are hardcoded - see
fpl_rival/fixtures/simulation_fixtures.py:rival_uncertainty_scenario_a/b.
"""

import unittest

from fpl_rival.fixtures import simulation_fixtures as fx
from fpl_rival.simulation.captain_battle import run_captain_battle
from fpl_rival.simulation.models import SimulationConfig


def _run(builder, num_simulations=30_000, seed=42):
    chris, league, projections, candidates = builder()
    config = SimulationConfig(num_simulations=num_simulations, random_seed=seed, rival_captain_mode="probabilistic")
    result = run_captain_battle(chris, league, projections, candidates, config)
    return {c.player_name: c.result for c in result.candidates}, result


class TestRivalCaptainUncertaintyAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario_a, cls.result_a = _run(fx.rival_uncertainty_scenario_a)
        cls.scenario_b, cls.result_b = _run(fx.rival_uncertainty_scenario_b)

    def test_scenario_a_favours_the_differential_captain(self):
        # Leader is very likely to captain Salah, the player Chris also
        # owns -> differentiating with Palmer should win the mini-league objective.
        self.assertGreater(self.scenario_a["Palmer"].prob_finish_first, self.scenario_a["Salah"].prob_finish_first)
        self.assertEqual(self.result_a.objective_winner.player_name, "Palmer")

    def test_scenario_b_favours_matching_the_higher_mean_captain(self):
        # Leader's captain is now a toss-up -> Salah (higher mean, and only
        # cancels against the leader half the time now) should win instead.
        self.assertGreater(self.scenario_b["Salah"].prob_finish_first, self.scenario_b["Palmer"].prob_finish_first)
        self.assertEqual(self.result_b.objective_winner.player_name, "Salah")

    def test_optimal_captain_flips_between_scenarios(self):
        self.assertNotEqual(self.result_a.objective_winner.player_name, self.result_b.objective_winner.player_name)

    def test_only_the_rival_probability_assumption_changed(self):
        # Sanity check on the fixture itself: expected points for each
        # candidate should be (almost) unaffected by the rival's captain
        # assumption, since Chris's own squad/captain choice didn't change -
        # only the mini-league probabilities should move.
        self.assertAlmostEqual(
            self.scenario_a["Salah"].expected_gameweek_points, self.scenario_b["Salah"].expected_gameweek_points, delta=0.5
        )


if __name__ == "__main__":
    unittest.main()
