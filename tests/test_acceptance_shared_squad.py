"""Stage 3 mandatory acceptance test (item 12): shared player exposure.

Verifies that shared player outcomes are modelled correctly: two managers
with identical squads must show ZERO relative variance (their totals move
in perfect lockstep, since every player is the same shared draw), and
increasing the number of differing XI slots must increase the relative
variance between them.
"""

import unittest

import numpy as np

from fpl_rival.fixtures import simulation_fixtures as fx
from fpl_rival.simulation.manager_outcomes import simulate_manager_gameweek
from fpl_rival.simulation.monte_carlo import collect_required_player_ids
from fpl_rival.simulation.player_outcomes import sample_all_players


def _gap_std(num_shared: int, num_simulations: int = 20_000, seed: int = 42) -> float:
    chris, rival, league, projections = fx.shared_squad_scenario(num_shared)
    rng = np.random.default_rng(seed)
    ids = collect_required_player_ids(league)
    points = sample_all_players(ids, projections, num_simulations, rng)
    chris_scores = simulate_manager_gameweek(chris, points, np.full(num_simulations, chris.captain_id))
    rival_scores = simulate_manager_gameweek(rival, points, np.full(num_simulations, rival.captain_id))
    return float(np.std(chris_scores - rival_scores))


class TestSharedPlayerExposure(unittest.TestCase):
    def test_identical_squads_produce_exactly_zero_variance(self):
        # 11/11 shared, same captain -> every simulation must cancel exactly.
        std = _gap_std(num_shared=11)
        self.assertEqual(std, 0.0)

    def test_low_differentiation_produces_low_relative_variance(self):
        low_std = _gap_std(num_shared=10)  # 1/11 differ
        high_std = _gap_std(num_shared=5)  # 6/11 differ
        self.assertGreater(high_std, low_std)

    def test_variance_increases_monotonically_with_differing_players(self):
        stds = [_gap_std(num_shared=n) for n in (11, 9, 7, 5, 3)]
        for earlier, later in zip(stds, stds[1:]):
            self.assertLessEqual(earlier, later)
        self.assertLess(stds[0], stds[-1])


if __name__ == "__main__":
    unittest.main()
