"""Unit tests for fpl_rival.simulation.monte_carlo - reranking, aggregation, reproducibility."""

import unittest

import numpy as np

from fpl_rival.simulation.models import LeagueState, ManagerState, PlayerProjection, SimulationConfig
from fpl_rival.simulation.monte_carlo import compute_league_outcome, run_simulation


def _single_player_manager(entry_id, name, position, current_total, player_id):
    """A manager whose entire gameweek total is controlled directly: XI is
    just one player (who is therefore the captain, so scores 2x)."""
    return ManagerState(
        entry_id=entry_id, name=name, current_total_points=current_total, current_league_position=position,
        starting_xi=(player_id,), captain_id=player_id,
    )


class TestRerankingTieRule(unittest.TestCase):
    """4 hand-picked simulations with known totals, verifying the documented
    "1 + count strictly greater" competition-ranking tie rule."""

    def setUp(self):
        # sim:                 0     1     2     3
        chris_half = np.array([50.0, 50.0, 50.0, 50.0])  # chris total always 100
        rival_a_half = np.array([50.0, 55.0, 45.0, 55.0])  # 100, 110, 90, 110
        rival_b_half = np.array([45.0, 45.0, 40.0, 60.0])  # 90, 90, 80, 120
        self.chris = _single_player_manager(1, "Chris", 2, 0, 1)
        rival_a = _single_player_manager(2, "RivalA", 1, 0, 2)
        rival_b = _single_player_manager(3, "RivalB", 3, 0, 3)
        self.league = LeagueState(league_id=1, league_name="L", managers=(self.chris, rival_a, rival_b))
        self.player_points = {1: chris_half, 2: rival_a_half, 3: rival_b_half}
        self.captains = {1: np.full(4, 1), 2: np.full(4, 2), 3: np.full(4, 3)}

    def test_expected_ranks_per_simulation(self):
        result = compute_league_outcome(self.league, self.player_points, self.captains, chris_entry_id=1, num_simulations=4)
        # sim0: chris=100 ties RivalA=100, both beat RivalB=90 -> chris rank 1 (tie shares best position)
        # sim1: chris=100 < RivalA=110, > RivalB=90 -> chris rank 2
        # sim2: chris=100 > RivalA=90 > RivalB=80 -> chris rank 1
        # sim3: chris=100 < RivalA=110 < RivalB=120 -> chris rank 3
        # expected_position = mean([1,2,1,3]) = 1.75
        self.assertAlmostEqual(result.expected_position, 1.75)
        # prob_finish_first = fraction of sims with rank==1 -> sims 0,2 -> 2/4
        self.assertAlmostEqual(result.prob_finish_first, 0.5)

    def test_tied_leader_counts_as_finishing_first(self):
        # sim0 is an exact tie for the lead - our documented rule counts it as rank 1.
        single_sim_points = {k: v[:1] for k, v in self.player_points.items()}
        single_sim_captains = {k: v[:1] for k, v in self.captains.items()}
        result = compute_league_outcome(self.league, single_sim_points, single_sim_captains, chris_entry_id=1, num_simulations=1)
        self.assertEqual(result.prob_finish_first, 1.0)

    def test_leader_gap_is_zero_when_chris_is_the_leader(self):
        # sim2: chris=100 is the strict leader that simulation -> gap must be 0.
        idx = 2
        single_sim_points = {k: v[idx : idx + 1] for k, v in self.player_points.items()}
        single_sim_captains = {k: v[idx : idx + 1] for k, v in self.captains.items()}
        result = compute_league_outcome(self.league, single_sim_points, single_sim_captains, chris_entry_id=1, num_simulations=1)
        self.assertEqual(result.expected_leader_gap, 0.0)

    def test_rival_ahead_and_overtake_are_complementary_up_to_ties(self):
        result = compute_league_outcome(self.league, self.player_points, self.captains, chris_entry_id=1, num_simulations=4)
        # vs RivalB: chris beats RivalB in sims 0,1,2 (100>90,100>90,100>80) and loses sim3 (100<120) -> overtake=3/4, ahead=1/4
        self.assertAlmostEqual(result.rival_overtake_prob[3], 0.75)
        self.assertAlmostEqual(result.rival_ahead_prob[3], 0.25)


class TestProbabilityBounds(unittest.TestCase):
    def test_all_probabilities_between_0_and_1(self):
        rng_players = {
            1: PlayerProjection(1, "P1", 6.0, 3.0),
            2: PlayerProjection(2, "P2", 5.0, 3.0),
            3: PlayerProjection(3, "P3", 4.0, 3.0),
        }
        chris = ManagerState(entry_id=1, name="Chris", current_total_points=100, current_league_position=2, starting_xi=(1, 2, 3), captain_id=1)
        rival = ManagerState(entry_id=2, name="Rival", current_total_points=105, current_league_position=1, starting_xi=(1, 2, 3), captain_id=2)
        league = LeagueState(league_id=1, league_name="L", managers=(chris, rival))
        result = run_simulation(league, rng_players, chris_entry_id=1, chris_captain_id=1, config=SimulationConfig(num_simulations=2000, random_seed=1))
        for prob in (result.prob_finish_first, result.prob_top3, result.prob_move_up, result.prob_move_down):
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
        for prob in list(result.rival_overtake_prob.values()) + list(result.rival_ahead_prob.values()):
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)


class TestReproducibility(unittest.TestCase):
    def test_same_seed_gives_identical_result(self):
        projections = {
            1: PlayerProjection(1, "P1", 6.0, 3.0),
            2: PlayerProjection(2, "P2", 5.0, 3.0),
        }
        chris = ManagerState(entry_id=1, name="Chris", current_total_points=100, current_league_position=1, starting_xi=(1, 2), captain_id=1)
        rival = ManagerState(entry_id=2, name="Rival", current_total_points=95, current_league_position=2, starting_xi=(1, 2), captain_id=2)
        league = LeagueState(league_id=1, league_name="L", managers=(chris, rival))
        config = SimulationConfig(num_simulations=5000, random_seed=99)
        result1 = run_simulation(league, projections, chris_entry_id=1, chris_captain_id=1, config=config)
        result2 = run_simulation(league, projections, chris_entry_id=1, chris_captain_id=1, config=config)
        self.assertEqual(result1, result2)

    def test_probabilities_stable_as_simulation_count_increases(self):
        projections = {
            1: PlayerProjection(1, "P1", 6.0, 4.0),
            2: PlayerProjection(2, "P2", 5.5, 4.0),
        }
        chris = ManagerState(entry_id=1, name="Chris", current_total_points=100, current_league_position=1, starting_xi=(1, 2), captain_id=1)
        rival = ManagerState(entry_id=2, name="Rival", current_total_points=100, current_league_position=2, starting_xi=(1, 2), captain_id=2)
        league = LeagueState(league_id=1, league_name="L", managers=(chris, rival))

        small = run_simulation(league, projections, 1, 1, SimulationConfig(num_simulations=2000, random_seed=1))
        large = run_simulation(league, projections, 1, 1, SimulationConfig(num_simulations=50_000, random_seed=2))
        # Independent seeds, very different sample sizes - should still agree
        # within a generous tolerance if the estimator is behaving.
        self.assertAlmostEqual(small.prob_finish_first, large.prob_finish_first, delta=0.06)


if __name__ == "__main__":
    unittest.main()
