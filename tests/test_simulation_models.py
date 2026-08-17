"""Unit tests for fpl_rival.simulation.models - structural validation."""

import unittest

from fpl_rival.simulation.exceptions import InvalidManagerStateError
from fpl_rival.simulation.models import LeagueState, ManagerState, PlayerProjection, SimulationConfig


def _manager(entry_id=1, xi=(1, 2, 3), captain=1, position=1, points=100):
    return ManagerState(
        entry_id=entry_id,
        name=f"M{entry_id}",
        current_total_points=points,
        current_league_position=position,
        starting_xi=xi,
        captain_id=captain,
    )


class TestManagerState(unittest.TestCase):
    def test_empty_xi_rejected(self):
        with self.assertRaises(InvalidManagerStateError):
            ManagerState(entry_id=1, name="M", current_total_points=0, current_league_position=1, starting_xi=(), captain_id=1)

    def test_duplicate_xi_players_rejected(self):
        with self.assertRaises(InvalidManagerStateError):
            _manager(xi=(1, 1, 2), captain=1)

    def test_captain_must_be_in_xi(self):
        with self.assertRaises(InvalidManagerStateError):
            _manager(xi=(1, 2, 3), captain=99)

    def test_with_captain_returns_new_instance(self):
        m = _manager(xi=(1, 2, 3), captain=1)
        m2 = m.with_captain(2)
        self.assertEqual(m.captain_id, 1)
        self.assertEqual(m2.captain_id, 2)
        self.assertIsNot(m, m2)


class TestLeagueState(unittest.TestCase):
    def test_duplicate_entry_ids_rejected(self):
        with self.assertRaises(InvalidManagerStateError):
            LeagueState(league_id=1, league_name="L", managers=(_manager(1), _manager(1)))

    def test_too_few_managers_rejected(self):
        with self.assertRaises(InvalidManagerStateError):
            LeagueState(league_id=1, league_name="L", managers=(_manager(1),))

    def test_too_many_managers_rejected(self):
        managers = tuple(_manager(i) for i in range(51))
        with self.assertRaises(InvalidManagerStateError):
            LeagueState(league_id=1, league_name="L", managers=managers)

    def test_supports_two_to_fifty_managers(self):
        league2 = LeagueState(league_id=1, league_name="L", managers=(_manager(1), _manager(2)))
        self.assertEqual(len(league2.managers), 2)
        league50 = LeagueState(league_id=1, league_name="L", managers=tuple(_manager(i) for i in range(50)))
        self.assertEqual(len(league50.managers), 50)

    def test_manager_lookup(self):
        league = LeagueState(league_id=1, league_name="L", managers=(_manager(1), _manager(2)))
        self.assertEqual(league.manager(1).entry_id, 1)
        self.assertIsNone(league.manager(999))


class TestSimulationConfig(unittest.TestCase):
    def test_default_is_50000_simulations(self):
        self.assertEqual(SimulationConfig().num_simulations, 50_000)

    def test_invalid_rival_captain_mode_rejected(self):
        with self.assertRaises(ValueError):
            SimulationConfig(rival_captain_mode="sometimes")

    def test_zero_simulations_rejected(self):
        with self.assertRaises(ValueError):
            SimulationConfig(num_simulations=0)


class TestPlayerProjection(unittest.TestCase):
    def test_negative_std_rejected(self):
        with self.assertRaises(Exception):
            PlayerProjection(1, "P", 5.0, -1.0)


if __name__ == "__main__":
    unittest.main()
