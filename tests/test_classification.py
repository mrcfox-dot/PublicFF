"""Unit tests for fpl_rival.intelligence.classification."""

import unittest

from fpl_rival.intelligence.classification import classify_risk_behaviour, classify_transfer_behaviour
from fpl_rival.intelligence.config import EngineConfig
from fpl_rival.intelligence.models import GameweekSquad, ManagerData, Pick


def _make_squad(elements, captain, event=1):
    picks = tuple(
        Pick(element=e, position=i + 1, multiplier=2 if e == captain else 1, is_captain=e == captain, is_vice_captain=False)
        for i, e in enumerate(elements)
    )
    return GameweekSquad(event=event, picks=picks)


class TestTransferBehaviour(unittest.TestCase):
    def test_insufficient_data_below_min_gameweeks(self):
        config = EngineConfig(min_gameweeks_for_classification=3)
        profile = {"gameweeks_played": 2, "average_transfers_per_gameweek": 5.0}
        result = classify_transfer_behaviour(profile, config)
        self.assertEqual(result["classification"], "Insufficient data")

    def test_very_active_band(self):
        config = EngineConfig()
        profile = {"gameweeks_played": 5, "average_transfers_per_gameweek": 3.5}
        result = classify_transfer_behaviour(profile, config)
        self.assertEqual(result["classification"], "Very active")

    def test_very_patient_band(self):
        config = EngineConfig()
        profile = {"gameweeks_played": 5, "average_transfers_per_gameweek": 0.1}
        result = classify_transfer_behaviour(profile, config)
        self.assertEqual(result["classification"], "Very patient")

    def test_metrics_are_exposed(self):
        config = EngineConfig()
        profile = {"gameweeks_played": 5, "average_transfers_per_gameweek": 1.5}
        result = classify_transfer_behaviour(profile, config)
        self.assertEqual(result["metrics"]["average_transfers_per_gameweek"], 1.5)
        self.assertEqual(result["metrics"]["gameweeks_played"], 5)


class TestRiskBehaviour(unittest.TestCase):
    def test_insufficient_data_below_min_gameweeks(self):
        config = EngineConfig(min_gameweeks_for_classification=3)
        manager = ManagerData(entry_id=1)
        profile = {"gameweeks_played": 1, "hits_taken": 0, "average_transfers_per_gameweek": 1.0}
        consensus = {"managers_with_squad_data": 0, "player_ownership": [], "manager_average_overlap_by_entry": {}}
        result = classify_risk_behaviour(manager, profile, consensus, config)
        self.assertEqual(result["classification"], "Insufficient data")

    def test_aggressive_manager(self):
        config = EngineConfig(min_gameweeks_for_classification=1)
        elements = list(range(1, 16))
        squad = _make_squad(elements, captain=elements[0])
        manager = ManagerData(entry_id=1, squads_by_event={1: squad})
        profile = {
            "gameweeks_played": 5,
            "hits_taken": 8,  # at/above saturation (6) -> hits component = 1.0
            "average_transfers_per_gameweek": 5.0,  # above saturation (3.0) -> component = 1.0
            "captain_concentration": {"unique_captains_used": 5, "total_captain_data_points": 5},  # max variety
        }
        # every owned player is 0% owned elsewhere in the league -> fully differential
        consensus = {
            "managers_with_squad_data": 2,
            "player_ownership": [{"element": e, "mini_league_ownership_percentage": 0.0} for e in elements],
            "manager_average_overlap_by_entry": {1: 0.0},  # 0% overlap with league -> max deviation
        }
        result = classify_risk_behaviour(manager, profile, consensus, config)
        self.assertEqual(result["classification"], "Aggressive")
        self.assertIsNotNone(result["risk_score"])
        self.assertGreaterEqual(result["risk_score"], config.risk_aggressive_threshold)
        # every component must be visible, not just the final label
        for name in ("hits", "differential_ownership", "captain_variety", "transfer_frequency", "squad_deviation"):
            self.assertIn(name, result["components"])

    def test_conservative_manager(self):
        config = EngineConfig(min_gameweeks_for_classification=1)
        elements = list(range(1, 16))
        squad = _make_squad(elements, captain=elements[0])
        manager = ManagerData(entry_id=1, squads_by_event={1: squad})
        profile = {
            "gameweeks_played": 5,
            "hits_taken": 0,
            "average_transfers_per_gameweek": 0.2,
            "captain_concentration": {"unique_captains_used": 1, "total_captain_data_points": 5},
        }
        # every owned player is 100% owned elsewhere -> pure template, zero differential
        consensus = {
            "managers_with_squad_data": 2,
            "player_ownership": [{"element": e, "mini_league_ownership_percentage": 100.0} for e in elements],
            "manager_average_overlap_by_entry": {1: 100.0},  # perfectly template -> zero deviation
        }
        result = classify_risk_behaviour(manager, profile, consensus, config)
        self.assertEqual(result["classification"], "Conservative")
        self.assertLessEqual(result["risk_score"], config.risk_conservative_threshold)


if __name__ == "__main__":
    unittest.main()
