"""Unit tests for fpl_rival.intelligence.strategy - built with hand-crafted
data models directly, no fixtures module and no network required."""

import unittest

from fpl_rival.intelligence.config import EngineConfig
from fpl_rival.intelligence.models import LeagueData, ManagerData
from fpl_rival.intelligence.strategy import (
    ATTACK,
    BALANCED,
    DEFEND,
    DESPERATE,
    INSUFFICIENT_SEASON_DATA,
    classify_strategy_state,
)


def _mgr(entry_id, position, points):
    return ManagerData(entry_id=entry_id, league_position=position, total_points=points)


class TestInsufficientSeasonData(unittest.TestCase):
    def test_no_completed_gameweek_returns_insufficient(self):
        chris = _mgr(1, 1, 100)
        league = LeagueData(league_id=1, league_name="L", managers=(chris,), latest_completed_event=None)
        result = classify_strategy_state(chris, league, [], EngineConfig())
        self.assertEqual(result["state"], INSUFFICIENT_SEASON_DATA)

    def test_chris_unscored_returns_insufficient(self):
        chris = ManagerData(entry_id=1, league_position=None, total_points=None)
        league = LeagueData(league_id=1, league_name="L", managers=(chris,), latest_completed_event=5)
        result = classify_strategy_state(chris, league, [], EngineConfig())
        self.assertEqual(result["state"], INSUFFICIENT_SEASON_DATA)


class TestDefendBoundary(unittest.TestCase):
    def test_leader_with_thin_margin_is_balanced_not_defend(self):
        config = EngineConfig()
        chris = _mgr(1, 1, 1000)
        rival = _mgr(2, 2, 990)  # only 10 points ahead, below defend_min_margin_points (15)
        league = LeagueData(league_id=1, league_name="L", managers=(chris, rival), latest_completed_event=20)
        relevant = [{"entry_id": 2, "points_gap": -10, "position_distance": 1}]
        result = classify_strategy_state(chris, league, relevant, config)
        self.assertEqual(result["state"], BALANCED)

    def test_leader_with_exact_threshold_margin_is_defend(self):
        config = EngineConfig()
        chris = _mgr(1, 1, 1000)
        rival = _mgr(2, 2, 1000 - config.defend_min_margin_points)
        league = LeagueData(league_id=1, league_name="L", managers=(chris, rival), latest_completed_event=20)
        relevant = [{"entry_id": 2, "points_gap": -config.defend_min_margin_points, "position_distance": 1}]
        result = classify_strategy_state(chris, league, relevant, config)
        self.assertEqual(result["state"], DEFEND)


class TestAttackVsDesperateBoundary(unittest.TestCase):
    def test_small_deficit_is_balanced(self):
        config = EngineConfig()
        chris = _mgr(2, 2, 990)
        leader = _mgr(1, 1, 995)  # 5 points behind - below attack_min_deficit_points (10)
        league = LeagueData(league_id=1, league_name="L", managers=(leader, chris), latest_completed_event=20)
        result = classify_strategy_state(chris, league, [], config)
        self.assertEqual(result["state"], BALANCED)

    def test_large_deficit_with_time_left_is_attack(self):
        config = EngineConfig()
        chris = _mgr(2, 2, 950)
        leader = _mgr(1, 1, 980)  # 30 points behind, 18 gws remaining -> ~1.7 pts/gw, not desperate
        league = LeagueData(league_id=1, league_name="L", managers=(leader, chris), latest_completed_event=20)
        result = classify_strategy_state(chris, league, [], config)
        self.assertEqual(result["state"], ATTACK)

    def test_large_deficit_with_no_time_left_is_desperate(self):
        config = EngineConfig()
        chris = _mgr(2, 2, 900)
        leader = _mgr(1, 1, 980)  # 80 points behind, 1 gw remaining -> 80 pts/gw required
        league = LeagueData(league_id=1, league_name="L", managers=(leader, chris), latest_completed_event=37)
        result = classify_strategy_state(chris, league, [], config)
        self.assertEqual(result["state"], DESPERATE)


if __name__ == "__main__":
    unittest.main()
