"""Unit tests for fpl_rival.intelligence.consensus."""

import unittest

from fpl_rival.intelligence.config import EngineConfig
from fpl_rival.intelligence.consensus import build_league_consensus
from fpl_rival.intelligence.models import GameContext, GameweekSquad, LeagueData, ManagerData, Pick


def _squad(elements, captain, event=1):
    picks = tuple(
        Pick(element=e, position=i + 1, multiplier=2 if e == captain else 1, is_captain=e == captain, is_vice_captain=False)
        for i, e in enumerate(elements)
    )
    return GameweekSquad(event=event, picks=picks)


CTX = GameContext(element_names={i: f"P{i}" for i in range(1, 20)})


class TestLeagueConsensus(unittest.TestCase):
    def setUp(self):
        # 3 managers, 3-player "squads" for simplicity.
        # Player 1: owned by all three (100%). Player 2: owned by two (66.7%).
        # Players 3, 4, 5, 6: owned by exactly one manager each (unique).
        a = ManagerData(entry_id=1, manager_name="A", squads_by_event={1: _squad([1, 2, 3], captain=1)})
        b = ManagerData(entry_id=2, manager_name="B", squads_by_event={1: _squad([1, 2, 4], captain=1)})
        c = ManagerData(entry_id=3, manager_name="C", squads_by_event={1: _squad([1, 5, 6], captain=5)})
        self.league = LeagueData(league_id=1, league_name="L", managers=(a, b, c))
        self.config = EngineConfig()

    def test_ownership_percentages(self):
        result = build_league_consensus(self.league, CTX, self.config)
        by_element = {row["element"]: row["mini_league_ownership_percentage"] for row in result["player_ownership"]}
        self.assertEqual(by_element[1], 100.0)
        self.assertAlmostEqual(by_element[2], 66.7, places=1)
        self.assertAlmostEqual(by_element[3], 33.3, places=1)

    def test_most_common_captain(self):
        result = build_league_consensus(self.league, CTX, self.config)
        self.assertEqual(result["most_common_captain"]["element"], 1)
        self.assertAlmostEqual(result["most_common_captain"]["captain_ownership_percentage"], 66.7, places=1)

    def test_unique_players(self):
        result = build_league_consensus(self.league, CTX, self.config)
        unique_elements = {row["element"] for row in result["unique_players"]}
        self.assertEqual(unique_elements, {3, 4, 5, 6})

    def test_average_pairwise_overlap_is_a_percentage(self):
        result = build_league_consensus(self.league, CTX, self.config)
        self.assertIsNotNone(result["average_pairwise_squad_overlap_pct"])
        self.assertTrue(0 <= result["average_pairwise_squad_overlap_pct"] <= 100)

    def test_no_squad_data_returns_empty_with_note(self):
        empty_league = LeagueData(league_id=1, league_name="L", managers=(ManagerData(entry_id=1),))
        result = build_league_consensus(empty_league, CTX, self.config)
        self.assertEqual(result["managers_with_squad_data"], 0)
        self.assertIsNotNone(result["note"])
        self.assertEqual(result["player_ownership"], [])


if __name__ == "__main__":
    unittest.main()
