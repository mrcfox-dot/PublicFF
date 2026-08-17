"""Unit tests for fpl_rival.intelligence.threats (shields/threats/weapons)."""

import unittest

from fpl_rival.intelligence.config import EngineConfig
from fpl_rival.intelligence.models import GameContext, GameweekSquad, LeagueData, ManagerData, Pick
from fpl_rival.intelligence.threats import build_threat_analysis


def _squad(elements, captain=None, event=1):
    picks = tuple(
        Pick(element=e, position=i + 1, multiplier=2 if e == captain else 1, is_captain=e == captain, is_vice_captain=False)
        for i, e in enumerate(elements)
    )
    return GameweekSquad(event=event, picks=picks)


CTX = GameContext(element_names={i: f"P{i}" for i in range(1, 30)})


class TestThreatAnalysis(unittest.TestCase):
    def setUp(self):
        self.config = EngineConfig(high_ownership_threshold_pct=50.0, low_ownership_threshold_pct=20.0)
        # Chris owns {1, 2, 3}. Two relevant rivals both own {2, 10, 11} -
        # so 10/11 are threats (100% among relevant rivals, Chris lacks them),
        # 2 is a shield (Chris owns it, and it's heavily owned around him),
        # 1 and 3 are weapons (Chris owns them, 0% ownership among rivals).
        self.chris = ManagerData(entry_id=1, league_position=1, total_points=100, squads_by_event={1: _squad([1, 2, 3])})
        self.rival_a = ManagerData(entry_id=2, league_position=2, total_points=90, squads_by_event={1: _squad([2, 10, 11], captain=10)})
        self.rival_b = ManagerData(entry_id=3, league_position=3, total_points=80, squads_by_event={1: _squad([2, 10, 11], captain=10)})
        self.league = LeagueData(league_id=1, league_name="L", managers=(self.chris, self.rival_a, self.rival_b))
        self.relevant_rivals = [
            {"entry_id": 2, "points_gap": -10, "position_distance": 1},
            {"entry_id": 3, "points_gap": -20, "position_distance": 2},
        ]

    def test_threats_detected(self):
        result = build_threat_analysis(self.chris, self.league, self.relevant_rivals, CTX, self.config)
        threat_elements = {t["element"] for t in result["threats"]}
        self.assertEqual(threat_elements, {10, 11})
        for t in result["threats"]:
            self.assertEqual(t["ownership_pct_among_relevant_rivals"], 100.0)

    def test_weapons_detected(self):
        result = build_threat_analysis(self.chris, self.league, self.relevant_rivals, CTX, self.config)
        weapon_elements = {w["element"] for w in result["weapons"]}
        self.assertEqual(weapon_elements, {1, 3})

    def test_shields_detected(self):
        result = build_threat_analysis(self.chris, self.league, self.relevant_rivals, CTX, self.config)
        shield_elements = {s["element"] for s in result["shields"]}
        self.assertEqual(shield_elements, {2})

    def test_captain_ownership_also_counts_as_threat(self):
        # Both relevant rivals captain player 10 -> should still appear as a
        # threat via captain_pct even if ownership pct alone were lower.
        result = build_threat_analysis(self.chris, self.league, self.relevant_rivals, CTX, self.config)
        ten = next(t for t in result["threats"] if t["element"] == 10)
        self.assertEqual(ten["captain_pct_among_relevant_rivals"], 100.0)

    def test_no_relevant_rivals_returns_note(self):
        result = build_threat_analysis(self.chris, self.league, [], CTX, self.config)
        self.assertEqual(result["threats"], [])
        self.assertIsNotNone(result["note"])

    def test_chris_without_squad_returns_note(self):
        chris_no_squad = ManagerData(entry_id=1)
        result = build_threat_analysis(chris_no_squad, self.league, self.relevant_rivals, CTX, self.config)
        self.assertIsNotNone(result["note"])


if __name__ == "__main__":
    unittest.main()
