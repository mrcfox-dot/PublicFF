"""Unit tests for fpl_rival.intelligence.rival_analysis."""

import unittest

from fpl_rival.intelligence.config import EngineConfig
from fpl_rival.intelligence.models import GameContext, GameweekSquad, ManagerData, Pick
from fpl_rival.intelligence.rival_analysis import compare_managers

CTX = GameContext(element_names={i: f"P{i}" for i in range(1, 30)})


def _squad(event, xi, bench, captain, vice):
    picks = []
    for i, e in enumerate(xi, start=1):
        picks.append(Pick(element=e, position=i, multiplier=2 if e == captain else 1, is_captain=e == captain, is_vice_captain=e == vice))
    for i, e in enumerate(bench, start=len(xi) + 1):
        picks.append(Pick(element=e, position=i, multiplier=0, is_captain=False, is_vice_captain=False))
    return GameweekSquad(event=event, picks=tuple(picks))


class TestCompareManagers(unittest.TestCase):
    def setUp(self):
        self.config = EngineConfig(exposure_weight_squad_overlap=0.7, exposure_weight_captain_overlap=0.3)

    def test_squad_overlap_15_and_xi(self):
        chris_xi = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        chris_bench = [12, 13, 14, 15]
        rival_xi = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]  # 10/11 shared in XI
        rival_bench = [16, 17, 18, 19]  # 0/4 shared on bench

        chris = ManagerData(entry_id=1, squads_by_event={1: _squad(1, chris_xi, chris_bench, captain=1, vice=2)})
        rival = ManagerData(entry_id=2, squads_by_event={1: _squad(1, rival_xi, rival_bench, captain=20, vice=2)})

        result = compare_managers(chris, rival, CTX, self.config)
        self.assertEqual(result["squad_overlap_15"]["shared_count"], 10)
        self.assertAlmostEqual(result["squad_overlap_15"]["shared_pct_of_chris_squad"], 66.7, places=1)
        self.assertEqual(result["squad_overlap_xi"]["shared_count"], 10)
        self.assertIn("P20", result["players_rival_only"])
        self.assertIn("P11", result["players_chris_only"])

    def test_captain_overlap_across_multiple_gameweeks(self):
        chris_squads = {
            1: _squad(1, list(range(1, 12)), [12, 13, 14, 15], captain=1, vice=2),
            2: _squad(2, list(range(1, 12)), [12, 13, 14, 15], captain=1, vice=2),
            3: _squad(3, list(range(1, 12)), [12, 13, 14, 15], captain=2, vice=1),
        }
        rival_squads = {
            1: _squad(1, list(range(1, 12)), [12, 13, 14, 15], captain=1, vice=2),  # matches
            2: _squad(2, list(range(1, 12)), [12, 13, 14, 15], captain=3, vice=2),  # differs
            3: _squad(3, list(range(1, 12)), [12, 13, 14, 15], captain=2, vice=1),  # matches
        }
        chris = ManagerData(entry_id=1, squads_by_event=chris_squads)
        rival = ManagerData(entry_id=2, squads_by_event=rival_squads)

        result = compare_managers(chris, rival, CTX, self.config)
        self.assertEqual(result["captain_exposure"]["comparable_gameweeks"], 3)
        self.assertEqual(result["captain_exposure"]["matching_gameweeks"], 2)
        self.assertAlmostEqual(result["captain_exposure"]["captain_overlap_pct"], 66.7, places=1)

    def test_effective_exposure_formula(self):
        # Identical squads and identical single-gameweek captain -> both
        # components are 1.0, so exposure should be exactly 100%.
        xi = list(range(1, 12))
        bench = [12, 13, 14, 15]
        chris = ManagerData(entry_id=1, squads_by_event={1: _squad(1, xi, bench, captain=1, vice=2)})
        rival = ManagerData(entry_id=2, squads_by_event={1: _squad(1, xi, bench, captain=1, vice=2)})
        result = compare_managers(chris, rival, CTX, self.config)
        self.assertEqual(result["effective_exposure_pct"], 100.0)

        # Completely disjoint squads and different captains -> 0%.
        rival_xi = [21, 22, 23, 24, 25, 26, 27, 28, 29, 20, 19]
        rival_bench = [16, 17, 18, 30]
        rival2 = ManagerData(entry_id=3, squads_by_event={1: _squad(1, rival_xi, rival_bench, captain=21, vice=22)})
        result2 = compare_managers(chris, rival2, CTX, self.config)
        self.assertEqual(result2["effective_exposure_pct"], 0.0)

    def test_missing_squad_data_reports_gap_not_crash(self):
        chris = ManagerData(entry_id=1)  # no squads_by_event at all
        rival = ManagerData(entry_id=2, squads_by_event={1: _squad(1, list(range(1, 12)), [12, 13, 14, 15], 1, 2)})
        result = compare_managers(chris, rival, CTX, self.config)
        self.assertIsNone(result["squad_overlap_15"])
        self.assertTrue(result["data_gaps"])


if __name__ == "__main__":
    unittest.main()
