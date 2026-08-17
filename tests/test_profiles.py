"""Unit tests for fpl_rival.intelligence.profiles."""

import unittest

from fpl_rival.intelligence.config import EngineConfig
from fpl_rival.intelligence.models import (
    ChipEvent,
    GameContext,
    GameweekHistoryRow,
    GameweekSquad,
    ManagerData,
    Pick,
)
from fpl_rival.intelligence.profiles import build_manager_profile

CTX = GameContext(
    element_names={i: f"P{i}" for i in range(1, 30)},
    element_types={1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 1, 13: 2, 14: 3, 15: 4},
    chip_windows={"wildcard": [(2, 19), (20, 38)], "bboost": [(1, 19), (20, 38)]},
)


def _squad(event, captain, active_chip=None, points_on_bench=None):
    xi = list(range(1, 12))
    bench = [12, 13, 14, 15]
    picks = [
        Pick(element=e, position=i + 1, multiplier=2 if e == captain else 1, is_captain=e == captain, is_vice_captain=False, element_type=CTX.element_types.get(e))
        for i, e in enumerate(xi)
    ]
    picks += [
        Pick(element=e, position=i + 12, multiplier=0, is_captain=False, is_vice_captain=False, element_type=CTX.element_types.get(e))
        for i, e in enumerate(bench)
    ]
    return GameweekSquad(event=event, picks=tuple(picks), active_chip=active_chip, points_on_bench=points_on_bench)


class TestManagerProfile(unittest.TestCase):
    def test_chips_remaining_accounts_for_used_windows(self):
        manager = ManagerData(
            entry_id=1,
            chips_used=(ChipEvent(name="wildcard", event=5),),  # used in first-half window (2-19)
            gameweek_history=(GameweekHistoryRow(event=1),),
        )
        profile = build_manager_profile(manager, CTX, EngineConfig())
        self.assertEqual(profile["chips_remaining"]["wildcard"], 1)  # second-half window untouched
        self.assertEqual(profile["chips_remaining"]["bboost"], 2)  # never used

    def test_hits_and_average_transfers(self):
        history = tuple(
            GameweekHistoryRow(event=e, event_transfers=2, event_transfers_cost=4 if e == 3 else 0)
            for e in range(1, 6)
        )
        manager = ManagerData(entry_id=1, gameweek_history=history)
        profile = build_manager_profile(manager, CTX, EngineConfig())
        self.assertEqual(profile["total_transfers"], 10)
        self.assertEqual(profile["average_transfers_per_gameweek"], 2.0)
        self.assertEqual(profile["hits_taken"], 1)
        self.assertEqual(profile["points_lost_to_hits"], 4)

    def test_captain_history_and_concentration(self):
        squads = {1: _squad(1, captain=1), 2: _squad(2, captain=1), 3: _squad(3, captain=6)}
        manager = ManagerData(entry_id=1, squads_by_event=squads)
        profile = build_manager_profile(manager, CTX, EngineConfig())
        self.assertEqual(len(profile["captain_history"]), 3)
        self.assertEqual(profile["captain_concentration"]["unique_captains_used"], 2)
        self.assertAlmostEqual(profile["captain_concentration"]["concentration_pct"], 66.7, places=1)

    def test_formation_history(self):
        squads = {1: _squad(1, captain=1)}
        manager = ManagerData(entry_id=1, squads_by_event=squads)
        profile = build_manager_profile(manager, CTX, EngineConfig())
        # XI element types: 1 GK(1), DEF(2,3,4,5)=4, MID(6,7,8,9)=4, FWD(10,11)=2
        self.assertEqual(profile["formation_history"], [{"event": 1, "formation": "4-4-2"}])

    def test_bench_points_sourced_from_history_not_picks(self):
        history = (GameweekHistoryRow(event=1, points_on_bench=7), GameweekHistoryRow(event=2, points_on_bench=3))
        manager = ManagerData(entry_id=1, gameweek_history=history)
        profile = build_manager_profile(manager, CTX, EngineConfig())
        self.assertEqual(profile["bench_points"]["total_season"], 10)
        self.assertEqual(profile["bench_points"]["by_event"], {1: 7, 2: 3})

    def test_never_invents_missing_data(self):
        manager = ManagerData(entry_id=1)  # nothing retrieved at all
        profile = build_manager_profile(manager, CTX, EngineConfig())
        self.assertIsNone(profile["captain"])
        self.assertEqual(profile["current_squad"], [])
        self.assertIsNone(profile["total_transfers"])
        self.assertTrue(profile["data_gaps"])


if __name__ == "__main__":
    unittest.main()
