"""Unit tests for pre-season handling (fpl_rival.intelligence.preseason + engine dispatch)."""

import unittest

from fpl_rival.intelligence import engine
from fpl_rival.intelligence.config import EngineConfig
from fpl_rival.intelligence.models import GameContext, LeagueData, ManagerData
from fpl_rival.intelligence.preseason import build_preseason_report


class TestPreseasonReport(unittest.TestCase):
    def setUp(self):
        self.chris = ManagerData(entry_id=1, manager_name="Chris", team_name="Team Chris")
        self.rival = ManagerData(entry_id=2, manager_name="Rival", team_name="Team Rival")
        self.league = LeagueData(league_id=99, league_name="L", managers=(self.chris, self.rival), latest_completed_event=None)

    def test_membership_is_reported(self):
        report = build_preseason_report(self.league, self.chris)
        self.assertEqual(report["number_of_competitors"], 1)
        self.assertEqual(report["total_managers_in_league"], 2)
        entry_ids = {m["entry_id"] for m in report["membership"]}
        self.assertEqual(entry_ids, {1, 2})

    def test_no_dummy_live_data_invented(self):
        report = build_preseason_report(self.league, self.chris)
        for m in report["membership"]:
            self.assertNotIn("total_points", m)
            self.assertNotIn("league_position", m)

    def test_lists_what_activates_and_whats_blocked(self):
        report = build_preseason_report(self.league, self.chris)
        self.assertTrue(report["activates_after_gw1"])
        self.assertTrue(report["currently_blocked"])
        for blocked in report["currently_blocked"]:
            self.assertIn("area", blocked)
            self.assertIn("reason", blocked)


class TestEngineDispatchesToPreseason(unittest.TestCase):
    def test_engine_returns_preseason_mode_when_no_gw_completed(self):
        chris = ManagerData(entry_id=1)
        league = LeagueData(league_id=1, league_name="L", managers=(chris,), latest_completed_event=None)
        report = engine.run(league, 1, GameContext(), EngineConfig())
        self.assertEqual(report["mode"], "preseason")
        self.assertIn("preseason", report)
        self.assertNotIn("strategy", report)

    def test_engine_raises_for_unknown_manager(self):
        league = LeagueData(league_id=1, league_name="L", managers=(ManagerData(entry_id=1),), latest_completed_event=None)
        with self.assertRaises(engine.UnknownManagerError):
            engine.run(league, 999, GameContext(), EngineConfig())


if __name__ == "__main__":
    unittest.main()
