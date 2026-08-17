"""Stage 2 requirement 9: scenarios A-E, run purely against synthetic fixtures.

No network access happens anywhere in this file - see fpl_rival/fixtures.
"""

import unittest

from fpl_rival.fixtures import synthetic
from fpl_rival.intelligence import engine
from fpl_rival.intelligence.config import EngineConfig
from fpl_rival.intelligence.strategy import ATTACK, DEFEND, DESPERATE


class ScenarioA_Defend(unittest.TestCase):
    """Chris leads league by 70 points late in the season -> DEFEND."""

    def test_strategy_is_defend(self):
        league, ctx, chris_id = synthetic.scenario_a_defend()
        report = engine.run(league, chris_id, ctx, EngineConfig())
        self.assertEqual(report["mode"], "active")
        self.assertEqual(report["strategy"]["state"], DEFEND)
        self.assertEqual(report["strategy"]["metrics"]["points_ahead_of_nearest_below"], 70)

    def test_is_synthetic_flag_set(self):
        league, ctx, chris_id = synthetic.scenario_a_defend()
        self.assertTrue(league.is_synthetic)
        for m in league.managers:
            self.assertTrue(m.is_synthetic)
            self.assertGreaterEqual(m.entry_id, 900000)


class ScenarioB_Attack(unittest.TestCase):
    """Chris trails leader by 30 points with 6 gameweeks remaining -> ATTACK."""

    def test_strategy_is_attack(self):
        league, ctx, chris_id = synthetic.scenario_b_attack()
        report = engine.run(league, chris_id, ctx, EngineConfig())
        self.assertEqual(report["strategy"]["state"], ATTACK)
        self.assertEqual(report["strategy"]["metrics"]["points_behind_leader"], 30)
        self.assertEqual(report["strategy"]["metrics"]["gameweeks_remaining"], 6)


class ScenarioC_Desperate(unittest.TestCase):
    """Chris trails leader by 80 points with 3 gameweeks remaining -> DESPERATE."""

    def test_strategy_is_desperate(self):
        league, ctx, chris_id = synthetic.scenario_c_desperate()
        report = engine.run(league, chris_id, ctx, EngineConfig())
        self.assertEqual(report["strategy"]["state"], DESPERATE)
        self.assertEqual(report["strategy"]["metrics"]["points_behind_leader"], 80)
        self.assertEqual(report["strategy"]["metrics"]["gameweeks_remaining"], 3)


class ScenarioD_IdenticalSquads(unittest.TestCase):
    """Two managers with (near-)identical squads -> overlap/exposure should reflect it."""

    def test_overlap_and_exposure_are_high_for_twin(self):
        league, ctx, chris_id = synthetic.scenario_d_identical_squads()
        report = engine.run(league, chris_id, ctx, EngineConfig())

        twin_id = 900032
        different_id = 900033
        twin_comparison = report["comparisons"][twin_id]
        different_comparison = report["comparisons"][different_id]

        self.assertEqual(twin_comparison["squad_overlap_15"]["shared_pct_of_chris_squad"], 100.0)
        self.assertEqual(twin_comparison["captain_exposure"]["captain_overlap_pct"], 100.0)
        self.assertEqual(twin_comparison["effective_exposure_pct"], 100.0)
        self.assertEqual(twin_comparison["players_chris_only"], [])
        self.assertEqual(twin_comparison["players_rival_only"], [])

        # the contrasting manager should show meaningfully less overlap
        self.assertLess(
            different_comparison["squad_overlap_15"]["shared_pct_of_chris_squad"],
            twin_comparison["squad_overlap_15"]["shared_pct_of_chris_squad"],
        )


class ScenarioE_Threats(unittest.TestCase):
    """A rival owns several high-mini-league-ownership players Chris lacks."""

    def test_threats_are_detected(self):
        league, ctx, chris_id = synthetic.scenario_e_threats()
        report = engine.run(league, chris_id, ctx, EngineConfig())

        threats = report["threats"]["threats"]
        self.assertTrue(threats, "expected at least one detected threat player")

        threat_elements = {t["element"] for t in threats}
        # The three templated rivals all share this exact starting XI/bench -
        # every one of those elements should be flagged, since Chris owns none.
        template_elements = {p.element for squad in [league.manager(900041).latest_squad] for p in squad.picks}
        self.assertTrue(template_elements.issubset(threat_elements))

        for t in threats:
            self.assertGreaterEqual(t["ownership_pct_among_relevant_rivals"], 50.0)

        # weapons should be Chris's own, low-owned-among-rivals players
        weapons = report["threats"]["weapons"]
        self.assertTrue(weapons)
        chris_elements = {p.element for p in league.manager(chris_id).latest_squad.picks}
        for w in weapons:
            self.assertIn(w["element"], chris_elements)
            self.assertLessEqual(w["ownership_pct_among_relevant_rivals"], 20.0)


if __name__ == "__main__":
    unittest.main()
