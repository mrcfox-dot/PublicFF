"""Unit tests for fpl_rival.intelligence.relevance.

Key requirement under test: relevant-rival selection is driven by points
gap and position, NOT a hardcoded count like "always exactly 5".
"""

import unittest

from fpl_rival.intelligence.config import EngineConfig
from fpl_rival.intelligence.models import LeagueData, ManagerData
from fpl_rival.intelligence.relevance import compute_relevant_rivals


def _mgr(entry_id, position, points):
    return ManagerData(entry_id=entry_id, league_position=position, total_points=points)


class TestRelevantRivalSelection(unittest.TestCase):
    def test_relevance_responds_to_points_gap_not_a_fixed_count(self):
        # Disable the position-distance auto-include so rank-2 neighbours only
        # qualify via their score - isolates the points-gap effect cleanly.
        config = EngineConfig(max_relevant_rivals=20, always_relevant_position_distance=1)

        def make_league(gap_at_distance_2):
            chris = _mgr(6, 6, 500)
            managers = [chris]
            # rank-1 neighbours: always relevant regardless (distance 1)
            managers.append(_mgr(5, 5, 495))
            managers.append(_mgr(7, 7, 505))
            # rank-2 managers: relevance depends entirely on the points gap given
            managers.append(_mgr(4, 4, 500 - gap_at_distance_2))
            managers.append(_mgr(8, 8, 500 + gap_at_distance_2))
            # distant outliers with a huge, fixed gap - present in both leagues
            # so max_abs_gap is comparable and never themselves relevant
            for i, pos in enumerate((1, 2, 3, 9, 10, 11, 12)):
                managers.append(_mgr(100 + i, pos, 500 - 1000 if pos < 6 else 500 + 1000))
            return chris, LeagueData(league_id=1, league_name="L", managers=tuple(managers))

        chris_close, close_league = make_league(gap_at_distance_2=10)
        close_result = compute_relevant_rivals(chris_close, close_league, config)

        chris_far, far_league = make_league(gap_at_distance_2=850)
        far_result = compute_relevant_rivals(chris_far, far_league, config)

        self.assertGreater(
            len(close_result["relevant_rivals"]), len(far_result["relevant_rivals"]),
            "a rank-2 manager close in points should be pulled in as relevant more often than a distant one",
        )
        # Critically: neither result is a hardcoded count like "always 5".
        self.assertNotEqual(len(close_result["relevant_rivals"]), 5)
        self.assertNotIn(4, {r["entry_id"] for r in far_result["relevant_rivals"]})
        self.assertIn(4, {r["entry_id"] for r in close_result["relevant_rivals"]})

    def test_immediate_neighbours_are_always_relevant_regardless_of_points_gap(self):
        config = EngineConfig()
        chris = _mgr(1, 5, 500)
        # Enormous points gap, but rank distance 1 - should still be "always_relevant".
        neighbour = _mgr(2, 6, -100000)
        far_manager = _mgr(3, 50, 499)  # close in points, but very far in rank
        league = LeagueData(league_id=1, league_name="L", managers=(chris, neighbour, far_manager))
        result = compute_relevant_rivals(chris, league, config)
        neighbour_entry = next(r for r in result["relevant_rivals"] if r["entry_id"] == 2)
        self.assertTrue(neighbour_entry["always_relevant"])

    def test_max_relevant_rivals_cap_is_respected(self):
        config = EngineConfig(max_relevant_rivals=3, always_relevant_position_distance=0, relevance_score_threshold=0.0)
        chris = _mgr(1, 1, 1000)
        rivals = [_mgr(i, i, 1000 - i) for i in range(2, 20)]
        league = LeagueData(league_id=1, league_name="L", managers=tuple([chris] + rivals))
        result = compute_relevant_rivals(chris, league, config)
        self.assertLessEqual(len(result["relevant_rivals"]), 3)

    def test_min_relevant_rivals_is_respected_even_with_high_threshold(self):
        config = EngineConfig(relevance_score_threshold=0.999, always_relevant_position_distance=0, min_relevant_rivals=2)
        chris = _mgr(1, 1, 1000)
        rivals = [_mgr(2, 2, 500), _mgr(3, 3, 100)]
        league = LeagueData(league_id=1, league_name="L", managers=(chris,) + tuple(rivals))
        result = compute_relevant_rivals(chris, league, config)
        self.assertGreaterEqual(len(result["relevant_rivals"]), 2)

    def test_no_scored_rivals_returns_note_not_crash(self):
        config = EngineConfig()
        chris = _mgr(1, 1, 1000)
        unscored = ManagerData(entry_id=2, league_position=None, total_points=None)
        league = LeagueData(league_id=1, league_name="L", managers=(chris, unscored))
        result = compute_relevant_rivals(chris, league, config)
        self.assertEqual(result["relevant_rivals"], [])
        self.assertIsNotNone(result["note"])


if __name__ == "__main__":
    unittest.main()
