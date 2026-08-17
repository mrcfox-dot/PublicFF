"""Unit tests for fpl_rival.prediction.features."""

import unittest

from fpl_rival.prediction import features
from fpl_rival.prediction.models import CaptainObservation

SALAH, PALMER, HAALAND = 1, 2, 3
SQUAD = (SALAH, PALMER, HAALAND)


def _hist(captains):
    """captains: list of (gw, captain_id), all owning the full SQUAD."""
    return tuple(CaptainObservation(gameweek=gw, owned_player_ids=SQUAD, captain_id=cap) for gw, cap in captains)


class TestPersonalLoyaltyRate(unittest.TestCase):
    def test_rate_computed_correctly(self):
        hist = _hist([(1, SALAH), (2, SALAH), (3, PALMER), (4, SALAH)])
        owned, captained, rate = features.personal_loyalty_rate(hist, SALAH)
        self.assertEqual(owned, 4)
        self.assertEqual(captained, 3)
        self.assertAlmostEqual(rate, 0.75)

    def test_never_owned_returns_zero_not_error(self):
        owned, captained, rate = features.personal_loyalty_rate((), 999)
        self.assertEqual((owned, captained, rate), (0, 0, 0.0))


class TestRecencyWeightedRate(unittest.TestCase):
    def test_recent_captaincy_weighted_more_than_old(self):
        # Salah captained only in the OLDEST gw; Palmer captained only in the MOST RECENT gw.
        hist = _hist([(1, SALAH), (2, PALMER), (3, PALMER), (4, PALMER)])
        recent_salah = features.recency_weighted_rate(hist, SALAH, target_gameweek=5, decay_rate=0.5)
        recent_palmer = features.recency_weighted_rate(hist, PALMER, target_gameweek=5, decay_rate=0.5)
        self.assertGreater(recent_palmer, recent_salah)

    def test_decay_rate_is_configurable(self):
        hist = _hist([(1, PALMER), (2, PALMER), (3, PALMER), (4, SALAH)])
        # a slower decay (closer to 1.0) should weight the single old Salah
        # gameweek relatively less than a faster decay does, relative to
        # the mostly-recent Palmer choices - so Salah's rate should be
        # lower under slow decay than under an even slower/faster comparison
        # is not guaranteed in general, but the function must at least
        # respond to the parameter (different decay -> different result).
        rate_a = features.recency_weighted_rate(hist, SALAH, target_gameweek=5, decay_rate=0.9)
        rate_b = features.recency_weighted_rate(hist, SALAH, target_gameweek=5, decay_rate=0.2)
        self.assertNotEqual(rate_a, rate_b)


class TestConcentration(unittest.TestCase):
    def test_concentration_share(self):
        hist = _hist([(1, SALAH), (2, SALAH), (3, PALMER), (4, HAALAND)])
        total, share = features.concentration_share(hist, SALAH)
        self.assertEqual(total, 4)
        self.assertAlmostEqual(share, 0.5)

    def test_concentration_index_is_high_for_single_player(self):
        hist = _hist([(gw, SALAH) for gw in range(1, 11)])
        self.assertEqual(features.concentration_index(hist), 1.0)

    def test_concentration_index_is_lower_when_spread_out(self):
        hist = _hist([(1, SALAH), (2, PALMER), (3, HAALAND), (4, SALAH), (5, PALMER), (6, HAALAND)])
        idx = features.concentration_index(hist)
        self.assertLess(idx, 1.0)
        self.assertGreater(idx, 0.0)

    def test_empty_history_index_is_zero(self):
        self.assertEqual(features.concentration_index(()), 0.0)


class TestLeagueConsensusRate(unittest.TestCase):
    def test_pooled_across_other_managers(self):
        others = {
            10: _hist([(1, SALAH), (2, SALAH)]),
            11: _hist([(1, PALMER)]),
        }
        rate = features.league_consensus_rate(others, SALAH)
        self.assertAlmostEqual(rate, 2 / 3)

    def test_no_other_managers_returns_zero(self):
        self.assertEqual(features.league_consensus_rate({}, SALAH), 0.0)


class TestRecentWindowStats(unittest.TestCase):
    def test_window_smaller_than_history(self):
        hist = _hist([(1, PALMER), (2, SALAH), (3, SALAH), (4, SALAH), (5, PALMER)])
        captained, window = features.recent_window_stats(hist, SALAH, window_size=3)
        self.assertEqual(window, 3)  # gws 3,4,5
        self.assertEqual(captained, 2)  # gw3=SALAH, gw4=SALAH, gw5=PALMER -> 2 of 3

    def test_window_larger_than_history_uses_all(self):
        hist = _hist([(1, SALAH)])
        captained, window = features.recent_window_stats(hist, SALAH, window_size=10)
        self.assertEqual(window, 1)
        self.assertEqual(captained, 1)


class TestHistoricalResponseToExpectedPoints(unittest.TestCase):
    def test_none_when_no_ep_data_supplied(self):
        hist = _hist([(1, SALAH)])
        result = features.historical_response_to_expected_points(hist, historical_ep_by_gw=None)
        self.assertIsNone(result["follows_highest_projected_rate"])

    def test_follows_highest_rate_computed_when_ep_supplied(self):
        hist = _hist([(1, SALAH), (2, PALMER)])
        ep_by_gw = {
            1: {SALAH: 9.0, PALMER: 5.0, HAALAND: 4.0},  # captained the highest (SALAH) -> follows
            2: {SALAH: 9.0, PALMER: 5.0, HAALAND: 4.0},  # captained PALMER, not highest -> does not follow
        }
        result = features.historical_response_to_expected_points(hist, ep_by_gw)
        self.assertAlmostEqual(result["follows_highest_projected_rate"], 0.5)
        self.assertAlmostEqual(result["differential_rate"], 0.5)
        self.assertEqual(result["gameweeks_with_ep_data"], 2)


if __name__ == "__main__":
    unittest.main()
