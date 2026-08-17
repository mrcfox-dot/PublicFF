"""Unit tests for fpl_rival.prediction.history_store - uses a tempfile path,
never the repo's real data/ directory."""

import tempfile
import unittest
from pathlib import Path

from fpl_rival.prediction import history_store
from fpl_rival.prediction.models import CaptainObservation, LeagueHistory, PredictionConfig
from fpl_rival.prediction.prediction_engine import predict_captain

SALAH, PALMER = 1, 2
SQUAD = (SALAH, PALMER)


def _prediction():
    history = LeagueHistory({1: (CaptainObservation(gameweek=1, owned_player_ids=SQUAD, captain_id=SALAH),)})
    return predict_captain(1, "Dave", SQUAD, target_gameweek=2, league_history=history, config=PredictionConfig())


class TestHistoryStore(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self._tmpdir.name) / "2026-27.jsonl"

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_save_and_load_round_trips(self):
        prediction = _prediction()
        record = history_store.prediction_to_record(prediction, season="2026-27")
        history_store.save_prediction(record, self.path)

        loaded = history_store.load_predictions(self.path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["rival_entry_id"], 1)
        self.assertEqual(loaded[0]["gameweek"], 2)
        self.assertIsNone(loaded[0]["actual_captain_id"])

    def test_filters_work(self):
        prediction = _prediction()
        record = history_store.prediction_to_record(prediction, season="2026-27")
        history_store.save_prediction(record, self.path)
        history_store.save_prediction({**record, "rival_entry_id": 2}, self.path)

        self.assertEqual(len(history_store.load_predictions(self.path, rival_entry_id=1)), 1)
        self.assertEqual(len(history_store.load_predictions(self.path, rival_entry_id=999)), 0)

    def test_record_actual_outcome_updates_matching_record(self):
        prediction = _prediction()
        record = history_store.prediction_to_record(prediction, season="2026-27")
        history_store.save_prediction(record, self.path)

        updated_count = history_store.record_actual_outcome(self.path, "2026-27", gameweek=2, rival_entry_id=1, actual_captain_id=SALAH)
        self.assertEqual(updated_count, 1)

        loaded = history_store.load_predictions(self.path)
        self.assertEqual(loaded[0]["actual_captain_id"], SALAH)
        self.assertIsNotNone(loaded[0]["outcome_recorded_at"])

    def test_record_actual_outcome_no_match_returns_zero(self):
        updated_count = history_store.record_actual_outcome(self.path, "2026-27", gameweek=99, rival_entry_id=1, actual_captain_id=SALAH)
        self.assertEqual(updated_count, 0)

    def test_prediction_outcome_pairs_only_includes_resolved_records(self):
        prediction = _prediction()
        record = history_store.prediction_to_record(prediction, season="2026-27")
        history_store.save_prediction(record, self.path)  # unresolved
        history_store.record_actual_outcome(self.path, "2026-27", gameweek=2, rival_entry_id=1, actual_captain_id=SALAH)

        records = history_store.load_predictions(self.path)
        pairs = history_store.prediction_outcome_pairs(records)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["actual_captain_id"], SALAH)
        self.assertIsInstance(list(pairs[0]["candidate_probabilities"].keys())[0], int)


if __name__ == "__main__":
    unittest.main()
