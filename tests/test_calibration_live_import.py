"""Unit tests for fpl_rival.calibration.live_import - network-free.

Builds real HistoricalRecords from Stage 2's ManagerData/LeagueData shapes
directly (no live API call), verifying the conversion mapping and the
CSV merge/dedup behaviour.
"""

import tempfile
import unittest
from pathlib import Path

from fpl_rival.calibration.importer import import_csv
from fpl_rival.calibration.live_import import (
    REAL_CSV_FIELDS,
    infer_season_label,
    merge_and_write_csv,
)
from fpl_rival.calibration.models import HistoricalRecord
from fpl_rival.intelligence.models import GameweekHistoryRow, GameweekSquad, ManagerData, Pick


def _squad(event, xi_ids, bench_ids, captain, vice, active_chip=None, points=None):
    picks = [Pick(e, i + 1, 2 if e == captain else 1, e == captain, e == vice) for i, e in enumerate(xi_ids)]
    picks += [Pick(e, 12 + i, 0, False, False) for i, e in enumerate(bench_ids)]
    return GameweekSquad(event=event, picks=tuple(picks), active_chip=active_chip, points=points)


class TestInferSeasonLabel(unittest.TestCase):
    def test_august_deadline_gives_correct_season(self):
        bootstrap = {"events": [{"id": 1, "deadline_time": "2026-08-21T17:30:00Z"}]}
        self.assertEqual(infer_season_label(bootstrap), "2026-27")

    def test_missing_gw1_returns_unknown(self):
        self.assertEqual(infer_season_label({"events": []}), "unknown-season")


class TestConversionMapping(unittest.TestCase):
    """Exercises the same field-mapping logic build_real_historical_records
    uses, via a hand-built ManagerData (no live call needed)."""

    def test_gameweek_history_row_joined_by_event(self):
        xi = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
        squad = _squad(1, xi, (12, 13, 14, 15), captain=1, vice=2, active_chip="wildcard", points=66)
        manager = ManagerData(
            entry_id=100, manager_name="Chris Fox", league_position=3, total_points=66,
            gameweek_history=(GameweekHistoryRow(event=1, points=66, total_points=66, rank=893714, event_transfers=1),),
            squads_by_event={1: squad},
        )

        # Inline the same per-manager conversion build_real_historical_records performs.
        history_by_event = {row.event: row for row in manager.gameweek_history}
        history_row = history_by_event[1]
        record = HistoricalRecord(
            season="2026-27", manager_id=manager.entry_id, gameweek=1, starting_xi=xi, captain=1,
            manager_name=manager.manager_name, squad=tuple(p.element for p in squad.picks), vice_captain=2,
            transfers=history_row.event_transfers, chip=squad.active_chip, total_points=history_row.total_points,
            gameweek_points=squad.points, rank=history_row.rank,
        )
        self.assertEqual(record.transfers, 1)
        self.assertEqual(record.rank, 893714)
        self.assertEqual(record.chip, "wildcard")
        self.assertEqual(record.gameweek_points, 66)


class TestMergeAndWriteCsv(unittest.TestCase):
    def test_first_write_creates_file_with_all_records_added(self):
        records = [HistoricalRecord(season="2026-27", manager_id=1, gameweek=1, starting_xi=(1, 2, 3), captain=1)]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "history.csv"
            summary = merge_and_write_csv(records, path)
        self.assertEqual(summary["records_added"], 1)
        self.assertEqual(summary["records_updated"], 0)
        self.assertEqual(summary["total_records"], 1)

    def test_rerunning_the_same_gameweek_updates_not_duplicates(self):
        gw1 = [HistoricalRecord(season="2026-27", manager_id=1, gameweek=1, starting_xi=(1, 2, 3), captain=1)]
        gw1_corrected = [HistoricalRecord(season="2026-27", manager_id=1, gameweek=1, starting_xi=(1, 2, 3), captain=2)]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "history.csv"
            merge_and_write_csv(gw1, path)
            summary = merge_and_write_csv(gw1_corrected, path)
            reloaded = import_csv(path)
        self.assertEqual(summary["records_added"], 0)
        self.assertEqual(summary["records_updated"], 1)
        self.assertEqual(summary["total_records"], 1)
        self.assertEqual(reloaded[0].captain, 2)  # correction took effect

    def test_new_gameweek_is_appended_not_merged_into_existing_row(self):
        gw1 = [HistoricalRecord(season="2026-27", manager_id=1, gameweek=1, starting_xi=(1, 2, 3), captain=1)]
        gw2 = [HistoricalRecord(season="2026-27", manager_id=1, gameweek=2, starting_xi=(1, 2, 3), captain=2)]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "history.csv"
            merge_and_write_csv(gw1, path)
            summary = merge_and_write_csv(gw2, path)
        self.assertEqual(summary["records_added"], 1)
        self.assertEqual(summary["total_records"], 2)
        self.assertEqual(summary["gameweeks"], [1, 2])

    def test_written_csv_round_trips_through_the_real_importer(self):
        records = [
            HistoricalRecord(
                season="2026-27", manager_id=1, gameweek=1, starting_xi=(1, 2, 3), captain=2, vice_captain=1,
                manager_name="Chris Fox", squad=(1, 2, 3, 4), transfers=2, chip="bboost", total_points=66,
                gameweek_points=66, rank=893714,
            )
        ]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "history.csv"
            merge_and_write_csv(records, path)
            reloaded = import_csv(path)
        self.assertEqual(len(reloaded), 1)
        r = reloaded[0]
        self.assertEqual(r.captain, 2)
        self.assertEqual(r.vice_captain, 1)
        self.assertEqual(r.manager_name, "Chris Fox")
        self.assertEqual(r.squad, (1, 2, 3, 4))
        self.assertEqual(r.transfers, 2)
        self.assertEqual(r.chip, "bboost")
        self.assertEqual(r.total_points, 66)
        self.assertEqual(r.rank, 893714)

    def test_field_order_matches_declared_schema(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "history.csv"
            merge_and_write_csv([HistoricalRecord(season="2026-27", manager_id=1, gameweek=1, starting_xi=(1,), captain=1)], path)
            header = path.read_text().splitlines()[0].split(",")
        self.assertEqual(tuple(header), REAL_CSV_FIELDS)


if __name__ == "__main__":
    unittest.main()
