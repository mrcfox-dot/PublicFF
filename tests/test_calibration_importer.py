"""Unit tests for fpl_rival.calibration.importer - dataset ingestion/normalization."""

import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from fpl_rival.calibration.importer import (
    import_csv,
    import_jsonl,
    import_long_format_csv,
    load_dataset,
    records_by_manager_and_season,
    summarize_dataset,
)
from fpl_rival.calibration.models import DatasetError


class TestNativeCsvImport(unittest.TestCase):
    def test_required_fields_parsed(self):
        content = "season,manager_id,gameweek,starting_xi,captain\n2026-27,1,1,1;2;3,2\n"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.csv"
            path.write_text(content)
            records = import_csv(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].starting_xi, (1, 2, 3))
        self.assertEqual(records[0].captain, 2)

    def test_optional_fields_default_sensibly_when_absent(self):
        content = "season,manager_id,gameweek,starting_xi,captain\n2026-27,1,1,1;2;3,2\n"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.csv"
            path.write_text(content)
            records = import_csv(path)
        r = records[0]
        self.assertIsNone(r.rank)
        self.assertIsNone(r.total_points)
        self.assertIsNone(r.chip)
        self.assertEqual(r.transfers, 0)

    def test_missing_required_field_fails_clearly(self):
        content = "season,manager_id,gameweek,captain\n2026-27,1,1,2\n"  # no starting_xi column at all
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.csv"
            path.write_text(content)
            with self.assertRaises(DatasetError):
                import_csv(path)

    def test_blank_rows_skipped(self):
        content = "season,manager_id,gameweek,starting_xi,captain\n2026-27,1,1,1;2;3,2\n,,,,\n"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.csv"
            path.write_text(content)
            records = import_csv(path)
        self.assertEqual(len(records), 1)

    def test_malformed_field_fails_clearly(self):
        content = "season,manager_id,gameweek,starting_xi,captain\n2026-27,notanumber,1,1;2;3,2\n"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.csv"
            path.write_text(content)
            with self.assertRaises(DatasetError):
                import_csv(path)


class TestNativeJsonlImport(unittest.TestCase):
    def test_round_trips_arrays(self):
        rows = [{"season": "2026-27", "manager_id": 1, "gameweek": 1, "starting_xi": [1, 2, 3], "captain": 1}]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows))
            records = import_jsonl(path)
        self.assertEqual(records[0].starting_xi, (1, 2, 3))


class TestLongFormatAdaptor(unittest.TestCase):
    def test_pivots_per_player_rows_into_one_record(self):
        content = textwrap.dedent("""\
            season,manager_id,gameweek,player_id,is_starting,is_captain,is_vice_captain
            2026-27,1,1,10,1,0,0
            2026-27,1,1,11,1,1,0
            2026-27,1,1,12,1,0,1
            2026-27,1,1,99,0,0,0
        """)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "long.csv"
            path.write_text(content)
            records = import_long_format_csv(path)
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(set(r.starting_xi), {10, 11, 12})
        self.assertEqual(r.captain, 11)
        self.assertEqual(r.vice_captain, 12)
        self.assertIn(99, r.squad)  # bench player, not in starting_xi
        self.assertNotIn(99, r.starting_xi)


class TestLoadDatasetDispatch(unittest.TestCase):
    def test_loads_directory_of_multiple_files(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.csv").write_text("season,manager_id,gameweek,starting_xi,captain\n2026-27,1,1,1;2;3,1\n")
            (Path(d) / "b.csv").write_text("season,manager_id,gameweek,starting_xi,captain\n2026-27,2,1,1;2;3,2\n")
            records = load_dataset(Path(d))
        self.assertEqual(len(records), 2)

    def test_unsupported_extension_fails_clearly(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.txt"
            path.write_text("nonsense")
            with self.assertRaises(DatasetError):
                load_dataset(path)

    def test_missing_path_fails_clearly(self):
        with self.assertRaises(DatasetError):
            load_dataset(Path("/nonexistent/path/data.csv"))


class TestChronologicalOrdering(unittest.TestCase):
    def test_records_by_manager_and_season_sorted_by_gameweek(self):
        content = "season,manager_id,gameweek,starting_xi,captain\n2026-27,1,3,1;2;3,1\n2026-27,1,1,1;2;3,1\n2026-27,1,2,1;2;3,1\n"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.csv"
            path.write_text(content)
            records = import_csv(path)
        grouped = records_by_manager_and_season(records)
        gws = [r.gameweek for r in grouped[("2026-27", 1)]]
        self.assertEqual(gws, [1, 2, 3])


class TestSummarizeDataset(unittest.TestCase):
    def test_empty_dataset_summary(self):
        summary = summarize_dataset([])
        self.assertEqual(summary.num_records, 0)
        self.assertTrue(summary.warnings)

    def test_fewer_than_two_managers_warns_about_consensus(self):
        content = "season,manager_id,gameweek,starting_xi,captain\n2026-27,1,1,1;2;3,1\n"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.csv"
            path.write_text(content)
            records = import_csv(path)
        summary = summarize_dataset(records)
        self.assertTrue(any("consensus" in w.lower() for w in summary.warnings))


if __name__ == "__main__":
    unittest.main()
