"""Regression test: the synthetic dataset export must be fully
deterministic across separate Python processes.

An earlier version seeded persona generation with Python's builtin
``hash()`` on a (str, int) tuple, which is randomized per-process
(PYTHONHASHSEED) unless explicitly disabled - two runs of the CLI produced
DIFFERENT synthetic datasets and therefore different, non-reproducible
results. This test runs the generator in two separate subprocesses (so it
cannot be fooled by in-process hash-seed caching) and asserts byte-for-byte
identical output.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fpl_rival.fixtures.calibration_fixtures import build_synthetic_dataset, write_synthetic_dataset_csv


class TestSyntheticDatasetDeterminism(unittest.TestCase):
    def test_same_process_calls_are_identical(self):
        rows1, ep1 = build_synthetic_dataset(num_gameweeks=10, managers_per_persona=1)
        rows2, ep2 = build_synthetic_dataset(num_gameweeks=10, managers_per_persona=1)
        self.assertEqual(rows1, rows2)
        self.assertEqual(ep1, ep2)

    def test_deterministic_across_separate_python_processes(self):
        # Each subprocess gets its own (possibly different) PYTHONHASHSEED
        # unless the code avoids hash() entirely for seeding - this is the
        # test that would have caught the original bug.
        script = (
            "from fpl_rival.fixtures.calibration_fixtures import build_synthetic_dataset;"
            "rows, ep = build_synthetic_dataset(num_gameweeks=10, managers_per_persona=1);"
            "print(sum(r['captain'] for r in rows))"
        )
        outputs = set()
        for _ in range(3):
            result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, cwd=str(Path(__file__).resolve().parent.parent))
            outputs.add(result.stdout.strip())
        self.assertEqual(len(outputs), 1, f"Synthetic dataset generation is not deterministic across processes: {outputs}")

    def test_written_csv_is_byte_identical_across_calls(self):
        with tempfile.TemporaryDirectory() as d:
            path1 = Path(d) / "a.csv"
            path2 = Path(d) / "b.csv"
            write_synthetic_dataset_csv(path1, num_gameweeks=10, managers_per_persona=1)
            write_synthetic_dataset_csv(path2, num_gameweeks=10, managers_per_persona=1)
            self.assertEqual(path1.read_text(), path2.read_text())


if __name__ == "__main__":
    unittest.main()
