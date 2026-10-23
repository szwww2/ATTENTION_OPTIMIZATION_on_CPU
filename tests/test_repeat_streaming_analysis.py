"""Exercise repeat exports with synthetic records, never performance evidence."""

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

from test_streaming_analysis import StreamingAnalysisTests

ROOT = Path(__file__).resolve().parent.parent


class RepeatStreamingTests(unittest.TestCase):
    def setUp(self):
        fixture = StreamingAnalysisTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.first = fixture.folder
        self.second = self.first.parent / "second_synthetic_run"
        shutil.copytree(self.first, self.second)
        self.out = self.first.parent / "repeat_output"

    def run_analysis(self, second):
        return subprocess.run(
            [sys.executable, str(ROOT / "tools/analyze_repeats.py"),
             "--input", str(self.first), str(second), "--out", str(self.out)],
            capture_output=True, text=True,
        )

    def test_streaming_is_in_table_and_plot(self):
        run = self.run_analysis(self.second)
        self.assertEqual(run.returncode, 0, run.stderr)
        record = json.loads((self.out / "analysis.json").read_text())
        self.assertEqual(record["configurations"], 60)
        self.assertEqual(set(record["plotted_routes"]),
                         {"full_matrix", "rowwise", "blocked_streaming"})
        with (self.out / "repeat_comparison.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 60)
        streaming = [r for r in rows if r["route"] == "blocked_streaming"]
        self.assertTrue(streaming)
        self.assertTrue(all(r["requested_key_block"] == "256" for r in streaming))
        self.assertGreater((self.out / "repeat_threads.png").stat().st_size, 1000)

    def test_same_directory_is_not_two_runs(self):
        run = self.run_analysis(self.first)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("different run directory", run.stderr)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main(defaultTest="RepeatStreamingTests")
