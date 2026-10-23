"""Exercise repeat exports with synthetic records, never performance evidence."""

import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import unittest

from test_streaming_analysis import StreamingAnalysisTests, write_csv

ROOT = Path(__file__).resolve().parent.parent


class RepeatStreamingTests(unittest.TestCase):
    def setUp(self):
        fixture = StreamingAnalysisTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.first = fixture.folder
        self.second = self.first.parent / "second_synthetic_run"
        shutil.copytree(self.first, self.second)
        # Deliberately altered parser fixtures, never claimed as measurements.
        self.rewrite_samples(self.first, scale=1.03)
        self.rewrite_samples(self.second, scale=1.11)
        self.set_timestamps(self.first, hour=0)
        self.set_timestamps(self.second, hour=1)
        self.out = self.first.parent / "repeat_output"

    def rewrite_samples(self, folder, scale=1.0, warmup=None, reps=None):
        path = folder / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if warmup is not None:
            manifest["warmup"] = warmup
        if reps is not None:
            manifest["reps"] = reps
        manifest["fixture_note"] = "Synthetic parser records, not measured performance"
        for job in manifest["jobs"]:
            rawpath = folder / (job["id"] + "_raw.csv")
            with rawpath.open(newline="", encoding="utf-8") as stream:
                raw = [r for r in csv.DictReader(stream) if int(r["sample"]) < manifest["reps"]]
            for row in raw:
                row["ms"] = float(row["ms"]) * scale
            write_csv(rawpath, raw)
            summarypath = folder / (job["id"] + "_summary.csv")
            with summarypath.open(newline="", encoding="utf-8") as stream:
                summaries = list(csv.DictReader(stream))
            for row in summaries:
                times = [float(r["ms"]) for r in raw
                         if (r["route"], r["mode"]) == (row["route"], row["mode"])]
                q1, _, q3 = statistics.quantiles(times, method="inclusive")
                median = statistics.median(times)
                row.update(warmup=manifest["warmup"], reps=manifest["reps"],
                           median_ms=median, min_ms=min(times), q1_ms=q1, q3_ms=q3,
                           iqr_ms=q3-q1,
                           gflops=2*int(row["L"])**2*(int(row["d"])+int(row["dv"]))/(median*1e6))
            write_csv(summarypath, summaries)
            envpath = folder / (job["id"] + "_environment.json")
            environment = json.loads(envpath.read_text(encoding="utf-8"))
            for field in ("warmup", "reps"):
                args = environment["benchmark_arguments"]
                args[args.index("--" + field) + 1] = str(manifest[field])
            environment["fixture_note"] = manifest["fixture_note"]
            envpath.write_text(json.dumps(environment), encoding="utf-8")
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def set_timestamps(self, folder, hour):
        for i, path in enumerate(sorted(folder.glob("*_environment.json"))):
            environment = json.loads(path.read_text(encoding="utf-8"))
            if hour is None:
                environment.pop("started_utc", None)
                environment.pop("finished_utc", None)
            else:
                start = datetime(2026, 1, 1, hour, tzinfo=timezone.utc) + timedelta(seconds=20*i)
                environment.update(started_utc=start.isoformat(),
                                   finished_utc=(start + timedelta(seconds=10)).isoformat())
            path.write_text(json.dumps(environment), encoding="utf-8")

    def skip_one_job(self, folder):
        path = folder / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        next(j for j in manifest["jobs"] if j["id"] == "serial_L512_d64_T1").update(
            status="unavailable", reason="synthetic memory budget exclusion")
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def run_analysis(self, second, *flags, third=None):
        inputs = [str(self.first), str(second)]
        if third is not None:
            inputs.append(str(third))
        return subprocess.run(
            [sys.executable, "-X", "utf8", str(ROOT / "tools/analyze_repeats.py"),
             "--input", *inputs, "--out", str(self.out), *flags],
            capture_output=True, text=True, encoding="utf-8",
        )

    def test_streaming_is_in_table_and_plot(self):
        third = self.first.parent / "third_synthetic_run"
        shutil.copytree(self.second, third)
        self.rewrite_samples(third, scale=1.08)
        self.set_timestamps(third, hour=2)
        run = self.run_analysis(self.second, "--require-complete", third=third)
        self.assertEqual(run.returncode, 0, run.stderr)
        record = json.loads((self.out / "analysis.json").read_text())
        self.assertEqual(record["runs"], 3)
        self.assertTrue(record["records_validated"])
        self.assertEqual(record["performance_stability"], "descriptive_only")
        self.assertEqual(record["run_timing"]["status"], "verified_non_overlapping")
        self.assertEqual(len(set(record["raw_sample_fingerprints"].values())), 3)
        self.assertEqual(record["sampling_protocol"]["reps"], 20)
        self.assertTrue(all(not skipped for skipped in record["unavailable"].values()))
        self.assertEqual(record["configurations"], 60)
        self.assertEqual(set(record["plotted_routes"]),
                         {"full_matrix", "rowwise", "blocked_streaming"})
        with (self.out / "repeat_comparison.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 60)
        streaming = [r for r in rows if r["route"] == "blocked_streaming"]
        self.assertTrue(streaming)
        self.assertTrue(all(r["requested_key_block"] == "256" for r in streaming))
        self.assertTrue(all(r["reps"] == "20" for r in rows))
        self.assertGreater((self.out / "repeat_threads.png").stat().st_size, 1000)

    def test_same_directory_is_not_two_runs(self):
        run = self.run_analysis(self.first)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("different run directory", run.stderr)
        self.assertFalse(self.out.exists())

    def test_copied_raw_samples_are_not_two_runs(self):
        copy = self.first.parent / "copied_synthetic_run"
        shutil.copytree(self.first, copy)
        self.set_timestamps(copy, hour=2)
        run = self.run_analysis(copy)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("Duplicate raw sample fingerprint", run.stderr)
        self.assertFalse(self.out.exists())

    def test_different_warmup_rejected(self):
        self.rewrite_samples(self.second, warmup=11)
        run = self.run_analysis(self.second)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("Different sampling protocol", run.stderr)
        self.assertFalse(self.out.exists())

    def test_different_reps_rejected(self):
        self.rewrite_samples(self.second, reps=19)
        run = self.run_analysis(self.second)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("Different sampling protocol", run.stderr)
        self.assertFalse(self.out.exists())

    def test_overlapping_timestamps_rejected(self):
        self.set_timestamps(self.second, hour=0)
        run = self.run_analysis(self.second)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("intervals overlap", run.stderr)
        self.assertFalse(self.out.exists())

    def test_legacy_missing_timestamps_report_unknown(self):
        self.set_timestamps(self.second, hour=None)
        run = self.run_analysis(self.second)
        self.assertEqual(run.returncode, 0, run.stderr)
        record = json.loads((self.out / "analysis.json").read_text())
        self.assertEqual(record["run_timing"]["status"], "unknown")
        self.assertIn("run timing unknown", record["interpretation"])

    def test_complete_requires_timestamps(self):
        self.set_timestamps(self.second, hour=None)
        run = self.run_analysis(self.second, "--require-complete")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("require start and finish timestamps", run.stderr)
        self.assertFalse(self.out.exists())

    def test_invalid_timestamps_rejected(self):
        path = next(self.second.glob("*_environment.json"))
        environment = json.loads(path.read_text(encoding="utf-8"))
        environment["finished_utc"] = environment["started_utc"]
        path.write_text(json.dumps(environment), encoding="utf-8")
        run = self.run_analysis(self.second)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("finish must follow start", run.stderr)
        self.assertFalse(self.out.exists())

    def test_unavailable_configs_preserved(self):
        for folder in (self.first, self.second):
            self.skip_one_job(folder)
        run = self.run_analysis(self.second)
        self.assertEqual(run.returncode, 0, run.stderr)
        record = json.loads((self.out / "analysis.json").read_text())
        self.assertEqual(record["configurations"], 54)
        self.assertTrue(all(len(skipped) == 1 for skipped in record["unavailable"].values()))

    def test_complete_rejects_unavailable_configs(self):
        for folder in (self.first, self.second):
            self.skip_one_job(folder)
        run = self.run_analysis(self.second, "--require-complete")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("require no unavailable configurations", run.stderr)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main(defaultTest="RepeatStreamingTests")
