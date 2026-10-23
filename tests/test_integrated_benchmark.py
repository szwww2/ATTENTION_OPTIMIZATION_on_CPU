"""Check the six-variant benchmark's CSVs and OpenMP failure handling."""
import csv
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
BINARY = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else ROOT / "build" / (
    "attn_bench_all.exe" if os.name == "nt" else "attn_bench_all")
TEST_TEMP = ROOT / "build" / "integrated_benchmark_tests"
TEST_TEMP.mkdir(parents=True, exist_ok=True)
VARIANTS = [(route, mode) for mode in ("scalar", "simd", "openmp")
            for route in ("full_matrix", "rowwise")]


def run_benchmark(folder, options, environment=None):
    env = os.environ.copy()
    # Make the test's permitted thread budget independent of the invoking shell.
    env["OMP_THREAD_LIMIT"] = "64"
    if environment:
        env.update(environment)
    return subprocess.run(
        [str(BINARY), *options, "--out", "run"], cwd=folder, env=env,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=60,
    )


def read_csv(folder, suffix):
    with (Path(folder) / ("run_" + suffix + ".csv")).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class IntegratedBenchmarkTests(unittest.TestCase):
    def test_six_variants_statistics_workspace_and_order(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            run = run_benchmark(folder, [
                "--mode", "all", "--threads", "2", "--L", "31", "--d", "31",
                "--dv", "33", "--query-block", "3", "--warmup", "1", "--reps", "6",
                "--seed", "42",
            ])
            self.assertEqual(run.returncode, 0, run.stderr)
            raw, summaries = read_csv(folder, "raw"), read_csv(folder, "summary")
            self.assertEqual(len(raw), 36)
            self.assertEqual(len(summaries), 6)
            self.assertEqual({(r["route"], r["mode"]) for r in summaries}, set(VARIANTS))

            for result in summaries:
                key = result["route"], result["mode"]
                samples = [r for r in raw if (r["route"], r["mode"]) == key]
                samples.sort(key=lambda r: int(r["sample"]))
                self.assertEqual([int(r["sample"]) for r in samples], list(range(6)))
                workers = 2 if result["mode"] == "openmp" else 1
                block = 3 if result["route"] == "rowwise" or result["mode"] == "openmp" else 0
                for row in [result, *samples]:
                    for field, expected in (("L", 31), ("d", 31), ("dv", 33), ("seed", 42),
                                            ("query_block", block), ("threads", workers),
                                            ("actual_threads", workers)):
                        self.assertEqual(int(row[field]), expected, (key, field))

                times = [float(r["ms"]) for r in samples]
                self.assertTrue(all(math.isfinite(t) and t > 0 for t in times))
                q1, _, q3 = statistics.quantiles(times, method="inclusive")
                median = statistics.median(times)
                expected_stats = {
                    "median_ms": median, "min_ms": min(times), "q1_ms": q1,
                    "q3_ms": q3, "iqr_ms": q3 - q1,
                    "gflops": 2 * 31 * 31 * (31 + 33) / (median * 1e6),
                }
                for field, expected in expected_stats.items():
                    self.assertTrue(math.isclose(float(result[field]), expected,
                                                 rel_tol=1e-12, abs_tol=1e-12), (key, field))
                workspace = 31 * 31 * 4 if result["route"] == "full_matrix" else workers * 3 * 31 * 4
                self.assertEqual(int(result["workspace_bytes"]), workspace, key)
                self.assertEqual(result["contract_pass"], "1")
                self.assertEqual(result["warmup"], "1")
                self.assertEqual(result["reps"], "6")
                for field in ("pre_max_abs", "pre_max_rel", "post_max_abs", "post_max_rel"):
                    self.assertTrue(math.isfinite(float(result[field])), (key, field))

            # Reconstruct the actual execution sequence from the recorded slots.
            # Each round moves the previous first variant to the end.
            expected_order = VARIANTS.copy()
            for round_index in range(6):
                round_rows = [r for r in raw if int(r["sample"]) == round_index]
                round_rows.sort(key=lambda r: int(r["order"]))
                self.assertEqual([int(r["order"]) for r in round_rows], list(range(6)))
                self.assertEqual([(r["route"], r["mode"]) for r in round_rows], expected_order)
                expected_order = expected_order[1:] + expected_order[:1]

    def test_invalid_mode_and_threads_produce_no_csv(self):
        cases = [
            ["--mode", "unknown"],
            ["--mode", "all", "--threads", "0"],
            ["--mode", "scalar", "--threads", "2"],
            ["--mode", "simd", "--threads", "2"],
        ]
        for options in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
                run = run_benchmark(folder, options)
                self.assertNotEqual(run.returncode, 0, run.stdout)
                self.assertFalse((Path(folder) / "run_raw.csv").exists())
                self.assertFalse((Path(folder) / "run_summary.csv").exists())

    def test_unavailable_thread_count_produces_no_csv(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            run = run_benchmark(
                folder, ["--mode", "openmp", "--threads", "2", "--L", "31"],
                {"OMP_THREAD_LIMIT": "1"},
            )
            self.assertNotEqual(run.returncode, 0, run.stdout)
            self.assertIn("OpenMP team size differs", run.stderr)
            self.assertFalse((Path(folder) / "run_raw.csv").exists())
            self.assertFalse((Path(folder) / "run_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
