"""Check CSV semantics and statistics independently of the C++ implementation."""
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
BINARY = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else ROOT / "build" / (
    "attn_bench.exe" if os.name == "nt" else "attn_bench")
TEST_TEMP = ROOT / "build" / "benchmark_tests"
TEST_TEMP.mkdir(parents=True, exist_ok=True)


class BenchmarkTests(unittest.TestCase):
    def test_odd_shape_statistics_and_metadata(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            prefix = Path(folder) / "odd"
            # Stay above coarse clock ticks; numerical suites still cover L=1 and L=7.
            command = [str(BINARY), "--L", "31", "--d", "31", "--dv", "33", "--seed", "42",
                       "--warmup", "2", "--reps", "7", "--query-block", "3", "--out", "odd"]
            run = subprocess.run(command, cwd=folder, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            with open(str(prefix) + "_raw.csv", newline="") as f:
                raw = list(csv.DictReader(f))
            with open(str(prefix) + "_summary.csv", newline="") as f:
                summary = list(csv.DictReader(f))
            self.assertEqual(len(raw), 14)
            self.assertEqual({r["route"] for r in summary}, {"full_matrix", "rowwise"})
            for r in summary:
                samples = [x for x in raw if x["route"] == r["route"]]
                self.assertEqual([int(x["sample"]) for x in samples], list(range(7)))
                times = [float(x["ms"]) for x in samples]
                self.assertTrue(all(math.isfinite(x) and x > 0 for x in times))
                q1, _, q3 = statistics.quantiles(times, method="inclusive")
                expected = [("median_ms", statistics.median(times)), ("min_ms", min(times)),
                            ("q1_ms", q1), ("q3_ms", q3), ("iqr_ms", q3-q1),
                            ("gflops", 2*31*31*(31+33)/(statistics.median(times)*1e6))]
                for field, value in expected:
                    self.assertTrue(math.isclose(float(r[field]), value, rel_tol=1e-12, abs_tol=1e-12), field)
                self.assertEqual(r["contract_pass"], "1")
                self.assertEqual(r["seed"], "42")
                for field in ["pre_max_abs", "pre_max_rel", "post_max_abs", "post_max_rel"]:
                    self.assertTrue(math.isfinite(float(r[field])))
                self.assertEqual(int(r["workspace_bytes"]), (31*31 if r["route"] == "full_matrix" else 3*31)*4)
                index = 0 if r["route"] == "full_matrix" else 1
                self.assertEqual([int(x["order"]) for x in samples], [(index+i)%2 for i in range(7)])
            old = {suffix: Path(str(prefix) + suffix).read_bytes()
                   for suffix in ("_raw.csv", "_summary.csv")}
            self.assertNotEqual(subprocess.run(command, cwd=folder, capture_output=True).returncode, 0)
            for suffix, content in old.items():
                self.assertEqual(Path(str(prefix) + suffix).read_bytes(), content)

    def test_invalid_arguments_do_not_write_results(self):
        cases = [("--L", "0"), ("--d", "-1"), ("--dv", "2x"), ("--reps", "0"),
                 ("--warmup", "0"), ("--query-block", "0"), ("--L", "4097")]
        for option, value in cases:
            with self.subTest(option=option, value=value), tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
                prefix = str(Path(folder) / "invalid")
                run = subprocess.run([str(BINARY), option, value, "--out", prefix], capture_output=True)
                self.assertNotEqual(run.returncode, 0)
                self.assertFalse(Path(prefix + "_raw.csv").exists())
                self.assertFalse(Path(prefix + "_summary.csv").exists())


class RunnerTests(unittest.TestCase):
    def test_metadata_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            prefix = Path(folder) / "recorded"
            command = [sys.executable, str(ROOT / "tools/run_scalar_bench.py"), "--out", str(prefix),
                       "--L", "31", "--d", "31", "--dv", "33", "--warmup", "1", "--reps", "3"]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            path = Path(str(prefix) + "_environment.json")
            original = path.read_bytes()
            metadata = json.loads(original)
            self.assertEqual(metadata["status"], "passed")
            self.assertEqual(metadata["exit_code"], 0)
            self.assertEqual(metadata["threads"], 1)
            self.assertEqual(metadata["binary_sha256"], hashlib.sha256(BINARY.read_bytes()).hexdigest())
            for name, digest in metadata["source_sha256"].items():
                self.assertFalse(Path(name).is_absolute())
                self.assertEqual(digest, hashlib.sha256((ROOT / name).read_bytes()).hexdigest())
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(path.read_bytes(), original)

    @unittest.skipUnless(os.name == "nt", "CIM fallback is Windows-specific")
    def test_denied_cpu_probe_is_optional(self):
        spec = importlib.util.spec_from_file_location("runner", ROOT / "tools/run_scalar_bench.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with mock.patch.object(runner.platform, "processor", return_value="test CPU"), \
             mock.patch.object(runner.subprocess, "run", side_effect=PermissionError("PRIVATE_MARKER")):
            info = runner.cpu_information()
        self.assertEqual(info["description"], "test CPU")
        self.assertEqual(info["windows_cim"], "unavailable: PermissionError")
        self.assertNotIn("PRIVATE_MARKER", json.dumps(info))


if __name__ == "__main__":
    unittest.main()
