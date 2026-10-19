"""Prove that numerical failures block both performance CSV files."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
TEST_TEMP = ROOT / "build" / "benchmark_tests"
TEST_TEMP.mkdir(parents=True, exist_ok=True)


class ContractGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(dir=TEST_TEMP)
        cls.addClassCleanup(cls.folder.cleanup)
        scratch = Path(cls.folder.name)
        compiler = os.environ.get("CXX", "g++")
        flags = ["-std=c++17", "-O3", "-fno-tree-vectorize", "-Wall", "-Wextra", "-Iinclude"]
        original = scratch / "full_matrix.o"
        subprocess.run([compiler, *flags,
                        "-Dfull_matrix_scalar_fp32=full_matrix_unmodified", "-c",
                        "src/full_matrix_scalar.cpp", "-o", str(original)], cwd=ROOT, check=True)
        cls.binary = scratch / ("fault_bench.exe" if os.name == "nt" else "fault_bench")
        sources = ["src/oracle_fp64.cpp", "src/rowwise_scalar.cpp", "src/input.cpp",
                   "src/metrics.cpp", "tests/fault_full_matrix.cpp", "tools/bench_scalar.cpp"]
        subprocess.run([compiler, *flags, *sources, str(original), "-o", str(cls.binary)],
                       cwd=ROOT, check=True)

    def check_failure(self, fail_at, mode, phase):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            env = dict(os.environ, ATTN_TEST_FAIL_AT=str(fail_at), ATTN_TEST_FAILURE=mode)
            # Each route has 1 pre-check + 2 warmups + 3 timed calls. Call 6 is last.
            run = subprocess.run([str(self.binary), "--L", "7", "--d", "3", "--dv", "5",
                                  "--warmup", "2", "--reps", "3", "--out", "failed"],
                                 cwd=folder, env=env, capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("fixed numerical contract " + phase, run.stderr)
            self.assertFalse((Path(folder) / "failed_raw.csv").exists())
            self.assertFalse((Path(folder) / "failed_summary.csv").exists())

    def test_finite_error_before_timing(self):
        self.check_failure(1, "finite", "before timing")

    def test_nan_before_timing(self):
        self.check_failure(1, "nan", "before timing")

    def test_error_after_timing(self):
        self.check_failure(6, "finite", "after timing")


if __name__ == "__main__":
    unittest.main()
