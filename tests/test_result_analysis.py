"""Reject damaged results before exporting comparisons; no benchmark rerun needed."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from analyze_results import load_grid


class ResultAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name) / "copied_grid"
        shutil.copytree(ROOT / "examples/integrated_20261021", self.folder)

    def change_csv(self, suffix, change):
        path = self.folder / ("serial_L512_d64_T1_" + suffix + ".csv")
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        change(rows)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def change_json(self, name, change):
        path = self.folder / name
        data = json.loads(path.read_text(encoding="utf-8"))
        change(data)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_portable_copy_and_known_measurements(self):
        # Original manifest prefixes do not exist in this temporary directory.
        rows, _, missing, hashes = load_grid(self.folder)
        self.assertEqual(len(rows), 40)
        self.assertEqual(sum(r["reps"] for r in rows), 800)
        self.assertEqual(len(hashes), 43)
        self.assertEqual(sum(j["variants"] for j in missing), 20)
        row = next(r for r in rows if r["job"] == "serial_L512_d64_T1"
                   and r["route"] == "full_matrix" and r["mode"] == "scalar")
        self.assertAlmostEqual(row["median_ms"], 27.926)
        self.assertAlmostEqual(row["q1_ms"], 25.66125)
        self.assertAlmostEqual(row["q3_ms"], 30.16625)

    def test_missing_sample(self):
        self.change_csv("raw", lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, "missing or duplicate"):
            load_grid(self.folder)

    def test_duplicate_sample(self):
        self.change_csv("raw", lambda rows: rows.append(rows[0].copy()))
        with self.assertRaisesRegex(ValueError, "missing or duplicate"):
            load_grid(self.folder)

    def test_nonfinite_time(self):
        self.change_csv("raw", lambda rows: rows[0].update(ms="nan"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_grid(self.folder)

    def test_incorrect_summary(self):
        self.change_csv("summary", lambda rows: rows[0].update(median_ms="1"))
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_grid(self.folder)

    def test_failed_contract(self):
        self.change_csv("summary", lambda rows: rows[0].update(contract_pass="0"))
        with self.assertRaisesRegex(ValueError, "contract failed"):
            load_grid(self.folder)

    def test_bad_order(self):
        self.change_csv("raw", lambda rows: rows[0].update(order="99"))
        with self.assertRaisesRegex(ValueError, "execution order"):
            load_grid(self.folder)

    def test_mixed_binary(self):
        self.change_json("serial_L512_d64_T1_environment.json", lambda d: d.update(binary_sha256="different"))
        with self.assertRaisesRegex(ValueError, "different source, binary"):
            load_grid(self.folder)

    def test_failed_environment(self):
        self.change_json("serial_L512_d64_T1_environment.json", lambda d: d.update(status="failed", exit_code=1))
        with self.assertRaisesRegex(ValueError, "failed environment"):
            load_grid(self.folder)

    def test_failed_grid(self):
        self.change_json("manifest.json", lambda d: d.update(status="failed"))
        with self.assertRaisesRegex(ValueError, "did not complete"):
            load_grid(self.folder)

    def test_environment_arguments_disagree(self):
        self.change_json("serial_L512_d64_T1_environment.json",
                         lambda d: d["benchmark_arguments"].__setitem__(-1, "123"))
        with self.assertRaisesRegex(ValueError, "arguments differ"):
            load_grid(self.folder)

    def test_missing_job_not_silently_omitted(self):
        self.change_json("manifest.json", lambda d: d["jobs"].pop())
        with self.assertRaisesRegex(ValueError, "planned jobs"):
            load_grid(self.folder)

    def test_unavailable_job_is_reported(self):
        self.change_json("manifest.json", lambda d: d["jobs"][0].update(status="unavailable", reason="test memory budget"))
        rows, _, missing, _ = load_grid(self.folder)
        self.assertEqual(len(rows), 36)
        self.assertTrue(any(j.get("reason") == "test memory budget" for j in missing))

    def test_missing_file(self):
        (self.folder / "serial_L512_d64_T1_raw.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            load_grid(self.folder)

    def test_cli_exports_and_refuses_overwrite(self):
        out = Path(self.temporary.name) / "analysis"
        command = [sys.executable, "-X", "utf8", str(ROOT / "tools/analyze_results.py"),
                   "--input", str(self.folder), "--out", str(out), "--tables-only"]
        run = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(run.returncode, 0, run.stderr)
        before = {p.name: p.read_bytes() for p in out.iterdir()}
        self.assertEqual(len(before), 3)
        report = json.loads(before["analysis.json"])
        self.assertEqual((report["runs"], report["configurations"], report["raw_samples"]), (14, 40, 800))
        again = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(before, {p.name: p.read_bytes() for p in out.iterdir()})


if __name__ == "__main__":
    unittest.main()
