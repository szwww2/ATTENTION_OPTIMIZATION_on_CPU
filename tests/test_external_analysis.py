"""Check external comparison/model evidence using a supplied completed dataset."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analyze_pytorch import load_api_grid
from analyze_roofline import traffic, check_calibration

API = Path(sys.argv.pop(1)) if len(sys.argv) > 1 else None
CALIBRATION = Path(sys.argv.pop(1)) if len(sys.argv) > 1 else None


class EvidenceTests(unittest.TestCase):
    def test_traffic_scenarios(self):
        self.assertEqual(traffic(4096, 64, 64, False, False), 5 * 1024**2)
        self.assertEqual(traffic(4096, 64, 64, True, False), 325 * 1024**2)
        self.assertEqual(traffic(4096, 64, 64, False, True), 8195 * 1024**2)
        self.assertEqual(traffic(4096, 64, 64, True, True), 8515 * 1024**2)

    @unittest.skipIf(API is None, "Pass a completed API grid directory")
    def test_api_data_and_corruption(self):
        rows, _, _ = load_api_grid(API)
        self.assertEqual(len(rows), 36)
        first = json.loads((API / "manifest.json").read_text())["jobs"][0]["id"]
        for corruption in (
            "backend",
            "summary",
            "raw",
            "threads",
            "missing_job",
            "block",
            "pool",
            "input_hash",
            "missing_route",
            "negative_error",
        ):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                copied = Path(tmp) / "grid"
                shutil.copytree(API, copied)
                if corruption == "missing_job":
                    path = copied / "manifest.json"
                    data = json.loads(path.read_text())
                    data["jobs"].pop()
                    path.write_text(json.dumps(data))
                elif corruption in (
                    "backend",
                    "threads",
                    "block",
                    "pool",
                    "input_hash",
                    "missing_route",
                ):
                    path = copied / first / "environment.json"
                    data = json.loads(path.read_text())
                    if corruption == "backend":
                        data["torch_backend_verified"] = False
                    elif corruption == "threads":
                        data["cpp_preflight_threads"] = 99
                    elif corruption == "block":
                        data["query_block"] = 1
                    elif corruption == "pool":
                        data["pool_policy"]["OMP_WAIT_POLICY"] = "ACTIVE"
                    elif corruption == "input_hash":
                        data["input_sha256"]["Q"] = "0" * 64
                    else:
                        data["unavailable_routes"] = []
                    path.write_text(json.dumps(data))
                else:
                    path = (
                        copied
                        / first
                        / ("summary.csv" if corruption == "negative_error" else corruption + ".csv")
                    )
                    lines = path.read_text().splitlines()
                    fields = lines[0].split(",")
                    values = lines[1].split(",")
                    field = (
                        "max_abs"
                        if corruption == "negative_error"
                        else "median_ms" if corruption == "summary" else "ms"
                    )
                    values[fields.index(field)] = "-1" if corruption == "negative_error" else "nan"
                    lines[1] = ",".join(values)
                    path.write_text("\n".join(lines) + "\n")
                with self.assertRaises(ValueError):
                    load_api_grid(copied)

    @unittest.skipIf(CALIBRATION is None, "Pass a completed calibration directory")
    def test_calibration_and_corruption(self):
        self.assertEqual(len(check_calibration(CALIBRATION)), 8)
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "calibration"
            shutil.copytree(CALIBRATION, copied)
            path = copied / "raw.csv"
            lines = path.read_text().splitlines()
            fields = lines[0].split(",")
            row = next(i for i in range(1, len(lines)) if lines[i].startswith("read_stream,"))
            values = lines[row].split(",")
            values[fields.index("checksum")] = "1"
            lines[row] = ",".join(values)
            path.write_text("\n".join(lines) + "\n")
            with self.assertRaises(ValueError):
                check_calibration(copied)


if __name__ == "__main__":
    unittest.main()
