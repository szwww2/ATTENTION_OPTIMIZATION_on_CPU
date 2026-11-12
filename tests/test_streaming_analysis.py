"""Synthetic three-route records test the parser, not measured streaming speed."""

import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from analyze_results import load_grid
from run_project_grid import memory_estimate


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class StreamingAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / "fixture"
        shutil.copytree(ROOT / "examples/integrated_20261021", self.folder)
        path = self.folder / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest.update(streaming=True, key_block=256, unavailable_routes=[])
        for job in manifest["jobs"]:
            job["variants"] = 6 if job["mode"] == "serial" else 3
            modes = ["scalar", "simd"] if job["mode"] == "serial" else ["openmp"]
            variants = [
                (r, m)
                for m in modes
                for r in ["full_matrix", "rowwise", "blocked_streaming"]
            ]
            for suffix in ["summary", "raw"]:
                csvpath = self.folder / (job["id"] + "_" + suffix + ".csv")
                with csvpath.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                # Reuse rowwise times solely to exercise parser validation.
                copies = [
                    dict(r, route="blocked_streaming")
                    for r in rows
                    if r["route"] == "rowwise"
                ]
                rows += copies
                for row in rows:
                    bq = min(manifest["query_block"], job["L"])
                    bk = min(manifest["key_block"], job["L"])
                    row.update(
                        key_block=bk if row["route"] == "blocked_streaming" else 0,
                        requested_query_block=manifest["query_block"],
                        requested_key_block=256,
                    )
                    if suffix == "summary" and row["route"] == "blocked_streaming":
                        row["workspace_bytes"] = (
                            4 * job["threads"] * (bq * bk + bq * job["dv"] + 2 * bq)
                        )
                    if suffix == "raw":
                        row["order"] = (
                            variants.index((row["route"], row["mode"]))
                            - int(row["sample"])
                        ) % len(variants)
                write_csv(csvpath, rows)
            envpath = self.folder / (job["id"] + "_environment.json")
            env = json.loads(envpath.read_text())
            env["benchmark_arguments"] += ["--streaming", "--key-block", "256"]
            env["variant_threads"] += [
                dict(r, route="blocked_streaming")
                for r in env["variant_threads"]
                if r["route"] == "rowwise"
            ]
            envpath.write_text(json.dumps(env))
        path.write_text(json.dumps(manifest))

    def mutate(self, suffix, field, value, route="blocked_streaming"):
        path = self.folder / ("serial_L512_d64_T1_" + suffix + ".csv")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        next(r for r in rows if r["route"] == route)[field] = value
        write_csv(path, rows)

    def test_all_routes_and_capped_blocks(self):
        rows, _, missing, _ = load_grid(self.folder)
        self.assertEqual(len(rows), 60)
        self.assertEqual(sum(r["reps"] for r in rows), 1200)
        self.assertEqual(missing, [])
        small = next(
            r for r in rows if r["route"] == "blocked_streaming" and r["L"] == 128
        )
        self.assertEqual(small["key_block"], 128)
        self.assertEqual(small["requested_key_block"], 256)
        self.assertGreater(
            memory_estimate(512, 64, 16, 4, "openmp", 20, True, 256),
            memory_estimate(512, 64, 16, 4, "openmp", 20),
        )

    def test_bad_summary_blocks_and_workspace(self):
        for field in [
            "key_block",
            "requested_query_block",
            "requested_key_block",
            "workspace_bytes",
        ]:
            with self.subTest(field=field):
                path = self.folder / "serial_L512_d64_T1_summary.csv"
                saved = path.read_bytes()
                self.mutate("summary", field, 999)
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    load_grid(self.folder)
                path.write_bytes(saved)

    def test_bad_raw_blocks_rejected(self):
        for field in [
            "key_block",
            "requested_query_block",
            "requested_key_block",
            "query_block",
        ]:
            with self.subTest(field=field):
                path = self.folder / "serial_L512_d64_T1_raw.csv"
                saved = path.read_bytes()
                self.mutate("raw", field, 999)
                with self.assertRaisesRegex(ValueError, "missing or duplicate"):
                    load_grid(self.folder)
                path.write_bytes(saved)

    def test_nonstreaming_route_key_block_zero(self):
        self.mutate("summary", "key_block", 256, "rowwise")
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            load_grid(self.folder)

    def test_manifest_requires_key_block_and_availability(self):
        path = self.folder / "manifest.json"
        manifest = json.loads(path.read_text())
        for value in [0, -1, "256", True]:
            manifest["key_block"] = value
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Invalid key block"):
                load_grid(self.folder)
        manifest["key_block"] = 256
        manifest["unavailable_routes"] = [{"route": "blocked_streaming"}]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "cannot be listed"):
            load_grid(self.folder)

    def test_missing_block_column(self):
        path = self.folder / "serial_L512_d64_T1_raw.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row.pop("requested_key_block")
        write_csv(path, rows)
        with self.assertRaisesRegex(ValueError, "block schema"):
            load_grid(self.folder)


if __name__ == "__main__":
    unittest.main()
