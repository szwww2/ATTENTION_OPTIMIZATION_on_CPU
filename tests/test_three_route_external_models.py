"""Pure-Python schema/model checks using synthetic records; no performance workload."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analyze_pytorch import load_api_grid
from analyze_roofline import traffic_rows, check_machine_conditions
from compare_pytorch import measure, save_csv, write_json
from run_pytorch_grid import study_settings


def api_fixture(folder, schema):
    """Synthetic times are only for verifying ingestion and quartile calculations."""
    shapes = [(128, 64, 1), (512, 64, 1), (2048, 64, 1), (4096, 64, 1),
              (512, 32, 1), (2048, 32, 1), (128, 64, 2), (4096, 64, 2)]
    jobs = [dict(id=f"L{l}_d{d}_T{t}", L=l, d=d, dv=d, threads=t,
                 status="passed", exit_code=0) for l, d, t in shapes]
    manifest = dict(status="passed", physical_cores=2, thread_grid=[1, 2],
                    warmup=1, reps=3, seed=7, jobs=jobs,
                    unavailable_routes=[] if schema == 2 else ["blocked_streaming"])
    if schema == 2:
        manifest.update(schema_version=2, query_block=16, key_block=256, requested_cpu_list=None)
    write_json(folder / "manifest.json", manifest)
    for job in jobs:
        case = folder / job["id"]
        case.mkdir()
        env = {k: "synthetic fixture" for k in (
            "source_revision", "runner_sha256", "torch_version", "torch_build", "numpy", "python",
            "cpu", "os", "timing_scope", "affinity", "power_mode", "correctness_scope")}
        env.update({k: job[k] for k in ("L", "d", "dv", "threads")})
        env.update(status="passed", exit_code=0, warmup=1, reps=3, seed=7,
                   bridge_build={"status": "synthetic fixture"}, source_dirty=False,
                   torch_backend_verified=True, torch_sdpa_operators=["aten::_scaled_dot_product_test"],
                   input_sha256={k: "a" * 64 for k in ("Q", "K", "V")},
                   cpp_preflight_threads=job["threads"], torch_intraop_threads=job["threads"],
                   torch_interop_threads=1, query_block=16,
                   unavailable_routes=manifest["unavailable_routes"],
                   pool_policy={"OMP_NUM_THREADS": str(job["threads"]),
                                "OMP_WAIT_POLICY": "PASSIVE", "KMP_BLOCKTIME": "0"})
        if schema == 2:
            env.update(schema_version=2, requested_query_block=16, requested_key_block=256,
                       key_block=min(job["L"], 256), requested_cpu_list=None,
                       blocked_actual_threads=job["threads"], thread_check_scope="synthetic fixture",
                       runtime_conditions={"fixture": True}, runtime_conditions_after={"fixture": True})
        write_json(case / "environment.json", env)
        mode = "simd" if job["threads"] == 1 else "simd_omp"
        variants = ["full_" + mode, "row_" + mode]
        if schema == 2:
            variants.append("blocked_" + mode)
        variants.append("pytorch_sdpa")
        save_csv(case / "raw.csv", [dict(variant=name, sample=sample,
                 order=(index - sample) % len(variants), ms=float(sample + 1))
                 for sample in range(3) for index, name in enumerate(variants)])
        save_csv(case / "summary.csv", [dict(variant=name, samples=3, median_ms=2.,
                 q1_ms=1.5, q3_ms=2.5, iqr_ms=1., max_abs=0., max_rel=0., worst_margin=-1e-4)
                 for name in variants])
    return jobs[0]["id"]


class ApiSchemaTests(unittest.TestCase):
    def test_frozen_study_blocks_and_cpu_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            protocol = dict(tuning={"query_block": 16}, cpu_order=[0, 1], physical_cores=2)
            write_json(folder / "protocol.json", protocol)
            frozen = dict(status="frozen", selected_key_block=128, source_revision="fixture",
                          protocol_sha256=hashlib.sha256((folder / "protocol.json").read_bytes()).hexdigest())
            path = folder / "frozen_settings.json"
            write_json(path, frozen)
            self.assertEqual(study_settings(path, None, None, [0, 1], 2)[:2], (16, 128))
            for bq, bk, cpus, cores in ((16, 256, [0, 1], 2), (8, 128, [0, 1], 2),
                                       (None, None, [1, 0], 2), (None, None, [0, 1], 4)):
                with self.subTest(bq=bq, bk=bk, cpus=cpus, cores=cores), self.assertRaises(ValueError):
                    study_settings(path, bq, bk, cpus, cores)
            protocol["tuning"]["query_block"] = 8
            write_json(folder / "protocol.json", protocol)
            with self.assertRaises(ValueError):
                study_settings(path, None, None, [0, 1], 2)

    def test_legacy_and_three_route_grids(self):
        for schema, expected in ((1, 24), (2, 32)):
            with self.subTest(schema=schema), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                api_fixture(folder, schema)
                rows, _, _ = load_api_grid(folder)
                self.assertEqual(len(rows), expected)
                blocked = [r for r in rows if r["variant"].startswith("blocked_")]
                self.assertEqual(len(blocked), 0 if schema == 1 else 8)

    def test_blocked_records_fail_closed(self):
        for corruption in ("key_block", "threads", "missing_variant", "order", "cpu_list"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                case = folder / api_fixture(folder, 2)
                if corruption in ("key_block", "threads", "cpu_list"):
                    path = case / "environment.json"
                    env = json.loads(path.read_text())
                    if corruption == "key_block":
                        env["key_block"] = 256  # L=128 requires the effective value 128.
                    elif corruption == "threads":
                        env["blocked_actual_threads"] = 2
                    else:
                        env["requested_cpu_list"] = "0"
                    write_json(path, env)
                else:
                    path = case / ("summary.csv" if corruption == "missing_variant" else "raw.csv")
                    with path.open(newline="") as stream:
                        rows = list(csv.DictReader(stream))
                    if corruption == "missing_variant":
                        rows = [r for r in rows if not r["variant"].startswith("blocked_")]
                    else:
                        rows[0]["order"] = "3"
                    save_csv(path, rows)
                with self.assertRaises(ValueError):
                    load_api_grid(folder)

    def test_four_variant_rotation_and_output_release_boundary(self):
        events, clock_values = [], iter(range(100))
        class Output:
            def __init__(self, name):
                self.name = name
            def __del__(self):
                events.append("release")
        def clock():
            events.append("clock")
            return next(clock_values) * 1000000
        calls = {k: lambda k=k: Output(k) for k in "abcd"}
        raw, _ = measure(calls, None, 1, 4, lambda output, reference: {"max_abs": 0}, clock)
        self.assertEqual([r["variant"] for r in raw], list("abcdbcdacdabdabc"))
        self.assertEqual(len(raw), 16)
        # Every timer's start/end calls are adjacent: prior-output deletion must be outside.
        indices = [i for i, event in enumerate(events) if event == "clock"]
        self.assertTrue(all(indices[i + 1] == indices[i] + 1 for i in range(0, len(indices), 2)))

    def test_ordered_cpu_policy_allows_different_masks_per_thread_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            api_fixture(folder, 2)
            manifest_path = folder / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["cpu_order"] = [3, 1]
            for job in manifest["jobs"]:
                job["cpu_ids"] = manifest["cpu_order"][:job["threads"]]
                path = folder / job["id"] / "environment.json"
                env = json.loads(path.read_text())
                env["requested_cpu_list"] = ",".join(map(str, job["cpu_ids"]))
                env["runtime_conditions"] = {"allowed_cpu_ids": sorted(job["cpu_ids"])}
                env["runtime_conditions_after"] = env["runtime_conditions"]
                write_json(path, env)
            write_json(manifest_path, manifest)
            self.assertEqual(len(load_api_grid(folder)[0]), 32)
            path = folder / manifest["jobs"][-1]["id"] / "environment.json"
            env = json.loads(path.read_text())
            env["runtime_conditions"]["allowed_cpu_ids"] = [0, 1]
            env["runtime_conditions_after"] = env["runtime_conditions"]
            write_json(path, env)
            with self.assertRaises(ValueError):
                load_api_grid(folder)


class TrafficModelTests(unittest.TestCase):
    def row(self, **extra):
        result = dict(route="blocked_streaming", L=4096, d=64, dv=64, mode="openmp",
                      threads=1, query_block=16, key_block=256,
                      workspace_bytes=20608, median_ms=100.)
        result.update(extra)
        return result

    def test_route_specific_bytes_and_unknown_counter(self):
        rows = traffic_rows([self.row()])
        self.assertEqual([r["dram_bytes_model"] for r in rows], [5 * 1024**2, 515 * 1024**2])
        self.assertEqual(rows[1]["online_state_bytes_per_thread"], 4 * 16 * 66)
        self.assertEqual(rows[1]["dram_bytes_counter"], "")
        self.assertEqual(rows[1]["F_mat"], 2 * 4096**2 * 128)
        self.assertIn("not bounds", rows[1]["assumptions"])

    def test_tails_threads_and_key_blocks(self):
        row = self.row(L=17, d=7, dv=5, query_block=4, key_block=3)
        rows = traffic_rows([row])
        self.assertEqual(rows[1]["dram_bytes_model"], 4 * 17 * 7 + 4 * 17 * 12 * 5 + 8 * 17 * 5)
        threaded = traffic_rows([dict(row, threads=8, workspace_bytes=9999)])
        self.assertEqual(rows[1]["dram_bytes_model"], threaded[1]["dram_bytes_model"])
        self.assertEqual(rows[1]["F_mat"], threaded[1]["F_mat"])
        different_bk = traffic_rows([dict(row, key_block=7)])
        self.assertEqual(rows[1]["dram_bytes_model"], different_bk[1]["dram_bytes_model"])

    def test_unmodeled_or_unrecorded_route_rejected(self):
        for row in (self.row(route="unknown"), self.row(key_block=0), self.row(query_block=0)):
            with self.assertRaises(ValueError):
                traffic_rows([row])

    def test_machine_compiler_and_flags(self):
        native = dict(cpu={"model": "fixture"}, os="Windows fixture",
                      build=["compiler=GCC fixture", "base_flags=-O3 -ffp-contract=off", "openmp_extra_flags=-fopenmp"])
        calibration = dict(cpu=native["cpu"], compiler="GCC fixture",
                           flags=["-O3", "-ffp-contract=off", "-fopenmp"])
        self.assertIn("legacy", check_machine_conditions(calibration, native))
        calibration["os"] = native["os"]
        self.assertNotIn("legacy", check_machine_conditions(calibration, native))
        for field, wrong in (("os", "Linux fixture"), ("compiler", "other"), ("flags", ["-O0"])):
            changed = copy.deepcopy(calibration)
            changed[field] = wrong
            with self.assertRaises(ValueError):
                check_machine_conditions(changed, native)

    def test_calibration_cpu_policy(self):
        order = [3, 1, 2, 4, 5, 6, 7, 8]
        native = dict(cpu={}, os="fixture", cpu_order=order, ac_line_status="online",
                      build=["compiler=fixture", "base_flags=-O3", "openmp_extra_flags=-fopenmp"])
        calibration = dict(cpu={}, os="fixture", compiler="fixture", flags=["-O3", "-fopenmp"],
                           cpu_order=order, runtime_conditions_by_threads={
                               str(t): {"allowed_cpu_ids": sorted(order[:t]), "ac_line_status": "online"}
                               for t in (1, 2, 4, 8)})
        self.assertIn("ordered CPU", check_machine_conditions(calibration, native))
        calibration["runtime_conditions_by_threads"]["8"]["allowed_cpu_ids"] = list(range(8))
        with self.assertRaises(ValueError):
            check_machine_conditions(calibration, native)


if __name__ == "__main__":
    unittest.main()
