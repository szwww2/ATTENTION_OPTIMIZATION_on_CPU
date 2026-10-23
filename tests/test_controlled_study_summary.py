"""Synthetic-only checks for pairing and frozen-record binding; no benchmark calls."""

import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from summarize_controlled_study import aggregate, check_binding, paired_speedups, main, load_study


def sample(batch, mode, threads, value, route="blocked_streaming", length=128, d=64):
    return dict(batch=batch, route=route, L=length, d=d, dv=d, mode=mode,
                threads=threads, median_ms=value)


class PairingTests(unittest.TestCase):
    def test_median_of_paired_ratios_not_ratio_of_medians(self):
        rows = []
        baseline, target = [1., 4., 100.], [1., 100., 2.]
        for batch, (base, current) in enumerate(zip(baseline, target), 1):
            rows += [sample(batch, "scalar", 1, base), sample(batch, "simd", 1, current),
                     sample(batch, "openmp", 1, 2 * base), sample(batch, "openmp", 2, 2 * current)]
        pairs, summary = paired_speedups(rows)
        for kind in ("SIMD_vs_scalar", "OpenMP_vs_OpenMP1"):
            row = next(r for r in summary if r["comparison"] == kind
                       and r["threads"] == (1 if kind.startswith("SIMD") else 2))
            self.assertEqual(row["median_paired_speedup"], 1.)
            self.assertEqual(row["min_paired_speedup"], .04)
            self.assertEqual(row["max_paired_speedup"], 50.)
            self.assertNotEqual(row["median_paired_speedup"], statistics.median(baseline) / statistics.median(target))
        self.assertEqual(len(pairs), 9)

    def test_full_plan_counts_and_same_route_shape_pairing(self):
        rows = []
        shapes = ((128, 64), (512, 64), (2048, 64), (4096, 64), (512, 32), (2048, 32))
        for batch in (1, 2, 3):
            for index, route in enumerate(("full_matrix", "rowwise", "blocked_streaming"), 1):
                for length, d in shapes:
                    base = float(batch * index * length * d)
                    rows += [sample(batch, "scalar", 1, base, route, length, d),
                             sample(batch, "simd", 1, base / 2, route, length, d)]
                for length in (128, 4096):
                    base = float(batch * index * length)
                    rows += [sample(batch, "openmp", t, base / t, route, length) for t in (1, 2, 4, 8)]
        pairs, summary = paired_speedups(rows)
        self.assertEqual((len(rows), len(pairs), len(summary)), (180, 126, 42))
        for row in summary:
            self.assertEqual(row["median_paired_speedup"],
                             2. if row["comparison"] == "SIMD_vs_scalar" else float(row["threads"]))

    def test_missing_baseline_duplicate_and_nonfinite_rejected(self):
        with self.assertRaisesRegex(ValueError, "baseline"):
            paired_speedups([sample(1, "simd", 1, 1.)])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            paired_speedups([sample(1, "scalar", 1, 1.)] * 2)
        with self.assertRaisesRegex(ValueError, "Invalid median"):
            paired_speedups([sample(1, "scalar", 1, float("nan"))])

    def test_aggregate_requires_exactly_three_distinct_batches(self):
        with self.assertRaisesRegex(ValueError, "three batches"):
            aggregate([sample(1, "scalar", 1, 1.), sample(1, "scalar", 1, 2.),
                       sample(3, "scalar", 1, 3.)], ("route",), "median_ms", "batch_median_ms")


class FrozenBindingTests(unittest.TestCase):
    def setUp(self):
        self.protocol = dict(source_revision="a" * 40, native_binary_sha256="b" * 64,
                             cpu_order=list(range(8)), physical_cores=16,
                             tuning={"query_block": 16},
                             main=dict(batches=3, configurations_per_batch=60, warmup=10, reps=20,
                                       seed=20261020, job_orders=["forward", "reverse", "rotate"]))
        self.freeze = {field: self.protocol[field] for field in ("source_revision", "native_binary_sha256", "cpu_order")}
        self.freeze.update(status="frozen", protocol_sha256="c" * 64, selected_key_block=256,
                           frozen_utc="2026-10-23T00:00:00+00:00")
        self.manifest = dict(streaming=True, unavailable_routes=[], cpu_order=list(range(8)),
                             physical_cores=16, job_order="forward", query_block=16, key_block=256,
                             warmup=10, reps=20, seed=20261020, jobs=[{"id": "synthetic_job"}])
        self.environments = {"synthetic_job": dict(source_revision="a" * 40, binary_sha256="b" * 64,
                             source_dirty=False, started_utc="2026-10-23T00:01:00+00:00")}

    def check(self, freeze=None, manifest=None, environments=None, digest=None):
        check_binding(self.protocol, freeze or self.freeze, digest or "c" * 64,
                      manifest or self.manifest, environments or self.environments, 1)

    def test_matching_frozen_records(self):
        self.check()

    def test_protocol_digest_and_settings_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.check(digest="d" * 64)
        for field, value in (("key_block", 128), ("job_order", "reverse"), ("reps", 19), ("cpu_order", [1, 0])):
            changed = dict(self.manifest, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(manifest=changed)

    def test_every_job_source_binary_and_freeze_time_checked(self):
        for field, value in (("source_revision", "wrong"), ("binary_sha256", "wrong"),
                             ("source_dirty", True), ("started_utc", "2026-10-22T00:01:00+00:00")):
            changed = copy.deepcopy(self.environments)
            changed["synthetic_job"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(environments=changed)

    def test_invalid_protocol_cannot_create_success_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "protocol.json").write_text(json.dumps(self.protocol), encoding="utf-8")
            (root / "frozen_settings.json").write_text(json.dumps(self.freeze), encoding="utf-8")
            (root / "study_status.json").write_text(json.dumps(dict(status="native_collected",
                jobs=[dict(batch=i, status="passed") for i in (1, 2, 3)])), encoding="utf-8")
            out = root / "analysis"
            with patch.object(sys, "argv", ["summary", "--input", str(root), "--out", str(out)]):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    main()
            self.assertFalse(out.exists())

    def test_controller_failure_or_incomplete_batches_rejected_before_grid_load(self):
        passed = [dict(batch=i, status="passed") for i in (1, 2, 3)]
        bad_states = [dict(status="failed", jobs=passed),
                      dict(status="native_collected", jobs=passed[:2]),
                      dict(status="native_collected", jobs=passed[:2] + [dict(batch=3, status="failed")]),
                      dict(status="native_collected", jobs=passed[:2] + [dict(batch=2, status="passed")])]
        for state in bad_states:
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "study_status.json").write_text(json.dumps(state), encoding="utf-8")
                with patch("summarize_controlled_study.load_grid") as loader:
                    with self.assertRaisesRegex(ValueError, "Study controller"):
                        load_study(root)
                    loader.assert_not_called()

    def test_frozen_state_mismatch_or_missing_copy_rejected_before_grid_load(self):
        for missing in (False, True):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                protocol = dict(self.protocol, frozen_state={"source": "original"})
                protocol_path = root / "protocol.json"
                protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
                freeze = dict(self.freeze, protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest())
                if not missing:
                    freeze["frozen_state"] = {"source": "changed"}
                (root / "frozen_settings.json").write_text(json.dumps(freeze), encoding="utf-8")
                (root / "study_status.json").write_text(json.dumps(dict(status="native_collected",
                    jobs=[dict(batch=i, status="passed") for i in (1, 2, 3)])), encoding="utf-8")
                with patch("summarize_controlled_study.load_grid") as loader:
                    with self.assertRaisesRegex(ValueError, "frozen_state mismatch"):
                        load_study(root)
                    loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
