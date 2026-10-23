"""Small correctness/lifecycle tests; no performance assertions."""

import ctypes as ct
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import numpy as np
from attention_bridge import Bridge, FP32, check_output
from compare_pytorch import measure


class MeasurementTests(unittest.TestCase):
    def test_order_count_quantiles(self):
        counts = {"a": 0, "b": 0, "c": 0}

        def call(name):
            counts[name] += 1
            return np.zeros((1, 1), np.float32)

        clock = iter(range(1000))
        raw, summary = measure(
            {k: lambda k=k: call(k) for k in counts},
            np.zeros((1, 1)),
            2,
            4,
            check_output,
            lambda: next(clock) * 1000000,
        )
        self.assertEqual(counts, dict.fromkeys(counts, 7))  # precheck + warmup + timed
        self.assertEqual([r["variant"] for r in raw], list("abcbcac ababc".replace(" ", "")))
        self.assertTrue(
            all(r["samples"] == 4 and r["median_ms"] == 1 and r["iqr_ms"] == 0 for r in summary)
        )

    def test_corrupt_outputs_fail_before_summary(self):
        for value in (float("nan"), float("inf"), 1.0):
            for corrupt_at in (1, 6):  # precheck, final timed output
                count = [0]

                def call():
                    count[0] += 1
                    return np.array([[value if count[0] == corrupt_at else 0]], dtype=np.float32)

                with self.assertRaises(ValueError):
                    measure({"a": call}, np.zeros((1, 1)), 2, 3, check_output)


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = Bridge()

    def test_all_routes_and_tails(self):
        for shape in ((1, 1, 1), (3, 7, 5), (7, 31, 33), (17, 32, 64)):
            q, k, v = self.bridge.inputs(*shape, 20261020)
            reference = self.bridge.prepare(q, k, v)
            for route in range(9):
                threads = 2 if route in (4, 5, 8) else 1
                self.assertEqual(self.bridge.lib.gp8_threads(threads), threads)
                a = self.bridge.run(route, min(shape[0], 16), threads, min(shape[0], 4))
                b = self.bridge.run(route, min(shape[0], 16), threads, min(shape[0], 4))
                self.assertFalse(np.shares_memory(a, b))
                check_output(a, reference)
                check_output(b, reference)
                if route in (6, 7, 8):
                    self.assertEqual(self.bridge.last_actual_threads[route], threads)

    def test_seed_and_uniform_attention(self):
        first = self.bridge.inputs(3, 7, 5, 20261020)
        again = self.bridge.inputs(3, 7, 5, 20261020)
        other = self.bridge.inputs(3, 7, 5, 20261021)
        for a, b, c in zip(first, again, other):
            np.testing.assert_array_equal(a, b)
            self.assertFalse(np.array_equal(a, c))
        q, k, v = first
        q.fill(0)
        ref = self.bridge.prepare(q, k, v)
        np.testing.assert_allclose(
            ref, np.broadcast_to(v.astype(np.float64).mean(axis=0), v.shape), atol=1e-14
        )
        for route in range(9):
            check_output(self.bridge.run(route, 3, 1, 2), ref)

    def test_reject_invalid_arguments(self):
        q, k, v = self.bridge.inputs(3, 7, 5, 1)
        self.bridge.prepare(q, k, v)
        for route, bq, threads in (
            (9, 1, 1),
            (0, 0, 1),
            (0, 4, 1),
            (0, 1, 2),
            (4, 1, 0),
            (4, 1, 65),
        ):
            with self.assertRaises(RuntimeError):
                self.bridge.run(route, bq, threads)
        p = q.ctypes.data_as(FP32)
        self.assertEqual(self.bridge.lib.gp8_run(0, 3, 7, 5, 1, 1, p, p, p, p, 21, 15, 0), 1)
        self.assertEqual(
            self.bridge.lib.gp8_run(0, 2**31 - 1, 7, 5, 1, 1, p, p, p, p, 21, 15, 15), 1
        )
        with self.assertRaises(ValueError):
            self.bridge.prepare(q[:, ::2], k, v)

    def test_blocked_argument_rejection(self):
        self.bridge.prepare(*self.bridge.inputs(3, 7, 5, 1))
        for route, bq, bk, threads in ((6, 1, 0, 1), (7, 1, 4, 1),
                                       (8, 0, 1, 2), (7, 1, 1, 2), (8, 1, 1, 0)):
            with self.assertRaises(RuntimeError):
                self.bridge.run(route, bq, threads, bk)

    def test_torch_same_inputs(self):
        import torch

        torch.set_num_threads(1)
        for shape in ((1, 1, 1), (3, 7, 5), (7, 31, 33), (17, 32, 64)):
            arrays = self.bridge.inputs(*shape, 9)
            ref = self.bridge.prepare(*arrays)
            tensors = [torch.from_numpy(a).view(1, 1, *a.shape) for a in arrays]
            for tensor, array in zip(tensors, arrays):
                self.assertEqual(tensor.data_ptr(), array.ctypes.data)
            with torch.inference_mode():
                result = torch.nn.functional.scaled_dot_product_attention(
                    *tensors, dropout_p=0, is_causal=False
                )
            check_output(result.numpy().reshape(ref.shape), ref)


if __name__ == "__main__":
    unittest.main()
