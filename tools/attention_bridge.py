"""NumPy adapter for the existing C++ inputs, FP64 oracle and attention routes."""

import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FP32 = ct.POINTER(ct.c_float)
FP64 = ct.POINTER(ct.c_double)
SIZE = ct.c_size_t


def check_output(output, reference):
    """Apply the project's unchanged elementwise FP64 contract; fail closed."""
    if (
        output.shape != reference.shape
        or output.dtype != np.float32
        or reference.dtype != np.float64
    ):
        raise ValueError("Output/reference shape or dtype mismatch")
    if not np.isfinite(output).all() or not np.isfinite(reference).all():
        raise ValueError("Non-finite output/reference")
    error = np.abs(output.astype(np.float64) - reference)
    margin = error - (1e-4 + 1e-3 * np.abs(reference))
    if np.any(margin > 0):
        raise ValueError("FP64 correctness check failed")
    return dict(
        max_abs=float(error.max()),
        max_rel=float((error / np.maximum(1e-8, np.abs(reference))).max()),
        worst_margin=float(margin.max()),
    )


class Bridge:
    def __init__(self):
        folder = ROOT / "build/python_bridge"
        self.build = json.loads((folder / "build.json").read_text(encoding="utf-8"))
        if self.build["status"] != "passed":
            raise ValueError("Build the Python bridge successfully first")
        library = folder / self.build["library"]
        if hashlib.sha256(library.read_bytes()).hexdigest() != self.build["binary_sha256"]:
            raise ValueError("Bridge binary differs from build record")
        for name, digest in self.build["source_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Bridge sources changed; rebuild first: " + name)
        # Some Windows GCC distributions keep runtime DLLs beside g++.
        # Keep the directory handle alive for the lifetime of this adapter.
        self.dll_directory = None
        compiler = shutil.which(os.environ.get("CXX", "g++"))
        if os.name == "nt" and compiler:
            self.dll_directory = os.add_dll_directory(str(Path(compiler).resolve().parent))
        self.lib = ct.CDLL(str(library))
        self.lib.gp8_fill.argtypes = [ct.c_int] * 3 + [ct.c_uint64] + [FP32] * 3 + [SIZE] * 2
        self.lib.gp8_oracle.argtypes = [ct.c_int] * 3 + [FP32] * 3 + [FP64] + [SIZE] * 3
        self.lib.gp8_run.argtypes = [ct.c_int] * 6 + [FP32] * 4 + [SIZE] * 3
        self.lib.gp8_threads.argtypes = [ct.c_int]
        for name in ("gp8_fill", "gp8_oracle", "gp8_run", "gp8_threads"):
            getattr(self.lib, name).restype = ct.c_int

    @staticmethod
    def checked(code):
        if code:
            raise RuntimeError(f"C++ bridge returned error {code}")

    def inputs(self, length, d, dv, seed):
        if not (1 <= length <= 8192 and 1 <= d <= 4096 and 1 <= dv <= 4096 and 0 <= seed < 2**64):
            raise ValueError("Input dimensions/seed out of range")
        q = np.empty((length, d), np.float32)
        k = np.empty_like(q)
        v = np.empty((length, dv), np.float32)
        self.checked(
            self.lib.gp8_fill(
                length, d, dv, seed, *(a.ctypes.data_as(FP32) for a in (q, k, v)), q.size, v.size
            )
        )
        return q, k, v

    def prepare(self, q, k, v):
        if any(a.dtype != np.float32 or a.ndim != 2 or not a.flags.c_contiguous for a in (q, k, v)):
            raise ValueError("Expected contiguous two-dimensional FP32 inputs")
        if q.shape != k.shape or q.shape[0] != v.shape[0]:
            raise ValueError("Incompatible Q/K/V shapes")
        self.shape = (q.shape[0], q.shape[1], v.shape[1])
        self.arrays = (q, k, v)  # Keep input buffers alive for the C pointers.
        self.pointers = tuple(a.ctypes.data_as(FP32) for a in self.arrays)
        self.nq, self.nv = q.size, v.size
        ref = np.empty(v.shape, np.float64)
        self.checked(
            self.lib.gp8_oracle(
                *self.shape, *self.pointers, ref.ctypes.data_as(FP64), self.nq, self.nv, ref.size
            )
        )
        return ref

    def run(self, route, bq=16, threads=1):
        # Both output allocation and the C++ workspace lifetime are inside the call.
        output = np.empty((self.shape[0], self.shape[2]), np.float32)
        self.checked(
            self.lib.gp8_run(
                route,
                *self.shape,
                bq,
                threads,
                *self.pointers,
                output.ctypes.data_as(FP32),
                self.nq,
                self.nv,
                output.size,
            )
        )
        return output
