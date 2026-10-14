#!/usr/bin/env python3
"""Recompute the C++ results with NumPy.

1. splitmix64: q/k/v must be bit-identical to the dumps
2. float64 must match the FP64 oracle
3. float32 (BLAS, different summation order) must still be inside the tolerance

Usage: python tools/crosscheck_numpy.py <prefix> <L> <d> <dv> <seed>
"""
import sys

import numpy as np

MASK = (1 << 64) - 1


def splitmix64_stream(seed, n):
    """Bit-exact reimplementation of attn::SplitMix64 (see include/attn.h)."""
    state = seed & MASK
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        state = (state + 0x9E3779B97F4A7C15) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        z = z ^ (z >> 31)
        out[i] = (z >> 11) * (1.0 / 9007199254740992.0)
    return out


def make_tensor(n, seed, lo=-1.0, hi=1.0):
    u = splitmix64_stream(seed, n)
    return (lo + (hi - lo) * u).astype(np.float32)


def main():
    prefix, L, d, dv, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    q = np.loadtxt(f"{prefix}_q.txt", dtype=np.float32).reshape(L, d)
    k = np.loadtxt(f"{prefix}_k.txt", dtype=np.float32).reshape(L, d)
    v = np.loadtxt(f"{prefix}_v.txt", dtype=np.float32).reshape(L, dv)
    o_ref = np.loadtxt(f"{prefix}_oref.txt", dtype=np.float64).reshape(L, dv)
    o32 = np.loadtxt(f"{prefix}_o32.txt", dtype=np.float32).reshape(L, dv)

    problems = []

    # 1. generator
    q2 = make_tensor(L * d, seed * 3 + 1)
    k2 = make_tensor(L * d, seed * 3 + 2)
    v2 = make_tensor(L * dv, seed * 3 + 3)
    gen_ok = (np.array_equal(q.ravel(), q2) and np.array_equal(k.ravel(), k2)
              and np.array_equal(v.ravel(), v2))
    print(f"[1] splitmix64 reproduction        : {'OK' if gen_ok else 'MISMATCH'} "
          f"(q/k/v {'bit-identical' if gen_ok else 'differ'})")
    if not gen_ok:
        problems.append("input generator mismatch")

    # 2. FP64 oracle
    qd, kd, vd = q.astype(np.float64), k.astype(np.float64), v.astype(np.float64)
    s = (qd @ kd.T) / np.sqrt(d)
    s -= s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=1, keepdims=True)
    ref_np = p @ vd
    oracle_diff = np.max(np.abs(ref_np - o_ref))
    oracle_ok = oracle_diff < 1e-12 * max(1.0, np.max(np.abs(ref_np)))
    print(f"[2] NumPy FP64 vs C++ FP64 oracle  : {'OK' if oracle_ok else 'MISMATCH'} "
          f"(max |diff| = {oracle_diff:.3e})")
    if not oracle_ok:
        problems.append("oracle disagreement")

    # 3. NumPy FP32 inside the tolerance?
    s32 = (q @ k.T) / np.float32(np.sqrt(d))
    s32 = s32 - s32.max(axis=1, keepdims=True)
    p32 = np.exp(s32, dtype=np.float32)
    p32 = p32 / p32.sum(axis=1, keepdims=True, dtype=np.float32)
    o_np32 = (p32 @ v).astype(np.float32)
    diff = np.abs(o_np32.astype(np.float64) - ref_np)
    margin = diff - (1e-4 + 1e-3 * np.abs(ref_np))
    np32_ok = bool(np.all(np.isfinite(o_np32)) and np.max(margin) <= 0)
    print(f"[3] NumPy FP32 vs contract         : {'OK' if np32_ok else 'FAIL'} "
          f"(max_abs = {diff.max():.3e}, worst_margin = {margin.max():.3e})")
    if not np32_ok:
        problems.append("numpy fp32 outside contract")

    diff_cpp = np.abs(o32.astype(np.float64) - ref_np)
    print(f"    (info) C++ FP32 max_abs = {diff_cpp.max():.3e}, "
          f"NumPy FP32 max_abs = {diff.max():.3e}, "
          f"|C++ - NumPy| max = {np.max(np.abs(o32.astype(np.float64) - o_np32.astype(np.float64))):.3e}")

    if problems:
        print("CROSS-CHECK FAILED: " + "; ".join(problems))
        return 1
    print("CROSS-CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
