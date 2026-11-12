"""Compare allocating CPU APIs on shared inputs; keep separate from native timings."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parent.parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def measure(calls, reference, warmup, reps, validate, clock=perf_counter_ns):
    """Check first, rotate call order, measure, then check every final output."""
    names = list(calls)
    errors = {}
    for name in names:
        result = calls[name]()
        errors[name] = validate(result, reference)
        del result
    for sample in range(warmup):
        for pos in range(len(names)):
            result = calls[names[(sample + pos) % len(names)]]()
            del result
    raw, last = [], {}
    for sample in range(reps):
        for pos in range(len(names)):
            name = names[(sample + pos) % len(names)]
            start = clock()
            result = calls[name]()
            elapsed = (clock() - start) / 1e6
            if not 0 < elapsed < float("inf"):
                raise ValueError("Invalid elapsed time")
            # Replacing/releasing an earlier output happens after the timer.
            last[name] = result
            del result
            raw.append(dict(variant=name, sample=sample, order=pos, ms=elapsed))
    for name in names:
        post = validate(last[name], reference)
        errors[name] = {k: max(v, post[k]) for k, v in errors[name].items()}
    summary = []
    for name in names:
        values = [r["ms"] for r in raw if r["variant"] == name]
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
        summary.append(
            dict(
                variant=name,
                samples=len(values),
                median_ms=statistics.median(values),
                q1_ms=q1,
                q3_ms=q3,
                iqr_ms=q3 - q1,
                **errors[name]
            )
        )
    return raw, summary


def run_case(args):
    from runtime_conditions import apply_cpu_affinity, capture_conditions

    apply_cpu_affinity(args.cpu_list)
    # Configure pools before importing either runtime. Separate pools still exist.
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import numpy as np
    import torch
    from attention_bridge import Bridge, check_output
    from run_scalar_bench import cpu_information

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    env = dict(
        schema_version=2,
        status="running",
        started_utc=datetime.now(timezone.utc).isoformat(),
        L=args.L,
        d=args.d,
        dv=args.dv,
        threads=args.threads,
        seed=args.seed,
        warmup=args.warmup,
        reps=args.reps,
        requested_query_block=args.query_block,
        requested_key_block=args.key_block,
        query_block=min(args.L, args.query_block),
        key_block=min(args.L, args.key_block),
        timing_scope="allocating CPU API: Python dispatch, output allocation, workspace allocation/reset/release and compute; output release outside timer",
        correctness_scope="precheck and last timed output for each variant; not every sample",
        requested_cpu_list=args.cpu_list,
        affinity="requested CPU list applied" if args.cpu_list else "not changed by runner",
        runtime_conditions=capture_conditions(),
        power_mode="not changed by this runner",
        cpu=cpu_information(),
        os=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        pool_policy={
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OMP_THREAD_LIMIT",
                "OMP_WAIT_POLICY",
                "KMP_BLOCKTIME",
                "OMP_PROC_BIND",
                "OMP_PLACES",
            )
        },
        pool_note="GCC OpenMP and PyTorch runtimes are separate; equal requested thread counts do not imply equal placement",
        torch_version=torch.__version__,
        torch_build=torch.__config__.show(),
        unavailable_routes=[],
        thread_check_scope="blocked: actual compute team checked in every API call; full/row: preflight probe only; PyTorch: requested pool configuration",
    )
    try:
        env["source_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        env["source_dirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        )
        env["runner_sha256"] = {
            f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
            for f in ("tools/compare_pytorch.py", "tools/attention_bridge.py", "tools/runtime_conditions.py")
        }
        bridge = Bridge()
        env["bridge_build"] = bridge.build
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        env["torch_intraop_threads"] = torch.get_num_threads()
        env["torch_interop_threads"] = torch.get_num_interop_threads()
        env["cpp_preflight_threads"] = bridge.lib.gp8_threads(args.threads)
        if (
            env["cpp_preflight_threads"] != args.threads
            or env["torch_intraop_threads"] != args.threads
        ):
            raise ValueError("Thread configuration was not accepted")
        q, k, v = bridge.inputs(args.L, args.d, args.dv, args.seed)
        ref = bridge.prepare(q, k, v)
        input_hash = lambda: {
            name: hashlib.sha256(a.tobytes()).hexdigest()
            for name, a in zip(("Q", "K", "V"), (q, k, v))
        }
        env["input_sha256"] = input_hash()
        qt, kt, vt = [torch.from_numpy(a).view(1, 1, *a.shape) for a in (q, k, v)]
        if any(t.data_ptr() != a.ctypes.data for t, a in zip((qt, kt, vt), (q, k, v))):
            raise ValueError("Torch inputs must share the original FP32 buffers")
        sdpa = lambda: torch.nn.functional.scaled_dot_product_attention(
            qt, kt, vt, dropout_p=0.0, is_causal=False
        )
        route = 2 if args.threads == 1 else 4
        blocked_route = 7 if args.threads == 1 else 8
        mode = "simd" if args.threads == 1 else "simd_omp"
        calls = {
            "full_" + mode: lambda: bridge.run(route, env["query_block"], args.threads),
            "row_" + mode: lambda: bridge.run(route + 1, env["query_block"], args.threads),
            "blocked_" + mode: lambda: bridge.run(blocked_route, env["query_block"], args.threads, env["key_block"]),
            "pytorch_sdpa": sdpa,
        }

        def validate(output, reference):
            if isinstance(output, torch.Tensor):
                output = output.squeeze(0).squeeze(0).numpy()
            return check_output(output, reference)

        with torch.inference_mode():
            # Identify actual CPU operators outside the measurement loop.
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as profiler:
                probe = sdpa()
            validate(probe, ref)
            del probe
            env["torch_sdpa_operators"] = sorted(
                {e.key for e in profiler.key_averages() if "attention" in e.key}
            )
            env["torch_backend_verified"] = any(
                "_scaled_dot_product_" in k for k in env["torch_sdpa_operators"]
            )
            if not env["torch_backend_verified"]:
                raise ValueError("Profiler did not identify the CPU SDPA backend")
            write_json(out / "environment.json", env)
            raw, summary = measure(calls, ref, args.warmup, args.reps, validate)
        env["blocked_actual_threads"] = bridge.last_actual_threads[blocked_route]
        if env["blocked_actual_threads"] != args.threads:
            raise ValueError("Blocked compute team mismatch")
        if input_hash() != env["input_sha256"]:
            raise ValueError("An implementation modified its input")
        env["runtime_conditions_after"] = capture_conditions()
        if env["runtime_conditions_after"] != env["runtime_conditions"]:
            raise ValueError("Runtime conditions changed during API comparison")
        save_csv(out / "raw.csv", raw)
        save_csv(out / "summary.csv", summary)
        env.update(status="passed", exit_code=0)
    except Exception as error:
        env.update(status="failed", exit_code=1, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        env["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "environment.json", env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--L", type=int, default=512)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--dv", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--query-block", type=int, default=16)
    parser.add_argument("--key-block", type=int, default=256)
    parser.add_argument("--cpu-list", default=None, help="Optional allowed CPU indices; inherited by workers")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()
    if not (
        1 <= args.L <= 4096
        and 1 <= args.d <= 128
        and 1 <= args.dv <= 128
        and 1 <= args.threads <= 16
        and 1 <= args.query_block <= 4096
        and 1 <= args.key_block <= 4096
        and 1 <= args.warmup <= 1000
        and 3 <= args.reps <= 1000
        and 0 <= args.seed < 2**64
    ):
        parser.error("Dimensions, threads or sample counts exceed this experiment's bounds")
    if Path(args.out).exists():
        parser.error("Output exists; use a new --out directory")
    run_case(args)
    print("Saved API comparison to", args.out)


if __name__ == "__main__":
    main()
