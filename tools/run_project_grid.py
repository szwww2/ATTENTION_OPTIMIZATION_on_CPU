"""Run the available proposal experiment grid and record missing routes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
MIB = 1024 * 1024


def bounded(low, high):
    def parse(text):
        value = int(text)
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"expected {low}..{high}")
        return value
    return parse


def memory_estimate(length, width, block, threads, mode, reps):
    """Conservative planning allowance, not a measured or guaranteed RSS bound."""
    variants = 4 if mode == "serial" else 2
    bq = min(block, length)
    # Q, K, V, spare output, FP64 reference, and the oracle's score row.
    tensors = 24 * length * width + 8 * length
    scratch = (2 * length * length + 2 * bq * length) if mode == "serial" else (
        length * length + threads * bq * length)
    route_buffers = 4 * (scratch + variants * length * width) + 8 * variants * reps
    reserve = 4 * length * length  # Spare full score matrix allowance.
    return (tensors + route_buffers + reserve) * 5 // 4 + 64 * MIB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-cores", type=bounded(1, 4096), required=True,
                        help="confirmed physical core count, not logical processor count")
    parser.add_argument("--out", required=True, help="new directory within the repository")
    parser.add_argument("--memory-mib", type=bounded(1, 1048576), default=512)
    parser.add_argument("--query-block", type=bounded(1, 4096), default=16)
    parser.add_argument("--warmup", type=bounded(1, 1000), default=10)
    parser.add_argument("--reps", type=bounded(3, 1000), default=20)
    parser.add_argument("--seed", type=bounded(0, 2**64 - 1), default=20261020)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    try:
        relative_out = out.relative_to(ROOT)
    except ValueError:
        parser.error("--out must be within the repository")
    if out.exists():
        parser.error("Output directory already exists; choose a new --out")

    threads = sorted({t for t in (1, 2, 4, min(8, args.physical_cores)) if t <= args.physical_cores})
    jobs = []
    missing = []
    shapes = [(128, 64), (512, 64), (2048, 64), (4096, 64), (512, 32), (2048, 32)]
    planned = [(length, width, "serial", 1) for length, width in shapes]
    planned += [(length, 64, "openmp", t) for length in (128, 4096) for t in threads]
    for length, width, mode, workers in planned:
        name = f"{mode}_L{length}_d{width}_T{workers}"
        estimate = memory_estimate(length, width, args.query_block, workers, mode, args.reps)
        job = dict(id=name, L=length, d=width, dv=width, mode=mode, threads=workers,
                   variants=4 if mode == "serial" else 2, estimated_bytes=estimate,
                   prefix=(relative_out / name).as_posix(), status="planned")
        if estimate > args.memory_mib * MIB:
            job.update(status="unavailable", reason="memory planning allowance exceeds budget")
        jobs.append(job)
        missing.append(dict(L=length, d=width, dv=width, mode=mode, threads=workers,
                            route="blocked_streaming", variants=2 if mode == "serial" else 1,
                            status="unavailable", reason="not integrated"))

    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="planned",
                    dry_run=args.dry_run, physical_cores=args.physical_cores, thread_grid=threads,
                    query_block=args.query_block, block_setting="fixed for this run; not a blocked-route tuning result",
                    warmup=args.warmup, reps=args.reps, seed=args.seed, memory_budget_mib=args.memory_mib,
                    memory_estimate="buffers + one spare L*L FP32 matrix, 25% margin, 64 MiB reserve; not measured RSS",
                    jobs=jobs, unavailable_routes=missing)
    out.mkdir(parents=True)
    manifest_path = out / "manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    save()
    print(f"Plan: {len(jobs)} runs; {sum(j['variants'] for j in jobs if j['status'] == 'planned')} runnable configurations; "
          f"{sum(j['variants'] for j in missing)} blocked-route configurations not integrated.", flush=True)
    if args.dry_run:
        return 0
    manifest["status"] = "running"
    for job in jobs:
        if job["status"] == "unavailable":
            continue
        job["status"] = "running"
        save()
        command = [sys.executable, "tools/run_scalar_bench.py", "--suite", "optimized", "--out", job["prefix"],
                   "--mode", job["mode"], "--threads", str(job["threads"]), "--L", str(job["L"]),
                   "--d", str(job["d"]), "--dv", str(job["dv"]), "--query-block", str(args.query_block),
                   "--warmup", str(args.warmup), "--reps", str(args.reps), "--seed", str(args.seed)]
        print("Run", job["id"], flush=True)
        try:
            result = subprocess.run(command, cwd=ROOT)
            job["exit_code"] = result.returncode
        except OSError as error:
            job.update(exit_code=1, error_type=type(error).__name__)
        job["status"] = "passed" if job["exit_code"] == 0 else "failed"
        if job["exit_code"]:
            manifest["status"] = "failed"
            save()
            print("Stopped after failure:", job["id"], file=sys.stderr)
            return 1
        save()
    manifest["status"] = "completed_available_runs"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
