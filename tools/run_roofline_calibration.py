"""Build and measure AVX2 arithmetic/read-stream reference workloads."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

from build_integrated import ROOT, BASE_FLAGS, CXX
from compare_pytorch import save_csv, write_json
from run_scalar_bench import cpu_information
from runtime_conditions import apply_cpu_affinity, capture_conditions, parse_cpu_list
import platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cpu-order", help="use first T listed processor IDs per calibration workload")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        parser.error("Use a new --out directory")
    out.mkdir(parents=True)
    env = dict(
        status="running",
        started_utc=datetime.now(timezone.utc).isoformat(),
        cpu=cpu_information(),
        os=platform.platform(),
        affinity="explicit CPU list per job" if args.cpu_order else "not pinned",
        cpu_order=parse_cpu_list(args.cpu_order) if args.cpu_order else None,
        runtime_conditions_by_threads={},
        power_mode="not changed by runner",
        buffer_bytes=512 * 1024**2,
        interpretation="read payload and independent AVX2 mul+add references, not strict attention performance bounds",
        warmup=2,
        reps=7,
        compute_iterations=10000000,
        read_passes=4,
        timing_scope="workload plus OpenMP launch/reduction; data initialization excluded",
        arithmetic_check="0, 1 and 17 AVX2 iterations checked against scalar FP32 recurrence before timing",
        openmp_environment={
            k: os.environ.get(k)
            for k in ("OMP_WAIT_POLICY", "OMP_PROC_BIND", "OMP_PLACES", "OMP_THREAD_LIMIT")
        },
    )
    try:
        source = ROOT / "tools/calibrate_roofline.cpp"
        binary = (
            ROOT
            / "build"
            / ("roofline_calibration.exe" if os.name == "nt" else "roofline_calibration")
        )
        binary.parent.mkdir(exist_ok=True)
        subprocess.run(
            [CXX, *BASE_FLAGS, "-fopenmp", str(source), "-o", str(binary)], cwd=ROOT, check=True
        )
        env.update(
            compiler=subprocess.check_output([CXX, "--version"], text=True).splitlines()[0],
            flags=[*BASE_FLAGS, "-fopenmp"],
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            source_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            source_dirty=bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True
                ).strip()
            ),
        )
        raw = []
        for threads in (1, 2, 4, 8):
            if args.cpu_order:
                cpus=parse_cpu_list(args.cpu_order)
                if len(cpus)<threads:raise ValueError("Insufficient CPUs in calibration order")
                apply_cpu_affinity(",".join(map(str,cpus[:threads])))
            before=capture_conditions()
            result = subprocess.run(
                [str(binary), str(threads)], capture_output=True, text=True, check=True
            )
            rows = list(csv.DictReader(io.StringIO(result.stdout)))
            after=capture_conditions()
            if before != after:raise ValueError("Calibration runtime conditions changed")
            env["runtime_conditions_by_threads"][str(threads)] = before
            if len(rows) != 14:
                raise ValueError("Unexpected sample count")
            for row in rows:
                if int(row["actual_threads"]) != threads:
                    raise ValueError("Wrong actual thread count")
                ms, work, rate = map(float, (row["ms"], row["work_units"], row["units_per_second"]))
                if not all(math.isfinite(x) and x > 0 for x in (ms, work, rate)):
                    raise ValueError("Invalid sample")
                if not math.isclose(rate, work / (ms / 1000), rel_tol=1e-12):
                    raise ValueError("Invalid rate")
            raw.extend(rows)
        summary = []
        for threads in (1, 2, 4, 8):
            for workload in ("avx2_mul_add", "read_stream"):
                rows = [
                    r for r in raw if int(r["threads"]) == threads and r["workload"] == workload
                ]
                if sorted(int(r["sample"]) for r in rows) != list(range(7)):
                    raise ValueError("Missing/duplicate sample")
                times = [float(r["ms"]) for r in rows]
                q1, _, q3 = statistics.quantiles(times, n=4, method="inclusive")
                median = statistics.median(times)
                summary.append(
                    dict(
                        workload=workload,
                        threads=threads,
                        samples=7,
                        median_ms=median,
                        q1_ms=q1,
                        q3_ms=q3,
                        giga_units_per_second=float(rows[0]["work_units"]) / median / 1e6,
                        unit="GFLOP/s" if workload == "avx2_mul_add" else "GB/s payload",
                    )
                )
        save_csv(out / "raw.csv", raw)
        save_csv(out / "summary.csv", summary)
        env.update(status="passed", exit_code=0)
    except Exception as error:
        env.update(status="failed", exit_code=1, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        env["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "environment.json", env)


if __name__ == "__main__":
    main()
