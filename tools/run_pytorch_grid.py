"""Run six serial CPU API shapes and the two proposal thread-scaling shapes."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
from compare_pytorch import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-cores", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.physical_cores <= 4096 or args.out.exists():
        parser.error("Need a valid core count and new output directory")
    args.out.mkdir(parents=True)
    shapes = [(128, 64), (512, 64), (2048, 64), (4096, 64), (512, 32), (2048, 32)]
    threads = sorted(
        {t for t in (1, 2, 4, min(8, args.physical_cores)) if t <= args.physical_cores}
    )
    jobs = [
        dict(id=f"L{l}_d{d}_T1", L=l, d=d, dv=d, threads=1, status="planned") for l, d in shapes
    ]
    jobs += [
        dict(id=f"L{l}_d64_T{t}", L=l, d=64, dv=64, threads=t, status="planned")
        for l in (128, 4096)
        for t in threads
        if t > 1
    ]
    manifest = dict(
        status="running",
        started_utc=datetime.now(timezone.utc).isoformat(),
        physical_cores=args.physical_cores,
        thread_grid=threads,
        warmup=10,
        reps=20,
        seed=20261020,
        jobs=jobs,
        unavailable_routes=["blocked_streaming"],
        timing="allocating API, separate from native grid",
    )
    path = args.out / "manifest.json"
    write_json(path, manifest)
    for job in jobs:
        job["status"] = "running"
        write_json(path, manifest)
        command = [
            sys.executable,
            "-X",
            "utf8",
            str(ROOT / "tools/compare_pytorch.py"),
            "--out",
            str(args.out.resolve() / job["id"]),
        ]
        for field in ("L", "d", "dv", "threads"):
            command += ["--" + field, str(job[field])]
        with (args.out / (job["id"] + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        job.update(
            status="passed" if result.returncode == 0 else "failed", exit_code=result.returncode
        )
        print(job["id"], job["status"], flush=True)
        if result.returncode:
            manifest["status"] = "failed"
            write_json(path, manifest)
            return result.returncode
        write_json(path, manifest)
    manifest.update(status="passed", finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(path, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
