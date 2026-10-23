"""Run six serial CPU API shapes and the two proposal thread-scaling shapes."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from compare_pytorch import ROOT, write_json
from runtime_conditions import parse_cpu_list


def study_settings(path, query_block, key_block, cpu_order, physical_cores):
    """Reuse the native study's frozen blocks and CPU plan, rejecting overrides."""
    frozen = json.loads(path.read_text(encoding="utf-8"))
    protocol_bytes = path.with_name("protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    if (frozen.get("status") != "frozen" or
        hashlib.sha256(protocol_bytes).hexdigest() != frozen.get("protocol_sha256")):
        raise ValueError("Invalid frozen settings or changed study protocol")
    bq, bk = protocol["tuning"]["query_block"], frozen["selected_key_block"]
    if (query_block is not None and query_block != bq or
        key_block is not None and key_block != bk):
        raise ValueError("Block override differs from the frozen study")
    if cpu_order != protocol["cpu_order"] or physical_cores != protocol["physical_cores"]:
        raise ValueError("CPU plan differs from the frozen study")
    return bq, bk, dict(frozen_settings_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       source_revision=frozen["source_revision"],
                       protocol_sha256=frozen["protocol_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-cores", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--query-block", type=int, default=None)
    parser.add_argument("--key-block", type=int, default=None)
    parser.add_argument("--frozen-settings", type=Path, help="Reuse a controlled native study's block sizes and CPU plan")
    affinity = parser.add_mutually_exclusive_group()
    affinity.add_argument("--cpu-list", default=None)
    affinity.add_argument("--cpu-order", default=None, help="Use first T CPU IDs for each thread count")
    parser.add_argument("--thread-scaling-note", default="", help="Recorded annotation for heterogeneous CPU sets")
    args = parser.parse_args()
    cpu_order = parse_cpu_list(args.cpu_order) if args.cpu_order else None
    study = None
    if args.frozen_settings:
        try:
            args.query_block, args.key_block, study = study_settings(
                args.frozen_settings, args.query_block, args.key_block, cpu_order, args.physical_cores)
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if revision != study["source_revision"]:
            parser.error("Current revision differs from the frozen study")
    args.query_block = 16 if args.query_block is None else args.query_block
    args.key_block = 256 if args.key_block is None else args.key_block
    if (not 1 <= args.physical_cores <= 4096 or args.out.exists()
        or not 1 <= args.query_block <= 4096 or not 1 <= args.key_block <= 4096):
        parser.error("Need a valid core count and new output directory")
    shapes = [(128, 64), (512, 64), (2048, 64), (4096, 64), (512, 32), (2048, 32)]
    threads = sorted(
        {t for t in (1, 2, 4, min(8, args.physical_cores)) if t <= args.physical_cores}
    )
    if cpu_order and len(cpu_order) < max(threads):
        parser.error("CPU order is shorter than the maximum thread count")
    jobs = [
        dict(id=f"L{l}_d{d}_T1", L=l, d=d, dv=d, threads=1, status="planned") for l, d in shapes
    ]
    jobs += [
        dict(id=f"L{l}_d64_T{t}", L=l, d=64, dv=64, threads=t, status="planned")
        for l in (128, 4096)
        for t in threads
        if t > 1
    ]
    if cpu_order:
        for job in jobs:
            job["cpu_ids"] = cpu_order[:job["threads"]]
    manifest = dict(
        schema_version=2,
        status="running",
        started_utc=datetime.now(timezone.utc).isoformat(),
        physical_cores=args.physical_cores,
        thread_grid=threads,
        warmup=10,
        reps=20,
        seed=20261020,
        jobs=jobs,
        unavailable_routes=[],
        query_block=args.query_block,
        key_block=args.key_block,
        native_study=study,
        requested_cpu_list=args.cpu_list,
        thread_scaling_note=args.thread_scaling_note,
        timing="allocating API, separate from native grid",
    )
    if cpu_order:
        manifest["cpu_order"] = cpu_order
    args.out.mkdir(parents=True)
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
        command += ["--query-block", str(args.query_block), "--key-block", str(args.key_block)]
        if cpu_order:
            command += ["--cpu-list", ",".join(map(str, job["cpu_ids"]))]
        elif args.cpu_list is not None:
            command += ["--cpu-list", args.cpu_list]
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
