"""Run the small scalar benchmark and record its environment (standard library only)."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def cpu_information():
    """Detailed Windows metadata is optional; a denied CIM query is not fatal."""
    info = {"description": platform.processor(), "logical_processors": os.cpu_count()}
    if os.name == "nt":
        command = ("Get-CimInstance Win32_Processor | Select-Object "
                   "Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress")
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=8, check=True)
            info["windows_cim"] = json.loads(result.stdout)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            # Keep the reason, never local paths or the shell's potentially private stderr.
            info["windows_cim"] = "unavailable: " + type(error).__name__
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/scalar", help="new output prefix, relative to repository")
    parser.add_argument("--suite", choices=("scalar", "optimized"), default="scalar")
    args, extra = parser.parse_known_args()
    root = Path(__file__).resolve().parent.parent
    prefix = Path(args.out)
    if not prefix.is_absolute():
        prefix = root / prefix
    env_file = Path(str(prefix) + "_environment.json")
    outputs = [env_file, Path(str(prefix) + "_raw.csv"), Path(str(prefix) + "_summary.csv")]
    if any(p.exists() for p in outputs):
        parser.error("Output already exists; choose another --out prefix")
    name = "attn_bench_all" if args.suite == "optimized" else "attn_bench"
    binary = root / "build" / (name + (".exe" if os.name == "nt" else ""))
    build_record = root / "build" / ("integrated_build.txt" if args.suite == "optimized" else "scalar_build.txt")
    if not binary.exists() or not build_record.exists():
        parser.error("Build the selected suite first (build.bat/build.sh or tools/build_integrated.py)")
    build_lines = build_record.read_text(encoding="utf-8").splitlines()
    if args.suite == "optimized" and "status=passed" not in build_lines:
        parser.error("The integrated build did not finish successfully")

    def git(*command):
        return subprocess.check_output(["git", *command], cwd=root, text=True, encoding="utf-8").strip()

    sources = [*root.glob("src/*.cpp"), *root.glob("src/*.h"), *root.glob("include/*.h"),
               root / "tools/bench_scalar.cpp", root / "tools/run_scalar_bench.py",
               root / "tools/build_integrated.py", root / "tools/run_project_grid.py", root / "build.bat", root / "build.sh"]
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": git("rev-parse", "HEAD"), "source_dirty": bool(git("status", "--porcelain")),
        "source_sha256": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "build": build_lines, "suite": args.suite, "benchmark_arguments": extra,
        "os": platform.platform(), "cpu": cpu_information(), "threads": 1 if args.suite == "scalar" else "per variant",
        "openmp_environment": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY") if k in os.environ},
        "affinity": "not pinned by the harness", "power_mode": "not controlled by the harness",
        "timing_scope": "complete native kernel including resets; allocation, inputs, oracle, checks and I/O excluded",
        "quartiles": "inclusive linear interpolation at p*(n-1)", "status": "running",
    }
    prefix.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    # Relative paths also work with Windows C++ runtimes using a narrow main().
    run = subprocess.run([str(binary), "--out", os.path.relpath(prefix, root), *extra], cwd=root)
    if run.returncode == 0:
        with outputs[2].open(newline="", encoding="utf-8") as stream:
            metadata["variant_threads"] = [
                {key: row[key] for key in ("route", "mode", "threads", "actual_threads")}
                for row in csv.DictReader(stream)]
        metadata["thread_check_scope"] = "OpenMP preflight team probe; dynamic teams disabled; timed calls not instrumented"
    metadata.update(finished_utc=datetime.now(timezone.utc).isoformat(), exit_code=run.returncode,
                    status="passed" if run.returncode == 0 else "failed")
    env_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return run.returncode


if __name__ == "__main__":
    sys.exit(main())
