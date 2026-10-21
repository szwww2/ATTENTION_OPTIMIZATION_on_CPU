"""Plot measured reference envelopes and export explicit DRAM traffic scenarios."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from analyze_results import load_grid, read_csv
from compare_pytorch import save_csv, write_json


def traffic(length, d, dv, full, refill):
    """Cold operands; ordinary output stores; optional K/V refill for each query.

    Full score spills between score/softmax/PV phases: 2S+2S+S.
    Row scratch stays in cache. These are scenarios, not measured bounds.
    """
    kv_visits = length if refill else 1
    operands = 4 * length * d + 4 * length * (d + dv) * kv_visits + 8 * length * dv
    return operands + (20 * length * length if full else 0)


def check_calibration(folder):
    env = json.loads((folder / "environment.json").read_text(encoding="utf-8"))
    if env.get("status") != "passed" or env.get("exit_code") != 0:
        raise ValueError("Failed calibration")
    raw, summary = read_csv(folder / "raw.csv"), read_csv(folder / "summary.csv")
    if len(raw) != 56 or len(summary) != 8:
        raise ValueError("Incomplete calibration")
    expected = {(w, t) for w in ("read_stream", "avx2_mul_add") for t in (1, 2, 4, 8)}
    if {(r["workload"], int(r["threads"])) for r in summary} != expected:
        raise ValueError("Missing workload")
    for row in summary:
        samples = [
            s for s in raw if (s["workload"], s["threads"]) == (row["workload"], row["threads"])
        ]
        if sorted(int(s["sample"]) for s in samples) != list(range(7)):
            raise ValueError("Invalid sample IDs")
        for s in samples:
            if (
                s["actual_threads"] != s["threads"]
                or not math.isfinite(float(s["ms"]))
                or float(s["ms"]) <= 0
            ):
                raise ValueError("Invalid sample")
            expected_work = (
                512 * 1024**2 * 4
                if s["workload"] == "read_stream"
                else 10000000 * 8 * 16 * int(s["threads"])
            )
            if float(s["work_units"]) != expected_work:
                raise ValueError("Invalid work count")
            if not math.isfinite(float(s["checksum"])) or float(s["checksum"]) <= 0:
                raise ValueError("Invalid checksum")
            if s["workload"] == "read_stream" and float(s["checksum"]) != 512 * 1024**2:
                raise ValueError("Invalid read checksum")
            if not math.isclose(
                float(s["units_per_second"]), expected_work / float(s["ms"]) * 1000, rel_tol=1e-12
            ):
                raise ValueError("Invalid sample rate")
        times = [float(s["ms"]) for s in samples]
        q1, _, q3 = statistics.quantiles(times, n=4, method="inclusive")
        med = statistics.median(times)
        for field, value in dict(
            median_ms=med, q1_ms=q1, q3_ms=q3, giga_units_per_second=expected_work / med / 1e6
        ).items():
            if not math.isclose(float(row[field]), value, rel_tol=1e-12):
                raise ValueError("Invalid summary")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new output directory")
    calibration = check_calibration(args.calibration)
    native, environment, _, hashes = load_grid(args.grid)
    calibration_env = json.loads(
        (args.calibration / "environment.json").read_text(encoding="utf-8")
    )
    if calibration_env["cpu"] != environment["cpu"]:
        raise ValueError("Calibration and native grid describe different CPUs")
    # Limit the score-spill illustration to the largest proposal shape.
    selected = [
        r for r in native if r["L"] == 4096 and r["d"] == r["dv"] == 64 and r["mode"] != "scalar"
    ]
    if not selected:
        raise ValueError("No available L4096 SIMD/OpenMP cases")
    rows = []
    for r in selected:
        flops = 2 * r["L"] ** 2 * (r["d"] + r["dv"])
        for refill in (False, True):
            count = traffic(r["L"], r["d"], r["dv"], r["route"] == "full_matrix", refill)
            rows.append(
                dict(
                    route=r["route"],
                    mode=r["mode"],
                    L=r["L"],
                    d=r["d"],
                    dv=r["dv"],
                    threads=r["threads"],
                    scenario=(
                        "kv_refill_per_query_score_spill"
                        if refill
                        else "operands_once_cold_score_spill"
                    ),
                    F_mat=flops,
                    native_median_ms=r["median_ms"],
                    effective_gflops=flops / r["median_ms"] / 1e6,
                    dram_bytes_model=count,
                    arithmetic_intensity_model=flops / count,
                    dram_bytes_counter="",
                    traffic_evidence="model_assumption",
                    assumptions=("Q once; K/V per query" if refill else "Q/K/V once")
                    + "; cold ordinary O stores incl RFO; full score 5S; row scratch cached; scenarios are not bounds",
                )
            )
    args.out.mkdir(parents=True)
    save_csv(args.out / "traffic_scenarios.csv", rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    intensity = np.logspace(-2, 3, 200)
    for t in (1, 2, 4, 8):
        get = lambda w: float(
            next(
                r["giga_units_per_second"]
                for r in calibration
                if r["workload"] == w and int(r["threads"]) == t
            )
        )
        axes[0].loglog(
            intensity,
            np.minimum(get("avx2_mul_add"), get("read_stream") * intensity),
            label=f"T={t}",
        )
    axes[0].set(
        xlabel="Assumed arithmetic intensity (FLOP/byte)",
        ylabel="Reference GFLOP/s",
        title="Read-stream reference envelopes",
    )
    axes[0].legend()
    for route, label in (("full_matrix", "Full matrix"), ("rowwise", "Rowwise")):
        values = sorted(
            (r for r in selected if r["route"] == route and r["mode"] == "openmp"),
            key=lambda r: r["threads"],
        )
        axes[1].plot(
            [r["threads"] for r in values], [r["gflops"] for r in values], "o-", label=label
        )
    axes[1].set(
        xlabel="OpenMP threads",
        ylabel="Measured effective GFLOP/s",
        title="Attention: L=4096, d=dv=64",
        xticks=[1, 2, 4, 8],
    )
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Calibration and attention throughput — separate measurements", fontsize=15)
    fig.savefig(args.out / "roofline_reference.png", dpi=160)
    plt.close(fig)
    write_json(
        args.out / "analysis.json",
        dict(
            status="passed",
            native_environment=environment,
            input_sha256=hashes,
            calibration_sha256={
                f: hashlib.sha256((args.calibration / f).read_bytes()).hexdigest()
                for f in ("raw.csv", "summary.csv", "environment.json")
            },
            analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            interpretation="No measured DRAM intensities, automatic bottleneck classification or percentage-of-peak claims",
        ),
    )
    (args.out / "README.md").write_text(
        "# Machine reference and attention throughput\n\n"
        "![Reference curves and measured throughput](roofline_reference.png)\n\n"
        "Left: measured AVX2 mul+add and read-stream reference curves. Right: native attention throughput "
        "from the selected grid. The panels use separate experiments; these are not verified hardware limits.\n\n"
        "`traffic_scenarios.csv` contains two assumed DRAM traffic models for L=4096, d=dv=64. "
        "Both assume cold ordinary output stores (including RFO), full score spilling with 5S traffic, "
        "and row scratch retained in cache. They differ in whether K/V arrive once or per query. "
        "These scenarios are not measured lower/upper bounds. Actual DRAM bytes are unknown.\n\n"
        "See [model and calibration details]("
        + Path(
            os.path.relpath(
                Path(__file__).resolve().parent.parent / "docs/ROOFLINE_REFERENCE.md", args.out
            )
        ).as_posix()
        + ").\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
