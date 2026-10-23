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


def traffic_rows(selected):
    """Explicit DRAM scenarios; workspace capacity is evidence for neither traffic nor residency."""
    rows = []
    for r in selected:
        route, length, d, dv = r["route"], r["L"], r["d"], r["dv"]
        if route not in ("full_matrix", "rowwise", "blocked_streaming"):
            raise ValueError("No traffic model for route: " + route)
        blocked = route == "blocked_streaming"
        bq, bk = r.get("query_block", 0), r.get("key_block", 0)
        if blocked and (not 1 <= bq <= length or not 1 <= bk <= length):
            raise ValueError("Blocked model requires effective query/key block sizes")
        flops = 2 * length**2 * (d + dv)
        for refill in (False, True):
            if blocked:
                visits = (length + bq - 1) // bq if refill else 1
                count = 4 * length * d + 4 * length * (d + dv) * visits + 8 * length * dv
                scenario = ("kv_refill_per_query_block_cached_tile" if refill
                            else "operands_once_cold_cached_tile")
                assumptions = ("Q once; K/V per query block" if refill else "Q/K/V once")
                assumptions += "; query block, score tile and online m/l/u remain cached; no DRAM scratch spills"
            else:
                count = traffic(length, d, dv, route == "full_matrix", refill)
                scenario = "kv_refill_per_query_score_spill" if refill else "operands_once_cold_score_spill"
                assumptions = ("Q once; K/V per query" if refill else "Q/K/V once")
                assumptions += "; full score 5S" if route == "full_matrix" else "; row scratch cached"
            rows.append(dict(
                route=route, mode=r["mode"], L=length, d=d, dv=dv, threads=r["threads"],
                query_block=bq, key_block=bk if blocked else 0,
                workspace_bytes_capacity=r["workspace_bytes"],
                online_state_bytes_per_thread=4 * bq * (dv + 2) if blocked else 0,
                scenario=scenario, F_mat=flops, native_median_ms=r["median_ms"],
                effective_gflops=flops / r["median_ms"] / 1e6,
                dram_bytes_model=count, arithmetic_intensity_model=flops / count,
                dram_bytes_counter="", traffic_evidence="model_assumption",
                assumptions=assumptions + "; cold ordinary O stores incl RFO; scenarios are not bounds",
            ))
    return rows


def check_machine_conditions(calibration, native):
    """Check available compatibility evidence, retaining support for legacy calibration records."""
    if calibration["cpu"] != native["cpu"]:
        raise ValueError("Calibration and native grid describe different CPUs")
    if "os" in calibration and calibration["os"] != native["os"]:
        raise ValueError("Calibration and native grid describe different operating systems")
    build = dict(line.split("=", 1) for line in native["build"] if "=" in line)
    if calibration["compiler"] != build.get("compiler"):
        raise ValueError("Calibration and native grid use different compilers")
    expected_flags = (build["base_flags"] + " " + build["openmp_extra_flags"]).split()
    if calibration["flags"] != expected_flags:
        raise ValueError("Calibration and native grid use different base/OpenMP flags")
    scope = ("CPU, OS, compiler and flags" if "os" in calibration
             else "CPU, compiler and flags; legacy calibration has no OS record")
    order = native.get("cpu_order")
    if order or calibration.get("cpu_order"):
        if not order or calibration.get("cpu_order") != order:
            raise ValueError("Calibration and native grid use different ordered CPU policies")
        for threads in (1, 2, 4, 8):
            conditions = calibration.get("runtime_conditions_by_threads", {}).get(str(threads), {})
            if conditions.get("allowed_cpu_ids") != sorted(order[:threads]):
                raise ValueError("Calibration process affinity differs from ordered policy")
            for field in ("ac_line_status", "power_scheme_guid"):
                if field in native and conditions.get(field) != native[field]:
                    raise ValueError("Calibration power conditions differ from native grid")
        scope += "; ordered CPU eligibility and recorded power conditions"
    return scope


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
    parser.add_argument("--thread-scaling-note", default="", help="Annotation for heterogeneous CPU sets")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new output directory")
    calibration = check_calibration(args.calibration)
    native, environment, _, hashes = load_grid(args.grid)
    calibration_env = json.loads(
        (args.calibration / "environment.json").read_text(encoding="utf-8")
    )
    compatibility_scope = check_machine_conditions(calibration_env, environment)
    # Limit the score-spill illustration to the largest proposal shape.
    selected = [
        r for r in native if r["L"] == 4096 and r["d"] == r["dv"] == 64 and r["mode"] != "scalar"
    ]
    if not selected:
        raise ValueError("No available L4096 SIMD/OpenMP cases")
    rows = traffic_rows(selected)
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
    for route, label in (("full_matrix", "Full matrix"), ("rowwise", "Rowwise"),
                         ("blocked_streaming", "Blocked streaming")):
        values = sorted(
            (r for r in selected if r["route"] == route and r["mode"] == "openmp"),
            key=lambda r: r["threads"],
        )
        if not values:
            continue
        regular = [r for r in values if r["threads"] <= 4]
        line, = axes[1].plot(
            [r["threads"] for r in regular], [r["gflops"] for r in regular], "o-", label=label
        )
        separate = [r for r in values if r["threads"] > 4]
        axes[1].plot([r["threads"] for r in separate], [r["gflops"] for r in separate],
                     "D", color=line.get_color())
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
    if args.thread_scaling_note:
        fig.supxlabel(args.thread_scaling_note, fontsize=10)
    fig.savefig(args.out / "roofline_reference.png", dpi=160)
    plt.close(fig)
    # Horizontal alternatives change only assumed bytes, never the measured time.
    fig, panels = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, threads, mode in zip(panels, (1, 4), ("simd", "openmp")):
        rates = {r["workload"]: float(r["giga_units_per_second"])
                 for r in calibration if int(r["threads"]) == threads}
        ax.loglog(intensity, np.minimum(rates["avx2_mul_add"], rates["read_stream"] * intensity),
                  color="#6b7280", linestyle="--", label="Measured reference workloads")
        for route, label in (("full_matrix", "Full"), ("rowwise", "Rowwise"), ("blocked_streaming", "Blocked")):
            points = [r for r in rows if r["route"] == route and r["mode"] == mode and r["threads"] == threads]
            xs = [r["arithmetic_intensity_model"] for r in points]
            ys = [r["effective_gflops"] for r in points]
            ax.plot(xs, ys, "o:", label=label)
        ax.set(xlabel="Assumed FLOP/byte (two scenarios per route)", ylabel="Measured effective GFLOP/s",
               title=f"T={threads}, {mode}; L=4096, d=dv=64")
        ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.suptitle("Model placement: one selected native batch, DRAM bytes not measured", fontsize=14)
    fig.supxlabel("Dotted spans are alternative assumptions, not confidence intervals or measured traffic bounds.", fontsize=10)
    fig.savefig(args.out / "roofline_model_placement.png", dpi=160)
    plt.close(fig)
    write_json(
        args.out / "analysis.json",
        dict(
            status="passed",
            native_environment=environment,
            calibration_environment=calibration_env,
            compatibility_scope=compatibility_scope,
            thread_scaling_note=args.thread_scaling_note,
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
        "![Two assumed traffic scenarios per route](roofline_model_placement.png)\n\n"
        "Left: measured AVX2 mul+add and read-stream reference curves. Right: native attention throughput "
        "from the selected grid. The panels use separate experiments; these are not verified hardware limits.\n\n"
        "`traffic_scenarios.csv` contains two assumed DRAM traffic scenarios per available route for L=4096, d=dv=64. "
        "Both assume cold ordinary output stores (including RFO). Full/row scenarios retain the original "
        "score-spill/cached-row assumptions. Blocked scenarios assume cached query, score and online state; "
        "K/V arrive once or once per query block. Bq/Bk and scratch capacity are recorded separately. "
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
