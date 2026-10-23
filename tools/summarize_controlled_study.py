"""Validate three frozen native batches and summarize within-batch paired speedups."""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

from analyze_results import BLOCK_FIELDS, key, load_grid
from analyze_repeats import check_repeat_evidence
from compare_pytorch import save_csv, write_json


ROUTES = ("full_matrix", "rowwise", "blocked_streaming")
LABELS = ("Full matrix", "Rowwise", "Blocked streaming")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None, "Timestamp needs a timezone")
    return parsed


def check_binding(protocol, freeze, protocol_sha, manifest, environments, batch):
    """Bind every saved job to the plan frozen before the native measurements."""
    require(freeze.get("status") == "frozen", "Settings were not frozen")
    require(freeze.get("protocol_sha256") == protocol_sha, "Protocol SHA256 differs from freeze")
    for field in ("source_revision", "native_binary_sha256", "cpu_order"):
        require(freeze.get(field) == protocol.get(field), "Protocol/freeze mismatch: " + field)
    require(protocol["main"]["batches"] == 3 and protocol["main"]["configurations_per_batch"] == 60,
            "This summary requires the complete three-batch, 60-configuration plan")
    require(manifest.get("streaming") is True and not manifest["unavailable_routes"], "All three routes are required")
    require(manifest.get("cpu_order") == freeze["cpu_order"], "Grid CPU order differs from freeze")
    require(manifest["physical_cores"] == protocol["physical_cores"], "Grid core budget differs from plan")
    require(manifest.get("job_order") == protocol["main"]["job_orders"][batch - 1], "Grid batch order differs from plan")
    require(manifest["query_block"] == protocol["tuning"]["query_block"], "Grid query block differs from plan")
    require(manifest["key_block"] == freeze["selected_key_block"], "Grid key block differs from freeze")
    for field in ("warmup", "reps", "seed"):
        require(manifest[field] == protocol["main"][field], "Grid sampling differs from plan: " + field)
    expected_jobs = {job["id"] for job in manifest["jobs"]}
    require(set(environments) == expected_jobs, "Missing or extra job environment")
    frozen_at = timestamp(freeze["frozen_utc"])
    for name, env in environments.items():
        require(env["source_revision"] == freeze["source_revision"], name + ": source revision differs from freeze")
        require(env["binary_sha256"] == freeze["native_binary_sha256"], name + ": binary differs from freeze")
        require(env["source_dirty"] is False, name + ": source tree was not clean")
        require(timestamp(env["started_utc"]) >= frozen_at, name + ": native job predates freeze")


def aggregate(rows, fields, value, prefix):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[f] for f in fields), []).append(row)
    result = []
    for config, samples in sorted(groups.items()):
        require(sorted(r["batch"] for r in samples) == [1, 2, 3], "Each configuration needs exactly three batches")
        samples = sorted(samples, key=lambda r: r["batch"])
        values = [r[value] for r in samples]
        record = dict(zip(fields, config))
        record.update({f"batch{r['batch']}_{value}": r[value] for r in samples})
        record.update({"median_" + prefix: statistics.median(values),
                       "min_" + prefix: min(values), "max_" + prefix: max(values)})
        result.append(record)
    return result


def paired_speedups(batch_rows):
    """Divide medians inside each batch, then summarize the three paired ratios."""
    lookup = {}
    for row in batch_rows:
        require(math.isfinite(row["median_ms"]) and row["median_ms"] > 0, "Invalid median time")
        identity = tuple(row[f] for f in ("batch", "route", "L", "d", "dv", "mode", "threads"))
        require(identity not in lookup, "Duplicate timing configuration")
        lookup[identity] = row
    pairs = []
    for identity, row in sorted(lookup.items()):
        batch, route, length, d, dv, mode, threads = identity
        if mode == "simd":
            kind, baseline_mode, baseline_threads = "SIMD_vs_scalar", "scalar", 1
        elif mode == "openmp":
            kind, baseline_mode, baseline_threads = "OpenMP_vs_OpenMP1", "openmp", 1
        else:
            continue
        baseline_key = (batch, route, length, d, dv, baseline_mode, baseline_threads)
        require(baseline_key in lookup, "Missing within-batch baseline")
        baseline = lookup[baseline_key]
        ratio = baseline["median_ms"] / row["median_ms"]
        require(math.isfinite(ratio) and ratio > 0, "Invalid paired speedup")
        pairs.append(dict(batch=batch, comparison=kind, route=route, L=length, d=d, dv=dv,
                          threads=threads, baseline_mode=baseline_mode, baseline_threads=baseline_threads,
                          baseline_median_ms=baseline["median_ms"], target_median_ms=row["median_ms"],
                          speedup=ratio))
    fields = ("comparison", "route", "L", "d", "dv", "threads", "baseline_mode", "baseline_threads")
    return pairs, aggregate(pairs, fields, "speedup", "paired_speedup")


def load_study(folder):
    status_path = folder / "study_status.json"
    study_status = read_json(status_path)
    require(study_status.get("status") == "native_collected", "Study controller did not complete successfully")
    jobs = study_status.get("jobs")
    require(isinstance(jobs, list) and len(jobs) == 3
            and all(isinstance(job, dict) and job.get("status") == "passed"
                    and type(job.get("batch")) is int for job in jobs)
            and sorted(job["batch"] for job in jobs) == [1, 2, 3],
            "Study controller needs exactly three passed batches")
    protocol_path, freeze_path = folder / "protocol.json", folder / "frozen_settings.json"
    protocol, freeze = read_json(protocol_path), read_json(freeze_path)
    require(freeze.get("protocol_sha256") == sha(protocol_path), "Protocol SHA256 differs from freeze")
    if "frozen_state" in protocol or "frozen_state" in freeze:
        require("frozen_state" in protocol and "frozen_state" in freeze
                and protocol["frozen_state"] == freeze["frozen_state"],
                "Protocol/freeze frozen_state mismatch")
    folders = [folder / f"native_{i}" for i in (1, 2, 3)]
    grids = [load_grid(path) for path in folders]
    require(all(len(grid[0]) == 60 for grid in grids), "Expected exactly 60 configurations per batch")
    first = grids[0]
    require(all(grid[1] == first[1] for grid in grids), "Different builds or recorded conditions across batches")
    configs = [{key(row) for row in grid[0]} for grid in grids]
    require(all(config == configs[0] for config in configs), "Different configurations across batches")
    sampling, fingerprints, unavailable, timing = check_repeat_evidence(folders, grids, require_complete=True)
    require(sampling == {f: protocol["main"][f] for f in ("warmup", "reps", "seed")},
            "Measured sampling differs from study protocol")
    for batch, (path, grid) in enumerate(zip(folders, grids), 1):
        manifest = read_json(path / "manifest.json")
        environments = {job["id"]: read_json(path / (job["id"] + "_environment.json"))
                        for job in manifest["jobs"]}
        check_binding(protocol, freeze, sha(protocol_path), manifest, environments, batch)
    batch_rows = [dict(batch=batch, **row) for batch, grid in enumerate(grids, 1) for row in grid[0]]
    fields = ("route", "mode", "L", "d", "dv", "threads", "query_block", "workspace_bytes",
              "warmup", "reps", "seed", *BLOCK_FIELDS)
    timing_summary = aggregate(batch_rows, fields, "median_ms", "batch_median_ms")
    pairs, speedups = paired_speedups(batch_rows)
    require(len(timing_summary) == 60 and len(batch_rows) == 180, "Incomplete batch summary")
    require(len(pairs) == 126 and len(speedups) == 42, "Incomplete paired comparisons")
    record = dict(records_validated=True, batches=3, configurations_per_batch=60,
                  raw_samples=sum(row["reps"] for row in batch_rows),
                  environment=first[1], frozen_settings=freeze, sampling_protocol=sampling,
                  protocol_sha256=sha(protocol_path), frozen_settings_sha256=sha(freeze_path),
                  study_status_sha256=sha(status_path), controller_status=study_status["status"],
                  raw_sample_fingerprints=fingerprints, run_timing=timing, unavailable=unavailable,
                  input_sha256={f"native_{i}": grid[3] for i, grid in enumerate(grids, 1)},
                  interpretation="Within-batch ratios first; median/min/max of three batches, not confidence intervals; no pooled samples")
    return batch_rows, timing_summary, pairs, speedups, record


def plot_threads(out, timings, speedups, cpu_note):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#2166ac", "#d6604d", "#1b7837")
    footer = (cpu_note + "\nT=8 is shown separately (diamond); CPU eligibility is not worker pinning. "
              "Temperature/frequency were not fixed.\nMedian and min-max of three batch medians or paired ratios; not confidence intervals.")
    for data, value, ylabel, filename, title in (
        (timings, "batch_median_ms", "Native call time (ms)", "thread_latency.png", "Three batches: native call latency"),
        (speedups, "paired_speedup", "Speedup vs OpenMP T=1", "thread_speedup.png", "Three batches: within-batch paired OpenMP speedup"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
        for ax, length in zip(axes, (128, 4096)):
            for route, label, color in zip(ROUTES, LABELS, colors):
                rows = sorted((r for r in data if r["route"] == route and r["L"] == length
                               and r["d"] == r["dv"] == 64
                               and (r.get("mode") == "openmp" or r.get("comparison") == "OpenMP_vs_OpenMP1")),
                              key=lambda r: r["threads"])
                require([r["threads"] for r in rows] == [1, 2, 4, 8], "Plot requires T=1/2/4/8")
                for group, marker, style, legend in ((rows[:3], "o", "-", label), (rows[3:], "D", "none", None)):
                    med = [r["median_" + value] for r in group]
                    ax.errorbar([r["threads"] for r in group], med,
                                yerr=[[m - r["min_" + value] for m, r in zip(med, group)],
                                      [r["max_" + value] - m for m, r in zip(med, group)]],
                                color=color, marker=marker, linestyle=style, capsize=4, label=legend)
            if value == "paired_speedup":
                ax.axhline(1, color="0.5", linewidth=.8, linestyle=":")
            ax.set(title=f"L={length}, d=dv=64", xlabel="OpenMP threads", ylabel=ylabel,
                   xticks=[1, 2, 4, 8], xticklabels=["1", "2", "4", "8 (separate)"])
            ax.grid(alpha=.2)
            ax.legend(fontsize=9)
        fig.suptitle(title, fontsize=15)
        fig.supxlabel(footer, fontsize=9)
        fig.savefig(out / filename, dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Root containing protocol, frozen settings and native_1..3")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu-class-note", default="CPU classes not specified")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new output directory")
    rows, timings, pairs, speedups, record = load_study(args.input)
    args.out.mkdir(parents=True)
    save_csv(args.out / "batch_timings.csv", rows)
    save_csv(args.out / "timing_summary.csv", timings)
    save_csv(args.out / "paired_speedups_by_batch.csv", pairs)
    save_csv(args.out / "paired_speedup_summary.csv", speedups)
    plot_threads(args.out, timings, speedups, args.cpu_class_note)
    record.update(status="passed", cpu_class_note=args.cpu_class_note,
                  analysis_sha256=sha(Path(__file__)), timing_summary_rows=len(timings),
                  paired_rows=len(pairs), paired_summary_rows=len(speedups))
    write_json(args.out / "analysis.json", record)  # Publish success only after all outputs exist.
    print("Validated 3 x 60 configurations; saved paired summaries and two figures")


if __name__ == "__main__":
    main()
