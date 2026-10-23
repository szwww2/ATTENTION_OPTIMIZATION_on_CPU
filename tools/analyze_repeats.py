"""Validate separate native grids and compare run medians without pooling samples."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics

from analyze_results import BLOCK_FIELDS, load_grid, key
from compare_pytorch import save_csv, write_json


def check_repeat_evidence(folders, grids, require_complete=False):
    """Check recorded provenance; this does not establish performance stability."""
    unavailable = {f"run{i+1}": grid[2] for i, grid in enumerate(grids)}
    if require_complete and any(unavailable.values()):
        raise ValueError("Complete repeats require no unavailable configurations")

    protocols = [
        {tuple(row[field] for field in ("warmup", "reps", "seed")) for row in grid[0]}
        for grid in grids
    ]
    if len(protocols[0]) != 1 or any(p != protocols[0] for p in protocols):
        raise ValueError("Different sampling protocol (warmup, reps or seed)")
    protocol = dict(zip(("warmup", "reps", "seed"), next(iter(protocols[0]))))

    fingerprints = {}
    for i, grid in enumerate(grids):
        raw_hashes = {name: digest for name, digest in grid[3].items() if name.endswith("_raw.csv")}
        if not raw_hashes:
            raise ValueError("No raw sample hashes available")
        fingerprints[f"run{i+1}"] = hashlib.sha256(
            json.dumps(raw_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    if len(set(fingerprints.values())) != len(fingerprints):
        raise ValueError("Duplicate raw sample fingerprint; copied records are not independent repeats")

    def timestamp(value):
        if value is None:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError("timezone required")
            return parsed
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("Invalid recorded run timestamp") from error

    timing_runs, complete_intervals, job_intervals = {}, [], []
    for i, (folder, grid) in enumerate(zip(folders, grids)):
        name = f"run{i+1}"
        intervals, missing = [], []
        for job in sorted({row["job"] for row in grid[0]}):
            environment = json.loads((folder / (job + "_environment.json")).read_text(encoding="utf-8"))
            start = timestamp(environment.get("started_utc"))
            finish = timestamp(environment.get("finished_utc"))
            if start is None or finish is None:
                missing.append(job)
                continue
            if finish <= start:
                raise ValueError("Recorded run finish must follow start")
            intervals.append((start, finish))
            job_intervals.append((start, finish))
        timing_runs[name] = dict(
            status="unknown" if missing else "recorded",
            missing_timestamp_jobs=missing,
        )
        if not missing:
            start = min(pair[0] for pair in intervals)
            finish = max(pair[1] for pair in intervals)
            timing_runs[name].update(started_utc=start.isoformat(), finished_utc=finish.isoformat())
            complete_intervals.append((start, finish))

    for intervals in (job_intervals, complete_intervals):
        ordered = sorted(intervals)
        if any(current[0] < previous[1] for previous, current in zip(ordered, ordered[1:])):
            raise ValueError("Recorded run intervals overlap; sequential repeats required")
    known = len(complete_intervals) == len(grids)
    if require_complete and not known:
        raise ValueError("Complete repeats require start and finish timestamps for every job")
    timing = dict(
        status="verified_non_overlapping" if known else "unknown",
        scope="recorded job start/finish timestamps; not independent proof of acquisition",
        runs=timing_runs,
    )
    return protocol, fingerprints, unavailable, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true",
                        help="require all configurations and recorded non-overlapping run times")
    args = parser.parse_args()
    if len(args.input) < 2 or args.out.exists():
        parser.error("Need at least two grids and a new output directory")
    if len({p.resolve() for p in args.input}) != len(args.input):
        parser.error("Each input must be a different run directory")
    grids = [load_grid(p) for p in args.input]
    first = grids[0]
    if any(g[1] != first[1] for g in grids):
        raise ValueError("Different build or recorded conditions; do not combine")
    groups = [{key(r): r for r in g[0]} for g in grids]
    if any(set(g) != set(groups[0]) for g in groups):
        raise ValueError("Different configuration sets")
    protocol, fingerprints, unavailable, timing = check_repeat_evidence(
        args.input, grids, args.require_complete
    )
    rows = []
    for config, row in groups[0].items():
        medians = [g[config]["median_ms"] for g in groups]
        result = {k: row[k] for k in ("route", "mode", "L", "d", "dv", "threads", "query_block", "warmup", "reps", "seed")}
        result.update({k: row[k] for k in BLOCK_FIELDS if k in row})
        result.update({f"run{i+1}_median_ms": v for i, v in enumerate(medians)})
        result.update({f"run{i+1}_iqr_ms": g[config]["iqr_ms"] for i, g in enumerate(groups)})
        result.update(
            median_of_run_medians_ms=statistics.median(medians),
            min_run_median_ms=min(medians),
            max_run_median_ms=max(medians),
            run_median_range_over_median=(max(medians) - min(medians)) / statistics.median(medians),
        )
        rows.append(result)
    args.out.mkdir(parents=True)
    save_csv(args.out / "repeat_comparison.csv", rows)
    record = dict(
        status="passed",
        records_validated=True,
        performance_stability="descriptive_only",
        require_complete=args.require_complete,
        runs=len(grids),
        configurations=len(rows),
        raw_samples=sum(sum(r["reps"] for r in g[0]) for g in grids),
        environment=first[1],
        sampling_protocol=protocol,
        raw_sample_fingerprints=fingerprints,
        unavailable=unavailable,
        run_timing=timing,
        input_sha256={f"run{i+1}": g[3] for i, g in enumerate(grids)},
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation=(
            "recorded non-overlapping runs; "
            if timing["status"] == "verified_non_overlapping"
            else "run timing unknown; "
        ) + "medians not pooled; range is descriptive, not a confidence interval",
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    routes = sorted({r["route"] for r in first[0]})
    record["plotted_routes"] = routes
    fig, axes = plt.subplots(len(routes), 2, figsize=(12, 4 * len(routes)),
                             squeeze=False, layout="constrained")
    for ri, route in enumerate(routes):
        for ci, length in enumerate((128, 4096)):
            ax = axes[ri, ci]
            for run, grid in enumerate(grids):
                selected = sorted(
                    (
                        r
                        for r in grid[0]
                        if r["route"] == route and r["mode"] == "openmp" and r["L"] == length
                    ),
                    key=lambda r: r["threads"],
                )
                ax.errorbar(
                    [r["threads"] for r in selected],
                    [r["median_ms"] for r in selected],
                    yerr=[
                        [r["median_ms"] - r["q1_ms"] for r in selected],
                        [r["q3_ms"] - r["median_ms"] for r in selected],
                    ],
                    marker="o",
                    capsize=3,
                    label=f"Run {run+1}",
                )
            ax.set(
                title=f"{route.replace('_',' ').title()}, L={length}",
                xlabel="OpenMP threads",
                ylabel="Native time (ms)",
                xticks=[1, 2, 4, 8],
            )
            ax.grid(alpha=0.2)
            ax.legend()
    fig.suptitle(f"{len(grids)} separate runs: medians with Q1-Q3 of each run", fontsize=15)
    fig.savefig(args.out / "repeat_threads.png", dpi=150)
    plt.close(fig)
    write_json(args.out / "analysis.json", record)
    print(f"Checked {len(grids)} grids, {len(rows)} configurations each")


if __name__ == "__main__":
    main()
