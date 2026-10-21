"""Validate separate native grids and compare run medians without pooling samples."""

import argparse
import hashlib
from pathlib import Path
import statistics

from analyze_results import load_grid, key
from compare_pytorch import save_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input) < 2 or args.out.exists():
        parser.error("Need at least two grids and a new output directory")
    grids = [load_grid(p) for p in args.input]
    first = grids[0]
    if any(g[1] != first[1] for g in grids):
        raise ValueError("Different build or recorded conditions; do not combine")
    groups = [{key(r): r for r in g[0]} for g in grids]
    if any(set(g) != set(groups[0]) for g in groups):
        raise ValueError("Different configuration sets")
    rows = []
    for config, row in groups[0].items():
        medians = [g[config]["median_ms"] for g in groups]
        result = {k: row[k] for k in ("route", "mode", "L", "d", "dv", "threads", "query_block")}
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
        runs=len(grids),
        configurations=len(rows),
        raw_samples=sum(sum(r["reps"] for r in g[0]) for g in grids),
        environment=first[1],
        input_sha256={f"run{i+1}": g[3] for i, g in enumerate(grids)},
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation="separate sequential runs; medians not pooled; range is descriptive, not a confidence interval",
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ri, route in enumerate(("full_matrix", "rowwise")):
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
    fig.suptitle(f"{len(grids)} sequential runs: medians with Q1-Q3 of each run", fontsize=15)
    fig.savefig(args.out / "repeat_threads.png", dpi=150)
    plt.close(fig)
    write_json(args.out / "analysis.json", record)
    print(f"Checked {len(grids)} grids, {len(rows)} configurations each")


if __name__ == "__main__":
    main()
