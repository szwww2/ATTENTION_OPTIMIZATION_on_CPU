"""Summarize Tang's timing CSVs and make two static, reproducible SVG figures.

Run from the repository root:
  python tools/analyze_tang.py docs/evidence/tang_msvc_L128.csv docs/evidence/tang_msvc_L4096.csv
"""

import argparse
import csv
import os
from pathlib import Path
from statistics import median

# Keep Matplotlib's cache in the ignored build directory, not the user's profile.
cache_dir = Path("build/matplotlib-cache")
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt


def save_svg(fig, path):
    fig.savefig(path)
    # Matplotlib emits trailing spaces inside SVG path data. The whitespace is
    # semantically irrelevant, and removing it keeps git diff --check clean.
    clean = "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
    path.write_text(clean + "\n", encoding="utf-8")


def read_rows(paths):
    rows = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                key = (int(row["L"]), row["route"], row["mode"], int(row["T"]))
                if key in rows:
                    raise ValueError(f"Duplicate timing configuration: {key}")
                samples = [float(value) for value in row["samples_ms"].split(";")]
                if len(samples) != int(row["reps"]):
                    raise ValueError(f"Incorrect sample count for {key}")
                if abs(median(samples) - float(row["median_ms"])) > 1e-6:
                    raise ValueError(f"Incorrect median for {key}")
                if row["contract_pass"] != "1":
                    raise ValueError(f"FP64 contract failed for {key}")
                row["samples"] = samples
                rows[key] = row
    return rows


def summarize(rows):
    summary = []
    for length in sorted({key[0] for key in rows}):
        for route in ("full_matrix", "rowwise"):
            scalar = rows[length, route, "scalar", 1]
            simd = rows[length, route, "simd", 1]
            omp = sorted(
                (key[3], row)
                for key, row in rows.items()
                if key[0] == length and key[1] == route and key[2] == "omp"
            )
            base = next((row for threads, row in omp if threads == 1), None)
            if base is None:
                raise ValueError(f"Missing OpenMP T=1 for L={length}, route={route}")
            base_ms = float(base["median_ms"])
            scalar_ms = float(scalar["median_ms"])
            simd_ms = float(simd["median_ms"])
            for threads, row in omp:
                omp_ms = float(row["median_ms"])
                speedup = base_ms / omp_ms
                summary.append({
                    "L": length,
                    "route": route,
                    "T": threads,
                    "Bq": row["Bq"],
                    "scalar_ms": f"{scalar_ms:.6f}",
                    "simd_ms": f"{simd_ms:.6f}",
                    "simd_vs_scalar": f"{scalar_ms / simd_ms:.4f}",
                    "omp1_ms": f"{base_ms:.6f}",
                    "ompT_ms": f"{omp_ms:.6f}",
                    "speedup": f"{speedup:.4f}",
                    "efficiency": f"{speedup / threads:.4f}",
                    "sample_min_ms": f"{min(row['samples']):.6f}",
                    "sample_max_ms": f"{max(row['samples']):.6f}",
                })
    return summary


def save_summary(summary, outdir, prefix):
    path = outdir / f"{prefix}_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return path


def save_scaling_figure(summary, outdir, prefix, condition):
    lengths = sorted({row["L"] for row in summary})
    fig, axes = plt.subplots(1, len(lengths), figsize=(10, 4), sharey=True)
    if len(lengths) == 1:
        axes = [axes]
    colors = {"full_matrix": "#225ea8", "rowwise": "#d95f0e"}
    labels = {"full_matrix": "Full matrix", "rowwise": "Row-wise"}
    for ax, length in zip(axes, lengths):
        relevant = [row for row in summary if row["L"] == length]
        ticks = sorted({row["T"] for row in relevant})
        ax.plot(ticks, ticks, "--", color="0.55", label="Ideal", linewidth=1.5)
        for route, marker in (("full_matrix", "o"), ("rowwise", "s")):
            points = [row for row in relevant if row["route"] == route]
            values = [float(row["speedup"]) for row in points]
            low = [value - float(row["omp1_ms"]) / float(row["sample_max_ms"])
                   for value, row in zip(values, points)]
            high = [float(row["omp1_ms"]) / float(row["sample_min_ms"]) - value
                    for value, row in zip(values, points)]
            ax.errorbar([row["T"] for row in points], values, yerr=[low, high],
                        marker=marker, color=colors[route], label=labels[route],
                        linewidth=2, capsize=2)
        ax.set(title=f"L = {length}", xlabel="OpenMP threads (T)", xticks=ticks)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Speedup vs same OpenMP route at T=1")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(f"Static query-block scaling ({condition}; d=dv=64, Bq=16)")
    fig.tight_layout()
    path = outdir / f"{prefix}_scaling.svg"
    save_svg(fig, path)
    fig.savefig(outdir / f"{prefix}_scaling.png", dpi=180)
    plt.close(fig)
    return path


def save_simd_figure(summary, outdir, prefix, condition):
    lengths = sorted({row["L"] for row in summary})
    fig, ax = plt.subplots(figsize=(6.5, 4))
    centers = list(range(len(lengths)))
    width = 0.34
    for route, offset, color, label in (
        ("full_matrix", -width / 2, "#225ea8", "Full matrix"),
        ("rowwise", width / 2, "#d95f0e", "Row-wise"),
    ):
        values = [next(float(row["simd_vs_scalar"]) for row in summary
                       if row["L"] == length and row["route"] == route) for length in lengths]
        bars = ax.bar([center + offset for center in centers], values, width,
                      color=color, label=label)
        ax.bar_label(bars, fmt="%.2f×", padding=3)
    ax.set(xticks=centers, xticklabels=[str(length) for length in lengths],
           xlabel="Sequence length (L)", ylabel="Scalar median / SIMD median")
    ax.set_ylim(0, max(float(row["simd_vs_scalar"]) for row in summary) * 1.2)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.suptitle(f"Scalar to explicit AVX2 comparison ({condition}; d=dv=64, Bq=16)")
    fig.tight_layout()
    path = outdir / f"{prefix}_simd.svg"
    save_svg(fig, path)
    fig.savefig(outdir / f"{prefix}_simd.png", dpi=180)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--outdir", type=Path, default=Path("docs/evidence"))
    parser.add_argument("--prefix", default="tang_msvc")
    parser.add_argument("--condition", default="MSVC exploratory")
    args = parser.parse_args()
    rows = read_rows(args.csv)
    summary = summarize(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    for path in (save_summary(summary, args.outdir, args.prefix),
                 save_scaling_figure(summary, args.outdir, args.prefix, args.condition),
                 save_simd_figure(summary, args.outdir, args.prefix, args.condition)):
        print(path)


if __name__ == "__main__":
    main()
