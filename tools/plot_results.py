"""Plots for the validated records returned by analyze_results.py."""
import hashlib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


COLORS = {("full_matrix", "scalar"): "#325780", ("rowwise", "scalar"): "#d67c30",
          ("full_matrix", "simd"): "#19877e", ("rowwise", "simd"): "#9a5eb0",
          ("full_matrix", "openmp"): "#325780", ("rowwise", "openmp"): "#d67c30"}


def timing_line(ax, rows, xfield, label):
    rows = sorted(rows, key=lambda r: r[xfield])
    first = rows[0]
    ax.errorbar([r[xfield] for r in rows], [r["median_ms"] for r in rows],
                yerr=[[r["median_ms"] - r["q1_ms"] for r in rows],
                      [r["q3_ms"] - r["median_ms"] for r in rows]],
                fmt="o-" if first["route"] == "full_matrix" else "s--", capsize=4, lw=1.8, label=label,
                color=COLORS[(first["route"], first["mode"])])


def finish(fig, out, name, title, note):
    fig.suptitle(title, fontsize=17, fontweight="bold", x=.065, ha="left")
    fig.text(.065, .02, note, fontsize=9, color="#465569", va="bottom")
    fig.tight_layout(rect=(.035, .12, .99, .92), w_pad=3)
    fig.savefig(out / (name + ".png"), dpi=180, facecolor="white")
    plt.close(fig)


def make_plots(out, rows):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.axisbelow": True})
    serial = [r for r in rows if r["mode"] in ("scalar", "simd")]
    widths = sorted({(r["d"], r["dv"]) for r in serial})
    if widths:
        fig, axes = plt.subplots(1, len(widths), figsize=(6 * len(widths), 5.8), squeeze=False)
        for ax, (d, dv) in zip(axes[0], widths):
            subset = [r for r in serial if (r["d"], r["dv"]) == (d, dv)]
            for route, mode in sorted({(r["route"], r["mode"]) for r in subset}):
                timing_line(ax, [r for r in subset if (r["route"], r["mode"]) == (route, mode)],
                            "L", route.replace("_", " ") + " / " + mode)
            lengths = sorted({r["L"] for r in subset})
            ax.set(xscale="log", yscale="log", title=f"d = {d}, dv = {dv}, T = 1",
                   xlabel="Sequence length L (log scale)", ylabel="Time in ms (log scale; lower is faster)")
            ax.set_xticks(lengths, [str(x) for x in lengths])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(alpha=.2, which="major")
            ax.legend(fontsize=9)
        finish(fig, out, "serial_latency", "Single-thread attention timings",
               "Median and Q1–Q3 from individual samples; not confidence intervals.\n"
               "Recorded batch only. SIMD also changes loop order; see analysis.json for machine conditions.")

    parallel = [r for r in rows if r["mode"] == "openmp"]
    shapes = sorted({(r["L"], r["d"], r["dv"]) for r in parallel})
    if shapes:
        fig, axes = plt.subplots(1, len(shapes), figsize=(6 * len(shapes), 5.8), squeeze=False)
        for ax, shape in zip(axes[0], shapes):
            subset = [r for r in parallel if (r["L"], r["d"], r["dv"]) == shape]
            for route in ("full_matrix", "rowwise"):
                timing_line(ax, [r for r in subset if r["route"] == route], "threads", route.replace("_", " "))
            ax.set(title=f"L = {shape[0]}, d = {shape[1]}, dv = {shape[2]}", xlabel="Requested OpenMP threads",
                   ylabel="Time in ms (lower is faster)", ylim=(0, None))
            ax.set_xticks(sorted({r["threads"] for r in subset}))
            ax.grid(alpha=.2)
            ax.legend(fontsize=10)
        finish(fig, out, "thread_latency", "SIMD + OpenMP: time versus thread count",
               "Median and Q1–Q3; thread counts were checked by a preflight probe, not inside timed calls.\n"
               "Each thread setting is a separate run. See analysis.json for recorded affinity and power settings.")

    # Show actual scratch reservations, independent of timing and input/output storage.
    scalar = [r for r in rows if r["mode"] == "scalar"]
    if scalar:
        fig, ax = plt.subplots(figsize=(10, 5.8))
        for route in ("full_matrix", "rowwise"):
            points = sorted({(r["L"], r["workspace_bytes"]) for r in scalar if r["route"] == route})
            ax.plot([p[0] for p in points], [p[1]/1024 for p in points], "o-",
                    color=COLORS[(route, "scalar")], label=route.replace("_", " "), lw=2)
        ax.set(xscale="log", yscale="log", xlabel="Sequence length L (log scale)",
               ylabel="Reserved scratch in KiB (log scale)")
        lengths = sorted({r["L"] for r in scalar})
        ax.set_xticks(lengths, [str(x) for x in lengths])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.2)
        ax.legend()
        blocks = sorted({r["query_block"] for r in scalar if r["route"] == "rowwise"})
        finish(fig, out, "workspace", "Single-thread workspace reservation",
               f"Rowwise effective Bq: {blocks}. Full matrix: 4 L² bytes; rowwise: 4 Bq L bytes.\n"
               "Scratch capacity only: excludes Q/K/V/output and does not measure DRAM traffic, cache hits or process RSS.")
    return {"matplotlib": matplotlib.__version__,
            "plot_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
