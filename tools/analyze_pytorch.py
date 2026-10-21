"""Validate API timings and export a separate PyTorch comparison table/plot."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from analyze_results import read_csv
from compare_pytorch import save_csv, write_json


def load_api_grid(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "passed":
        raise ValueError("API grid did not finish")
    if not isinstance(manifest["physical_cores"], int) or manifest["physical_cores"] < 1:
        raise ValueError("Invalid core budget")
    threads = sorted(
        {
            t
            for t in (1, 2, 4, min(8, manifest["physical_cores"]))
            if t <= manifest["physical_cores"]
        }
    )
    if manifest["thread_grid"] != threads or manifest["unavailable_routes"] != [
        "blocked_streaming"
    ]:
        raise ValueError("Thread plan or availability record mismatch")
    expected = {
        (l, d, 1) for l, d in ((128, 64), (512, 64), (2048, 64), (4096, 64), (512, 32), (2048, 32))
    }
    expected.update((l, 64, t) for l in (128, 4096) for t in threads if t > 1)
    jobs = manifest["jobs"]
    if len(jobs) != len(expected) or {(j["L"], j["d"], j["threads"]) for j in jobs} != expected:
        raise ValueError("API grid shape/thread plan incomplete")
    rows, hashes, common = (
        [],
        {"manifest.json": hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest()},
        None,
    )
    inputs_by_shape = {}
    for job in jobs:
        name = job["id"]
        if name != f"L{job['L']}_d{job['d']}_T{job['threads']}":
            raise ValueError("Invalid job id")
        paths = [folder / name / f for f in ("environment.json", "raw.csv", "summary.csv")]
        env = json.loads(paths[0].read_text(encoding="utf-8"))
        if (
            job["status"] != "passed"
            or job["exit_code"] != 0
            or env["status"] != "passed"
            or env["exit_code"] != 0
        ):
            raise ValueError("Failed case")
        if not env["torch_backend_verified"] or not any(
            "_scaled_dot_product_" in k for k in env["torch_sdpa_operators"]
        ):
            raise ValueError("Unverified backend")
        if any(env[f] != job[f] for f in ("L", "d", "dv", "threads")) or job["d"] != job["dv"]:
            raise ValueError("Environment/plan mismatch")
        if any(env[f] != manifest[f] for f in ("warmup", "reps", "seed")):
            raise ValueError("Sampling mismatch")
        shape = (job["L"], job["d"], job["dv"])
        if set(env["input_sha256"]) != {"Q", "K", "V"} or any(
            len(h) != 64 for h in env["input_sha256"].values()
        ):
            raise ValueError("Invalid input hashes")
        if shape in inputs_by_shape and inputs_by_shape[shape] != env["input_sha256"]:
            raise ValueError("Different inputs across thread settings")
        inputs_by_shape[shape] = env["input_sha256"]
        if (
            env["cpp_preflight_threads"] != job["threads"]
            or env["torch_intraop_threads"] != job["threads"]
            or env["torch_interop_threads"] != 1
        ):
            raise ValueError("Thread evidence mismatch")
        if env["query_block"] != min(job["L"], 16) or env["unavailable_routes"] != [
            "blocked_streaming"
        ]:
            raise ValueError("Block setting or missing-route record mismatch")
        policy = env["pool_policy"]
        if (
            policy["OMP_NUM_THREADS"] != str(job["threads"])
            or policy["OMP_WAIT_POLICY"] != "PASSIVE"
            or policy["KMP_BLOCKTIME"] != "0"
        ):
            raise ValueError("Pool policy mismatch")
        conditions = {
            f: env[f]
            for f in (
                "bridge_build",
                "source_revision",
                "source_dirty",
                "runner_sha256",
                "torch_version",
                "torch_build",
                "numpy",
                "python",
                "cpu",
                "os",
                "timing_scope",
                "affinity",
                "power_mode",
                "correctness_scope",
                "unavailable_routes",
            )
        }
        conditions["pool_policy_except_threads"] = {
            k: v for k, v in policy.items() if k != "OMP_NUM_THREADS"
        }
        if common is None:
            common = conditions
        if conditions != common:
            raise ValueError("API cases used different builds/environments")
        raw, summary = read_csv(paths[1]), read_csv(paths[2])
        mode = "simd" if job["threads"] == 1 else "simd_omp"
        names = ["full_" + mode, "row_" + mode, "pytorch_sdpa"]
        if (
            len(summary) != 3
            or {s["variant"] for s in summary} != set(names)
            or len(raw) != 3 * env["reps"]
        ):
            raise ValueError("Missing or extra variants/samples")
        for row in summary:
            samples = [r for r in raw if r["variant"] == row["variant"]]
            if sorted(int(r["sample"]) for r in samples) != list(range(env["reps"])):
                raise ValueError("Missing/duplicate samples")
            if any(
                int(r["order"]) != (names.index(row["variant"]) - int(r["sample"])) % 3
                for r in samples
            ):
                raise ValueError("Invalid rotated order")
            values = [float(r["ms"]) for r in samples]
            if any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError("Invalid timing")
            q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
            derived = dict(median_ms=statistics.median(values), q1_ms=q1, q3_ms=q3, iqr_ms=q3 - q1)
            if int(row["samples"]) != env["reps"]:
                raise ValueError("Wrong sample count")
            for field, value in derived.items():
                if not math.isclose(float(row[field]), value, rel_tol=1e-12):
                    raise ValueError("Summary differs from raw")
            for field in ("max_abs", "max_rel", "worst_margin"):
                if not math.isfinite(float(row[field])):
                    raise ValueError("Non-finite error")
            if float(row["max_abs"]) < 0 or float(row["max_rel"]) < 0:
                raise ValueError("Negative absolute/relative error")
            if float(row["worst_margin"]) > 0:
                raise ValueError("Correctness failed")
            rows.append(
                dict(
                    job=name,
                    L=job["L"],
                    d=job["d"],
                    dv=job["dv"],
                    threads=job["threads"],
                    **row,
                    backend=(
                        ";".join(env["torch_sdpa_operators"])
                        if row["variant"] == "pytorch_sdpa"
                        else "custom"
                    ),
                )
            )
        hashes.update(
            {f"{name}/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        )
    return rows, common, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new output directory")
    rows, env, hashes = load_api_grid(args.input)
    args.out.mkdir(parents=True)
    save_csv(args.out / "comparison.csv", rows)
    record = dict(
        status="passed",
        configurations=len(rows),
        raw_samples=sum(int(r["samples"]) for r in rows),
        environment=env,
        input_sha256=hashes,
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    max_threads = max(r["threads"] for r in rows)
    for ax, t in zip(axes, (1, max_threads)):
        for prefix, label in (
            ("full_", "Full matrix"),
            ("row_", "Rowwise"),
            ("pytorch_", "PyTorch SDPA"),
        ):
            selected = sorted(
                (
                    r
                    for r in rows
                    if r["threads"] == t and r["d"] == 64 and r["variant"].startswith(prefix)
                ),
                key=lambda r: r["L"],
            )
            x = [r["L"] for r in selected]
            y = [float(r["median_ms"]) for r in selected]
            errors = [
                [float(r["median_ms"]) - float(r["q1_ms"]) for r in selected],
                [float(r["q3_ms"]) - float(r["median_ms"]) for r in selected],
            ]
            ax.errorbar(x, y, yerr=errors, marker="o", capsize=3, label=label)
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="L (sequence length)",
            ylabel="Allocating API time (ms)",
            title=f"T={t}, d=dv=64",
        )
        ax.grid(alpha=0.2)
        ax.legend()
        lengths = sorted({r["L"] for r in rows if r["threads"] == t and r["d"] == 64})
        ax.set_xticks(lengths, labels=[str(length) for length in lengths])
    fig.suptitle("CPU API time: median and Q1-Q3 (not CI); custom T1=SIMD, T>1=OpenMP", fontsize=13)
    fig.savefig(args.out / "pytorch_api.png", dpi=160)
    plt.close(fig)
    write_json(args.out / "analysis.json", record)


if __name__ == "__main__":
    main()
