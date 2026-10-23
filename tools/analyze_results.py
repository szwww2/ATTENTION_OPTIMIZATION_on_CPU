"""Check a saved project grid, then export a combined table and plots."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

CONFIG = (
    "route",
    "mode",
    "L",
    "d",
    "dv",
    "seed",
    "query_block",
    "threads",
    "actual_threads",
)
BLOCK_FIELDS = ("key_block", "requested_query_block", "requested_key_block")
INTEGERS = CONFIG[2:] + ("warmup", "reps", "workspace_bytes")
MEASURES = (
    "median_ms",
    "min_ms",
    "q1_ms",
    "q3_ms",
    "iqr_ms",
    "gflops",
    "pre_max_abs",
    "pre_max_rel",
    "post_max_abs",
    "post_max_rel",
)
ENV_FIELDS = (
    "source_revision",
    "source_dirty",
    "source_sha256",
    "binary_sha256",
    "build",
    "suite",
    "os",
    "cpu",
    "affinity",
    "power_mode",
    "openmp_environment",
    "timing_scope",
    "quartiles",
    "thread_check_scope",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def key(row):
    return tuple(row[field] for field in CONFIG) + tuple(
        row[f] for f in BLOCK_FIELDS if f in row
    )


def check_run(folder, job, manifest):
    """Return checked summaries. All source files stay read-only."""
    name = job["id"]
    require(re.fullmatch(r"[A-Za-z0-9_-]+", name), "Invalid job id")
    # A copied grid is portable: resolve by id beside its manifest, not its old prefix.
    paths = [
        folder / (name + suffix)
        for suffix in ("_environment.json", "_summary.csv", "_raw.csv")
    ]
    environment = json.loads(paths[0].read_text(encoding="utf-8"))
    require(job["status"] == "passed" and job["exit_code"] == 0, f"{name}: failed job")
    require(
        environment["status"] == "passed" and environment["exit_code"] == 0,
        f"{name}: failed environment record",
    )
    require(
        environment["quartiles"] == "inclusive linear interpolation at p*(n-1)",
        f"{name}: unsupported quartile method",
    )
    args = environment["benchmark_arguments"]
    expected_args = {
        "--" + field: str(job[field]) for field in ("mode", "threads", "L", "d", "dv")
    }
    expected_args.update(
        {"--" + field: str(manifest[field]) for field in ("warmup", "reps", "seed")}
    )
    expected_args["--query-block"] = str(manifest["query_block"])
    streaming = manifest.get("streaming", False)
    if streaming:
        require(args.count("--streaming") == 1, f"{name}: missing streaming argument")
        args = [arg for arg in args if arg != "--streaming"]
        expected_args["--key-block"] = str(manifest["key_block"])
    require(
        len(args) == 2 * len(expected_args)
        and dict(zip(args[::2], args[1::2])) == expected_args,
        f"{name}: environment arguments differ from manifest",
    )
    summaries, raw = read_csv(paths[1]), read_csv(paths[2])
    modes = {"serial": ("scalar", "simd"), "openmp": ("openmp",)}
    require(job["mode"] in modes, f"{name}: unsupported grid mode")
    variants = [
        (route, mode)
        for mode in modes[job["mode"]]
        for route in (
            ("full_matrix", "rowwise", "blocked_streaming")
            if streaming
            else ("full_matrix", "rowwise")
        )
    ]
    require(
        len(summaries) == job["variants"] == len(variants), f"{name}: missing summaries"
    )
    require(
        {(r["route"], r["mode"]) for r in summaries} == set(variants),
        f"{name}: incorrect variant set",
    )

    for row in summaries + raw:
        require(
            all((field in row) == streaming for field in BLOCK_FIELDS),
            f"{name}: unexpected block schema",
        )
    groups = defaultdict(list)
    for sample in raw:
        for field in (
            CONFIG[2:] + ("sample", "order") + (BLOCK_FIELDS if streaming else ())
        ):
            sample[field] = int(sample[field])
        sample["ms"] = float(sample["ms"])
        require(
            math.isfinite(sample["ms"]) and sample["ms"] > 0,
            f"{name}: non-finite or non-positive timing",
        )
        groups[key(sample)].append(sample)

    for row in summaries:
        for field in INTEGERS + (BLOCK_FIELDS if streaming else ()):
            row[field] = int(row[field])
        for field in MEASURES:
            row[field] = float(row[field])
            require(
                math.isfinite(row[field]) and row[field] >= 0,
                f"{name}: invalid {field}",
            )
        require(row["contract_pass"] == "1", f"{name}: numerical contract failed")
        expected = {field: job[field] for field in ("L", "d", "dv", "threads")}
        expected.update(
            seed=manifest["seed"],
            warmup=manifest["warmup"],
            reps=manifest["reps"],
            actual_threads=job["threads"],
        )
        block = min(manifest["query_block"], job["L"])
        expected["query_block"] = (
            block
            if row["route"] in ("rowwise", "blocked_streaming")
            or row["mode"] == "openmp"
            else 0
        )
        expected["workspace_bytes"] = (
            4 * job["L"] ** 2
            if row["route"] == "full_matrix"
            else 4 * job["threads"] * block * job["L"]
        )
        if streaming:
            bk = min(manifest["key_block"], job["L"])
            expected.update(
                key_block=bk if row["route"] == "blocked_streaming" else 0,
                requested_query_block=manifest["query_block"],
                requested_key_block=manifest["key_block"],
            )
            if row["route"] == "blocked_streaming":
                expected["workspace_bytes"] = (
                    4 * job["threads"] * (block * bk + block * job["dv"] + 2 * block)
                )
        require(
            all(row[field] == value for field, value in expected.items()),
            f"{name}: summary configuration differs from manifest or workspace rule",
        )
        samples = groups.pop(key(row), [])
        require(
            sorted(s["sample"] for s in samples) == list(range(row["reps"])),
            f"{name}: missing or duplicate samples",
        )
        variant_index = variants.index((row["route"], row["mode"]))
        require(
            all(
                s["order"] == (variant_index - s["sample"]) % len(variants)
                for s in samples
            ),
            f"{name}: incorrect rotating execution order",
        )
        times = [s["ms"] for s in samples]
        q1, _, q3 = statistics.quantiles(times, method="inclusive")
        median = statistics.median(times)
        recalculated = dict(
            median_ms=median,
            min_ms=min(times),
            q1_ms=q1,
            q3_ms=q3,
            iqr_ms=q3 - q1,
            gflops=2 * row["L"] ** 2 * (row["d"] + row["dv"]) / (median * 1e6),
        )
        for field, value in recalculated.items():
            require(
                math.isclose(row[field], value, rel_tol=1e-10, abs_tol=1e-12),
                f"{name}: {field} does not match raw samples",
            )
        row.update(recalculated)
        row.update(job=name, iqr_over_median=(q3 - q1) / median)
    require(not groups, f"{name}: unexpected raw sample groups")
    recorded_threads = {
        (r["route"], r["mode"], int(r["threads"]), int(r["actual_threads"]))
        for r in environment["variant_threads"]
    }
    require(
        recorded_threads
        == {
            (r["route"], r["mode"], r["threads"], r["actual_threads"])
            for r in summaries
        },
        f"{name}: environment thread record differs from CSV",
    )
    return summaries, environment, paths


def check_plan(manifest):
    """Require every planned shape/thread setting, including explicitly unavailable jobs."""
    cores = manifest["physical_cores"]
    require(isinstance(cores, int) and cores >= 1, "Invalid physical core budget")
    threads = sorted({t for t in (1, 2, 4, min(8, cores)) if t <= cores})
    require(manifest["thread_grid"] == threads, "Incomplete thread plan")
    expected = {
        (length, width, width, "serial", 1)
        for length, width in (
            (128, 64),
            (512, 64),
            (2048, 64),
            (4096, 64),
            (512, 32),
            (2048, 32),
        )
    }
    expected.update(
        (length, 64, 64, "openmp", t) for length in (128, 4096) for t in threads
    )
    fields = ("L", "d", "dv", "mode", "threads")
    jobs = manifest["jobs"]
    require(
        len(jobs) == len(expected)
        and {tuple(j[f] for f in fields) for j in jobs} == expected,
        "Missing or duplicate planned jobs",
    )
    missing = manifest["unavailable_routes"]
    require(type(manifest.get("streaming", False)) is bool, "Invalid streaming flag")
    if manifest.get("streaming", False):
        require(
            type(manifest.get("key_block")) is int
            and 1 <= manifest["key_block"] <= 4096,
            "Invalid key block",
        )
        require(
            not missing, "Integrated streaming routes cannot be listed as unavailable"
        )
        require(
            all(j["variants"] == (6 if j["mode"] == "serial" else 3) for j in jobs),
            "Incorrect streaming variant counts",
        )
        return
    require(
        len(missing) == len(expected)
        and {tuple(j[f] for f in fields) for j in missing} == expected,
        "Missing blocked-route availability records",
    )
    require(
        all(
            j["route"] == "blocked_streaming"
            and j["status"] == "unavailable"
            and j["reason"]
            and j["variants"] == (2 if j["mode"] == "serial" else 1)
            for j in missing
        ),
        "Unexpected blocked-route status; extend the analyzer before adding new routes",
    )


def load_grid(folder):
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(
        manifest["status"] == "completed_available_runs"
        and manifest["dry_run"] is False,
        "Grid did not complete; a dry run or failed run cannot be plotted",
    )
    require(
        manifest["reps"] >= 3
        and manifest["warmup"] >= 1
        and manifest["query_block"] >= 1,
        "Invalid sampling or block settings",
    )
    check_plan(manifest)
    ids = [job["id"] for job in manifest["jobs"]]
    require(len(ids) == len(set(ids)), "Duplicate job ids")
    rows, skipped, hashes, common = [], list(manifest["unavailable_routes"]), {}, None
    paths = [manifest_path]
    for job in manifest["jobs"]:
        if job["status"] == "unavailable":
            require(bool(job["reason"]), "Unavailable job needs a reason")
            skipped.append(job)
            continue
        checked, environment, files = check_run(folder, job, manifest)
        conditions = {field: environment[field] for field in ENV_FIELDS}
        if common is None:
            common = conditions
        require(
            conditions == common,
            f"{job['id']}: different source, binary or recorded environment; analyze separately",
        )
        rows.extend(checked)
        paths.extend(files)
    require(rows, "No completed configurations")
    require(
        len({key(r) for r in rows}) == len(rows), "Duplicate configurations across jobs"
    )
    for path in paths:
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows, common, skipped, hashes


def write_tables(out, rows, environment, skipped, hashes, plot_info=None):
    with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    record = dict(
        runs=len({r["job"] for r in rows}),
        configurations=len(rows),
        raw_samples=sum(r["reps"] for r in rows),
        environment=environment,
        unavailable=skipped,
        input_sha256=hashes,
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plotting=plot_info,
        python_version=sys.version.split()[0],
    )
    (out / "analysis.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    lines = [
        "# Saved grid results",
        "",
        f"{record['runs']} runs, {len(rows)} configurations, {record['raw_samples']} raw timings. "
        "Summary statistics were checked against every raw sample.",
        "",
        f"Measured source: `{environment['source_revision']}`. Dirty tree: `{environment['source_dirty']}`.",
        f"Affinity: {environment['affinity']}. Power mode: {environment['power_mode']}.",
        "",
        "Points show medians; error bars span Q1 to Q3 of individual timings, not confidence intervals. "
        "Thread points come from separate runs. Workspace means allocated scratch, not DRAM traffic or total process memory.",
        "",
        "| Run | Route / mode | Median ms | Q1–Q3 ms | IQR / median | Scratch KiB |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['job']} | {row['route']} / {row['mode']} | {row['median_ms']:.4f} | "
            f"{row['q1_ms']:.4f}–{row['q3_ms']:.4f} | {row['iqr_over_median']:.1%} | {row['workspace_bytes']/1024:g} |"
        )
    lines += [
        "",
        "## Largest relative spreads",
        "",
        "Ordered by IQR / median to help choose repeat runs. This is a descriptive ranking, not a significance test.",
        "",
    ]
    for row in sorted(rows, key=lambda r: r["iqr_over_median"], reverse=True)[:5]:
        lines.append(
            f"- {row['job']}, {row['route']} / {row['mode']}: {row['iqr_over_median']:.1%}."
        )
    lines += ["", "## Not measured", ""]
    for reason in sorted(
        {f"{j.get('route', j.get('id'))}: {j['reason']}" for j in skipped}
    ):
        lines.append("- " + reason)
    (out / "RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="directory containing manifest.json and its CSVs",
    )
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument(
        "--tables-only", action="store_true", help="no plotting dependency required"
    )
    args = parser.parse_args()
    try:
        require(not args.out.exists(), "Output already exists; choose a new --out")
        rows, environment, skipped, hashes = load_grid(args.input)
        if not args.tables_only:
            from plot_results import make_plots
        args.out.mkdir(parents=True)
        plot_info = None
        if not args.tables_only:
            plot_info = make_plots(args.out, rows)
        write_tables(args.out, rows, environment, skipped, hashes, plot_info)
        print(
            f"Checked {len(rows)} configurations / {sum(r['reps'] for r in rows)} samples; saved to {args.out}"
        )
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        parser.exit(1, f"Analysis failed: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
