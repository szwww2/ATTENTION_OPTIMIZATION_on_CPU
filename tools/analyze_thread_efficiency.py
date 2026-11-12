"""Compute OpenMP parallel efficiency from the validated three-batch study."""

import argparse
from pathlib import Path

from compare_pytorch import save_csv, write_json
from summarize_controlled_study import aggregate, load_study, require, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Saved three-batch study")
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new output directory")

    # Reuse the full raw-data, version and same-batch baseline checks.
    _, _, pairs, _, evidence = load_study(args.input)
    rows = []
    for pair in pairs:
        if pair["comparison"] != "OpenMP_vs_OpenMP1":
            continue
        threads = pair["threads"]
        require(type(threads) is int and threads > 0, "Invalid thread count")
        row = dict(pair)
        row["parallel_efficiency"] = pair["speedup"] / threads
        rows.append(row)

    fields = ("route", "L", "d", "dv", "threads")
    summary = aggregate(rows, fields, "parallel_efficiency", "parallel_efficiency")
    require(len(rows) == 72 and len(summary) == 24, "Incomplete OpenMP study")

    args.out.mkdir(parents=True)
    save_csv(args.out / "efficiency_by_batch.csv", rows)
    save_csv(args.out / "efficiency_summary.csv", summary)
    write_json(args.out / "analysis.json", dict(
        status="passed",
        definition="E(T) = [OpenMP time at T=1 / OpenMP time at T] / T, within each batch",
        interpretation="Median/min/max across three batch efficiencies; not CPU utilization",
        limitation="T=8 uses a mixed CPU class set on this machine; not equal-core scaling",
        source_revision=evidence["frozen_settings"]["source_revision"],
        protocol_sha256=evidence["protocol_sha256"],
        frozen_settings_sha256=evidence["frozen_settings_sha256"],
        study_status_sha256=evidence["study_status_sha256"],
        raw_sample_fingerprints=evidence["raw_sample_fingerprints"],
        analysis_sha256=sha(Path(__file__)),
        batch_rows=len(rows), summary_rows=len(summary),
    ))
    print("Validated the saved study; wrote 72 paired efficiencies and 24 summaries")


if __name__ == "__main__":
    main()
