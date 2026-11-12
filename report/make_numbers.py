#!/usr/bin/env python3
"""Compute every number used in the GP8 report from the archived measurements.

Reads the 23 October 2026 study (examples/controlled_20261023 and
examples/thread_efficiency_20261023) of the team repository, recomputes all
statistics from raw samples, checks them against the archived summary tables,
and writes:

  generated/numbers.tex   \\gpn{key} macros used in the running text
  generated/tab_*.tex     tables included by main.tex
  generated/checks.txt    agreement log (recomputed vs archived values)
  figures/*.pdf           all figures

Usage:  python make_numbers.py [--repo PATH_TO_REPOSITORY]
The default repository is the parent directory of this script's folder, so the
script works unchanged when it is committed as <repo>/report/make_numbers.py.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROUTES = ["full_matrix", "rowwise", "blocked_streaming"]
RNAME = {"full_matrix": "Full matrix", "rowwise": "Row-wise",
         "blocked_streaming": "Blocked streaming"}
RSHORT = {"full_matrix": "full", "rowwise": "row", "blocked_streaming": "blk"}
SHAPES = [(128, 64), (512, 64), (2048, 64), (4096, 64), (512, 32), (2048, 32)]
THREADS = [1, 2, 4, 8]

# One categorical order for every figure (first three slots of the reference
# palette validate all-pairs); PyTorch and machine references are neutral.
COLOR = {"full_matrix": "#2a78d6", "rowwise": "#eb6834",
         "blocked_streaming": "#1baf7a"}
MARK = {"full_matrix": "o", "rowwise": "s", "blocked_streaming": "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
REF = "#52514e"

NUM = {}          # key -> formatted string for \gpn
CHECKS = []       # agreement log lines


# ----------------------------------------------------------------- helpers
def q(values, p):
    """Inclusive linear interpolation at p*(n-1); same rule as the harness."""
    return float(np.percentile(np.asarray(values, dtype=float), 100 * p))


def put(key, value, fmt=None):
    if key in NUM:
        raise KeyError("duplicate number key " + key)
    if fmt is None:
        NUM[key] = str(value)
    else:
        NUM[key] = format(value, fmt)


def fmt_ms(x):
    """Latency in ms with about four significant digits."""
    if x >= 100:
        return f"{x:.1f}"
    if x >= 10:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.3f}"
    return f"{x:.4f}"


def tex_int(n):
    s = f"{int(n):,}"
    return s.replace(",", "{,}")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check(label, recomputed, archived, tol):
    diff = abs(recomputed - archived)
    ok = diff <= tol
    CHECKS.append((ok, label, recomputed, archived, diff))
    return ok


# ------------------------------------------------------------ native data
def load_native(study):
    """samples[(batch, route, mode, L, d, T)] = list of ms (20 per key)."""
    samples = defaultdict(list)
    meta = {}
    for b in (1, 2, 3):
        for path in sorted(glob.glob(str(study / f"native_{b}" / "*_raw.csv"))):
            job = Path(path).name[: -len("_raw.csv")]
            for r in read_csv(path):
                key = (b, r["route"], r["mode"], int(r["L"]), int(r["d"]), int(r["threads"]))
                samples[key].append(float(r["ms"]))
                meta[key] = job
    return samples, meta


def native_stats(samples):
    st = {}
    for key, xs in samples.items():
        st[key] = dict(n=len(xs), median=q(xs, .5), q1=q(xs, .25), q3=q(xs, .75),
                       min=min(xs))
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(HERE.parent))
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    study = repo / "examples" / "controlled_20261023"
    eff_dir = repo / "examples" / "thread_efficiency_20261023"
    gen = HERE / "generated"
    figs = HERE / "figures"
    gen.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)

    protocol = json.load(open(study / "protocol.json", encoding="utf-8"))
    frozen = json.load(open(study / "frozen_settings.json", encoding="utf-8"))
    env1 = json.load(open(study / "native_1" / "serial_L2048_d64_T1_environment.json", encoding="utf-8"))
    cache = json.load(open(repo / "examples" / "machine_20261023" / "cache_wmi.json", encoding="utf-8"))
    topo = read_csv(repo / "examples" / "machine_20261023" / "cpu_topology.csv")

    # ---------------------------------------------------- setup numbers
    put("srcrev", protocol["source_revision"][:7])
    put("binsha", protocol["native_binary_sha256"][:12])
    put("warmup", protocol["main"]["warmup"])
    put("reps", protocol["main"]["reps"])
    put("batches", protocol["main"]["batches"])
    put("seed", protocol["main"]["seed"])
    put("tuneseed", protocol["tuning"]["seed"])
    put("configs", protocol["main"]["configurations_per_batch"])
    put("bq", 16)
    put("bk", frozen["selected_key_block"])
    put("cores", cache["NumberOfCores"])
    put("logical", cache["NumberOfLogicalProcessors"])
    put("ltwo", cache["L2CacheSize"] // 1024)
    put("lthree", cache["L3CacheSize"] // 1024)
    hi = [int(r["core"]) for r in topo if r["efficiency_class"] == "1"]
    lo = [int(r["core"]) for r in topo if r["efficiency_class"] == "0"]
    put("pcores", f"CPU{min(hi)}--{max(hi)}")
    put("ecores", f"CPU{min(lo)}--{max(lo)}")
    put("npcores", len(hi))
    compiler = [x for x in env1["build"] if x.startswith("compiler=")][0].split("=", 1)[1]
    put("gccver", re.search(r"(\d+\.\d+\.\d+)", compiler).group(1))
    put("osver", env1["os"].replace("Windows-11-", "").replace("-SP0", ""))
    put("cpuname", env1["cpu"]["windows_cim"]["Name"].replace("(R)", "").replace("(TM)", ""))
    started = protocol["created_utc"][11:16]
    status = json.load(open(study / "study_status.json", encoding="utf-8"))
    put("studystart", started)
    put("nativeend", status["jobs"][-1]["finished_utc"][11:16])
    calenv = json.load(open(study / "calibration" / "environment.json", encoding="utf-8"))
    put("studyend", calenv["finished_utc"][11:16])
    put("studydate", protocol["created_utc"][:10])

    # ---------------------------------------------------- native stats
    samples, jobs = load_native(study)
    st = native_stats(samples)
    nraw = sum(len(v) for v in samples.values())
    put("nraw", tex_int(nraw))
    put("ninst", len(st))
    assert len(st) == 180 and all(v["n"] == 20 for v in st.values())

    # check against per-job summaries written by the harness
    bad = 0
    nsum = 0
    errs = []
    for b in (1, 2, 3):
        for path in sorted(glob.glob(str(study / f"native_{b}" / "*_summary.csv"))):
            for r in read_csv(path):
                key = (b, r["route"], r["mode"], int(r["L"]), int(r["d"]), int(r["threads"]))
                s = st[key]
                nsum += 1
                for f, g in (("median", "median_ms"), ("q1", "q1_ms"), ("q3", "q3_ms"), ("min", "min_ms")):
                    if not check(f"native b{b} {key[1:]} {f}", s[f], float(r[g]), 1e-9 * max(1, s[f])):
                        bad += 1
                errs.append(dict(key=key, pre_abs=float(r["pre_max_abs"]), post_abs=float(r["post_max_abs"]),
                                 pre_rel=float(r["pre_max_rel"]), post_rel=float(r["post_max_rel"]),
                                 passed=r["contract_pass"] == "1", ws=int(r["workspace_bytes"])))
    CHECKS.append((bad == 0, f"{nsum} harness summary rows vs raw samples (median/Q1/Q3/min)", nsum, nsum, 0))

    # batch_timings.csv
    for r in read_csv(study / "summary" / "batch_timings.csv"):
        key = (int(r["batch"]), r["route"], r["mode"], int(r["L"]), int(r["d"]), int(r["threads"]))
        s = st[key]
        check(f"batch_timings {key} median", s["median"], float(r["median_ms"]), 1e-9 * s["median"])
        check(f"batch_timings {key} iqr/median", (s["q3"] - s["q1"]) / s["median"],
              float(r["iqr_over_median"]), 1e-9)

    def bm(route, mode, L, d, T=1):
        return [st[(b, route, mode, L, d, T)]["median"] for b in (1, 2, 3)]

    def agg(xs):
        return median(xs), min(xs), max(xs)

    # timing_summary.csv
    for r in read_csv(study / "summary" / "timing_summary.csv"):
        xs = bm(r["route"], r["mode"], int(r["L"]), int(r["d"]), int(r["threads"]))
        m, lo_, hi_ = agg(xs)
        check(f"timing_summary {r['route']} {r['mode']} L{r['L']} d{r['d']} T{r['threads']}",
              m, float(r["median_batch_median_ms"]), 1e-9 * m)
        check("timing_summary min", lo_, float(r["min_batch_median_ms"]), 1e-9 * m)
        check("timing_summary max", hi_, float(r["max_batch_median_ms"]), 1e-9 * m)

    # ---------------------------------------------------- correctness
    put("npass", sum(e["passed"] for e in errs))
    maxabs = max(errs, key=lambda e: max(e["pre_abs"], e["post_abs"]))
    put("maxabs", f"{max(maxabs['pre_abs'], maxabs['post_abs']):.2e}".replace("e-0", r"\times10^{-") + "}")
    maxabs_v = max(max(e["pre_abs"], e["post_abs"]) for e in errs)
    put("atolratio", f"{1e-4 / maxabs_v:.0f}")
    maxrel = max(errs, key=lambda e: max(e["pre_rel"], e["post_rel"]))
    put("maxrel", f"{max(maxrel['pre_rel'], maxrel['post_rel']):.3f}")
    put("maxrelshape", f"$L={maxrel['key'][3]}$, $d=d_v={maxrel['key'][4]}$")
    same = all(e["pre_abs"] == e["post_abs"] and e["pre_rel"] == e["post_rel"] for e in errs)
    put("prepostsame", "identical" if same else "not identical")

    # ---------------------------------------------------- variability
    iqrm = [(s["q3"] - s["q1"]) / s["median"] for s in st.values()]
    put("nvarten", sum(x > 0.10 for x in iqrm))
    put("nvartwenty", sum(x > 0.20 for x in iqrm))
    put("varmax", f"{100 * max(iqrm):.0f}")
    worst = max(st, key=lambda k: (st[k]["q3"] - st[k]["q1"]) / st[k]["median"])
    MODE = {"scalar": "scalar", "simd": "AVX2", "openmp": "AVX2\\,+\\,OpenMP"}
    put("varmaxcfg", f"{RNAME[worst[1]].lower()}, {MODE[worst[2]]}, $L={worst[3]}$, $T={worst[5]}$")
    groups = defaultdict(dict)
    for e in errs:
        b_, route_, mode_, L_, d_, T_ = e["key"]
        groups[(b_, mode_, L_, d_, T_)][route_] = (e["pre_abs"], e["pre_rel"], e["post_abs"], e["post_rel"])
    same_fr = sum(1 for g in groups.values() if g["full_matrix"] == g["rowwise"])
    put("fullrowsame", f"{same_fr} of {len(groups)}")
    big = [k for k in st if k[3] >= 2048]
    put("varmedbig", f"{100 * median([(st[k]['q3'] - st[k]['q1']) / st[k]['median'] for k in big]):.1f}")
    # batch-to-batch spread of the batch medians
    spread = []
    for r in ROUTES:
        for L, d in SHAPES:
            for m in ("scalar", "simd"):
                xs = bm(r, m, L, d)
                spread.append((max(xs) - min(xs)) / median(xs))
    put("bspreadmed", f"{100 * median(spread):.1f}")
    put("bspreadmax", f"{100 * max(spread):.1f}")

    # pooled 60-sample medians (robustness check of the summary statistic)
    def pooled(route, mode, L, d, T=1):
        xs = []
        for b in (1, 2, 3):
            xs += samples[(b, route, mode, L, d, T)]
        return q(xs, .5)

    # ---------------------------------------------------- workspaces
    ws = {}
    for e in errs:
        b, route, mode, L, d, T = e["key"]
        ws[(route, mode, L, d, T)] = e["ws"]
    for r in ROUTES:
        put(f"ws-{RSHORT[r]}-4096", tex_int(ws[(r, "simd", 4096, 64, 1)]))
        put(f"ws-{RSHORT[r]}-4096-T8", tex_int(ws[(r, "openmp", 4096, 64, 8)]))
    put("wsratio", tex_int(round(ws[("full_matrix", "simd", 4096, 64, 1)] / ws[("blocked_streaming", "simd", 4096, 64, 1)])))
    put("wsfullmib", ws[("full_matrix", "simd", 4096, 64, 1)] // 2**20)
    put("wsrowkib", ws[("rowwise", "simd", 4096, 64, 1)] // 1024)

    # ---------------------------------------------------- paired speedups
    paired_rows = read_csv(study / "summary" / "paired_speedups_by_batch.csv")
    arch_pair = {}
    for r in paired_rows:
        arch_pair[(int(r["batch"]), r["comparison"], r["route"], int(r["L"]), int(r["d"]), int(r["threads"]))] = float(r["speedup"])
    simd_sp = {}
    for r in ROUTES:
        for L, d in SHAPES:
            xs = []
            for b in (1, 2, 3):
                x = st[(b, r, "scalar", L, d, 1)]["median"] / st[(b, r, "simd", L, d, 1)]["median"]
                check(f"paired SIMD/scalar b{b} {r} L{L} d{d}", x, arch_pair[(b, "SIMD_vs_scalar", r, L, d, 1)], 1e-12)
                xs.append(x)
            simd_sp[(r, L, d)] = agg(xs)
    omp_sp = {}
    for r in ROUTES:
        for L in (128, 4096):
            for T in THREADS:
                xs = []
                for b in (1, 2, 3):
                    x = st[(b, r, "openmp", L, 64, 1)]["median"] / st[(b, r, "openmp", L, 64, T)]["median"]
                    check(f"paired OMP b{b} {r} L{L} T{T}", x, arch_pair[(b, "OpenMP_vs_OpenMP1", r, L, 64, T)], 1e-12)
                    xs.append(x)
                omp_sp[(r, L, T)] = agg(xs)
    for r in read_csv(study / "summary" / "paired_speedup_summary.csv"):
        if r["comparison"] == "SIMD_vs_scalar":
            m = simd_sp[(r["route"], int(r["L"]), int(r["d"]))][0]
        else:
            m = omp_sp[(r["route"], int(r["L"]), int(r["threads"]))][0]
        check(f"paired summary {r['comparison']} {r['route']} L{r['L']} d{r['d']} T{r['threads']}",
              m, float(r["median_paired_speedup"]), 1e-12)
    # efficiency
    for r in read_csv(eff_dir / "efficiency_by_batch.csv"):
        b, route, L, T = int(r["batch"]), r["route"], int(r["L"]), int(r["threads"])
        x = st[(b, route, "openmp", L, 64, 1)]["median"] / st[(b, route, "openmp", L, 64, T)]["median"] / T
        check(f"efficiency b{b} {route} L{L} T{T}", x, float(r["parallel_efficiency"]), 1e-12)

    for r in ROUTES:
        for L, d in SHAPES:
            m, lo_, hi_ = simd_sp[(r, L, d)]
            put(f"simd-{RSHORT[r]}-{L}-{d}", f"{m:.2f}")
        for L in (128, 4096):
            for T in THREADS:
                m, lo_, hi_ = omp_sp[(r, L, T)]
                put(f"omp-{RSHORT[r]}-{L}-T{T}", f"{m:.2f}")
                put(f"eff-{RSHORT[r]}-{L}-T{T}", f"{100 * m / T:.0f}")
    allsimd = [simd_sp[(r, L, d)][0] for r in ROUTES for L, d in SHAPES]
    put("simdmin", f"{min(allsimd):.2f}")
    put("simdmax", f"{max(allsimd):.2f}")
    for r in ROUTES:
        xs = [simd_sp[(r, L, d)][0] for L, d in SHAPES]
        put(f"simdmin-{RSHORT[r]}", f"{min(xs):.2f}")
        put(f"simdmax-{RSHORT[r]}", f"{max(xs):.2f}")
    t4_4096 = [omp_sp[(r, 4096, 4)][0] for r in ROUTES]
    t8_4096 = [omp_sp[(r, 4096, 8)][0] for r in ROUTES]
    put("t4min", f"{min(t4_4096):.2f}")
    put("t4max", f"{max(t4_4096):.2f}")
    put("t8min", f"{min(t8_4096):.2f}")
    put("t8max", f"{max(t8_4096):.2f}")
    put("e4min", f"{100 * min(t4_4096) / 4:.0f}")
    put("e4max", f"{100 * max(t4_4096) / 4:.0f}")
    put("e8min", f"{100 * min(t8_4096) / 8:.0f}")
    put("e8max", f"{100 * max(t8_4096) / 8:.0f}")
    t4_128 = [omp_sp[(r, 128, 4)][0] for r in ROUTES]
    t8_128 = [omp_sp[(r, 128, 8)][0] for r in ROUTES]
    put("s128t4min", f"{min(t4_128):.2f}")
    put("s128t4max", f"{max(t4_128):.2f}")
    put("s128t8min", f"{min(t8_128):.2f}")
    put("s128t8max", f"{max(t8_128):.2f}")
    t2_4096 = [omp_sp[(r, 4096, 2)][0] for r in ROUTES]
    put("t2min", f"{min(t2_4096):.2f}")
    put("t2max", f"{max(t2_4096):.2f}")

    # T8 vs T4: in how many (route, batch) pairs is T4 faster?
    for L in (128, 4096):
        wins = 0
        ratios = []
        for r in ROUTES:
            for b in (1, 2, 3):
                t4 = st[(b, r, "openmp", L, 64, 4)]["median"]
                t8 = st[(b, r, "openmp", L, 64, 8)]["median"]
                wins += t4 < t8
                ratios.append(t8 / t4)
        put(f"t4wins-{L}", wins)
        put(f"t8overt4-{L}-min", f"{min(ratios):.2f}")
        put(f"t8overt4-{L}-max", f"{max(ratios):.2f}")
    # batch 2 at L=128 is the exception
    exc = [b for b in (1, 2, 3) if all(st[(b, r, "openmp", 128, 64, 8)]["median"] < st[(b, r, "openmp", 128, 64, 4)]["median"] for r in ROUTES)]
    put("t8excbatch", ",".join(map(str, exc)) if exc else "none")

    # ---------------------------------------------------- route comparisons
    def paired_ratio(r1, m1, r2, m2, L, d, T=1):
        return [st[(b, r1, m1, L, d, T)]["median"] / st[(b, r2, m2, L, d, T)]["median"] for b in (1, 2, 3)]

    for L, d in SHAPES:
        x = median(paired_ratio("rowwise", "scalar", "blocked_streaming", "scalar", L, d))
        put(f"rowoverblk-scalar-{L}-{d}", f"{x:.2f}")
        put(f"blkgain-scalar-{L}-{d}", f"{100 * (1 - 1 / x):.0f}")
        x = median(paired_ratio("full_matrix", "scalar", "rowwise", "scalar", L, d))
        put(f"fulloverrow-scalar-{L}-{d}", f"{x:.3f}")
        sm = [median(bm(r, "simd", L, d)) for r in ROUTES]
        put(f"simdspread-{L}-{d}", f"{100 * (max(sm) / min(sm) - 1):.1f}")
    frs = [median(paired_ratio("full_matrix", "scalar", "rowwise", "scalar", L, d)) for L, d in SHAPES]
    put("fullrowsc-min", f"{min(frs):.3f}")
    put("fullrowsc-max", f"{max(frs):.3f}")
    put("fullrowsc-dev", f"{100 * max(abs(x - 1) for x in frs):.1f}")
    s4 = [median(bm(r, "simd", 4096, 64)) for r in ROUTES]
    put("simd4096min", f"{min(s4):.0f}")
    put("simd4096max", f"{max(s4):.0f}")
    gains = [100 * (1 - 1 / median(paired_ratio("rowwise", "scalar", "blocked_streaming", "scalar", L, d))) for L, d in SHAPES]
    put("blkgainmin", f"{min(gains):.0f}")
    put("blkgainmax", f"{max(gains):.0f}")
    sp = [100 * (max(median(bm(r, 'simd', L, d)) for r in ROUTES) / min(median(bm(r, 'simd', L, d)) for r in ROUTES) - 1) for L, d in SHAPES]
    put("simdspreadmax", f"{max(sp):.1f}")
    fr = [median(paired_ratio("full_matrix", "simd", "rowwise", "simd", L, d)) for L, d in SHAPES]
    put("fullrowsimdmin", f"{100 * (min(fr) - 1):.1f}")
    put("fullrowsimdmax", f"{100 * (max(fr) - 1):.1f}")
    fr4 = median(paired_ratio("full_matrix", "simd", "rowwise", "simd", 4096, 64))
    put("fullrowsimd4096", f"{100 * (fr4 - 1):.1f}")

    # cells in which full-matrix has the lowest median, and its margin over the next route
    full_best = []
    cells = [(m, L, d, 1) for L, d in SHAPES for m in ("scalar", "simd")]
    cells += [("openmp", L, 64, T) for L in (128, 4096) for T in THREADS]
    for m, L, d, T in cells:
        v = {r: median(bm(r, m, L, d, T)) for r in ROUTES}
        if min(v, key=v.get) == "full_matrix":
            nxt = min(v[r] for r in ROUTES if r != "full_matrix")
            full_best.append(((m, L, d, T), 100 * (nxt / v["full_matrix"] - 1)))
    put("fullbestcells", len(full_best))
    put("fullbestmargin", f"{max(x for _, x in full_best):.1f}" if full_best else "0.0")
    CHECKS.append((sorted(c for c, _ in full_best) == [("openmp", 4096, 64, 8), ("simd", 2048, 32, 1)],
                   "full-matrix has the lowest median only at (AVX2, L=2048, d=32) and (OpenMP T=8, L=4096)", len(full_best), 2, 0))

    # OpenMP T1 vs SIMD serial (different processes, same batch)
    for L in (128, 4096):
        xs = []
        for r in ROUTES:
            xs.append(median(paired_ratio(r, "openmp", r, "simd", L, 64)))
        put(f"omp1oversimd-{L}-min", f"${100 * (min(xs) - 1):+.1f}$")
        put(f"omp1oversimd-{L}-max", f"${100 * (max(xs) - 1):+.1f}$")

    # fork/join cost estimate at L=128 from T2/T4: c_T = t_T - t_1/T
    over = []
    for r in ROUTES:
        for T in (2, 4):
            for b in (1, 2, 3):
                t1 = st[(b, r, "openmp", 128, 64, 1)]["median"]
                tt = st[(b, r, "openmp", 128, 64, T)]["median"]
                over.append(1000 * (tt - t1 / T))
    put("ovhmin", f"{min(over):.0f}")
    put("ovhmax", f"{max(over):.0f}")
    put("ovhmed", f"{median(over):.0f}")
    t1_128 = [median(bm(r, "openmp", 128, 64, 1)) for r in ROUTES]
    put("t1-128-min", f"{1000 * min(t1_128):.0f}")
    put("t1-128-max", f"{1000 * max(t1_128):.0f}")

    # ---------------------------------------------------- d scaling model
    # t(d) = L^2 (a + b d) with d = dv; two points give a and b exactly.
    dsc = {}
    for r in ROUTES:
        for mode in ("scalar", "simd"):
            for L in (512, 2048):
                rho = median([st[(b, r, mode, L, 64, 1)]["median"] / st[(b, r, mode, L, 32, 1)]["median"] for b in (1, 2, 3)])
                t32 = median(bm(r, mode, L, 32))
                t64 = median(bm(r, mode, L, 64))
                tot_ns = t64 * 1e6 / L**2
                a_ns = (2 / rho - 1) * tot_ns
                dsc[(r, mode, L)] = dict(rho=rho, share=2 / rho - 1, a=a_ns, tot=tot_ns)
                put(f"rho-{RSHORT[r]}-{mode}-{L}", f"{rho:.2f}")
                put(f"share-{RSHORT[r]}-{mode}-{L}", f"{100 * (2 / rho - 1):.0f}")
                put(f"ans-{RSHORT[r]}-{mode}-{L}", f"{a_ns:.1f}")
                put(f"totns-{RSHORT[r]}-{mode}-{L}", f"{tot_ns:.1f}")
    simd_share = [dsc[(r, "simd", L)]["share"] for r in ROUTES for L in (512, 2048)]
    put("simdsharemin", f"{100 * min(simd_share):.0f}")
    put("simdsharemax", f"{100 * max(simd_share):.0f}")
    sc_share = [dsc[(r, "scalar", L)]["share"] for r in ROUTES for L in (512, 2048)]
    put("scsharemin", f"{100 * min(sc_share):.0f}")
    put("scsharemax", f"{100 * max(sc_share):.0f}")
    simd_a = [dsc[(r, "simd", L)]["a"] for r in ROUTES for L in (512, 2048)]
    put("simdamin", f"{min(simd_a):.0f}")
    put("simdamax", f"{max(simd_a):.0f}")
    ab = [dsc[(r, "simd", L)]["share"] / (1 - dsc[(r, "simd", L)]["share"]) for r in ROUTES for L in (512, 2048)]
    put("aoverbmin", f"{min(ab):.0f}")
    put("aoverbmax", f"{max(ab):.0f}")
    simd_tot = [dsc[(r, "simd", L)]["tot"] for r in ROUTES for L in (512, 2048)]
    put("simdtotmin", f"{min(simd_tot):.0f}")
    put("simdtotmax", f"{max(simd_tot):.0f}")
    rho_simd = [dsc[(r, "simd", L)]["rho"] for r in ROUTES for L in (512, 2048)]
    put("rhosimdmin", f"{min(rho_simd):.2f}")
    put("rhosimdmax", f"{max(rho_simd):.2f}")
    # per-score time at L=4096
    for r in ROUTES:
        for mode in ("scalar", "simd"):
            put(f"pers-{RSHORT[r]}-{mode}-4096", f"{median(bm(r, mode, 4096, 64)) * 1e6 / 4096**2:.1f}")

    # paired-ratio spread across batches vs absolute spread
    pspread = [(simd_sp[k][2] - simd_sp[k][1]) / simd_sp[k][0] for k in simd_sp]
    put("pairspreadmed", f"{100 * median(pspread):.1f}")
    put("pairspreadmax", f"{100 * max(pspread):.1f}")
    # decomposition of the full-matrix SIMD speedup at L=4096 (per-score medians)
    pf = median(bm("full_matrix", "scalar", 4096, 64))
    pb = median(bm("blocked_streaming", "scalar", 4096, 64))
    ps = median(bm("full_matrix", "simd", 4096, 64))
    put("loopfactor", f"{pf / pb:.2f}")
    put("vecfactor", f"{pb / ps:.2f}")
    for L_ in (128, 512, 2048, 4096):
        put(f"colws-{L_}", f"{L_ * 64 // 1024}")
    put("amdmin", f"{1 / max(simd_share):.2f}")
    put("amdmax", f"{1 / min(simd_share):.2f}")
    put("rescalepct", f"{100 / 128:.1f}")
    put("nblk-4096", 4096 // 16)
    put("nblkthr-4096", 4096 // 16 // 8)
    put("nblk-128", 128 // 16)

    # ---------------------------------------------------- tuning
    tune = {}
    for b in (1, 2, 3):
        rows = read_csv(study / "tuning" / f"batch{b}_raw.csv")
        summ = {int(r["key_block"]): r for r in read_csv(study / "tuning" / f"batch{b}_summary.csv")}
        for bk in (128, 256, 512):
            xs = [float(r["ms"]) for r in rows if int(r["key_block"]) == bk]
            assert len(xs) == 20
            tune[(b, bk)] = dict(median=q(xs, .5), q1=q(xs, .25), q3=q(xs, .75),
                                 ws=int(rows[[int(r["key_block"]) for r in rows].index(bk)]["workspace_bytes"]))
            check(f"tuning b{b} Bk{bk} median", tune[(b, bk)]["median"], float(summ[bk]["median_ms"]), 1e-9)
    sel = {}
    for c in frozen["candidates"]:
        bk = c["key_block"]
        meds = [tune[(b, bk)]["median"] for b in (1, 2, 3)]
        check(f"tuning selection value Bk{bk}", median(meds), c["selection_value_ms"], 1e-9)
        sel[bk] = median(meds)
    chosen = min(sel, key=lambda k: (sel[k], k))
    assert chosen == frozen["selected_key_block"]
    vals = sorted(sel.values())
    put("tunespread", f"{100 * (vals[-1] / vals[0] - 1):.1f}")
    for bk in (128, 256, 512):
        put(f"tune-{bk}", f"{sel[bk]:.1f}")
    tb = []
    for b in (1, 2, 3):
        xs = [tune[(b, bk)]["median"] for bk in (128, 256, 512)]
        tb.append(xs)
    btw = [(max(tune[(b, bk)]["median"] for b in (1, 2, 3)) - min(tune[(b, bk)]["median"] for b in (1, 2, 3))) / sel[bk] for bk in (128, 256, 512)]
    put("tunebatchspread", f"{100 * min(btw):.0f}--{100 * max(btw):.0f}")
    winners = [min((128, 256, 512), key=lambda k: tune[(b, k)]["median"]) for b in (1, 2, 3)]
    put("tunewinners", ", ".join(str(w) for w in winners))

    # ---------------------------------------------------- calibration
    cal_raw = read_csv(study / "calibration" / "raw.csv")
    cal = {}
    for wl in ("avx2_mul_add", "read_stream"):
        for T in THREADS:
            rows = [r for r in cal_raw if r["workload"] == wl and int(r["threads"]) == T]
            ms = [float(r["ms"]) for r in rows]
            work = float(rows[0]["work_units"])
            med = q(ms, .5)
            cal[(wl, T)] = dict(ms=med, rate=work / (med / 1000) / 1e9)
    for r in read_csv(study / "calibration" / "summary.csv"):
        c = cal[(r["workload"], int(r["threads"]))]
        check(f"calibration {r['workload']} T{r['threads']} median", c["ms"], float(r["median_ms"]), 1e-9)
        check(f"calibration {r['workload']} T{r['threads']} rate", c["rate"], float(r["giga_units_per_second"]), 1e-9)
    for T in THREADS:
        put(f"calC-T{T}", f"{cal[('avx2_mul_add', T)]['rate']:.1f}")
        put(f"calB-T{T}", f"{cal[('read_stream', T)]['rate']:.1f}")
        put(f"calridge-T{T}", f"{cal[('avx2_mul_add', T)]['rate'] / cal[('read_stream', T)]['rate']:.1f}")
        slow = cal[("avx2_mul_add", T)]["ms"] / cal[("avx2_mul_add", 1)]["ms"]
        put(f"calslow-T{T}", f"{100 * (slow - 1):.0f}")
        put(f"calspeed-T{T}", f"{T / slow:.2f}")
        put(f"caleff-T{T}", f"{100 / slow:.0f}")
    calref = {T: T * cal[("avx2_mul_add", 1)]["ms"] / cal[("avx2_mul_add", T)]["ms"] for T in THREADS}
    put("calt8overt4", f"{cal[('avx2_mul_add', 8)]['ms'] / cal[('avx2_mul_add', 4)]['ms']:.2f}")
    # implied speed of slowest T8 worker relative to CPU0 alone
    e8 = [omp_sp[(r, 4096, 8)][0] / 8 for r in ROUTES]
    put("slowfactor8min", f"{1 / max(e8):.1f}")
    put("slowfactor8max", f"{1 / min(e8):.1f}")
    put("calslowfactor8", f"{cal[('avx2_mul_add', 8)]['ms'] / cal[('avx2_mul_add', 1)]['ms']:.2f}")

    # ---------------------------------------------------- roofline model
    L, d, dv = 4096, 64, 64
    F = 2 * L * L * (d + dv)
    put("Fmat", f"{F / 1e9:.2f}")
    Nq = math.ceil(L / 16)
    base = 4 * L * (2 * d + 3 * dv)
    S = 4 * L * L
    model = {
        ("full_matrix", "once"): base + 5 * S,
        ("full_matrix", "refill"): 4 * L * d + 4 * L * L * (d + dv) + 8 * L * dv + 5 * S,
        ("rowwise", "once"): base,
        ("rowwise", "refill"): 4 * L * d + 4 * L * L * (d + dv) + 8 * L * dv,
        ("blocked_streaming", "once"): base,
        ("blocked_streaming", "refill"): 4 * L * d + 4 * L * (d + dv) * Nq + 8 * L * dv,
    }
    scen = read_csv(study / "roofline_analysis" / "traffic_scenarios.csv")
    for r in scen:
        kind = "once" if "once" in r["scenario"] else "refill"
        check(f"roofline bytes {r['route']} {r['scenario']}", model[(r["route"], kind)], float(r["dram_bytes_model"]), 0.5)
        check(f"roofline F {r['route']}", F, float(r["F_mat"]), 0.5)
    roof = {}
    for r in ROUTES:
        for mode, T in (("simd", 1), ("openmp", 4), ("openmp", 8)):
            t = median(bm(r, mode, L, 64, T))
            gf = F / (t / 1000) / 1e9
            for kind in ("once", "refill"):
                ai = F / model[(r, kind)]
                bw = model[(r, kind)] / (t / 1000) / 1e9
                C = cal[("avx2_mul_add", T)]["rate"]
                B = cal[("read_stream", T)]["rate"]
                roof[(r, mode, T, kind)] = dict(t=t, gf=gf, ai=ai, bw=bw, att=min(C, B * ai), C=C, B=B)
    for r in ROUTES:
        x = roof[(r, "simd", 1, "once")]
        put(f"gf-{RSHORT[r]}-T1", f"{x['gf']:.2f}")
        put(f"pctC-{RSHORT[r]}-T1", f"{100 * x['gf'] / x['C']:.1f}")
        x4 = roof[(r, "openmp", 4, "once")]
        put(f"gf-{RSHORT[r]}-T4", f"{x4['gf']:.1f}")
        put(f"pctC-{RSHORT[r]}-T4", f"{100 * x4['gf'] / x4['C']:.1f}")
        for kind in ("once", "refill"):
            put(f"ai-{RSHORT[r]}-{kind}", f"{roof[(r, 'simd', 1, kind)]['ai']:.2f}" if roof[(r, 'simd', 1, kind)]['ai'] < 100 else f"{roof[(r, 'simd', 1, kind)]['ai']:.0f}")
            put(f"mb-{RSHORT[r]}-{kind}", f"{model[(r, kind)] / 2**20:.1f}" if model[(r, kind)] < 2**30 else f"{model[(r, kind)] / 2**30:.2f}")
            put(f"bw-{RSHORT[r]}-{kind}-T1", f"{roof[(r, 'simd', 1, kind)]['bw']:.2f}")
            put(f"bw-{RSHORT[r]}-{kind}-T4", f"{roof[(r, 'openmp', 4, kind)]['bw']:.2f}")
            put(f"bwpct-{RSHORT[r]}-{kind}-T1", f"{100 * roof[(r, 'simd', 1, kind)]['bw'] / roof[(r, 'simd', 1, kind)]['B']:.1f}")
            put(f"bwpct-{RSHORT[r]}-{kind}-T4", f"{100 * roof[(r, 'openmp', 4, kind)]['bw'] / roof[(r, 'openmp', 4, kind)]['B']:.1f}")
            put(f"attfrac-{RSHORT[r]}-{kind}-T1", f"{100 * roof[(r, 'simd', 1, kind)]['gf'] / roof[(r, 'simd', 1, kind)]['att']:.0f}")
    put("trafficratio", f"{model[('full_matrix', 'once')] / model[('blocked_streaming', 'once')]:.0f}")

    # ---------------------------------------------------- PyTorch API comparison
    pt = {}
    pt_dirs = sorted(glob.glob(str(study / "pytorch" / "L*_d*_T*")))
    comp = {(r["job"], r["variant"]): r for r in read_csv(study / "pytorch_analysis" / "comparison.csv")}
    for path in pt_dirs:
        job = Path(path).name
        m = re.match(r"L(\d+)_d(\d+)_T(\d+)", job)
        L_, d_, T_ = map(int, m.groups())
        rows = read_csv(Path(path) / "raw.csv")
        summ = {r["variant"]: r for r in read_csv(Path(path) / "summary.csv")}
        envp = json.load(open(Path(path) / "environment.json", encoding="utf-8"))
        assert envp["status"] == "passed" and envp["torch_backend_verified"]
        for var in sorted({r["variant"] for r in rows}):
            xs = [float(r["ms"]) for r in rows if r["variant"] == var]
            med = q(xs, .5)
            check(f"pytorch {job} {var} median", med, float(summ[var]["median_ms"]), 1e-9)
            check(f"pytorch comparison.csv {job} {var}", med, float(comp[(job, var)]["median_ms"]), 1e-9)
            pt[(L_, d_, T_, var)] = dict(median=med, q1=q(xs, .25), q3=q(xs, .75),
                                        max_abs=float(summ[var]["max_abs"]))
    put("torchver", envp["torch_version"].replace("+cpu", ""))
    put("torchop", envp["torch_sdpa_operators"][0].replace("_", r"\_"))
    put("pyver", envp["python"])
    pt_cfgs = sorted({k[:3] for k in pt}, key=lambda c: (0, SHAPES.index((c[0], c[1]))) if c[2] == 1 else (1, c[0], c[2]))
    ratios = {}
    for c in pt_cfgs:
        L_, d_, T_ = c
        suffix = "" if T_ == 1 else "_omp"
        cust = {r: pt[(L_, d_, T_, {"full_matrix": "full", "rowwise": "row", "blocked_streaming": "blocked"}[r] + "_simd" + suffix)]["median"] for r in ROUTES}
        best = min(cust.values())
        ratios[c] = (best / pt[(L_, d_, T_, "pytorch_sdpa")]["median"], cust)
    rr = [ratios[c][0] for c in pt_cfgs]
    put("ptratiomin", f"{min(rr):.0f}")
    put("ptratiomax", f"{max(rr):.0f}")
    put("ptratio-2048-T1", f"{ratios[(2048, 64, 1)][0]:.0f}")
    put("ptratio-4096-T1", f"{ratios[(4096, 64, 1)][0]:.0f}")
    put("ptratio-4096-T4", f"{ratios[(4096, 64, 4)][0]:.0f}")
    put("ptratio-4096-T8", f"{ratios[(4096, 64, 8)][0]:.0f}")
    put("pt-4096-T4", fmt_ms(pt[(4096, 64, 4, "pytorch_sdpa")]["median"]))
    put("pt-4096-T1", fmt_ms(pt[(4096, 64, 1, "pytorch_sdpa")]["median"]))
    F2048 = 2 * 2048**2 * 128
    put("ptgf-2048-T1", f"{F2048 / (pt[(2048, 64, 1, 'pytorch_sdpa')]['median'] / 1000) / 1e9:.0f}")
    put("ptgf-4096-T1", f"{F / (pt[(4096, 64, 1, 'pytorch_sdpa')]['median'] / 1000) / 1e9:.0f}")
    pt_t4_gf = F / (pt[(4096, 64, 4, "pytorch_sdpa")]["median"] / 1000) / 1e9
    put("ptgf-4096-T4", f"{pt_t4_gf:.0f}")
    put("ptspeed-4096-T4", f"{pt[(4096, 64, 1, 'pytorch_sdpa')]['median'] / pt[(4096, 64, 4, 'pytorch_sdpa')]['median']:.2f}")
    put("ptspeed-4096-T8", f"{pt[(4096, 64, 1, 'pytorch_sdpa')]['median'] / pt[(4096, 64, 8, 'pytorch_sdpa')]['median']:.2f}")
    ptpers = pt[(4096, 64, 1, "pytorch_sdpa")]["median"] * 1e6 / 4096**2
    put("ptpers-4096", f"{ptpers:.2f}")
    put("ptpersratio", f"{min(dsc[(r, 'simd', L)]['a'] for r in ROUTES for L in (512, 2048)) / ptpers:.0f}")
    put("ptmaxabs", f"{max(v['max_abs'] for k, v in pt.items() if k[3] == 'pytorch_sdpa'):.1e}".replace("e-0", r"\times10^{-") + "}")
    # API vs native for the same custom route (T1 SIMD, L=4096)
    api_native = []
    for r in ROUTES:
        api = ratios[(4096, 64, 1)][1][r]
        nat = median(bm(r, "simd", 4096, 64))
        api_native.append(100 * (api / nat - 1))
    put("apinativemin", f"${min(api_native):+.1f}$")
    put("apinativemax", f"${max(api_native):+.1f}$")
    put("ptbatches", 1)
    put("ptsamples", tex_int(sum(1 for path in pt_dirs for _ in read_csv(Path(path) / "raw.csv"))))

    # ---------------------------------------------------- pooled-median robustness
    flips = 0
    for L, d in SHAPES:
        a = [median(bm(r, "scalar", L, d)) for r in ROUTES]
        b_ = [pooled(r, "scalar", L, d) for r in ROUTES]
        flips += int(np.argmin(a) != np.argmin(b_))
    pooled_t4 = all(pooled(r, "openmp", 4096, 64, 4) < pooled(r, "openmp", 4096, 64, 8) for r in ROUTES)
    CHECKS.append((pooled_t4, "pooled 60-sample medians also give T4 < T8 at L=4096 for all routes", 1, 1, 0))
    CHECKS.append((flips == 0, "pooled medians give the same fastest scalar route per shape", flips, 0, 0))

    # ---------------------------------------------------- tests (log)
    log = HERE / "test_logs" / "linux_gcc13.txt"
    total = 0
    if log.exists():
        txt = log.read_text(encoding="utf-8")
        for m in re.finditer(r"suite=(\S+) exit=(\d+) :: .*?(\d+) checks, (\d+) failures", txt):
            name, rc, n, f = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            assert rc == 0 and f == 0
            put("chk-" + name.replace("_", ""), tex_int(n))
            total += n
        put("chktotal", tex_int(total))
        m = re.search(r"NumPy FP64 vs C\+\+ FP64 oracle\s*: OK \(max \|diff\| = ([0-9.e+-]+)\)", txt)
        put("npdiff", f"{float(m.group(1)):.1e}".replace("e-", r"\times10^{-") + "}")
        put("sanok", "yes" if "7 of 7 exit=0" in txt else "no")
        mx = max(float(v) for v in re.findall(r"max_abs=([0-9.e+-]+)", txt))
        put("suitemaxabs", f"{mx:.1e}".replace("e-0", r"\times10^{-") + "}")
        put("suitemaxratio", f"{1e-4 / mx:.0f}")
        put("linuxgcc", re.search(r"g\+\+ .*? (\d+\.\d+\.\d+)", txt).group(1))

    # ---------------------------------------------------- tables
    write_tables(gen, st, bm, simd_sp, omp_sp, ws, dsc, tune, sel, cal, roof, model, pt, pt_cfgs, ratios)
    # ---------------------------------------------------- figures
    make_figures(figs, st, bm, simd_sp, omp_sp, calref, cal, roof, pt, pt_cfgs, ratios)

    # ---------------------------------------------------- outputs
    with open(gen / "numbers.tex", "w", encoding="utf-8") as f:
        f.write("% Generated by make_numbers.py -- do not edit by hand.\n")
        for k in sorted(NUM):
            f.write("\\expandafter\\def\\csname gpn@%s\\endcsname{%s}\n" % (k, NUM[k]))
    nbad = sum(1 for c in CHECKS if not c[0])
    with open(gen / "checks.txt", "w", encoding="utf-8") as f:
        f.write(f"{len(CHECKS)} comparisons, {nbad} disagreements\n")
        for ok, label, a, b, diff in CHECKS:
            if not ok:
                f.write(f"DISAGREE {label}: recomputed={a!r} archived={b!r} diff={diff!r}\n")
        f.write("All other comparisons agree within tolerance (relative 1e-9 for times, 1e-12 for ratios).\n")
    print(f"{len(NUM)} numbers, {len(CHECKS)} comparisons, {nbad} disagreements")


# ===================================================================== tables
def rng(lo, hi, f=fmt_ms):
    return f"{f(lo)}--{f(hi)}"


def write_tables(gen, st, bm, simd_sp, omp_sp, ws, dsc, tune, sel, cal, roof, model, pt, pt_cfgs, ratios):
    # Main route x mode table for L=128 and L=4096
    lines = []
    for L in (128, 4096):
        lines.append(r"\multicolumn{6}{l}{\textit{$L=%d$, $d=d_v=64$}}\\" % L)
        for r in ROUTES:
            cells = [RNAME[r]]
            for mode, T in (("scalar", 1), ("simd", 1), ("openmp", 4), ("openmp", 8)):
                xs = bm(r, mode, L, 64, T)
                cells.append(f"{fmt_ms(median(xs))} {{\\scriptsize[{fmt_ms(min(xs))}, {fmt_ms(max(xs))}]}}")
            b = ws[(r, "simd", L, 64, 1)]
            cells.append(human_bytes(b))
            lines.append(" & ".join(cells) + r"\\")
        if L == 128:
            lines.append(r"\addlinespace")
    (gen / "tab_main.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # SIMD speedup over scalar, six shapes
    lines = []
    for L, d in SHAPES:
        cells = [f"{L}", f"{d}"]
        for r in ROUTES:
            cells.append(fmt_ms(median(bm(r, "scalar", L, d))))
        for r in ROUTES:
            cells.append(fmt_ms(median(bm(r, "simd", L, d))))
        for r in ROUTES:
            m, lo, hi = simd_sp[(r, L, d)]
            cells.append(f"{m:.2f}")
        lines.append(" & ".join(cells) + r"\\")
    (gen / "tab_simd.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # d-scaling decomposition
    lines = []
    for mode in ("scalar", "simd"):
        for r in ROUTES:
            cells = [("Scalar" if mode == "scalar" else "AVX2") if r == ROUTES[0] else "", RNAME[r]]
            for L in (512, 2048):
                x = dsc[(r, mode, L)]
                cells += [f"{x['rho']:.2f}", f"{x['tot']:.1f}", f"{x['a']:.1f}", f"{100 * x['share']:.0f}\\%"]
            lines.append(" & ".join(cells) + r"\\")
        if mode == "scalar":
            lines.append(r"\addlinespace")
    (gen / "tab_dscale.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # thread scaling table (speedup and efficiency, L=128 and 4096)
    lines = []
    for L in (128, 4096):
        for r in ROUTES:
            cells = [f"{L}" if r == ROUTES[0] else "", RNAME[r], fmt_ms(median(bm(r, "openmp", L, 64, 1)))]
            for T in (2, 4, 8):
                m, lo, hi = omp_sp[(r, L, T)]
                cells.append(f"{m:.2f} {{\\scriptsize[{lo:.2f}, {hi:.2f}]}}")
            for T in (2, 4, 8):
                cells.append(f"{100 * omp_sp[(r, L, T)][0] / T:.0f}")
            lines.append(" & ".join(cells) + r"\\")
        if L == 128:
            lines.append(r"\addlinespace")
    lines.append(r"\addlinespace")
    cells = [r"\multicolumn{2}{l}{AVX2 reference loop}", fmt_ms(cal[("avx2_mul_add", 1)]["ms"])]
    for T in (2, 4, 8):
        cells.append(f"{T * cal[('avx2_mul_add', 1)]['ms'] / cal[('avx2_mul_add', T)]['ms']:.2f}")
    for T in (2, 4, 8):
        cells.append(f"{100 * cal[('avx2_mul_add', 1)]['ms'] / cal[('avx2_mul_add', T)]['ms']:.0f}")
    lines.append(" & ".join(cells) + r"\\")
    (gen / "tab_threads.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # tuning
    lines = []
    for bk in (128, 256, 512):
        cells = [str(bk), tex_int(tune[(1, bk)]["ws"])]
        for b in (1, 2, 3):
            cells.append(f"{tune[(b, bk)]['median']:.1f}")
        cells.append(f"{sel[bk]:.1f}")
        lines.append(" & ".join(cells) + r"\\")
    (gen / "tab_tuning.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # PyTorch
    lines = []
    prev = None
    for c in pt_cfgs:
        L_, d_, T_ = c
        if prev is not None and (prev[2] == 1) != (T_ == 1):
            lines.append(r"\addlinespace")
        prev = c
        ratio, cust = ratios[c]
        cells = [str(L_), str(d_), str(T_)] + [fmt_ms(cust[r]) for r in ROUTES]
        cells += [fmt_ms(pt[(L_, d_, T_, "pytorch_sdpa")]["median"]), f"{ratio:.1f}"]
        lines.append(" & ".join(cells) + r"\\")
    (gen / "tab_pytorch.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # roofline model table (L=4096, SIMD T1 and OpenMP T4)
    lines = []
    for r in ROUTES:
        for kind, label in (("once", "Q/K/V once"), ("refill", "K/V refetched")):
            x1 = roof[(r, "simd", 1, kind)]
            x4 = roof[(r, "openmp", 4, kind)]
            cells = [RNAME[r] if kind == "once" else "", label, human_bytes(model[(r, kind)]),
                     f"{x1['ai']:.2f}" if x1['ai'] < 100 else f"{x1['ai']:.0f}",
                     f"{x1['bw']:.2f}", f"{100 * x1['bw'] / x1['B']:.1f}\\%",
                     f"{x4['bw']:.2f}", f"{100 * x4['bw'] / x4['B']:.1f}\\%"]
            lines.append(" & ".join(cells) + r"\\")
    (gen / "tab_roofline.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def human_bytes(b):
    if b >= 2**30:
        return f"{b / 2**30:.2f} GiB"
    if b >= 2**20:
        v = b / 2**20
        return f"{v:.0f} MiB" if abs(v - round(v)) < 1e-9 else f"{v:.1f} MiB"
    if b >= 2**10:
        v = b / 2**10
        return f"{v:.0f} KiB" if abs(v - round(v)) < 1e-9 else f"{v:.1f} KiB"
    return f"{b} B"


# ==================================================================== figures
def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8,
        "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "lines.linewidth": 1.4, "lines.markersize": 4.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "legend.frameon": False,
        "axes.titlelocation": "left", "axes.titleweight": "bold",
    })


def make_figures(figs, st, bm, simd_sp, omp_sp, calref, cal, roof, pt, pt_cfgs, ratios):
    style()
    W = 6.85  # inches, full text width at 19 mm margins on A4

    # ---- Fig. SIMD: time per score and SIMD speedup across six shapes
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.3))
    labels = [f"{L}\n{d}" for L, d in SHAPES]
    x = np.arange(len(SHAPES))
    ax = axs[0]
    for r in ROUTES:
        for mode, ls, fill in (("scalar", "-", "white"), ("simd", "--", COLOR[r])):
            ys = [median(bm(r, mode, L, d)) * 1e6 / L**2 for L, d in SHAPES]
            ax.plot(x, ys, ls=ls, color=COLOR[r], marker=MARK[r], mfc=fill, mec=COLOR[r], mew=1.0)
    ax.set_xticks(x, labels)
    ax.set_xlabel("$L$ (top) and $d=d_v$ (bottom)")
    ax.set_ylabel("ns per score ($t/L^2$)")
    ax.set_ylim(0, 120)
    ax.axvline(3.5, color=GRID, lw=0.8)
    ax.set_title("(a) Latency per score, scalar vs AVX2", fontsize=8)
    ax = axs[1]
    wbar = 0.26
    for i, r in enumerate(ROUTES):
        ms = [simd_sp[(r, L, d)][0] for L, d in SHAPES]
        lo = [simd_sp[(r, L, d)][0] - simd_sp[(r, L, d)][1] for L, d in SHAPES]
        hi = [simd_sp[(r, L, d)][2] - simd_sp[(r, L, d)][0] for L, d in SHAPES]
        ax.bar(x + (i - 1) * wbar, ms, width=wbar - 0.03, color=COLOR[r],
               yerr=[lo, hi], error_kw=dict(ecolor=INK2, lw=0.7, capsize=1.5))
    ax.axhline(1, color=INK2, lw=0.7)
    ax.set_xticks(x, labels)
    ax.set_xlabel("$L$ (top) and $d=d_v$ (bottom)")
    ax.set_ylabel("AVX2 speedup over scalar (×)")
    ax.set_ylim(0, 2.6)
    ax.axvline(3.5, color=GRID, lw=0.8)
    ax.set_title("(b) Paired AVX2 speedup", fontsize=8)
    handles = [Patch(color=COLOR[r], label=RNAME[r]) for r in ROUTES]
    handles += [Line2D([], [], color=INK2, ls="-", marker="o", mfc="white", mec=INK2, label="scalar (a)"),
                Line2D([], [], color=INK2, ls="--", marker="o", mfc=INK2, mec=INK2, label="AVX2 (a)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04), handlelength=2.4)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2.0)
    fig.savefig(figs / "fig_simd.pdf")
    plt.close(fig)

    # ---- Fig. threads: latency and speedup at L=128 and L=4096
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.3))
    ax = axs[0]
    for r in ROUTES:
        ys = [median(bm(r, "openmp", 4096, 64, T)) for T in THREADS]
        lo = [ys[i] - min(bm(r, "openmp", 4096, 64, T)) for i, T in enumerate(THREADS)]
        hi = [max(bm(r, "openmp", 4096, 64, T)) - ys[i] for i, T in enumerate(THREADS)]
        ax.errorbar(THREADS, ys, yerr=[lo, hi], color=COLOR[r], marker=MARK[r], capsize=1.5, elinewidth=0.7, label=RNAME[r])
    ax.set_xscale("log", base=2)
    ax.set_xticks(THREADS, [str(t) for t in THREADS])
    ax.set_xlabel("threads $T$")
    ax.set_ylabel("latency (ms)")
    ax.set_ylim(0, None)
    ax.set_title("(a) Latency, $L=4096$", fontsize=8)
    for k, L in enumerate((128, 4096)):
        ax = axs[1 + k]
        ax.plot(THREADS, THREADS, ls=":", color=INK2, lw=0.9, label="ideal $S=T$")
        ax.plot(THREADS, [calref[T] for T in THREADS], ls="--", color=REF, lw=0.9, marker="x", ms=4,
                label="AVX2 reference loop")
        for r in ROUTES:
            ys = [omp_sp[(r, L, T)][0] for T in THREADS]
            lo = [omp_sp[(r, L, T)][0] - omp_sp[(r, L, T)][1] for T in THREADS]
            hi = [omp_sp[(r, L, T)][2] - omp_sp[(r, L, T)][0] for T in THREADS]
            ax.errorbar(THREADS, ys, yerr=[lo, hi], color=COLOR[r], marker=MARK[r], capsize=1.5, elinewidth=0.7, label=RNAME[r])
        ax.set_xscale("log", base=2)
        ax.set_xticks(THREADS, [str(t) for t in THREADS])
        ax.set_xlabel("threads $T$")
        ax.set_ylabel("speedup over OpenMP $T=1$")
        ax.set_ylim(0, 8.4)
        ax.set_title(f"({'bc'[k]}) Speedup, $L={L}$", fontsize=8)
    h, l = axs[2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.6)
    fig.savefig(figs / "fig_threads.pdf")
    plt.close(fig)

    # ---- Fig. PyTorch: API latency per configuration
    fig, ax = plt.subplots(figsize=(W, 2.45))
    x = np.arange(len(pt_cfgs))
    off = {"full_matrix": -0.18, "rowwise": -0.06, "blocked_streaming": 0.06}
    for r in ROUTES:
        ys = [ratios[c][1][r] for c in pt_cfgs]
        ax.scatter(x + off[r], ys, color=COLOR[r], marker=MARK[r], s=16, label=RNAME[r], zorder=3)
    ys = [pt[(c[0], c[1], c[2], "pytorch_sdpa")]["median"] for c in pt_cfgs]
    ax.scatter(x + 0.2, ys, color=INK, marker="D", s=14, label="PyTorch SDPA", zorder=3)
    for i, c in enumerate(pt_cfgs):
        ax.annotate(f"{ratios[c][0]:.0f}×", (x[i] + 0.2, ys[i]), xytext=(0, -9), textcoords="offset points",
                    ha="center", fontsize=6.3, color=INK2)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"$L$={c[0]}\n$d$={c[1]}\n$T$={c[2]}" for c in pt_cfgs], fontsize=6.3)
    n1 = sum(1 for c in pt_cfgs if c[2] == 1)
    ax.axvline(n1 - 0.5, color=GRID, lw=0.8)
    ax.set_ylabel("API call latency (ms, log)")
    ax.set_ylim(min(ys) / 2.2, None)
    ax.legend(loc="lower center", ncol=4, fontsize=6.6, bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(figs / "fig_pytorch.pdf")
    plt.close(fig)

    # ---- Fig. roofline placement
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.7))
    ai = np.logspace(-1, 3.2, 200)
    for T, shade in ((1, "#8c8b87"), (4, "#52514e")):
        C = cal[("avx2_mul_add", T)]["rate"]
        B = cal[("read_stream", T)]["rate"]
        ax.plot(ai, np.minimum(C, B * ai), color=shade, lw=1.0, ls="-" if T == 1 else "--")
        ax.text(ai[-1], C * 1.08, f"$T={T}$ reference: {C:.0f} GFLOP/s, {B:.0f} GB/s", ha="right", va="bottom", fontsize=6.3, color=shade)
    for r in ROUTES:
        for mode, T, fill in (("simd", 1, "white"), ("openmp", 4, COLOR[r])):
            a1 = roof[(r, mode, T, "refill")]["ai"]
            a2 = roof[(r, mode, T, "once")]["ai"]
            g = roof[(r, mode, T, "once")]["gf"]
            ax.plot([a1, a2], [g, g], color=COLOR[r], lw=0.8, alpha=0.8)
            ax.plot([a1, a2], [g, g], ls="none", marker=MARK[r], mfc=fill, mec=COLOR[r], mew=1.0,
                    label=f"{RNAME[r]}" if T == 1 else None)
    g1 = 2 * 4096**2 * 128 / (pt[(4096, 64, 1, "pytorch_sdpa")]["median"] / 1000) / 1e9
    g4 = 2 * 4096**2 * 128 / (pt[(4096, 64, 4, "pytorch_sdpa")]["median"] / 1000) / 1e9
    for g, lab in ((g1, "PyTorch SDPA, $T=1$ (API timing)"), (g4, "PyTorch SDPA, $T=4$ (API timing)")):
        ax.axhline(g, color=INK, lw=0.6, ls=":")
        ax.text(0.12, g * 1.07, lab, fontsize=6.3, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 1600)
    ax.set_ylim(2, 1500)
    ax.set_xlabel("modelled arithmetic intensity $F_{mat}/D_{DRAM}$ (FLOP/byte)")
    ax.set_ylabel("effective throughput $F_{mat}/t$ (GFLOP/s)")
    ax.legend(loc="lower right", fontsize=6.5, title="open: AVX2, $T=1$; filled: AVX2+OpenMP, $T=4$", title_fontsize=6.3)
    fig.tight_layout()
    fig.savefig(figs / "fig_roofline.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
