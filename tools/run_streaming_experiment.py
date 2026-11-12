"""Build and record Cui's scalar-only tile pilot; Python standard library only."""
import argparse
import csv
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["src/blocked_streaming_scalar.cpp", "src/oracle_fp64.cpp",
           "src/input.cpp", "src/metrics.cpp", "tools/bench_streaming.cpp"]
FLAGS = ["-std=c++17", "-O3", "-fno-tree-vectorize", "-ffp-contract=off",
         "-Wall", "-Wextra", "-Iinclude"]


def command(args):
    """在完整仓库执行命令，返回stdout；失败保留stderr并抛异常，不产生通过状态。"""
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True,
                            encoding="utf-8", errors="replace", timeout=600)
    if result.returncode:
        raise RuntimeError(f"{args}: exit {result.returncode}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def digest(path):
    """对文件字节生成SHA256；只读，用于区分相同git基线上的未提交实现。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(text, expected_keys, phase=0):
    """从原始CSV重算统计并核查预定网格/样本/契约/顺序/workspace；不运行kernel。"""
    groups = {}
    rounds = {}
    rows = list(csv.DictReader(io.StringIO(text)))
    for row in rows:
        key = tuple(int(row[k]) for k in ("L", "d", "dv", "key_block"))
        groups.setdefault(key, []).append(row)
        rounds.setdefault((key[:3], int(row["sample"])), []).append(row)
    if set(groups) != set(expected_keys):
        raise ValueError("Incomplete or unexpected experiment grid")
    for (shape, index), samples in rounds.items():
        choices = sorted(k[3] for k in expected_keys if k[:3] == shape)
        actual = sorted(samples, key=lambda r: int(r["order"]))
        if [int(r["order"]) for r in actual] != list(range(len(choices))):
            raise ValueError("Invalid execution order")
        if [int(r["key_block"]) for r in actual] != [choices[(j+index+phase)%len(choices)] for j in range(len(choices))]:
            raise ValueError("Candidate rotation mismatch")
    summaries = []
    for key, samples in sorted(groups.items()):
        L, d, dv, bk = key
        if len(samples) != 20 or sorted(int(r["sample"]) for r in samples) != list(range(20)):
            raise ValueError("Incomplete repetitions")
        times = [float(r["ms"]) for r in samples]
        for row in samples:
            expected = {"route":"blocked_streaming", "mode":"scalar", "threads":"1",
                        "query_block":"16", "warmup":"10", "reps":"20",
                        "seed":"20261022", "contract_pass":"1"}
            if any(row[k] != v for k,v in expected.items()):
                raise ValueError("Configuration or contract mismatch")
            eq, ek = min(16,L), min(bk,L)
            if int(row["workspace_bytes"]) != 4*(eq*ek+eq*dv+2*eq):
                raise ValueError("Workspace mismatch")
            if any(not math.isfinite(float(row[k])) or float(row[k])<0 for k in
                   ("pre_max_abs","pre_max_rel","post_max_abs","post_max_rel")):
                raise ValueError("Invalid errors")
        if not all(math.isfinite(t) and t>0 for t in times):
            raise ValueError("Invalid timings")
        q1, _, q3 = statistics.quantiles(times, method="inclusive")
        median = statistics.median(times)
        summaries.append(dict(L=L,d=d,dv=dv,query_block=16,key_block=bk,
            median_ms=median,q1_ms=q1,q3_ms=q3,iqr_ms=q3-q1,
            gflops=2*L*L*(d+dv)/(median*1e6),
            workspace_bytes=int(samples[0]["workspace_bytes"]),
            max_abs=max(float(r[k]) for r in samples for k in ("pre_max_abs","post_max_abs")),
            contract_pass=1))
    return summaries


def write_csv(path, rows):
    """写已核验的摘要；调用者保证输出目录全新，不覆盖其他实验。"""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """编译→调参→按中位数选择Bk→六形状pilot；保存状态、哈希和所有样本。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="new output directory, relative to repo")
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    metadata = {"status":"running", "started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "platform":platform.platform(), "python":platform.python_version(),
                "flags":FLAGS, "timing":"native complete scalar call; allocation/oracle/checks excluded",
                "selection_rule":"lowest tuning median; tie: smaller Bk; freeze before grid",
                "limits":"single local run, no affinity/power control; not final group speedup evidence"}
    try:
        compiler = shlex.split(os.environ.get("CXX", "g++"))
        metadata["compiler"] = command(compiler+["--version"])
        metadata["revision"] = command(["git","rev-parse","HEAD"]).strip()
        metadata["working_tree"] = command(["git","status","--short"])
        metadata["source_sha256"] = {p:digest(ROOT/p) for p in SOURCES+[
            "include/attn.h", "tools/run_streaming_experiment.py"]}
        metadata["thread_environment"] = {k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OMP_PROC_BIND","OMP_PLACES")}
        if hasattr(os, "sched_getaffinity"):
            metadata["allowed_cpus"] = sorted(os.sched_getaffinity(0))
        if platform.system()=="Linux":
            metadata["cpu"] = command(["lscpu","-J"])
        (ROOT/"build").mkdir(exist_ok=True)
        binary=ROOT/"build"/("attn_streaming_bench.exe" if os.name=="nt" else "attn_streaming_bench")
        build=compiler+FLAGS+SOURCES+["-o",str(binary)]
        metadata["build_command"]=build
        command(build)
        metadata["binary_sha256"]=digest(binary)
        tuning=command([str(binary),"tune"])
        tune_rows=summarize(tuning,[(2048,64,64,bk) for bk in (128,256,512)])
        (out/"tuning_raw.csv").write_text(tuning,encoding="utf-8")
        write_csv(out/"tuning_summary.csv",tune_rows)
        selected=min(tune_rows,key=lambda r:(r["median_ms"],r["key_block"]))["key_block"]
        metadata["selected_key_block"]=selected
        grid=command([str(binary),"grid",str(selected)])
        grid_rows=summarize(grid,[(L,d,d,selected) for L,d in
                                ((128,64),(512,64),(2048,64),(4096,64),(512,32),(2048,32))])
        (out/"grid_raw.csv").write_text(grid,encoding="utf-8")
        write_csv(out/"grid_summary.csv",grid_rows)
        metadata["status"]="passed"
        metadata["samples"]={"tuning":60,"grid":120}
        print(json.dumps({"selected_key_block":selected,"tuning":tune_rows,"grid":grid_rows},indent=2))
    except Exception as exc:
        metadata["status"]="failed"
        metadata["error"]=str(exc)
        raise
    finally:
        metadata["finished_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
        (out/"environment.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
