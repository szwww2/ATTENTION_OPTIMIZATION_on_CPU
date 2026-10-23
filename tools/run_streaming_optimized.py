"""Record a same-build scalar/SIMD/OpenMP pilot, keeping all raw timings."""
import argparse
import csv
import datetime
import hashlib
import io
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
from pathlib import Path
from build_streaming_optimized import ROOT, SOURCES, HEADERS, build

SHAPES = [(128,64,64),(512,64,64),(2048,64,64),(4096,64,64),(512,32,32),(2048,32,32)]


def summarize(text):
    """严格核验24配置×20样本、轮换顺序、实际team、scratch及契约，返回摘要。"""
    expected = {}
    for shape in SHAPES:
        modes = [("scalar",1),("simd",1),("simd_omp",1)]
        if shape[0] in (128,4096):
            modes += [("simd_omp",t) for t in (2,4,8)]
        for index,(mode,threads) in enumerate(modes):
            expected[shape+(mode,threads)] = (index,len(modes))
    groups = {}
    for r in csv.DictReader(io.StringIO(text)):
        key = tuple(int(r[k]) for k in ("L","d","dv"))+(r["mode"],int(r["requested_threads"]))
        if key not in expected:
            raise ValueError("unexpected configuration")
        groups.setdefault(key,[]).append(r)
    if set(groups) != set(expected):
        raise ValueError("incomplete grid")
    result = []
    for key,rows in groups.items():
        L,d,dv,mode,threads = key
        index,count = expected[key]
        if len(rows)!=20 or sorted(int(r["sample"]) for r in rows)!=list(range(20)):
            raise ValueError("missing or duplicate sample")
        for r in rows:
            settings = {"query_block":16,"key_block":256,"effective_bq":min(16,L),
                        "effective_bk":min(256,L),"seed":20261022,"warmup":10,"reps":20,"contract_pass":1,
                        "workspace_bytes":4*threads*min(16,L)*(min(256,L)+dv+2)}
            if any(int(r[k])!=v for k,v in settings.items()):
                raise ValueError("contract/configuration/scratch mismatch")
            if (int(r["order"])+int(r["sample"]))%count != index or not 0<=int(r["order"])<count:
                raise ValueError("invalid rotation")
            actual=int(r["actual_threads"])
            if not 1<=actual<=threads:
                raise ValueError("invalid actual team")
            for field in ("ms","pre_max_abs","pre_max_rel","post_max_abs","post_max_rel"):
                value=float(r[field])
                if not math.isfinite(value) or value<0 or (field=="ms" and value==0):
                    raise ValueError("invalid numeric record")
        times=[float(r["ms"]) for r in rows]
        q1,_,q3=statistics.quantiles(times,method="inclusive")
        teams=sorted({int(r["actual_threads"]) for r in rows})
        # 同一配置team变化时保留原始数据，但拒绝把不同team混为一个性能摘要。
        if len(teams)!=1:
            raise ValueError("variable actual team; do not pool timings")
        result.append(dict(L=L,d=d,dv=dv,mode=mode,requested_threads=threads,actual_threads=teams[0],
                           median_ms=statistics.median(times),iqr_ms=q3-q1,
                           workspace_bytes=int(rows[0]["workspace_bytes"]),
                           max_abs=max(float(r[k]) for r in rows for k in ("pre_max_abs","post_max_abs"))))
    for row in result:
        peers=[r for r in result if (r["L"],r["d"],r["dv"])==(row["L"],row["d"],row["dv"])]
        scalar=next(r["median_ms"] for r in peers if r["mode"]=="scalar")
        simd=next(r["median_ms"] for r in peers if r["mode"]=="simd")
        row["speedup_vs_scalar"]=scalar/row["median_ms"]
        row["speedup_vs_simd"]=simd/row["median_ms"]
    return result


def main():
    """建新目录、独立编译、运行固定矩阵并保存来源/运行时设置；失败也保留记录。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    out=(ROOT/args.out).resolve(); out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    # 绑定WSL可见core，明确控制动态team与忙等；不声称等于宿主物理核隔离。
    env.update(OMP_DYNAMIC="FALSE",OMP_PROC_BIND="close",OMP_PLACES="cores",OMP_WAIT_POLICY="PASSIVE")
    meta={"status":"running","started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "platform":platform.platform(),"python":platform.python_version(),
          "thread_environment":{k:v for k,v in env.items() if k.startswith(("OMP_","GOMP_","KMP_"))},
          "scope":"route3 only; fixed Bq16/Bk256; same-build scalar baseline; one local pilot"}
    def run(cmd):
        """同一环境执行并捕获输出；非零返回不视为通过。"""
        return subprocess.run(cmd,cwd=ROOT,env=env,text=True,capture_output=True,check=True,timeout=600).stdout
    try:
        meta["revision"]=run(["git","rev-parse","HEAD"]).strip()
        meta["working_tree"]=run(["git","status","--short"])
        meta["compiler"]=run(shlex.split(os.environ.get("CXX","g++"))+["--version"])
        if platform.system()=="Linux":
            meta["cpu"]=json.loads(run(["lscpu","-J"]))
            meta["allowed_cpus"]=sorted(os.sched_getaffinity(0))
        files=SOURCES+HEADERS+["tools/bench_streaming_optimized.cpp","tools/build_streaming_optimized.py",
                             "tools/run_streaming_optimized.py","tests/test_streaming_optimized.cpp"]
        meta["source_sha256"]={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
        binaries,meta["build_commands"]=build(benchmark=True)
        meta["binary_sha256"]=hashlib.sha256(binaries["bench"].read_bytes()).hexdigest()
        if platform.system()=="Linux":
            meta["linked_libraries"]=run(["ldd",str(binaries["bench"])])
        meta["tests"]=run([str(binaries["test"])])
        raw=run([str(binaries["bench"])])
        (out/"raw.csv").write_text(raw,encoding="utf-8")
        rows=summarize(raw)
        with (out/"summary.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        meta.update(status="passed",configurations=len(rows),samples=20*len(rows))
        print(json.dumps(rows,indent=2))
    except Exception as exc:
        meta.update(status="failed",error=str(exc))
        if isinstance(exc,subprocess.CalledProcessError):
            (out/"failed_stdout.txt").write_text(exc.stdout or "",encoding="utf-8")
            (out/"failed_stderr.txt").write_text(exc.stderr or "",encoding="utf-8")
        raise
    finally:
        meta["finished_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
        (out/"environment.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
