"""Build isolated AVX2/OpenMP streaming objects with GCC; optional ASan/UBSan."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["src/blocked_streaming_scalar.cpp", "src/blocked_streaming_optimized.cpp",
           "src/blocked_streaming_avx2.cpp", "src/oracle_fp64.cpp", "src/input.cpp", "src/metrics.cpp"]
HEADERS = ["include/attn.h", "include/attn_simd.h", "src/blocked_streaming_internal.h"]


def build(sanitize=False, benchmark=False):
    """独立编译每个源文件，仅AVX2核启用-mavx2；返回目标路径和实际命令供记录。"""
    folder = ROOT / "build" / ("streaming_sanitize" if sanitize else "streaming_optimized")
    folder.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get("CXX", "g++"))
    flags = ["-std=c++17", "-O1" if sanitize else "-O3", "-fno-tree-vectorize",
             "-ffp-contract=off", "-Wall", "-Wextra", "-Iinclude", "-fopenmp"]
    if sanitize:
        flags += ["-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-fno-pie", "-no-pie"]
    commands, objects = [], []
    for source in SOURCES:
        obj = folder / (Path(source).stem + ".o")
        cmd = compiler + flags + (["-mavx2"] if source.endswith("_avx2.cpp") else [])
        cmd += ["-c", source, "-o", str(obj)]
        subprocess.run(cmd, cwd=ROOT, check=True)
        commands.append(cmd)
        objects.append(str(obj))
    targets = {"test": "tests/test_streaming_optimized.cpp"}
    if benchmark:
        targets["bench"] = "tools/bench_streaming_optimized.cpp"
    binaries = {}
    for name, source in targets.items():
        binary = folder / (name + (".exe" if os.name == "nt" else ""))
        cmd = compiler + flags + [source] + objects + ["-o", str(binary)]
        subprocess.run(cmd, cwd=ROOT, check=True)
        commands.append(cmd)
        binaries[name] = binary
    return binaries, commands


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    paths, _ = build(args.sanitize, args.benchmark)
    for name, path in paths.items():
        print(f"{name}: {path}")
