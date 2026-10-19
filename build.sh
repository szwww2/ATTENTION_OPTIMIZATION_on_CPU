#!/usr/bin/env bash
# Build the scalar baselines, their suites and the helpers (Linux/macOS, g++ or clang++).
# No auto-vectorization on purpose: this build is the baseline.
set -euo pipefail

CXX=${CXX:-g++}
STD="-std=c++17"
SRC="src/oracle_fp64.cpp src/full_matrix_scalar.cpp src/rowwise_scalar.cpp src/input.cpp src/metrics.cpp"
BASEFLAGS="$STD -O3 -fno-tree-vectorize -Wall -Wextra -Iinclude"

mkdir -p build

echo "[1/4] tests   : build/attn_tests   (route 1, contract suite)"
"$CXX" $BASEFLAGS $SRC tests/test_contract.cpp -o build/attn_tests

echo "[2/4] harness : build/attn_bench   (scalar, -fno-tree-vectorize)"
"$CXX" $BASEFLAGS $SRC tools/bench_scalar.cpp -o build/attn_bench

echo "[3/4] dump    : build/attn_dump    (cross-check helper)"
"$CXX" $BASEFLAGS $SRC tools/dump_case.cpp -o build/attn_dump

echo "[4/4] blocks  : build/attn_blocks  (route 2, block sizes and row order)"
"$CXX" $BASEFLAGS $SRC tests/test_blocks.cpp -o build/attn_blocks

echo "flags=$BASEFLAGS" > build/scalar_build.txt
"$CXX" --version >> build/scalar_build.txt
echo "built: build/attn_tests  build/attn_bench  build/attn_dump  build/attn_blocks"
