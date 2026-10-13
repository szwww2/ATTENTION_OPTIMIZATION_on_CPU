#!/usr/bin/env bash
# Build the scalar baseline and its contract suite (Linux/macOS, g++ or clang++).
# No auto-vectorization on purpose: this build is the baseline.
set -euo pipefail

CXX=${CXX:-g++}
STD="-std=c++17"
SRC="src/oracle_fp64.cpp src/full_matrix_scalar.cpp src/input.cpp src/metrics.cpp"
BASEFLAGS="$STD -O3 -fno-tree-vectorize -Wall -Wextra -Iinclude"

mkdir -p build

echo "[1/1] tests   : build/attn_tests   (scalar, -fno-tree-vectorize)"
$CXX $BASEFLAGS $SRC tests/test_contract.cpp -o build/attn_tests

echo "built: build/attn_tests"
