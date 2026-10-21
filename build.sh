#!/usr/bin/env bash
# Build the scalar baselines, SIMD routes, OpenMP routes, tests, and helpers.
# Scalar flags disable auto-vectorization; SIMD uses explicit AVX2 intrinsics.
set -euo pipefail

CXX=${CXX:-g++}
STD="-std=c++17"
SRC="src/oracle_fp64.cpp src/full_matrix_scalar.cpp src/rowwise_scalar.cpp src/input.cpp src/metrics.cpp"
BASEFLAGS="$STD -O3 -fno-tree-vectorize -Wall -Wextra -Iinclude"
SIMDFLAGS="$STD -O3 -fno-tree-vectorize -mavx2 -Wall -Wextra -Iinclude"
OMPFLAGS="$SIMDFLAGS -fopenmp"

mkdir -p build

echo "[1/8] tests   : build/attn_tests   (route 1, contract suite)"
$CXX $BASEFLAGS $SRC tests/test_contract.cpp -o build/attn_tests

echo "[2/8] harness : build/attn_bench   (scalar, -fno-tree-vectorize)"
$CXX $BASEFLAGS $SRC tools/bench_scalar.cpp -o build/attn_bench

echo "[3/8] dump    : build/attn_dump    (cross-check helper)"
$CXX $BASEFLAGS $SRC tools/dump_case.cpp -o build/attn_dump

echo "[4/8] blocks  : build/attn_blocks  (route 2, block sizes and row order)"
$CXX $BASEFLAGS $SRC tests/test_blocks.cpp -o build/attn_blocks

echo "[5/8] simd kernels: build/attn_simd_kernels_tests (AVX2 intrinsics)"
$CXX $SIMDFLAGS tests/test_simd_kernels.cpp -o build/attn_simd_kernels_tests

echo "[6/8] simd routes : build/attn_simd_tests (full matrix and row-wise contract suite)"
$CXX $SIMDFLAGS -DATTN_TEST_SIMD $SRC src/simd_routes.cpp tests/test_contract.cpp -o build/attn_simd_tests

echo "[7/8] OpenMP routes: build/attn_omp_tests (static query blocks)"
$CXX $OMPFLAGS $SRC src/simd_routes.cpp src/omp_routes.cpp tests/test_omp_routes.cpp -o build/attn_omp_tests

echo "[8/8] Tang benchmark: build/attn_bench_tang (scalar, SIMD, OpenMP)"
$CXX $OMPFLAGS $SRC src/simd_routes.cpp src/omp_routes.cpp tools/bench_tang.cpp -o build/attn_bench_tang

echo "built: build/attn_tests  build/attn_bench  build/attn_dump  build/attn_blocks  build/attn_simd_kernels_tests  build/attn_simd_tests  build/attn_omp_tests  build/attn_bench_tang"
