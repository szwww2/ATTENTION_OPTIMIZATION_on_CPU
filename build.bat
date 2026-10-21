@echo off
REM Build the scalar baselines, SIMD routes, OpenMP routes, tests, and helpers.
REM Scalar flags disable auto-vectorization; SIMD uses explicit AVX2 intrinsics.
setlocal

set CXX=g++
set STD=-std=c++17
set SRC=src\oracle_fp64.cpp src\full_matrix_scalar.cpp src\rowwise_scalar.cpp src\input.cpp src\metrics.cpp
set BASEFLAGS=%STD% -O3 -fno-tree-vectorize -Wall -Wextra -Iinclude
set SIMDFLAGS=%STD% -O3 -fno-tree-vectorize -mavx2 -Wall -Wextra -Iinclude
set OMPFLAGS=%SIMDFLAGS% -fopenmp

if not exist build mkdir build

echo [1/8] tests   : attn_tests.exe   (route 1, contract suite)
%CXX% %BASEFLAGS% %SRC% tests\test_contract.cpp -o build\attn_tests.exe
if errorlevel 1 exit /b 1

echo [2/8] harness : attn_bench.exe   (scalar, -fno-tree-vectorize)
%CXX% %BASEFLAGS% %SRC% tools\bench_scalar.cpp -o build\attn_bench.exe
if errorlevel 1 exit /b 1

echo [3/8] dump    : attn_dump.exe    (cross-check helper)
%CXX% %BASEFLAGS% %SRC% tools\dump_case.cpp -o build\attn_dump.exe
if errorlevel 1 exit /b 1

echo [4/8] blocks  : attn_blocks.exe  (route 2, block sizes and row order)
%CXX% %BASEFLAGS% %SRC% tests\test_blocks.cpp -o build\attn_blocks.exe
if errorlevel 1 exit /b 1

echo [5/8] simd kernels: attn_simd_kernels_tests.exe (AVX2 intrinsics)
%CXX% %SIMDFLAGS% tests\test_simd_kernels.cpp -o build\attn_simd_kernels_tests.exe
if errorlevel 1 exit /b 1

echo [6/8] simd routes : attn_simd_tests.exe (full matrix and row-wise contract suite)
%CXX% %SIMDFLAGS% -DATTN_TEST_SIMD %SRC% src\simd_routes.cpp tests\test_contract.cpp -o build\attn_simd_tests.exe
if errorlevel 1 exit /b 1

echo [7/8] OpenMP routes: attn_omp_tests.exe (static query blocks)
%CXX% %OMPFLAGS% %SRC% src\simd_routes.cpp src\omp_routes.cpp tests\test_omp_routes.cpp -o build\attn_omp_tests.exe
if errorlevel 1 exit /b 1

echo [8/8] Tang benchmark: attn_bench_tang.exe (scalar, SIMD, OpenMP)
%CXX% %OMPFLAGS% %SRC% src\simd_routes.cpp src\omp_routes.cpp tools\bench_tang.cpp -o build\attn_bench_tang.exe
if errorlevel 1 exit /b 1

echo built: build\attn_tests.exe  build\attn_bench.exe  build\attn_dump.exe  build\attn_blocks.exe  build\attn_simd_kernels_tests.exe  build\attn_simd_tests.exe  build\attn_omp_tests.exe  build\attn_bench_tang.exe
endlocal
