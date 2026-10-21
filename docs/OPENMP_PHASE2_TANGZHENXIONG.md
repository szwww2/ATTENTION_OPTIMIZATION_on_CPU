# OpenMP phase 2: full-matrix and row-wise SIMD routes

Owner: Tang Zhenxiong. This phase adds static OpenMP query-block parallelism to the two SIMD routes available on the repository's current `main`. The row-wise route reuses the single-row SIMD computation; the full-matrix route now uses the same SIMD primitives in three whole-matrix phases. Blocked-streaming integration remains a separate phase until Cui Ziwei's route is available. Local exploratory timings are in `EXPERIMENT_PHASE3_TANGZHENXIONG.md`.

## Scheduling and workspace

`full_matrix_simd_omp_fp32` and `rowwise_simd_omp_fp32` divide output queries into blocks of `Bq` rows. OpenMP assigns blocks with `schedule(static)`. All input arrays are read-only, and each output row has exactly one owner. The parallel region is entered on every attention call; a later native timing run will include this overhead.

The full-matrix route keeps its existing `L * L` score workspace. Distinct query blocks write distinct rows of that matrix; OpenMP barriers separate score generation, row normalization and P*V. The row-wise route requires `T * Bq * L` float elements; worker `tid` uses the slice starting at `tid * Bq * L`. Row maxima, denominators, and row pointers are ordinary local variables inside the worker's call. The caller allocates both workspace and output before entering the kernel.

For example, `L=10`, `Bq=3`, and `T=2` produce four query blocks of lengths 3, 3, 3, and 1. The row-wise score scratch is 60 floats: 30 for each worker. The full-matrix score workspace is 100 floats, partitioned by query row.

`query_block=0` and non-positive thread counts fall back to one. The test uses `omp_set_dynamic(0)` so its requested counts are actually exercised; benchmark settings must record the thread count and affinity separately.

## Verification on Tang's Windows machine

- Compiler: MSVC 19.44.35228 with C++17, `/O2 /arch:AVX2 /openmp`.
- `build/attn_omp_tests_msvc.exe`: 418 checks, 0 failures. The suite exercises T=1, 2, 4 and Bq=1, 7, 512 for single-row, odd `d`/`dv`, equal scores, large finite score differences, and repeated calls. It checks the fixed FP64 oracle contract, bitwise agreement with the corresponding single-thread SIMD route, workspace boundary guards, and actual OpenMP thread creation.
- After the phase-3 `Bq=0` workspace fix and extra block-size case, the rebuilt
  OpenMP suite has 551 checks, 0 failures (2026-10-21). The 418 count above is
  the original phase-2 checkpoint.
- After the full-matrix three-phase correction, the latest suite has 635
  checks, 0 failures. It also compares the materialized P workspace bitwise.
- Rebuilt phase-1 `attn_simd_tests_msvc.exe` after the row-helper refactor: 267 checks, 0 failures.
- The GCC `-fopenmp` target is in `build.bat` and `build.sh`, but was not run on this machine because `g++` is not installed in the command environment.

To reproduce the OpenMP correctness suite from an x64 Developer Command Prompt in the repository root:

```bat
if not exist build mkdir build
cl /nologo /std:c++17 /O2 /arch:AVX2 /openmp /EHsc /Iinclude src\oracle_fp64.cpp src\full_matrix_scalar.cpp src\rowwise_scalar.cpp src\input.cpp src\metrics.cpp src\simd_routes.cpp src\omp_routes.cpp tests\test_omp_routes.cpp /Fo:build\ /Fe:build\attn_omp_tests_msvc.exe
build\attn_omp_tests_msvc.exe
```

The local exploratory timing method and limitations are in `EXPERIMENT_PHASE3_TANGZHENXIONG.md`. Correctness checks and the FP64 oracle stay outside timed regions.
