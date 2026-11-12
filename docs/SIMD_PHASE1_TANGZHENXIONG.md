# SIMD phase 1: full-matrix and row-wise AVX2 routes

Owner: Tang Zhenxiong. This phase adds complete basic SIMD implementations for the two scalar routes currently on `main`. It includes shared AVX2 dot-product and value-accumulation helpers, public route entry points, and the existing FP64 contract suite extended to both SIMD routes. Blocked-streaming integration will follow when that route is available. Timing comparisons are a separate phase; this phase makes no performance claim.

## Contract and implementation

`dot_avx2_fp32(a, b, n)` computes the FP32 dot product of two contiguous arrays. It uses unaligned AVX2 loads for each group of eight floats, reduces the eight lanes, and finishes any remaining elements with a scalar tail. `weighted_add_avx2_fp32(weight, values, accum, n)` broadcasts the weight and accumulates eight adjacent value dimensions at a time, followed by a scalar tail. Both helpers are inline to avoid a function call for every score or value row. Callers must compile with AVX2 enabled and run on an AVX2-capable CPU. No FMA instruction or approximate exponential is requested.

For the worked dot-product example in the learning discussion, the first eight products sum to 37 and the final two products sum to 1, giving 38. The tail is essential for dimensions such as 9, 10, 31, 33, or 65. The value-accumulation example uses nine value dimensions, also testing one AVX2 group plus a scalar tail.

SIMD changes the FP32 dot-product summation order, so the SIMD routes are not required to match the scalar routes bit for bit. The kernel test uses an FP32 rounding bound against an FP64 dot product. Every complete attention output also has to pass the existing fixed FP64 oracle contract. `test_contract.cpp` now checks both SIMD routes for finite output, equal scores, large finite score differences, odd `d` and `dv`, multiple query-block sizes, and repeated calls.

The SIMD value accumulation walks `V` by adjacent value dimensions, whereas the original scalar baseline walks key rows for each output dimension. Thus a future scalar-to-SIMD speedup includes the effects of this loop-order/locality change. The performance report should identify that combined change instead of attributing the whole difference to vector instructions alone.

## Verification on Tang's Windows machine

- Compiler: MSVC 19.44.35228, C++17, `/O2 /arch:AVX2`.
- `build/attn_simd_kernels_tests_msvc.exe`: 23 checks, 0 failures.
- `build/attn_simd_tests_msvc.exe`: 267 checks, 0 failures, including the original scalar checks.
- After the phase-3 `Bq=0` workspace fix and extra SIMD block-size case, the
  rebuilt route suite has 333 checks, 0 failures (2026-10-21). The 267 count
  above is the original phase-1 checkpoint.
- After aligning full-matrix SIMD with the incoming three-phase scalar route,
  the latest suite has 354 checks, 0 failures. The full-matrix workspace is
  now verified to contain row-normalized P.
- MSVC assembly generated from `src/simd_routes.cpp` shows `vmovups` (unaligned vector load), `vmulps` (eight FP32 products), and `vaddps` (vector accumulation) in both the score and value loops. Representative lines are `vmulps ymm1, ymm0, YMMWORD PTR [r8]` followed by `vaddps ymm2, ymm1, ymm2`, and later `vmulps ymm0, ymm3, YMMWORD PTR [r9+rcx]` followed by `vaddps ymm1, ymm0, YMMWORD PTR [rcx]`.
- The project `build.bat` and `build.sh` contain GCC AVX2 targets with `-mavx2 -fno-tree-vectorize`; those targets have not yet been run on this machine because `g++` is not installed in the command environment.

The low-level test includes lengths below, at, and above a multiple of eight and offsets input/output pointers to check unaligned access. The SIMD route tests use the same oracle and acceptance rule as the scalar suite.

To reproduce the MSVC checks from an x64 Developer Command Prompt in the repository root:

```bat
if not exist build mkdir build
cl /nologo /std:c++17 /O2 /arch:AVX2 /EHsc /Iinclude tests\test_simd_kernels.cpp /Fo:build\ /Fe:build\attn_simd_kernels_tests_msvc.exe
cl /nologo /std:c++17 /O2 /arch:AVX2 /EHsc /DATTN_TEST_SIMD /Iinclude src\oracle_fp64.cpp src\full_matrix_scalar.cpp src\rowwise_scalar.cpp src\input.cpp src\metrics.cpp src\simd_routes.cpp tests\test_contract.cpp /Fo:build\ /Fe:build\attn_simd_tests_msvc.exe
build\attn_simd_kernels_tests_msvc.exe
build\attn_simd_tests_msvc.exe
```

This MSVC build is a correctness check. The planned scalar-vs-SIMD timing comparison uses the separately flagged GCC builds after the benchmark harness is integrated.
