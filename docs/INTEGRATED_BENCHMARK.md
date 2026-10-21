# Integrated scalar, SIMD and OpenMP benchmark

Runs full-matrix and rowwise attention in scalar, AVX2 and AVX2+OpenMP modes through the same validation and timing code.

## Build and test

GCC, Python 3 and an AVX2-capable CPU are required. Set `CXX` if `g++` is not on PATH.

```text
python tools/build_integrated.py
build\attn_tests.exe
build\attn_blocks.exe
build\attn_simd_kernels_tests.exe
build\attn_simd_tests.exe
build\attn_omp_tests.exe
python tests/test_integrated_benchmark.py build/attn_bench_all.exe
```

Linux uses the same build script and executable names without `.exe`; validation so far is Windows/GCC only.
The benchmark checks AVX2 support before calling the SIMD kernels. The standalone SIMD tests require AVX2.
Scalar objects compile without `-mavx2`; the SIMD implementation uses explicit AVX2 intrinsics.
Automatic vectorization and FMA contraction are disabled. The original scalar-only build remains available.

## Run

```text
python tools/run_scalar_bench.py --suite optimized --mode all --threads 4 --L 512 --query-block 16 --out results/six_variants
```

Modes: `scalar` (two variants), `simd` (two), `serial` (scalar and SIMD, four), `openmp` (two), `all` (six).
`--threads` applies to OpenMP variants; scalar and serial SIMD use one thread.
Use a new output prefix for every run. See [SCALAR_BENCHMARK.md](SCALAR_BENCHMARK.md) for common fields and timing scope.

The additional CSV fields are `mode`, `threads` and `actual_threads`. The last field is verified by an OpenMP team probe before timing; timed calls are not instrumented. A mismatch stops the run. Dynamic teams are disabled.
All selected variants share inputs and an FP64 reference, pass validation before timing, and rotate execution order each round.
Rowwise OpenMP receives T*Bq*L floats of scratch; full-matrix OpenMP uses L*L.
Environment records include build flags, relevant source/header hashes and OpenMP environment settings.

## Proposal grid

```text
python tools/run_project_grid.py --physical-cores 16 --out results/grid_plan --dry-run
python tools/run_project_grid.py --physical-cores 16 --out results/grid_run
```

Replace 16 with the machine's verified physical-core budget. The plan uses all six proposal shapes for scalar/SIMD, plus L=128 and 4096 at T=1,2,4,min(8,P), removing duplicates and counts above P.
With P>=8 this is 40 available configurations across the two implemented routes. Blocked-streaming configurations are explicitly listed as unavailable, not integrated.
The default Bq=16 is fixed for this run, not a completed blocked-route tuning result. The manifest records memory preflight, parameters and each run's outcome; memory allowance is an estimate, not measured RSS.
Defaults are 10 warm-ups, 20 samples and a 512 MiB planning budget. `--warmup`, `--reps` and `--memory-mib` can override these before measurement. Execution stops on the first failed configuration.

## Integration changes

SIMD/OpenMP kernels and their tests originate from Tang Zhenxiong's branch at `f9f088c`. The shared benchmark and environment runner build on Chen Caoyu's scalar harness.

- Rowwise SIMD now normalizes each probability before PV, matching the scalar version. Tang's original version divided the accumulated output instead.
- SIMD PV still uses a different loop order from scalar PV. Results compare these implementations; they do not isolate vector instructions from locality and loop-order effects.
- The primitive test now rejects non-finite value-accumulation output. Both existing scalar suites remain in place.
- Scalar and SIMD cases in the comparison come from this integrated build. Earlier Tang/Chen measurements are kept separate.

CPU affinity and power mode are not controlled by these scripts. Repeat runs under documented conditions before making final performance claims. Workspace bytes are not measured DRAM traffic.
