# Integrated scalar, SIMD and OpenMP benchmark

Runs full-matrix and rowwise attention in scalar, AVX2 and AVX2+OpenMP modes through the same validation and timing code.

Add `--streaming --key-block 256` to include Cui's blocked-streaming route in
the same loop. `all` then runs nine variants; `serial` runs six and `openmp`
runs three. The default remains the previous two-route interface.

## Build and test

GCC, Python 3 and an AVX2-capable CPU are required. Set `CXX` if `g++` is not on PATH.

```text
python tools/build_integrated.py
build\attn_tests.exe
build\attn_blocks.exe
build\attn_simd_kernels_tests.exe
build\attn_simd_tests.exe
build\attn_omp_tests.exe
build\attn_streaming_tests.exe
build\attn_streaming_optimized_tests.exe
python tests/test_nine_mode_benchmark.py
python tests/test_integrated_benchmark.py build/attn_bench_all.exe
```

Linux uses the same build script and executable names without `.exe`; validation so far is Windows/GCC only.
The benchmark checks AVX2 support before calling the SIMD kernels. The standalone SIMD tests require AVX2.
Scalar objects compile without `-mavx2`; the SIMD implementation uses explicit AVX2 intrinsics.
Automatic vectorization and FMA contraction are disabled. The original scalar-only build remains available.

## Run

```text
python tools/run_scalar_bench.py --suite optimized --mode all --threads 4 --L 512 --query-block 16 --out results/six_variants
python tools/run_scalar_bench.py --suite optimized --streaming --key-block 256 --mode all --threads 4 --L 512 --query-block 16 --out results/nine_variants
```

Modes: `scalar` (two variants), `simd` (two), `serial` (scalar and SIMD, four), `openmp` (two), `all` (six).
`--threads` applies to OpenMP variants; scalar and serial SIMD use one thread.
Use a new output prefix for every run. See [SCALAR_BENCHMARK.md](SCALAR_BENCHMARK.md) for common fields and timing scope.
On the tested Windows/GCC build, use an ASCII relative output prefix such as
`results/nine_variants`. Non-ASCII output names can fail in the C++ filesystem
conversion; the repository itself can reside in a Chinese-named directory.

The additional CSV fields are `mode`, `threads` and `actual_threads`. Full/row
use a preflight team probe. Streaming records its actual compute team and checks
it after each timed call. A mismatch stops the run. Dynamic teams are disabled.
With streaming enabled, CSVs also record effective `key_block`,
`requested_query_block` and `requested_key_block`. Effective Bk is zero for
full/row. Streaming scratch includes all score tiles and online softmax state:
`T*(bq*bk + bq*dv + 2*bq)` floats. Blocks are clipped to L.
The input buffers are compared with their saved copies after all timed calls.
All selected variants share inputs and an FP64 reference, pass validation before timing, and rotate execution order each round.
Rowwise OpenMP receives T*Bq*L floats of scratch; full-matrix OpenMP uses L*L.
Environment records include build flags, relevant source/header hashes and OpenMP environment settings.

## Proposal grid

```text
python tools/run_project_grid.py --physical-cores 16 --out results/grid_plan --dry-run
python tools/run_project_grid.py --physical-cores 16 --out results/grid_run
python tools/run_project_grid.py --physical-cores 16 --streaming --key-block 256 --out results/three_routes
```

Replace 16 with the machine's verified physical-core budget. The plan uses all six proposal shapes for scalar/SIMD, plus L=128 and 4096 at T=1,2,4,min(8,P), removing duplicates and counts above P.
With P>=8 the default two-route grid has 40 configurations. Enabling streaming
produces 60 configurations and 1,200 samples at the default repetitions.
The default Bq=16 is fixed for this run, not a completed blocked-route tuning result. The manifest records memory preflight, parameters and each run's outcome; memory allowance is an estimate, not measured RSS.
Defaults are 10 warm-ups, 20 samples and a 512 MiB planning budget. `--warmup`, `--reps` and `--memory-mib` can override these before measurement. Execution stops on the first failed configuration.

## Integration changes

SIMD/OpenMP kernels and their tests originate from Tang Zhenxiong's branch at `f9f088c`. The shared benchmark and environment runner build on Chen Caoyu's scalar harness.

The integration branch now includes both Cui's `e12b93f` and Tang's `f9f088c`
in its Git history. Tang's merge retained the integrated versions of README,
build.bat, build.sh, include/attn.h, src/simd_routes.cpp and
tests/test_simd_kernels.cpp. Those versions contain the common build/benchmark
setup, the phase/normalization corrections below and the non-finite test fix.
Tang's additional tools, documents and original measurements are preserved.
His MSVC results and Cui's earlier results remain separate historical experiments;
they are not pooled into the new nine-variant measurements.

- Rowwise SIMD now normalizes each probability before PV, matching the scalar version. Tang's original version divided the accumulated output instead.
- SIMD PV still uses a different loop order from scalar PV. Results compare these implementations; they do not isolate vector instructions from locality and loop-order effects.
- The primitive test now rejects non-finite value-accumulation output. Both existing scalar suites remain in place.
- Scalar and SIMD cases in the comparison come from this integrated build. Earlier Tang/Chen measurements are kept separate.

The later rowwise workspace guard rejects element/byte-count overflow for
extremely large direct-API block or thread arguments. It preserves normal sizes
and does not change the attention loops. Rebuild after pulling this header change;
the runner rejects a binary whose recorded sources no longer match. Archived
23 October timings remain tied to their recorded source and executable, not to
this later build.

CPU affinity and power mode are not controlled by these scripts. Repeat runs under documented conditions before making final performance claims. Workspace bytes are not measured DRAM traffic.

After a grid completes, use [RESULT_ANALYSIS.md](RESULT_ANALYSIS.md) to check its saved records and produce comparison plots.
