# Tang's SIMD/OpenMP integration handoff

This is a proposed handoff for Cui Ziwei's blocked-streaming route and Chen
Caoyu's shared benchmark. It describes Tang's `dev/tangzhenxiong` development
branch. Do not assume these symbols are already available on GitHub `main`.

## Reusable SIMD primitives

Include `include/attn_simd.h` from an AVX2-enabled translation unit:

```cpp
float attn::dot_avx2_fp32(const float* q_row, const float* k_row, std::size_t d);
void attn::weighted_add_avx2_fp32(float weight, const float* v_row,
                                  float* accumulator, std::size_t dv);
```

Both accept unaligned FP32 arrays, handle 8-wide groups and a scalar tail,
and do not allocate. `dot_avx2_fp32` requires contiguous Q and K feature
rows; `weighted_add_avx2_fp32` requires a contiguous V row and output vector.
Neither implements softmax, online-max rescaling, tile scheduling or a
blocked-streaming workspace. The streaming owner must retain control of
those choices and ensure the FP64 oracle contract passes at every block size.

## Existing route signatures

`include/attn.h` now declares full-matrix and row-wise serial SIMD plus static
OpenMP entry points. OpenMP functions take `query_block`, `threads`, caller-
allocated `score_ws` and caller-allocated `out`. Required FP32 workspace:

- full-matrix SIMD / OpenMP: `L*L` floats;
- row-wise SIMD: `Bq*L` floats;
- row-wise OpenMP: `T*Bq*L` floats.

`Bq=0` and non-positive `threads` fall back to one in the corresponding
kernels and workspace helpers. All arrays are row-major, with Q/K of `L*d`
and V/output of `L*dv` floats. No kernel allocates or performs I/O. The
full-matrix route is **three-phase** (all scores, all normalized P, all P*V),
whereas row-wise completes each query row immediately. Full-matrix OpenMP
has barriers between phases; row-wise OpenMP has per-worker scratch.

For a blocked-streaming OpenMP variant, a practical analogous API would take
its tile sizes, thread count, caller workspace and output, and split over
independent query blocks with `schedule(static)`. The exact signature and
online-softmax state layout belong to Cui. A worker's online maximum,
denominator and value accumulator must not be shared with another worker.
No OpenMP region should be opened inside the inner dot or value loop.

## Test and benchmark integration

- Fixed contract: `|O32-O64| <= 1e-4 + 1e-3*|O64|` for every output element.
- SIMD and OpenMP tests already cover odd dimensions, non-divisible query
  blocks, equal scores, large score spread, repeated calls and T=1/2/4.
  Streaming additionally needs key-block-size/order tests with tolerance,
  not bitwise equality across online-softmax blockings.
- `tools/bench_tang.cpp` is a standalone two-route comparison and emits raw
  samples. Chen's fetched `dev/chencaoyu-harness` branch has a different,
  stronger scalar runner with environment recording and alternating order.
  After the branches are merged, use its provenance conventions and add all
  SIMD/OpenMP route calls to the single shared harness. Do not combine the
  old local main scalar numbers with Chen's three-phase scalar results.
- GCC targets in `build.bat` and `build.sh` are untested on Tang's machine.
  MSVC local builds use `/O2 /arch:AVX2 /openmp`; executable CPUs need AVX2.

The current local correctness results and raw timing limitations are in
`REVIEW_NOTES_TANGZHENXIONG.md` and `EXPERIMENT_PHASE3_TANGZHENXIONG.md`.
