# Streaming SIMD + static OpenMP handoff

Implemented locally on `dev/cuiziwei`, 2026-10-22, following the user's explicit
request to implement before discussing integration with teammates. This extends
route 3; it does not merge or replace Tang/Chen's draft route-1/2 work.

## Ownership and source

`include/attn_simd.h` is reused **unchanged** from Tang's remote-tracking snapshot
`f9f088c1ab7d705965712a0a50318e7a2abf059f`. Credit the shared dot-product and
weighted-value-add helpers to that work. This patch adds their use in the
streaming loop, AVX2 state rescaling/output division, baseline-ISA dispatch,
query-block parallelism, workspace validation, tests and a route-3 pilot.

- `src/blocked_streaming_avx2.cpp`: single-query-tile implementation; only this
  translation unit is compiled with `-mavx2`.
- `src/blocked_streaming_optimized.cpp`: public entry points, CPU/OS capability
  check, size/null checks and `schedule(static)` OpenMP query-block loop.
- `src/blocked_streaming_internal.h`: private tile entry, not a harness API.
- The existing scalar source and earlier saved measurements are preserved.

## Computation

AVX2 covers Q·K, weighted V accumulation, rescaling u when m increases, and
final u/l. Eight FP32 lanes, unaligned load/store, scalar tails, no approximate
reciprocal or fast exp. Max/exp/normalizer reduction stays scalar. Dot reduction
order differs from scalar, so use the existing FP64 tolerance, not scalar bitwise
equality. `-ffp-contract=off` and no fast-math avoid introducing FMA contraction.

OpenMP assigns whole query blocks, not key blocks: each query keeps its entire
online m/l/u recurrence on one worker. Each worker uses its own scratch slice,
and query output ranges are disjoint. No atomics, shared softmax reduction, or
key-block synchronization are needed. In the tested builds, SIMD and OpenMP
results are bitwise equal across thread counts because row arithmetic is unchanged.

## Public API (include/attn.h)

```cpp
bool blocked_streaming_avx2_supported();
void blocked_streaming_simd_fp32(
    const Shape&, const float* q, const float* k, const float* v,
    std::size_t Bq, std::size_t Bk, float* workspace, float* out);
std::size_t blocked_streaming_omp_workspace_elems(
    const Shape&, std::size_t Bq, std::size_t Bk, int threads);
void blocked_streaming_simd_omp_fp32(
    const Shape&, const float* q, const float* k, const float* v,
    std::size_t Bq, std::size_t Bk, int threads,
    float* workspace, float* out, int* actual_threads = nullptr);
```

Single-thread SIMD uses `blocked_streaming_workspace_elems`. OpenMP helper
returns **requested_threads × single-thread scratch**, in float elements,
including every score tile, maximum, normalizer and accumulator. There is no
padding or alignment prerequisite. Bq/Bk normalization matches scalar: zero→1,
above L→L. All dimensions and requested thread count must be positive; byte
overflow/null pointers are rejected before computation. Buffer lengths and
non-aliasing are caller responsibilities. `actual_threads`, if supplied, must
not alias another buffer. Finite, representable input/score assumptions match
the scalar implementation.

For Bq16/Bk256,dv64,L>=256: one worker uses 20,608 bytes; requested T8 uses
164,864 bytes. Runtime constraints/dynamic teams/nesting may reduce actual team
size. No global OpenMP setting is changed by the kernel. The optional actual
team output is written inside the same parallel region used for computation.

Application code does no dynamic allocation or I/O on valid calls. **The OpenMP
runtime may allocate internally**, particularly on first use or team changes.
Every timed call includes team-region entry, worksharing, barriers and exit;
the pilot's warm-ups exclude cold-start cost from the reported steady-state samples.

## Build and verification

GCC-compatible x86 compiler with OpenMP, Python 3 standard library:

```sh
python3 -B tools/build_streaming_optimized.py
OMP_DYNAMIC=FALSE build/streaming_optimized/test
OMP_THREAD_LIMIT=2 OMP_DYNAMIC=TRUE build/streaming_optimized/test
python3 -B tools/build_streaming_optimized.py --sanitize
OMP_DYNAMIC=FALSE build/streaming_sanitize/test
python3 -B tests/test_streaming_optimized_records.py
```

Only the AVX2 source gets AVX2 flags; scalar/oracle/dispatch/test objects use the
baseline ISA. All objects disable auto-vectorization and FMA contraction. Do
not add global `-march=native`, AVX2 flags or cross-TU LTO when adopting this build
without rechecking ISA isolation. `build.sh` remains the scalar-only build.
On GCC/Clang x86, dispatch checks AVX2 CPU/OS support and throws before writes
if unavailable; tests return skip code 77. Actual non-AVX2 hardware was not
available to verify that path. This is not yet a tested MSVC/ARM implementation.
Windows GCC build via the Python builder is provided structurally but untested.

Observed: **13,181 checks, zero failures**, max abs error **2.59116849e-6**;
normal runtime observed teams up to 8. ASan/UBSan passed the same suite. Under
OMP_THREAD_LIMIT=2/dynamic teams it passed with largest observed team 2.
Coverage includes seeds, tiny/odd dimensions, SIMD tails, zero/oversized tiles,
equal/large/rising scores, poisoned buffers, guards, repeated calls, thread
counts 1/2/3/4/8, fewer query blocks than workers, nested/concurrent independent
calls, invalid thread counts and scratch overflow. Sanitizers are memory/UB
checks, not a race-detector result; no race-detector run is claimed.

Existing 45/273/11,498 scalar checks and three scalar-record tests also passed.
Four optimized-record tests cover missing/duplicate samples, corruption and
variable-team rejection.

## Measurement and integration

```sh
python3 -B tools/run_streaming_optimized.py --out results/streaming_opt_new
```

See [the actual local pilot](../examples/streaming_optimized_20261022/README.md).
It rebuilds all modes together, uses Bq16/Bk256 frozen from the earlier pilot,
and records requested/effective blocks, requested/actual threads, total scratch,
source/binary hashes, environment, raw samples and median/IQR. It does not retune
blocks for SIMD. Six shapes × scalar/SIMD/OpenMP-T1, plus T2/4/8 at L128/4096 =
24 configurations, 480 timed calls. Each has ten warm-ups and twenty repetitions;
configuration order rotates. FP64 validation is before timing and after the last
call, not after every timed sample; input preservation is checked.

The driver sets OMP_DYNAMIC=FALSE, OMP_PROC_BIND=close, OMP_PLACES=cores,
OMP_WAIT_POLICY=PASSIVE; inherited OMP/GOMP/KMP settings are recorded. Binding
refers to WSL-visible topology, not verified host physical-core isolation.

For teammate discussion:

1. Confirm shared helper revision with Tang; keep a single shared helper file.
2. Register streaming scalar/SIMD/OpenMP in Chen's harness and include Bq/Bk,
   effective sizes, total scratch and actual teams in schemas/grouping.
3. Keep the AVX2 translation unit isolated; baseline modes stay scalar.
4. Choose a common machine/compiler/power/affinity policy and repeat runs before
   final claims. Use corrected route-1 phase order for cross-route comparisons.
5. Small L can be slower with OpenMP. Keep that evidence rather than silently
   dispatching to serial and labeling it multithreaded.

No commit, push, PR merge, or modification of teammate branches was performed.
