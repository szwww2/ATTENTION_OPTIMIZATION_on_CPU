# Blocked streaming scalar — Cui Ziwei

**Update:** SIMD and static OpenMP have now been implemented at the user's
request. See [optimized implementation and handoff](STREAMING_OPTIMIZED.md).
The scalar pilot below remains its original, separate experiment.

## Scope and status

Implemented on `dev/cuiziwei`, based on main `6d71c61`: query/key tiling, online
softmax, exact scratch sizing, checked test-only key-block schedules and local
tile experiments. This is scalar FP32, not SIMD or OpenMP. Those integrations
remain a separate team discussion. No teammate draft PR has been merged here.

## Public interface

Declarations are in `include/attn.h`, implementation in
`src/blocked_streaming_scalar.cpp`:

```cpp
std::size_t blocked_streaming_workspace_elems(
    const Shape& s, std::size_t query_block, std::size_t key_block);

void blocked_streaming_scalar_fp32(
    const Shape& s, const float* q, const float* k, const float* v,
    std::size_t query_block, std::size_t key_block,
    float* workspace, float* out);
```

- Shape dimensions must be positive; input dimensions and scratch byte counts
  must fit `size_t`. Invalid shape/overflow is rejected.
- `effective_Bq=min(L,max(1,Bq))`, likewise for Bk. Zero means one, matching
  the existing rowwise kernel convention. The helper and kernel use the same rule.
- Q/K are contiguous row-major L*d, V/O are L*dv. All buffers must be sufficiently
  sized and non-overlapping. Pointers are checked for null, not allocation size.
- Inputs and scaled dot products are expected to be finite and representable in
  FP32; NaN/Inf input handling is not part of the computation contract. The
  output contract check rejects non-finite results.
- Inputs are read-only. Workspace initial contents may be arbitrary. Output is
  fully overwritten; all resets are inside the call. Valid kernel calls have no
  dynamic allocation or I/O.
- Native workspace size, in **float elements**, is
  `bq*bk + 2*bq + bq*dv`: score tile, running maxima, normalizers and accumulators.
  This excludes Q/K/V, final output and the FP64 reference, but includes ALL
  streaming scratch. There is no separate packed K/V copy.

Example:

```cpp
attn::Shape s{2048,64,64};
attn::Tensors t;
t.resize(s);
t.fill_qkv(20261022);
t.ws.resize(attn::blocked_streaming_workspace_elems(s,16,256));
attn::blocked_streaming_scalar_fp32(
    s,t.q.data(),t.k.data(),t.v.data(),16,256,t.ws.data(),t.o.data());
// Compute oracle and call check_contract outside the timer.
```

## Computation and reuse

The query block is outermost; all key blocks are consumed before this query
block is written out. Within each key block, the score pass visits key j then
all local queries r, reusing that K row. The value pass likewise visits value j,
all local queries r, then contiguous value dimension y. This puts the same K/V
block near its uses across queries; it does not prove measured cache hits or
DRAM reduction.

For each query, after finding the block maximum:

1. On the first block, initialize the maximum directly, with zero l/u.
2. Otherwise set `new_m=max(old_m,block_max)` and
   `alpha=exp(old_m-new_m)`; multiply both old l and every component of u by alpha.
3. Overwrite scores with `exp(score-new_m)`; add their sum to l and their
   weighted V rows to u.
4. After the last key block, write `out=u/l`.

All arithmetic/accumulation is FP32. `std::exp` remains scalar. This in-place
form is algebraically the online merge without allocating a separate block
weighted-sum vector. Changing block sizes/order changes rounding, so bitwise
equality is required only for repeated calls using the identical schedule.

## Test-only schedule

`blocked_streaming_scalar_ordered_fp32` accepts `key_blocks` and `n_blocks`
after Bq/Bk and before workspace/output. The entries must be a permutation of
`[0, ceil(L/effective_Bk))`. They are block **indices**, not row offsets.

The function checks count, range and duplicates before writing any scratch or
output. Validation uses O(n_blocks²) comparisons without allocation. This is a
test hook; timings always use the natural-order entry point. Every ordering is
checked independently against the same FP64 oracle; a loose pairwise check is
not a substitute.

## Build and verify

```sh
bash build.sh
build/attn_tests
build/attn_blocks
build/attn_streaming_tests
python3 -B tests/test_streaming_records.py
```

Windows GCC: `build.bat`, then `build\attn_streaming_tests.exe`.
The Windows target was updated but not executed in this session.

Sanitizer reproduction on Linux/GCC:

```sh
g++ -std=c++17 -O1 -g -fno-tree-vectorize -ffp-contract=off \
  -fsanitize=address,undefined -fno-omit-frame-pointer -Iinclude \
  src/blocked_streaming_scalar.cpp src/oracle_fp64.cpp src/input.cpp \
  src/metrics.cpp tests/test_streaming.cpp -o build/attn_streaming_sanitize
build/attn_streaming_sanitize
```

Observed locally: original suites 45/273 checks passed; streaming 11,498 checks
passed in optimized and ASan/UBSan builds. Maximum absolute error across the
streaming suite was 3.54013512e-6, within the fixed elementwise
`1e-4 + 1e-3*abs(ref)` contract. The test covers six normal shapes with three
seeds, equal/large/rising-max scores, zero/one/tail/oversized blocks, natural,
reverse and shuffled schedules, input preservation, NaN-poisoned scratch/output,
guards, invalid schedules and size overflow. Assertions are not distinct
performance configurations.

## Scalar pilot

```sh
python3 -B tools/run_streaming_experiment.py --out results/streaming_new_run
```

Requires GCC-compatible C++17 compiler (`CXX` configurable) and Python 3; no new
packages. A new output directory is mandatory. The runner builds only the
streaming benchmark with `-O3 -fno-tree-vectorize -ffp-contract=off`, warms up
ten calls per candidate, and records twenty calls per configuration. Tuning
rotates candidate order each round. FP64 checks precede timing and follow the
last timed call; input preservation is also checked. These checks do not inspect
every timed output.

Tuning fixes Bq16 at L2048,d=dv64 and compares Bk128/256/512. Selection is lowest
median, tie-breaking by smaller Bk. The chosen Bk is frozen before the six-shape
pilot. Input seed is 20261022. Summary quartiles use inclusive interpolation;
GFLOP/s counts `2*L*L*(d+dv)` over complete kernel time including exp/resets.

Actual evidence: [local experiment](../examples/streaming_cuiziwei_20261022/README.md).
It is a single local pilot, not a claim of optimal Bk on every CPU or a final
three-route speedup comparison.

## Handoff to the shared harness (pending)

- Register `blocked_streaming` with the above kernel/helper, including Bq/Bk in
  CLI, raw/summary CSV, manifest, grouping keys and memory preflight.
- Report total streaming scratch, rather than calling it score-only scratch.
- `query_block`/`key_block` in this pilot record requested values; effective
  values follow the explicit clipping rule. Agree requested/effective fields
  with Chen before combining with his harness, which records effective Bq.
- Merge/review Chen's baseline phase-order correction before formal route
  comparisons. The existing main baseline is not changed by this work.
- Keep teammate Windows/GCC/MSVC results separate from this WSL/GCC pilot.
- Review SIMD/OpenMP separately: no intrinsics, SIMD helper dependency or thread
  runtime was added in this scalar phase. The subsequent optimized phase is
  documented in `STREAMING_OPTIMIZED.md`.
