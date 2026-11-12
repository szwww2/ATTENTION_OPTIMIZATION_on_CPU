# Adding a route

Notes for whoever writes the next kernel: blocked streaming, SIMD, OpenMP. I wrote them while doing route 1. A
route that keeps to this is covered by the suite we already have.

## How a route plugs in

I touched two places for route 1, and a new route touches the same two: one declaration in `include/attn.h`, one
file in `src/`.

```cpp
// how much scratch does the route need?
std::size_t blocked_streaming_workspace_elems(const Shape& s);

// I allocate the workspace and the output, you fill them in
void blocked_streaming_scalar_fp32(const Shape& s,
                                   const float* q, const float* k, const float* v,
                                   float* workspace, float* out);
```

The helper is not `inline`: its body ends up in a .cpp, so the keyword would do nothing.

The names are a guess and the shape of route 3 is its author's. What the block-order test needs is the block
schedule coming in from the caller; without it there is nothing to vary.

What the rest of the code assumes about a route:

- nothing allocates and nothing touches the disk inside it. I allocate once, outside the timed region, so the
  timer measures the kernel and not `malloc`.
- row-major FP32: `q` and `k` are L x d, `v` and `out` are L x dv.
- the row maximum comes off before exp, otherwise a large score spread turns into inf.
- same input, same output, every call. The repeated-call case leans on it.

## What the suite already gives you

A route that is added to the runner in `tests/test_contract.cpp` gets the required shapes for free: L = 1, L = 7
with d = 31 and dv = 33, L = 127, L = 129 with d = 65, L = 256 with dv = 33, equal scores, a large finite score
spread, and a repeated call. Each one is checked against the FP64 oracle with `|diff| <= 1e-4 + 1e-3 * |ref|`.

## The streaming part

Route 3 is the one whose answer depends on how the work is cut up, so it needs two things the others do not: a
second entry point that takes the block schedule from the caller, and a test that runs the same input with
different block sizes and block orders -- including sizes that do not divide L and sizes bigger than L.

Online softmax may differ bitwise because of rounding across block sizes, so those runs will not match exactly. Each run only
has to be within one tolerance of the same FP64 answer, `|O(Bk=a) - O(Bk=b)| <= 2 * (atol + rtol * |O64|)`.

Initialize (m, l, a) directly from the first non-empty block, as specified in the proposal.
Starting with (-inf, 0, 0) can also be mathematically valid when the first block has a finite maximum:
the old contribution is scaled by exp(-inf - finite) = 0. It does not inherently fail at L = 1.
Empty or all-masked blocks need separate handling; do not evaluate -inf - (-inf).

## Tools

`tools/bench_scalar.cpp` now times both scalar routes on the same Q/K/V buffers. Both routes pass the
FP64 contract before warm-up; timed rounds alternate which route runs first. A final contract check must
pass before raw and summary CSVs are written. `tools/run_scalar_bench.py` also records the environment.
See [SCALAR_BENCHMARK.md](SCALAR_BENCHMARK.md) for the timing boundary, columns and integration points.

Earlier timings used a different phase order in the full-matrix baseline and are not results for this
version. Warm-up and alternating order reduce some bias, but do not remove background load, frequency
changes or scheduling effects. Record the spread rather than attributing it to one cause without evidence.

`tools/dump_case.cpp` writes one case out as text files and `tools/crosscheck_numpy.py` recomputes it in NumPy,
comparing the generator (must be bit-identical), the FP64 result against the oracle, and the FP32 result against
the tolerance. Both write into `build/` or `results/`, both ignored.

## Build levels

| Level | Flags | Purpose |
|---|---|---|
| scalar | -O3 -fno-tree-vectorize | the baseline; the compiler must not supply the speedup |
| SIMD | -O3 plus an explicit ISA flag (-mavx2, -mavx512f, ...), same source | the level under study; keep the compiler report or the assembly as evidence |
| threads | the SIMD build plus -fopenmp, static split over query blocks | record thread count and affinity |

No `-march=native`: the build has to mean the same thing on any machine.

## Before you push

1. kernel and workspace helper in their own file;
2. wired into `tests/test_contract.cpp` and the build script;
3. `build/attn_tests.exe` ends with SUITE PASSED;
4. `build/attn_bench.exe` run for the route, with the headline numbers in the commit message or in the group.

The tolerance is fixed and is not adjusted per route.
