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

Online softmax cannot be bitwise stable across block sizes, so those runs will not match exactly. Each run only
has to be within one tolerance of the same FP64 answer, `|O(Bk=a) - O(Bk=b)| <= 2 * (atol + rtol * |O64|)`.

The first non-empty block has to set (m, l, a) straight from it. Merging into empty accumulators means -inf
minus -inf, and at L = 1 a division by zero.

## Tools

`tools/bench_scalar.cpp` times route 1 only, so a second route has to be added to it. It warms up, repeats,
prints the median and the minimum, and re-checks the contract on the last run. `--csv` gives one line per shape:
`L,d,dv,reps,median_ms,min_ms,gflops,workspace_MB,roundtrip_max_abs`.

The median at small L is the noisy one: two runs of the same binary gave 0.713 ms and 0.508 ms at L = 128, while
the minimum stayed at 0.502. That is warm-up, not the kernel, and it wants settling before these numbers go into
the report.

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
