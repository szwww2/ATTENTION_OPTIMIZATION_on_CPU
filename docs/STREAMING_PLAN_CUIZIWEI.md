# Cui Ziwei: scalar blocked streaming work plan

## Optimized extension completion

SIMD/static OpenMP implemented; 13,181 checks passed in optimized, ASan/UBSan
and constrained-team runs. Existing scalar regressions passed. Same-build pilot:
24 configurations/480 samples, every pre/post FP64 check passed. See
`STREAMING_OPTIMIZED.md` and `examples/streaming_optimized_20261022/` for actual
results/limits. Shared harness integration and team review remain pending;
no commit/push. The scalar-phase deferral below is historical, superseded by
the user's subsequent explicit SIMD/OpenMP request.

## SIMD/OpenMP extension initialization (2026-10-22)

User now explicitly requests working SIMD and multithreaded streaming versions
before discussing with teammates. Preserve the scalar implementation and its
saved pilot. Reuse Tang's `include/attn_simd.h` at f9f088c (local remote-tracking
snapshot); add an AVX2 translation unit, baseline-ISA dispatch/OpenMP wrapper,
thread workspace helper, tests and isolated build/measurement tools. Do not
merge draft PRs or modify routes 1/2. No commit/push requested.

Order: (1) implement and test unaligned/tail AVX2 and private-workspace static
query-block OpenMP; (2) run scalar regression, optimized suite, ASan/UBSan and
restricted-runtime team tests; (3) measure the six single-thread shapes and
L128/L4096 at requested T1/2/4/8, Bq16/Bk256 frozen, recording actual teams,
all raw samples, environment and hashes. Compile AVX2 only in its own object,
keep scalar auto-vectorization/FMA contraction disabled, and check CPU support
before entering the AVX2 object. Runtime team creation may allocate internally.
Risk: FP32 dot reduction changes; use existing oracle tolerance, not scalar
bitwise equality. WSL thread scaling is exploratory, not a final team result.

## Initialization (2026-10-22)

- Canonical checkout: course `opencode_work/team_project/ATTENTION_OPTIMIZATION_on_CPU` (user-approved first clone).
- Base: main `6d71c61`; local branch `dev/cuiziwei`. The teammate draft PRs are not merged here.
- Scope: scalar query/key tiling, online softmax, explicit scratch sizing, key-block permutation tests and a bounded tile experiment. SIMD/OpenMP integration is deferred for discussion.
- Reuse `attn::Shape`, SplitMix64, FP64 oracle and `check_contract`. Kernels accept row-major FP32 input and caller-owned output/scratch. Valid calls allocate nothing and do no I/O.
- Initial additions: `src/blocked_streaming_scalar.cpp`, declarations in `include/attn.h`, streaming tests, scalar build targets and an isolated route-3 experiment driver. Preserve existing route algorithms and teammate harness files.
- Existing run commands: `bash build.sh`, `build/attn_tests`, `build/attn_blocks`. New checks: streaming suite, address/undefined-behavior sanitizers, bounded scalar-only experiments.
- Risks: baseline phase order differs from Chen's pending fix; no formal route-1 speedup claims from this checkout. WSL machine/compiler differ from teammate Windows runs. Changes to Bq/Bk may change floating-point rounding; use the fixed FP64 contract, not bitwise equality across schedules.

## Ordered tasks

1. Implement scratch helper, natural-order scalar kernel and checked test-only schedule entry.
2. Verify tails, first/later maximum updates, permutations, repeated calls, input preservation and memory boundaries; run existing suites.
3. Measure Bq=16/Bk=128,256,512 at L2048,d=dv64; freeze a selected candidate for the six-shape scalar pilot; preserve samples/settings.
4. Document exact interface, results, limits and teammate handoff. No push/commit unless requested.

## Completion

- Scalar implementation, exact helper and schedule hook complete; all original
  suites and 11,498 streaming checks passed, including ASan/UBSan.
- Pilot completed: 60 tuning + 120 grid samples, all FP64 checks passed.
  Bk256 selected provisionally by the recorded rule; differences are within
  overlapping spreads. See `STREAMING_CUIZIWEI.md` and the example README.
- Shared benchmark/CLI integration is pending coordination with Chen's draft
  branch; SIMD/OpenMP is deferred by user request. No commit/push performed.
