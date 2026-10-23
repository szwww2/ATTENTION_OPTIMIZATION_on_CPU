# Streaming branch: team review and integration

The user authorized committing and publishing `dev/cuiziwei` for team review.
Earlier implementation notes saying "no commit/push" describe the status at
completion of those phases. Measurements were made before committing; their
recorded base SHA plus source hashes remain the measurement provenance.

## Required comparison matrix

| Route | Scalar | SIMD | SIMD + static OpenMP |
|---|---|---|---|
| Full matrix | main | Tang/Chen integration branch | Tang/Chen integration branch |
| Row-wise | main | Tang/Chen integration branch | Tang/Chen integration branch |
| Blocked streaming | dev/cuiziwei | dev/cuiziwei | dev/cuiziwei |

Remote heads checked before publishing: main `6d71c61`, Tang `f9f088c`, Chen
integration `e0d98b9`. The latter defines full_matrix/rowwise SIMD and OpenMP in
`src/simd_routes.cpp` and `src/omp_routes.cpp`. Presence on a branch is not
equivalent to integration into main or same-machine reproduction of all nine modes.

## Roles to coordinate (existing team responsibilities)

- **Cui:** streaming recurrence, tiling, scratch sizing, route-specific checks,
  results and explanation; present this branch for review.
- **Tang:** shared SIMD/OpenMP design and review across all three routes. The
  `attn_simd.h` here is byte-identical to his f9f088c file; retain shared ownership
  and avoid maintaining duplicate implementations.
- **Li:** baseline/oracle/numeric contract and validation; review that optimized
  variants retain the same attention problem and appropriate scalar baselines.
- **Chen:** unified harness/schema, environment capture, final experiment
  orchestration and plots, PyTorch/roofline integration.

Every route needs all three modes, but each owner need not independently rewrite
shared intrinsics or thread utilities. Agree review and integration ownership
before final merging; this checklist does not reassign the team's roles.

## Decisions for the team

1. **Integration base/order:** agree a common branch containing the reviewed
   baseline-phase correction and shared harness. Review this branch's additions
   to `include/attn.h`, build scripts and README alongside pending changes;
   preserve both sets of APIs. Do not overwrite the shared header wholesale.
2. **Shared AVX2 implementation/build:** retain one helper revision and isolate
   AVX2 compilation from scalar/dispatch objects. Keep consistent contraction,
   fast-math and auto-vectorization policies across the three routes.
3. **API and metadata:** register streaming scalar/SIMD/OMP with Bq and Bk,
   requested/effective block sizes, requested/actual threads, and ALL scratch.
   Streaming's optional actual-thread output is an extension to discuss for
   the other routes. This implementation rejects nonpositive thread counts.
4. **Workspace:** streaming uses `T*(bq*bk + bq*dv + 2*bq)` floats for requested
   T workers; full/row-wise have different requirements. Report each accurately,
   not all as score-only memory. Keep memory capacity distinct from DRAM traffic.
5. **Unified correctness:** run all nine modes against the same FP64 oracle,
   seeds and tolerance, including odd dimensions, tails, repeated calls and
   thread constraints. Our local tests do not certify the unmerged combined tree.
6. **Final performance protocol:** agree CPU/compiler, affinity, thread budget,
   wait policy, warm-up/repetitions and independent runs. Freeze blocks before
   comparison or document a fair per-mode tuning budget. Include complete
   parallel-region overhead and retain small-input slowdowns. Our WSL pilot is
   evidence for this branch, not a substitute for the common final experiment.

## Review entry points

- `docs/STREAMING_OPTIMIZED.md`: API, design, tests and reproduction commands.
- `docs/STREAMING_CUIZIWEI.md`: scalar recurrence, scratch and permutation checks.
- `examples/streaming_optimized_20261022/`: 24 configurations, 480 timings,
  environment/source hashes and same-build scalar comparisons.
- `examples/streaming_cuiziwei_20261022/`: earlier scalar block-selection pilot.

The user can share the branch for review; merging into main and creating a PR
are separate actions, not implied by publishing the branch.
