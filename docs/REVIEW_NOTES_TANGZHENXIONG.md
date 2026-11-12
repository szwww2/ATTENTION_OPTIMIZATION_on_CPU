# Tang code-review handoff (2026-10-21)

This is a local self-review plus a read-only review of the currently available
shared baseline. It is **not** a teammate's approval or a merged pull request.

## Checked and corrected

- The shared AVX2 helpers operate on eight adjacent FP32 values, use unaligned
  loads/stores, and finish a scalar tail. Complete routes keep row-max
  subtraction before `exp`; no heap allocation occurs inside either hot path.
- Full-matrix OpenMP assigns disjoint score/output rows and uses implicit
  barriers between all-score, all-softmax and all-P*V phases. Row-wise OpenMP gives
  each worker a separate `Bq*L` score slice and assigns disjoint output rows.
  Neither route opens a nested parallel region.
- The pre-existing `rowwise_workspace_elems(s, 0)` returned zero while the
  scalar and SIMD kernels interpreted `Bq=0` as one. The helper now also
  normalizes zero to one; SIMD and OpenMP tests include `Bq=0`.
- The benchmark was checked against the same FP64 oracle after every timed
  configuration, and it verifies the requested OpenMP worker count outside
  the timed region. The analysis script rechecks all raw sample counts,
  medians and contract flags before plotting.
- `git diff --check` and `git diff --cached --check` pass. Source and evidence
  remain on `dev/tangzhenxiong`; nothing has been pushed.

## Executed suites

| Suite | MSVC result |
| --- | ---: |
| Shared scalar contract | 45 checks, 0 failures |
| Tang low-level SIMD | 23 checks, 0 failures |
| SIMD plus shared contract | 354 checks, 0 failures |
| SIMD OpenMP | 635 checks, 0 failures |

The shared scalar block-decomposition test compiled but Windows Device Guard
blocked `attn_blocks_msvc.exe` from launching. This is an OS execution policy
result, not a test failure; it must be run on a permitted machine. The GCC
targets in `build.bat` and `build.sh` remain unexecuted here because `g++` is
not available in the command environment. No sanitizer or thread sanitizer
result is claimed.

## Shared-code review and pending external review

The baseline FP64 oracle, scalar kernels, input generator and contract checker
were inspected without changing their numerical rules. The scalar contract
suite passed; no additional confirmed defect was found in those files during
this review. The currently checked-out `tools/bench_scalar.cpp` measures only
full-matrix scalar, so Tang's comparison harness is in a separate file.
Chen Caoyu's fetched but unmerged `dev/chencaoyu-harness` branch contains a
two-scalar-route harness and changes the full-matrix scalar phase order.
Tang's full-matrix SIMD/OpenMP now follows that three-phase design, but the
local scalar benchmark still uses the older `main` baseline. Final fair
scalar-to-SIMD figures require branch integration and rerunning the harness.

Blocked streaming is still only an interface sketch in `docs/INTEGRATION.md`
in this checkout. Its scalar, SIMD and OpenMP routes cannot yet be reviewed or
integrated. The report also needs peer review of Tang's code and figures after
the branch is shared, plus reciprocal review of Cui's and Chen's contributions
when available. That is an external coordination task, not complete locally.
