# Report draft: Tang Zhenxiong's SIMD and OpenMP contribution

> **Historical draft from Tang's branch.** Keep its measurements separate from
> the integrated GCC experiment. Three statements below have since changed:
> full-matrix and rowwise SIMD now normalize probabilities **before** PV;
> the full-matrix phase-order fix is included; and blocked streaming is
> integrated in all three modes. Use the [current integration notes](INTEGRATED_BENCHMARK.md#integration-changes)
> for implementation details and [Chen's experiment section](REPORT_CHEN_SECTION_DRAFT.md)
> for the shared measurements. The original draft follows for reference.

**Draft, not final performance text.** It deliberately separates verified
implementation facts from exploratory single-machine timings.

We implemented two reusable AVX2 FP32 primitives for the attention hot loops.
The query-key dot product multiplies and accumulates eight adjacent feature
elements per vector iteration, then handles a scalar tail. The probability-
value loop broadcasts one unnormalized softmax weight and updates eight
adjacent value/output elements at a time, also with a scalar tail. The final
output is divided by the softmax denominator. Both the full-matrix and
row-wise routes use the same two inner-loop primitives, and each passes the
pre-existing independent FP64 oracle contract. The full-matrix route first
materializes all scores, then all normalized probabilities, then computes
P*V; the row-wise route completes each query row. AVX2 assembly evidence is saved in
`docs/evidence/tang_avx2_assembly.txt`. No approximate exponential or FMA
intrinsic was introduced. The change to P*V loop order also changes memory
locality, so a scalar-to-SIMD speedup cannot be attributed solely to SIMD.

For parallelism, independent query blocks are assigned with OpenMP
`schedule(static)`. Full-matrix query blocks own disjoint score and output
rows, with a barrier between the score, normalization and P*V phases.
Row-wise workers own disjoint output rows and separate `Bq*L`-float score
scratch. A parallel region is opened once per kernel call, not inside the
attention row's dot-product or value loop. Tests cover T=1,2,4 and several
block sizes and check the FP64 contract, bitwise equality with the serial SIMD
route, the full-matrix P workspace, repeated calls and workspace guards. The
most recent MSVC suites pass 23/23 low-level SIMD, 354/354 route-contract and
635/635 OpenMP checks.

The local MSVC timing harness was run at L=128 and L=4096, d=dv=64 and Bq=16,
with T=1,2,4,8. Each route was warmed up and measured repeatedly; raw samples,
compiler/CPU details and graph generation are in
`docs/EXPERIMENT_PHASE3_TANGZHENXIONG.md`. In the corrected-code local run,
T=8 OpenMP speedups over the same OpenMP route's T=1 were 3.09x/3.51x at
L=128 and 5.69x/5.79x at L=4096 (full-matrix/row-wise). These are exploratory
observations: several T=2/4 samples were substantially affected by thread
placement or system noise. They should not become final headline figures
without the agreed group-machine repeat and review of raw sample
distributions. Also, Chen Caoyu's newer full-matrix scalar baseline uses the
same three-phase order but is still on a separate branch; the current local
scalar-to-SIMD ratio compares unlike phase orders. The MSVC scalar build did
not disable auto-vectorization, unlike the project's GCC comparison flags.

The blocked-streaming route is owned by Cui Ziwei and has not yet been
integrated in this checkout. When it is available, we will reuse the dot and
weighted-add primitives where its tile layout permits, add static query-block
OpenMP with per-worker online-softmax scratch, validate against the same FP64
contract and extend the experiment table. This paragraph must be updated
after that integration; no blocked-streaming performance claim is made now.
