# Tang phase 3: two-route SIMD and OpenMP timing

Status: runnable harness, raw samples, summary and figures are complete for
full-matrix and row-wise routes. These are **exploratory local MSVC results**, not
final group-wide performance claims. Blocked streaming is not in this checkout.
The full-matrix SIMD/OpenMP kernels were corrected to a three-phase
score-matrix -> normalized-P -> P*V schedule after the first exploratory runs.
Only the `tang_msvc_phaseorder_*` files measure the corrected kernels.

## Knowledge and comparison rules

For each route, SIMD speedup is the median scalar latency divided by the median
serial SIMD latency. OpenMP speedup is `S(T)=t_omp(1)/t_omp(T)` for the **same**
OpenMP route, and efficiency is `E(T)=S(T)/T`. For example, 100 ms at T=1 and
40 ms at T=4 give 2.5x speedup and 62.5% efficiency. A correctness test at T=4
does not itself measure either number. `T=1` OpenMP is kept separate from the
non-OpenMP SIMD route to include OpenMP runtime and code-generation effects.

## Reproduction

- Date: 2026-10-21, Tang's Windows 11 machine, build 10.0.26200.
- CPU: Intel Core Ultra 9 275HX, 24 logical processors. The Windows
  `GetLogicalProcessorInformationEx` output is saved in
  `docs/evidence/tang_cpu_topology_2026-10-21.csv`. Eight logical processors
  report `EfficiencyClass=1`, with mask `0xC03C03`; the others report class 0.
  These are Windows classifications, not a guarantee of fixed clock speed or
  per-thread placement.
- Compiler: MSVC 19.44.35228, C++17, `/O2 /arch:AVX2 /openmp` for every route.
  This local comparison does **not** disable automatic vectorization in the
  scalar files. The group's GCC build uses `-fno-tree-vectorize` and should be
  run separately for the controlled compiler-level comparison. GCC was not
  available in Tang's command environment. An MSVC `/Qvec-report:1` compile
  emitted no vectorized-loop report for the scalar sources, but that is not
  the same as explicitly disabling all compiler vectorization.
- Base revision: local `main` at `6d71c61`. Chen Caoyu's separate
  `dev/chencaoyu-harness` branch has since changed the full-matrix scalar
  baseline to the same three-phase schedule. This branch is not merged here,
  so the full-matrix scalar-to-SIMD ratios below compare against the older
  scalar implementation and **must not** be used as the final fair ratio.
- Shape: `d=dv=64`; `L=128` and `4096`; `Bq=16`; deterministic seed
  `20261005+L`; thread counts 1, 2, 4 and 8. The small run uses five warmups
  and 51 timed repetitions; the large run uses two warmups and seven timed
  repetitions. The kernel is timed with `steady_clock`; inputs, workspace and
  output are allocated beforehand. One FP64 oracle is computed before any
  timing, and every route is checked against it after timing. All reported
  rows have `contract_pass=1`.

From an x64 Visual Studio Developer Command Prompt, in the repository root:

```bat
call tools\build_tang_msvc.bat
build\cpu_topology_win.exe
start "" /b /wait /affinity C03C03 build\attn_bench_tang_msvc.exe 128 16 5 51 8
start "" /b /wait /affinity C03C03 build\attn_bench_tang_msvc.exe 4096 16 2 7 8
```

Each benchmark command emits CSV to standard output; redirect it to a file
when reproducing. `tools/analyze_tang.py` verifies sample counts, medians and
contract flags before generating the summary and figures:

```bat
python tools\analyze_tang.py docs\evidence\tang_msvc_phaseorder_L128.csv docs\evidence\tang_msvc_phaseorder_L4096.csv --prefix tang_msvc_phaseorder --condition "MSVC three-phase full-matrix exploratory"
```

The current-code files are `docs/evidence/tang_msvc_phaseorder_L128.csv` and
`docs/evidence/tang_msvc_phaseorder_L4096.csv`; the derived
`tang_msvc_phaseorder_summary.csv`, `tang_msvc_phaseorder_scaling.svg` and
`tang_msvc_phaseorder_simd.svg` are tracked alongside them. PNG copies are
included for quick viewing. Each raw CSV row includes every timed sample, not
just its median. `docs/evidence/tang_msvc_L4096.csv` is one **superseded
old-kernel, unpinned diagnostic**; other old runs were moved to local ignored
`results/tang_legacy/`. They must not be pooled or graphed with the corrected
code. The corrected full-matrix
route now materializes all scores, then all normalized probabilities, then
computes outputs; the row-wise route still completes one row at a time.

## What the local samples show, and what they do not

In the corrected-code run, scalar-to-SIMD medians were 2.22x and 2.28x for
full-matrix and row-wise at L=128, and 3.72x and 4.30x at L=4096. The
full-matrix comparison is against the older scalar phase order (see above).
In the same run, T=8 OpenMP speedup over OpenMP T=1 was 3.09x and 3.51x at
L=128, and 5.69x and 5.79x at L=4096. The corresponding T=8 efficiencies
were about 39%, 44%, 71% and 72%. These are **observations from one run**,
not robust hardware constants. The full CSV summary includes T=2 and T=4.

The evidence has important instability: at L=4096, corrected full-matrix T=2
was about 377 ms while T=1 was about 274 ms; individual full-matrix T=4
samples ranged from about 81 to 383 ms. At L=128, some timed samples are much slower than
their medians. The scaling plot's whiskers show each configuration's observed
sample range with its T=1 median held fixed; they are not confidence intervals.
Process affinity constrains eligible logical processors but does not pin each
OpenMP worker or eliminate OS activity, power management and thermal effects.
The safest interpretation is that substantial parallel speedup is *possible*
at T=8 on this machine, while exact scaling at T=2/4 and small-L overhead
remain unsettled and need a controlled group-machine repeat.

An old-kernel unpinned run showed full-matrix serial SIMD at about 526 ms but OpenMP T=1
at about 258 ms. Pinning both to one logical processor brought them to about
239 and 236 ms; this supports a scheduling/placement explanation and is why
the unpinned result must not be called an algorithmic OpenMP-T=1 gain.

Workspace is deterministic: full-matrix uses `4*L*L` bytes (64 KiB at L=128,
64 MiB at L=4096); row-wise SIMD uses `4*Bq*L` bytes (8 KiB and 256 KiB),
and row-wise OpenMP uses `4*T*Bq*L` bytes (64 KiB and 2 MiB at T=8). These
figures count score scratch only, not Q/K/V or output. The reduced scratch may
help locality at large L, but bandwidth was not measured here. SIMD versus
scalar also changes the P*V loop order, so their latency ratio is not solely
the effect of AVX2 instructions.

## What remains before final report numbers

1. Rerun after Chen's three-phase scalar baseline and shared harness are
   integrated, on the agreed group machine/compiler, under a documented
   affinity/power/background-load policy. Keep all raw repetitions.
2. Integrate blocked-streaming scalar/SIMD/OpenMP after its owner provides its
   interface and implementation; add it to the same harness and figures.
3. Coordinate the group report's bandwidth/roofline and PyTorch comparisons
   with Chen Caoyu. No such claim is made here.
