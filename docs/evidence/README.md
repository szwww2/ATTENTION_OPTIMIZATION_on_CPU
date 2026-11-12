# Tang's evidence index

**Current code (full-matrix is three-phase):**

- `tang_msvc_phaseorder_L128.csv`, `tang_msvc_phaseorder_L4096.csv`: every
  timed sample, median, workspace and FP64 contract result.
- `tang_msvc_phaseorder_summary.csv`: derived SIMD and OpenMP ratios.
- `tang_msvc_phaseorder_scaling.svg` and `tang_msvc_phaseorder_simd.svg`:
  reproducible vector figures; PNG copies are for convenient preview.
- `tang_avx2_assembly.txt`: AVX2 instruction evidence and regeneration command.
- `tang_cpu_topology_2026-10-21.csv`: Windows logical-processor efficiency
  classes used to choose the exploratory affinity mask.

**Superseded diagnostic:** `tang_msvc_L4096.csv` records one old-kernel
unpinned run that exposed the core-placement discrepancy. The other generated
old-kernel files were moved to the local, ignored `results/tang_legacy/`
directory; they can be reproduced or inspected locally but are not proposed
for the shared branch. Old-kernel speedups must **not** be mixed with the
current-code results or used as final report figures.

Methods, limitations and exact commands are in
`../EXPERIMENT_PHASE3_TANGZHENXIONG.md`.
