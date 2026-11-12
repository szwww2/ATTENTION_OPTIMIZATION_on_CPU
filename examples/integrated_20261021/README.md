# First integrated run — 21 October 2026

The shared benchmark ran both full-matrix and rowwise routes in scalar, AVX2 and AVX2+OpenMP modes. All 40 available proposal configurations passed the checks before and after timing: 14 runs, 40 summaries and 800 raw timings. Blocked streaming is not integrated yet; its missing configurations are listed in `manifest.json`.

Source revision: `2cb8ec5a41212710fa7aae25776d78ee191b7eb0`. The working tree was clean for all runs. Machine: Intel Core Ultra 7 356H, 16 physical/logical cores, Windows 11, GCC 16.2.0. Each configuration uses Bq=16 where applicable, 10 warm-ups and 20 samples. Build flags and hashes are in each environment file.

```text
python tools/build_integrated.py
python tools/run_project_grid.py --physical-cores 16 --out results/grid_20261021_run1
```

The files were copied unchanged from that results directory, so manifest prefixes retain their original relative locations. For each run, read `_summary.csv`, then `_raw.csv` and `_environment.json` for the samples and settings.

This is an initial run, not a stable performance claim. CPU affinity and power mode were not controlled, and small cases are noisy: rowwise OpenMP at L=128, T=2 has IQR/median about 72.4%. Repeat under documented conditions before drawing final conclusions. SIMD changes PV loop order as well as vector instructions. Workspace bytes are allocated scratch, not measured DRAM traffic.

The integrated build passed the local scalar, workspace, SIMD and OpenMP suites, plus the benchmark, failure-gate and NumPy-checker tests. No Linux/macOS or sanitizer result is claimed. Run instructions are in [INTEGRATED_BENCHMARK.md](../../docs/INTEGRATED_BENCHMARK.md).
