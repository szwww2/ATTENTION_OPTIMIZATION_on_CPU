# Small scalar run - 2026-10-20 (UTC+8)

One local run on Windows 11, Intel Core Ultra 7 356H, GCC 16.2.0, one thread.
Source revision: `49cd7c6fa779b9e38725668905f5dd3d3a924650`.
The source tree was clean when measured; source and executable hashes are in `environment.json`.

```text
build.bat
python tools/run_scalar_bench.py --out results/scalar_20261020_run1
```

Each route used 10 warm-up calls and 20 timed calls per shape, d=dv=64, seed=20261020.
The rowwise query block was 64. The run produced 80 raw timing rows and four summary rows.
`raw.csv`, `summary.csv` and `environment.json` are unmodified copies of those outputs.
Machine-readable timestamps are UTC; the folder date is UTC+8.

| L | Route | Median (ms) | IQR (ms) | Score workspace (bytes) |
|---:|---|---:|---:|---:|
|128|full_matrix|1.3160|0.2080|65,536|
|128|rowwise|1.3015|0.1108|32,768|
|512|full_matrix|27.1250|5.4430|1,048,576|
|512|rowwise|27.6280|7.0695|131,072|

Both routes passed the fixed numerical contract before and after timing. The original
45-check contract suite and 273-check row/block-order suite passed. Four benchmark
tests independently checked statistics, invalid arguments, output protection and metadata;
three fault-gate tests checked that incorrect results do not produce performance CSVs.

The route medians are close relative to the observed spread. This single run establishes
that the measurement pipeline works, not a stable speedup or its cause. CPU affinity and
power mode were not controlled. More runs under a documented setup are needed for final
performance claims. The workspace reduction is allocated scratch capacity, not measured
DRAM traffic. No roofline or PyTorch result is established here.

The Windows compiler used for this run did not provide AddressSanitizer/UBSan runtime
libraries. Those checks and validation on Linux/macOS remain separate work.
