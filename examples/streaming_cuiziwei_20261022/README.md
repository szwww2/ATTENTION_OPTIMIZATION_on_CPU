# Cui Ziwei — blocked streaming scalar pilot, 2026-10-22

Actual local measurements, not teammate results or final group comparisons.
Machine: Intel Core i7-12700H as reported inside WSL2; GCC 11.4.0. One thread,
no manual affinity/power control. WSL virtual CPU topology is not a verified
physical-core topology. Exact OS, allowed CPUs, source/binary hashes, flags and
dirty working-tree state are saved in `environment.json`.

## Tile sweep

L2048,d=dv64,Bq16. Ten warm-ups and twenty measured calls per candidate;
candidate order rotates each round. All pre/post FP64 checks passed.

| Bk | median ms | IQR ms | total scratch bytes |
|---:|---:|---:|---:|
| 128 | 233.525 | 3.442 | 12,416 |
| 256 | 230.999 | 4.627 | 20,608 |
| 512 | 232.158 | 3.658 | 36,992 |

The predeclared lowest-median rule selected **Bk256**, then froze it for the
grid. Differences are about 0.5–1.1%, with overlapping interquartile ranges;
this is a provisional choice, not evidence of a reliably superior tile size.
Raw samples: `tuning_raw.csv`; recomputed statistics: `tuning_summary.csv`.

## Six-shape scalar grid

Requested Bq16/Bk256, with effective sizes clipped to L. Twenty timings each.

| L | d=dv | median ms | IQR ms | scratch bytes | max abs error |
|---:|---:|---:|---:|---:|---:|
| 128 | 64 | 0.932 | 0.094 | 12,416 | 1.361e-7 |
| 512 | 64 | 14.615 | 0.648 | 20,608 | 9.877e-8 |
| 2048 | 64 | 231.952 | 5.133 | 20,608 | 9.138e-8 |
| 4096 | 64 | 910.313 | 2.284 | 20,608 | 1.129e-7 |
| 512 | 32 | 7.233 | 0.413 | 18,560 | 9.148e-8 |
| 2048 | 32 | 120.449 | 2.139 | 18,560 | 7.527e-8 |

All output elements passed `abs(diff)<=1e-4+1e-3*abs(ref)` before timing and
after the last call; inputs were unchanged. No failures were dropped. There
are 60 tuning samples and 120 grid samples. The runner verifies sample counts,
parameter grid, execution order, finiteness, workspace and contract flags before
publishing a successful environment status.

For L>=256,dv64 the scratch stays at 20,608 bytes (~20.125 KiB) while L grows.
This is score tile + maxima + normalizers + accumulators. At L4096 a full score
matrix alone would occupy 67,108,864 bytes; this is a storage calculation, not
measured DRAM traffic or proof of speedup. Q/K/V and output still scale with L.

## Reproduce

```sh
python3 -B tools/run_streaming_experiment.py --out results/streaming_repeat
python3 -B tests/test_streaming_records.py
```

The runner rebuilds and writes a new directory, refusing overwrite. The
experiment base is main `6d71c61` plus uncommitted source identified by SHA256;
the old git revision alone does not identify the new implementation. These
samples must not be pooled with Chen's Windows runs or used to claim a SIMD
gain. Native timing includes full scalar call/state resets, excludes allocation,
generation, oracle and checks. No SIMD/OpenMP or PyTorch comparison ran here.
