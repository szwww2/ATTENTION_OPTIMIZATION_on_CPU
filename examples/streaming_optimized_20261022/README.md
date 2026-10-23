# Local streaming scalar / SIMD / OpenMP pilot

2026-10-22, i7-12700H as seen from WSL2, GCC 11.4.0. Complete source hashes,
per-object commands, binary hash, linked libraries and settings are in
`environment.json`; base main6d71c61 plus uncommitted changes. `raw.csv` contains
all 480 timings; `summary.csv` is recomputed from them with inclusive IQR.

Fixed requested Bq16/Bk256, seed20261022, ten warm-ups and twenty samples per
configuration; order rotates. Same-build scalar baseline, no previous-run or
other-machine timings mixed in. Six shapes × three single-thread modes plus
T2/4/8 for L128/L4096 = 24 configurations. All pre/post FP64 checks passed;
actual teams equaled requested teams on every timed call. No samples dropped.

## Single-thread SIMD

| L | d=dv | scalar median ms | SIMD median ms | ratio |
|---:|---:|---:|---:|---:|
|128|64|0.863|0.191|4.52×|
|512|64|13.699|3.108|4.41×|
|2048|64|218.939|54.492|4.02×|
|4096|64|875.303|202.662|4.32×|
|512|32|7.859|2.486|3.16×|
|2048|32|124.508|40.235|3.09×|

These are complete streaming-call ratios, including scalar exp and state updates,
not SIMD instruction throughput or cross-route speedups.

## Thread scaling, d=dv64

| mode | actual T | L128 median / IQR ms | L4096 median / IQR ms |
|---|---:|---:|---:|
|SIMD|1|0.191 / 0.011|202.662 / 4.478|
|SIMD+OpenMP|1|0.203 / 0.016|212.179 / 2.500|
|SIMD+OpenMP|2|0.216 / 0.039|102.856 / 2.183|
|SIMD+OpenMP|4|0.225 / 0.070|53.634 / 2.242|
|SIMD+OpenMP|8|0.453 / 0.134|37.375 / 3.252|

L4096 T8 is 5.42× faster than standalone SIMD, 5.68× than OpenMP-T1, and
23.42× than scalar in this run. T8 total scratch is 164,864 bytes vs 20,608
single-thread. At L128 T8 is **slower** than standalone SIMD (0.42× speedup);
small tasks do not amortize region/scheduling/wakeup costs. This is consistent
with overhead, not a measured attribution of each source of overhead.

OMP_DYNAMIC=FALSE, PROC_BIND=close, PLACES=cores, WAIT_POLICY=PASSIVE.
This is one local WSL run, with no host power/frequency control or verified
physical-core isolation. The chosen passive wait policy and rotating team sizes
are part of the experiment; results need not match persistent fixed-team runs.
Do not present these ratios as stable production or final team results.

## Reproduce / audit

```sh
python3 -B tools/run_streaming_optimized.py --out results/streaming_opt_repeat
python3 -B tests/test_streaming_optimized_records.py
```

Runner refuses an existing output directory. Raw records validate configuration,
counts/order, finite errors/times, fixed contract flags and exact scratch bytes.
It rejects mixed actual team sizes within one configuration. The native timer
includes scratch resets, CPU/argument checks and the entire parallel region;
allocation, input generation and FP64 validation are outside. No actual DRAM
traffic, roofline ceiling or PyTorch comparison was measured here.
