# Week 1 notes - Li Ruiqin

## Done

The FP64 oracle, the full-matrix baseline and the row-wise baseline are in, with two suites behind
them -- 45 checks in `attn_tests`, 273 in `attn_blocks`. Both build from `build.bat` on Windows or
`build.sh` anywhere else.

The oracle is never timed; routes 1 and 2 are. Route 1 stores the whole score matrix, route 2 keeps
one score row at a time. The suites compare both against the oracle, and route 2 is additionally run
at ten block sizes and five row orders, bit for bit.

## Results

All from my machine (see `docs/ENVIRONMENT.md`), d = dv = 64, median of N runs. They are provisional:
the timing side still needs finishing (that is Chen Caoyu's part), so at this point they are only
here to check that the shape is what we expect.

Route 1, full matrix:

| L | median_ms | GFLOP/s | ws_MB |
|---|---|---|---|
| 128 | 0.530 | 5.89 | 0.066 |
| 512 | 10.755 | 6.09 | 1.049 |
| 1024 | 45.164 | 6.23 | 4.194 |
| 2048 | 174.965 | 6.05 | 16.777 |

Route 2, row-wise:

| L | Bq = 1 | Bq = 64 | ws_MB at Bq = 1 |
|---|---|---|---|
| 128 | 0.582 | 0.520 | 0.001 |
| 512 | 11.934 | 11.677 | 0.002 |
| 1024 | 43.160 | 43.704 | 0.004 |
| 2048 | 187.350 | 179.605 | 0.008 |

The two routes take about the same time per shape. Route 2 is not faster here, it just needs far less
workspace: at L = 2048 that is 0.008 MB against 16.777 MB at Bq = 1, and 0.524 MB at Bq = 64.

Cross-check at L = 512: the generator reproduces q/k/v bit for bit, FP64 matches the oracle to
2.359e-16, and NumPy's own FP32 result (BLAS, different summation order) stays inside the contract at
4.700e-08.

One thing to watch: at L = 128 the median moves between runs -- 0.508 and 0.713 ms for the same
binary, with the minimum steady at 0.502. Three warm-ups are not enough at that size. More notes in
PR #2.

## Next steps

- blocked streaming -- Cui Ziwei. `docs/INTEGRATION.md` has a proposed signature; the shape is up to
  whoever writes it, it only has to let the caller pass the block schedule in.
- the SIMD and OpenMP levels -- Tang Zhenxiong. Scalar flags are in `build.bat` and `build.sh`, the
  three levels in `docs/INTEGRATION.md`.
- the harness, the PyTorch comparison and the roofline -- Chen Caoyu. `bench_scalar` times route 1
  only, and the torch environments on this machine are in `docs/ENVIRONMENT.md`.
