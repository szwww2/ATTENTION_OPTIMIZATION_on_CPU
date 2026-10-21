# Saved grid results

14 runs, 40 configurations, 800 raw timings. Summary statistics were checked against every raw sample.

Measured source: `2cb8ec5a41212710fa7aae25776d78ee191b7eb0`. Dirty tree: `False`.
Affinity: not pinned by the harness. Power mode: not controlled by the harness.

Points show medians; error bars span Q1 to Q3 of individual timings, not confidence intervals. Thread points come from separate runs. Workspace means allocated scratch, not DRAM traffic or total process memory.

| Run | Route / mode | Median ms | Q1–Q3 ms | IQR / median | Scratch KiB |
|---|---|---:|---:|---:|---:|
| serial_L128_d64_T1 | full_matrix / scalar | 1.4920 | 1.4485–2.2393 | 53.0% | 64 |
| serial_L128_d64_T1 | rowwise / scalar | 1.5300 | 1.3735–2.0118 | 41.7% | 8 |
| serial_L128_d64_T1 | full_matrix / simd | 0.9840 | 0.9280–1.3295 | 40.8% | 64 |
| serial_L128_d64_T1 | rowwise / simd | 1.0425 | 0.9470–1.4923 | 52.3% | 8 |
| serial_L512_d64_T1 | full_matrix / scalar | 27.9260 | 25.6613–30.1662 | 16.1% | 1024 |
| serial_L512_d64_T1 | rowwise / scalar | 29.4300 | 26.5043–32.9560 | 21.9% | 32 |
| serial_L512_d64_T1 | full_matrix / simd | 17.1150 | 15.2473–18.0793 | 16.5% | 1024 |
| serial_L512_d64_T1 | rowwise / simd | 17.4055 | 16.1412–22.0012 | 33.7% | 32 |
| serial_L2048_d64_T1 | full_matrix / scalar | 517.9915 | 494.0828–608.6092 | 22.1% | 16384 |
| serial_L2048_d64_T1 | rowwise / scalar | 552.6740 | 529.8128–666.8592 | 24.8% | 128 |
| serial_L2048_d64_T1 | full_matrix / simd | 261.6325 | 241.0430–381.7450 | 53.8% | 16384 |
| serial_L2048_d64_T1 | rowwise / simd | 280.0225 | 255.6662–350.1703 | 33.7% | 128 |
| serial_L4096_d64_T1 | full_matrix / scalar | 2074.9165 | 1925.8383–2147.1445 | 10.7% | 65536 |
| serial_L4096_d64_T1 | rowwise / scalar | 2207.2285 | 2077.4405–2243.4033 | 7.5% | 256 |
| serial_L4096_d64_T1 | full_matrix / simd | 1006.6080 | 919.8808–1073.3180 | 15.2% | 65536 |
| serial_L4096_d64_T1 | rowwise / simd | 1056.3380 | 1014.8257–1127.3942 | 10.7% | 256 |
| serial_L512_d32_T1 | full_matrix / scalar | 18.1835 | 17.7090–19.5275 | 10.0% | 1024 |
| serial_L512_d32_T1 | rowwise / scalar | 18.8000 | 17.6502–20.6610 | 16.0% | 32 |
| serial_L512_d32_T1 | full_matrix / simd | 14.5110 | 13.6850–15.8205 | 14.7% | 1024 |
| serial_L512_d32_T1 | rowwise / simd | 14.6800 | 13.8045–15.7820 | 13.5% | 32 |
| serial_L2048_d32_T1 | full_matrix / scalar | 331.7055 | 319.8438–338.7123 | 5.7% | 16384 |
| serial_L2048_d32_T1 | rowwise / scalar | 350.5005 | 335.1062–356.4695 | 6.1% | 128 |
| serial_L2048_d32_T1 | full_matrix / simd | 228.6085 | 220.8135–257.4860 | 16.0% | 16384 |
| serial_L2048_d32_T1 | rowwise / simd | 243.3690 | 235.4597–258.9800 | 9.7% | 128 |
| openmp_L128_d64_T1 | full_matrix / openmp | 0.9565 | 0.8635–1.3838 | 54.4% | 64 |
| openmp_L128_d64_T1 | rowwise / openmp | 0.9870 | 0.8908–1.1062 | 21.8% | 8 |
| openmp_L128_d64_T2 | full_matrix / openmp | 0.5195 | 0.4988–0.6958 | 37.9% | 64 |
| openmp_L128_d64_T2 | rowwise / openmp | 0.4925 | 0.4670–0.8237 | 72.4% | 16 |
| openmp_L128_d64_T4 | full_matrix / openmp | 0.6925 | 0.6675–0.7210 | 7.7% | 64 |
| openmp_L128_d64_T4 | rowwise / openmp | 0.6600 | 0.6455–0.6800 | 5.2% | 32 |
| openmp_L128_d64_T8 | full_matrix / openmp | 0.4800 | 0.4467–0.5012 | 11.4% | 64 |
| openmp_L128_d64_T8 | rowwise / openmp | 0.4195 | 0.3950–0.4447 | 11.9% | 64 |
| openmp_L4096_d64_T1 | full_matrix / openmp | 925.7210 | 903.0332–960.8077 | 6.2% | 65536 |
| openmp_L4096_d64_T1 | rowwise / openmp | 978.8660 | 967.9030–1021.1457 | 5.4% | 256 |
| openmp_L4096_d64_T2 | full_matrix / openmp | 468.9555 | 454.3775–473.9520 | 4.2% | 65536 |
| openmp_L4096_d64_T2 | rowwise / openmp | 479.4370 | 466.9930–503.2880 | 7.6% | 512 |
| openmp_L4096_d64_T4 | full_matrix / openmp | 297.9360 | 286.8202–316.0210 | 9.8% | 65536 |
| openmp_L4096_d64_T4 | rowwise / openmp | 307.9555 | 292.7125–331.7303 | 12.7% | 1024 |
| openmp_L4096_d64_T8 | full_matrix / openmp | 211.2915 | 206.0150–218.5655 | 5.9% | 65536 |
| openmp_L4096_d64_T8 | rowwise / openmp | 224.0580 | 214.7025–233.6920 | 8.5% | 2048 |

## Largest relative spreads

Ordered by IQR / median to help choose repeat runs. This is a descriptive ranking, not a significance test.

- openmp_L128_d64_T2, rowwise / openmp: 72.4%.
- openmp_L128_d64_T1, full_matrix / openmp: 54.4%.
- serial_L2048_d64_T1, full_matrix / simd: 53.8%.
- serial_L128_d64_T1, full_matrix / scalar: 53.0%.
- serial_L128_d64_T1, rowwise / simd: 52.3%.

## Not measured

- blocked_streaming: not integrated
