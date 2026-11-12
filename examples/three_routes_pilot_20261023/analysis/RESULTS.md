# Saved grid results

14 runs, 60 configurations, 1200 raw timings. Summary statistics were checked against every raw sample.

Measured source: `8df4cdcb42d3f1826add971c8dc74b1d0b9b5ce4`. Dirty tree: `False`.
Affinity: not pinned by the harness. Power mode: not controlled by the harness.

Points show medians; error bars span Q1 to Q3 of individual timings, not confidence intervals. Thread points come from separate runs. Workspace means allocated scratch, not DRAM traffic or total process memory.

| Run | Route / mode | Median ms | Q1–Q3 ms | IQR / median | Scratch KiB |
|---|---|---:|---:|---:|---:|
| serial_L128_d64_T1 | full_matrix / scalar | 2.9875 | 2.9248–3.0118 | 2.9% | 64 |
| serial_L128_d64_T1 | rowwise / scalar | 3.0140 | 2.9272–3.0375 | 3.7% | 8 |
| serial_L128_d64_T1 | blocked_streaming / scalar | 2.8155 | 2.6645–2.8450 | 6.4% | 12.125 |
| serial_L128_d64_T1 | full_matrix / simd | 2.2160 | 1.3580–2.2600 | 40.7% | 64 |
| serial_L128_d64_T1 | rowwise / simd | 2.2035 | 1.8908–2.2243 | 15.1% | 8 |
| serial_L128_d64_T1 | blocked_streaming / simd | 2.2275 | 1.8847–2.2757 | 17.6% | 12.125 |
| serial_L512_d64_T1 | full_matrix / scalar | 28.9790 | 27.4035–31.1382 | 12.9% | 1024 |
| serial_L512_d64_T1 | rowwise / scalar | 31.9760 | 30.7575–34.6497 | 12.2% | 32 |
| serial_L512_d64_T1 | blocked_streaming / scalar | 35.1530 | 31.5348–39.4532 | 22.5% | 20.125 |
| serial_L512_d64_T1 | full_matrix / simd | 18.2620 | 16.7675–19.4585 | 14.7% | 1024 |
| serial_L512_d64_T1 | rowwise / simd | 18.9655 | 18.4975–21.4992 | 15.8% | 32 |
| serial_L512_d64_T1 | blocked_streaming / simd | 18.1730 | 16.8262–19.8372 | 16.6% | 20.125 |
| serial_L2048_d64_T1 | full_matrix / scalar | 828.6205 | 816.3368–843.7468 | 3.3% | 16384 |
| serial_L2048_d64_T1 | rowwise / scalar | 871.9740 | 864.7458–888.7285 | 2.8% | 128 |
| serial_L2048_d64_T1 | blocked_streaming / scalar | 749.0925 | 743.0488–757.2787 | 1.9% | 20.125 |
| serial_L2048_d64_T1 | full_matrix / simd | 571.0155 | 552.3598–580.3433 | 4.9% | 16384 |
| serial_L2048_d64_T1 | rowwise / simd | 598.7235 | 593.3247–625.1163 | 5.3% | 128 |
| serial_L2048_d64_T1 | blocked_streaming / simd | 573.2430 | 564.1643–583.9360 | 3.4% | 20.125 |
| serial_L4096_d64_T1 | full_matrix / scalar | 2471.5470 | 2411.6190–2530.1852 | 4.8% | 65536 |
| serial_L4096_d64_T1 | rowwise / scalar | 2716.3095 | 2661.4172–2765.9182 | 3.8% | 256 |
| serial_L4096_d64_T1 | blocked_streaming / scalar | 2133.6240 | 2085.8610–2162.7293 | 3.6% | 20.125 |
| serial_L4096_d64_T1 | full_matrix / simd | 1192.6480 | 1164.4190–1230.5318 | 5.5% | 65536 |
| serial_L4096_d64_T1 | rowwise / simd | 1386.2235 | 1307.9415–1447.2330 | 10.0% | 256 |
| serial_L4096_d64_T1 | blocked_streaming / simd | 1207.9140 | 1142.7605–1272.4925 | 10.7% | 20.125 |
| serial_L512_d32_T1 | full_matrix / scalar | 29.7115 | 25.7298–43.0370 | 58.3% | 1024 |
| serial_L512_d32_T1 | rowwise / scalar | 28.8005 | 25.1178–34.6643 | 33.1% | 32 |
| serial_L512_d32_T1 | blocked_streaming / scalar | 27.0455 | 24.9883–32.9565 | 29.5% | 18.125 |
| serial_L512_d32_T1 | full_matrix / simd | 20.8640 | 17.6322–23.0162 | 25.8% | 1024 |
| serial_L512_d32_T1 | rowwise / simd | 21.3295 | 18.8972–25.7853 | 32.3% | 32 |
| serial_L512_d32_T1 | blocked_streaming / simd | 22.1965 | 18.0873–24.4712 | 28.8% | 18.125 |
| serial_L2048_d32_T1 | full_matrix / scalar | 429.7395 | 413.3428–463.7140 | 11.7% | 16384 |
| serial_L2048_d32_T1 | rowwise / scalar | 460.8890 | 451.1795–474.9585 | 5.2% | 128 |
| serial_L2048_d32_T1 | blocked_streaming / scalar | 408.8020 | 388.8473–424.9222 | 8.8% | 18.125 |
| serial_L2048_d32_T1 | full_matrix / simd | 294.9475 | 282.7180–309.8162 | 9.2% | 16384 |
| serial_L2048_d32_T1 | rowwise / simd | 320.2700 | 314.0240–346.3760 | 10.1% | 128 |
| serial_L2048_d32_T1 | blocked_streaming / simd | 304.8500 | 290.1035–324.5027 | 11.3% | 18.125 |
| openmp_L128_d64_T1 | full_matrix / openmp | 2.8835 | 2.4885–3.1233 | 22.0% | 64 |
| openmp_L128_d64_T1 | rowwise / openmp | 2.8080 | 2.5955–2.9973 | 14.3% | 8 |
| openmp_L128_d64_T1 | blocked_streaming / openmp | 2.6070 | 2.3942–2.9737 | 22.2% | 12.125 |
| openmp_L128_d64_T2 | full_matrix / openmp | 0.7855 | 0.6520–1.2583 | 77.2% | 64 |
| openmp_L128_d64_T2 | rowwise / openmp | 0.7550 | 0.6187–1.2338 | 81.5% | 16 |
| openmp_L128_d64_T2 | blocked_streaming / openmp | 0.9150 | 0.6050–1.3227 | 78.4% | 24.25 |
| openmp_L128_d64_T4 | full_matrix / openmp | 0.8275 | 0.7532–0.8800 | 15.3% | 64 |
| openmp_L128_d64_T4 | rowwise / openmp | 0.7700 | 0.6927–0.8280 | 17.6% | 32 |
| openmp_L128_d64_T4 | blocked_streaming / openmp | 0.7570 | 0.7313–0.8375 | 14.0% | 48.5 |
| openmp_L128_d64_T8 | full_matrix / openmp | 0.6885 | 0.5782–0.7340 | 22.6% | 64 |
| openmp_L128_d64_T8 | rowwise / openmp | 0.5935 | 0.5197–0.7012 | 30.6% | 64 |
| openmp_L128_d64_T8 | blocked_streaming / openmp | 0.5540 | 0.4903–0.7137 | 40.3% | 97 |
| openmp_L4096_d64_T1 | full_matrix / openmp | 2351.3445 | 2318.8010–2372.6350 | 2.3% | 65536 |
| openmp_L4096_d64_T1 | rowwise / openmp | 2635.1075 | 2592.5705–2689.1308 | 3.7% | 256 |
| openmp_L4096_d64_T1 | blocked_streaming / openmp | 2350.1520 | 2323.2047–2383.7865 | 2.6% | 20.125 |
| openmp_L4096_d64_T2 | full_matrix / openmp | 1169.1825 | 1157.6903–1194.0588 | 3.1% | 65536 |
| openmp_L4096_d64_T2 | rowwise / openmp | 1311.3770 | 1282.9965–1341.0628 | 4.4% | 512 |
| openmp_L4096_d64_T2 | blocked_streaming / openmp | 1174.2825 | 1154.0788–1195.2173 | 3.5% | 40.25 |
| openmp_L4096_d64_T4 | full_matrix / openmp | 627.1535 | 622.3452–644.2130 | 3.5% | 65536 |
| openmp_L4096_d64_T4 | rowwise / openmp | 691.4965 | 668.6355–718.8490 | 7.3% | 1024 |
| openmp_L4096_d64_T4 | blocked_streaming / openmp | 626.4775 | 621.6762–642.6363 | 3.3% | 80.5 |
| openmp_L4096_d64_T8 | full_matrix / openmp | 363.8385 | 344.2480–379.1633 | 9.6% | 65536 |
| openmp_L4096_d64_T8 | rowwise / openmp | 383.4535 | 369.3268–403.2210 | 8.8% | 2048 |
| openmp_L4096_d64_T8 | blocked_streaming / openmp | 362.7640 | 342.7348–383.4383 | 11.2% | 161 |

## Largest relative spreads

Ordered by IQR / median to help choose repeat runs. This is a descriptive ranking, not a significance test.

- openmp_L128_d64_T2, rowwise / openmp: 81.5%.
- openmp_L128_d64_T2, blocked_streaming / openmp: 78.4%.
- openmp_L128_d64_T2, full_matrix / openmp: 77.2%.
- serial_L512_d32_T1, full_matrix / scalar: 58.3%.
- serial_L128_d64_T1, full_matrix / simd: 40.7%.

## Unavailable configurations

None of the configurations in this grid was omitted.
