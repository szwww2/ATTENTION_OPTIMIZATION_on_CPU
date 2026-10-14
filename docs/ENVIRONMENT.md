# Environment

Here we record the machines the numbers came from.

## Machines

| who | CPU | cores | memory | OS | compiler |
|---|---|---|---|---|---|
| Li Ruiqin | AMD Ryzen 5 9600X, 3.9 GHz base, AVX2 and AVX-512 | 6 | 31.1 GB | Windows 11 Pro, build 10.0.26200 | g++ 16.1.0 (MSYS2, UCRT64) |
| Cui Ziwei | | | | | |
| Tang Zhenxiong | | | | | |
| Chen Caoyu | | | | | |

scalar everywhere: `-O3 -fno-tree-vectorize`. SIMD and threads flags are in `docs/INTEGRATION.md`.
6 cores here, so an OpenMP thread count past 6 does not mean much.

## PyTorch

Li Ruiqin's machine, 2026-10-14. Python 3.12.7 (Anaconda base), numpy 1.26.4.

| environment | torch |
|---|---|
| audiobook | 2.8.0+cu128, CUDA available |
| ceg5304 | 2.13.0+cu130, CUDA available |
| drones | 2.14.0+cpu |
| base, bigmodel, ceg5205, ceg5306, pianist | not installed |

## Run output

`build/` and `results/` are ignored.
