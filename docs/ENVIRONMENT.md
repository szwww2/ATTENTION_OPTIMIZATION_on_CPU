# Environment

Here we record the machines the numbers came from.

## Machines

| who | CPU | cores | memory | OS | compiler |
|---|---|---|---|---|---|
| Li Ruiqin | AMD Ryzen 5 9600X, 3.9 GHz base, AVX2 and AVX-512 | 6 | 31.1 GB | Windows 11 Pro, build 10.0.26200 | g++ 16.1.0 (MSYS2, UCRT64) |
| Cui Ziwei | | | | | |
| Tang Zhenxiong | | | | | |
| Chen Caoyu | Intel Core Ultra 7 356H, AVX2 | 16 (16 logical) | not recorded | Windows 11, build 10.0.26200 | g++ 16.2.0 (MinGW) |

scalar everywhere: `-O3 -fno-tree-vectorize`. SIMD and threads flags are in `docs/INTEGRATION.md`.
6 cores here, so an OpenMP thread count past 6 does not mean much.

## Measurement machine for the 23 October study

All numbers in the final report come from Chen Caoyu's laptop. The facts below are copied from the
records archived with the data, not retyped from memory.

| Item | Recorded value | Where |
|---|---|---|
| CPU | Intel(R) Core(TM) Ultra 7 356H, 16 cores / 16 logical processors | `examples/machine_20261023/cache_wmi.json` |
| Core classes | Windows efficiency class 1 on CPU0-3, class 0 on CPU4-15 | `examples/machine_20261023/cpu_topology.csv` |
| Caches (WMI totals) | L2 24576 KiB, L3 18432 KiB | `examples/machine_20261023/cache_wmi.json` |
| OS | Windows-11-10.0.26200-SP0 | every `*_environment.json` in `examples/controlled_20261023/` |
| Compiler and flags | g++.exe (GCC) 16.2.0; `-std=c++17 -O3 -fno-tree-vectorize -ffp-contract=off`, `-mavx2` on the two AVX2 files, `-fopenmp` on OpenMP files | same files, field `build` |
| Power | AC line online; clock frequency and temperature not measured or fixed | same files, field `runtime_conditions` |
| Python packages | Python 3.12.14, NumPy 2.5.4, PyTorch 2.13.0+cpu | `examples/controlled_20261023/pytorch/*/environment.json` |
| Memory size | not recorded | |

## PyTorch

Li Ruiqin's machine, 2026-10-14. Python 3.12.7 with numpy 1.26.4. torch is installed in a couple of
environments here (2.8.0 and 2.13.0 with CUDA, 2.14.0 CPU only), so the SDPA comparison can run on
this machine. The CPU-only build is the one to use; the version and the backend go in the report next
to the numbers.

## Run output

`build/` and `results/` are ignored.
