# Attention Optimization on CPU

Memory-Performance Trade-offs of Blocked and Streaming Attention on a CPU

Welcome to CEG5206 team project, Team GP8! Group members are: Li Ruiqin, Cui Ziwei, Tang Zhenxiong, Chen Caoyu.
Deadline 13 Nov 2026, 23:59.

## 测试工具进度（陈曹玉）

这次补的是统一测试和计时工具：把已有的全矩阵版、逐行版放到同一套程序里，用相同输入检查结果、测运行时间，并自动保存结果表和电脑配置。后面接入优化版后，可以按同一套规则比较。

- **已完成：** 两个标量版本的正确性检查、预热和重复计时、结果保存，以及工具自身的测试；已在本地 Windows 跑通，并上传一组 L=128、512 的示例数据。
- **有什么用：** 大家不用各写一套计时程序；后续比较各版本速度、整理图表和写实验报告，都能用这套工具输出的数据。
- **接下来：** 扩展到提案要求的全部实验配置，接入分块、SIMD 和多线程版本，再做性能比较与分析。PyTorch 对照和 roofline 按环境可行性推进。

目前完成了测试工具的第一版，完整性能实验和分析还没完成。新版本接入时需要对应的调用接口、工作区需求和运行参数。

[运行方法](docs/SCALAR_BENCHMARK.md) · [已跑的示例结果](examples/scalar_20261020/README.md)

## Objectives

Study how optimization choices affect the performance of scaled dot-product attention on a CPU, under one
numerical contract, one FP64 oracle and one harness:

- implement and validate the CPU attention kernels with the same numerical contract, as three routes
  (full-matrix, row-wise, blocked streaming) in scalar, basic SIMD and SIMD + static OpenMP form;
- identify when blocking and online softmax reduce memory traffic or improve locality;
- quantify the effects of basic SIMD dot-product and value-accumulation loops, and of OpenMP query-block
  parallelism;
- explain observed bottlenecks using throughput and workspace measurements, supported where feasible by
  bandwidth measurements and a roofline-style model.

A PyTorch CPU SDPA comparison is a target. A packed microkernel or a vectorized exponential is an optional
extension; at most one is taken, decided at the end of Week 4.

GPU kernels, multi-head fusion, training, causal
masking and KV-cache decoding are outside the project.

## Weeks

| Week | Planned outcome |
|---|---|
| 1 | Contract, CPU and compiler settings, FP64 oracle, full-matrix baseline, harness skeleton. PyTorch checked. |
| 2 | Row-wise and blocked-streaming kernels validated. Repeatable timing. Small CPU SDPA comparison if it runs. |
| 3 | Shared basic SIMD loops on all routes. Vectorization verified. Bounded tile parameters picked. Result tables drafted. |
| 4 | Static OpenMP. Core comparisons and target evidence. Figures drafted. Decision on at most one extension. |
| 5 | Results and interpretation finished. Extension finished. Report cross-reviewed. Code and settings frozen. |
| 6 | Frozen results reproduced, defects fixed, report finished. Last half-week kept as buffer. |

## Team roles

- Li Ruiqin: scalar baselines, FP64 oracle, validation tests
- Cui Ziwei: blocked streaming
- Tang Zhenxiong: SIMD and OpenMP
- Chen Caoyu: harness, PyTorch comparison, roofline

Each owner supplies their own tests, measurements and figures; the report is written together.

## How we work

`main` is shared, so please be careful: work on a dev branch of your own -- `dev/liruiqin` as an
example -- and push there.

A pull request then gives the others a chance to look before the work lands on `main`. A quick fix
can still go straight to `main`.

Writing a weekly note is also useful for the project report.

## Build and run

```
build.bat                    # Windows (MSYS2 g++)
bash build.sh                # Linux / macOS
build\attn_tests.exe         # both scalar routes; must end with SUITE PASSED
build\attn_blocks.exe        # route 2, block sizes and row orders, bit for bit
python tools/run_scalar_bench.py --out results/run1  # both scalar routes, L=128 and 512
```

On Linux or macOS the binaries drop the .exe suffix.

The small harness validates both routes before timing, alternates their order, and saves
raw samples, median/IQR, numerical error, workspace and environment information. It refuses
to overwrite earlier outputs. See [benchmark instructions](docs/SCALAR_BENCHMARK.md).
A small local example with raw samples and its limitations is in [examples/scalar_20261020](examples/scalar_20261020/README.md).
`python tests/test_benchmark.py build/attn_bench.exe` independently checks the CSV statistics
and invalid-argument handling (use `build/attn_bench` on Linux/macOS).
`python tests/test_contract_gate.py` builds a separate fault-injection binary to verify
that finite errors, NaN, and a failed final check cannot produce performance CSVs.

The full-matrix scalar route materializes the complete score matrix, normalizes all rows
in place to form P, then computes P*V. Rowwise uses the same row arithmetic but completes
each query row immediately. This phase-order correction changes the baseline; measurements
from older binaries should be kept separate.

The scalar build uses -O3 -fno-tree-vectorize to keep automatic vectorization out of the baseline. SIMD variants are measured separately.
