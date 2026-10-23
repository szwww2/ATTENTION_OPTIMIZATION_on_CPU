# Attention Optimization on CPU

Memory-Performance Trade-offs of Blocked and Streaming Attention on a CPU

Welcome to CEG5206 team project, Team GP8! Group members are: Li Ruiqin, Cui Ziwei, Tang Zhenxiong, Chen Caoyu.
Deadline 13 Nov 2026, 23:59.

## 测试工具进度（陈曹玉）

统一测试工具已接入全矩阵、逐行、分块三条路线的普通版、SIMD 版和 OpenMP 版，共九种实现。它们用相同输入验答案、计时，自动保存原始数据、汇总表和运行环境。

10 月 23 日接入崔子为的分块代码，补了块大小、完整工作区和实际线程数的记录。七套数值测试通过，完成一轮 60 配置、1,200 次计时的[三路线接入试跑](examples/three_routes_pilot_20261023/README.md)。这轮耗时有明显波动，前段与其他编译测试有重叠，先用来检查流程；正式比较前还要检查运行条件、确认块大小并复测。

10 月 22 日的两路线复测、PyTorch CPU 对照和 roofline 参考实验见[历史两路线结果](examples/validation_20261022/README.md)。三路线的外部对照和最终分析还在继续；实际 DRAM 流量尚未测量。

当前整合代码在 `dev/chencaoyu-integration`，见 [PR #5](https://github.com/szwww2/ATTENTION_OPTIMIZATION_on_CPU/pull/5)。它接在早期测试工具的 [PR #4](https://github.com/szwww2/ATTENTION_OPTIMIZATION_on_CPU/pull/4) 之上，两者仍待审查、处理与 `main` 的合并冲突。

分析脚本会先核对原始数据，再生成汇总表和比较图。[首轮历史数据](examples/integrated_20261021/README.md) · [重新生成的方法](docs/RESULT_ANALYSIS.md)。

[九版本的运行方法](docs/INTEGRATED_BENCHMARK.md) · [结果字段说明](docs/SCALAR_BENCHMARK.md) · [第一阶段示例](examples/scalar_20261020/README.md)

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

Chen's benchmark work now includes the shared scalar/SIMD/OpenMP tests, saved-result
checks and plots, an allocating PyTorch CPU comparison, and a roofline reference
experiment. Run instructions: [native results](docs/RESULT_ANALYSIS.md),
[PyTorch](docs/PYTORCH_COMPARISON.md), [roofline](docs/ROOFLINE_REFERENCE.md).
Blocked streaming now joins the native tests with `--streaming`; its PyTorch API
comparison and final three-route interpretation remain to be added.

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

## Blocked streaming: scalar, SIMD and OpenMP (Cui Ziwei)

Route 3 is implemented with explicit query/key blocks and online softmax.
Run `build/attn_streaming_tests` after `bash build.sh` (Windows: `.exe`).
See [the interface and verification guide](docs/STREAMING_CUIZIWEI.md) and
[the local scalar tile pilot](examples/streaming_cuiziwei_20261022/README.md).
SIMD and static OpenMP are now implemented for route 3. Build with
`python3 -B tools/build_streaming_optimized.py`, then run
`build/streaming_optimized/test`. See [optimized API/tests](docs/STREAMING_OPTIMIZED.md)
and [same-build measurements](examples/streaming_optimized_20261022/README.md).
The shared native harness now includes all three streaming modes.
See [the team review checklist](docs/STREAMING_TEAM_HANDOFF.md) for integration
ownership, interface decisions and the nine-mode comparison matrix.




The scalar build uses -O3 -fno-tree-vectorize to keep automatic vectorization out of the baseline. SIMD variants are measured separately.
