# Attention Optimization on CPU

Memory-Performance Trade-offs of Blocked and Streaming Attention on a CPU

Welcome to CEG5206 team project, Team GP8! Group members are: Li Ruiqin, Cui Ziwei, Tang Zhenxiong, Chen Caoyu.
Deadline 13 Nov 2026, 23:59.

## 本轮实验结果（陈曹玉）

已把三条路线的普通、SIMD 和 OpenMP 版接进统一测试，共九种实现。它们用相同输入检查答案，再计时。10 月 23 日完成三轮主实验，每轮 60 个配置，共 3,600 次原始计时；PyTorch CPU 对照、roofline 和线程效率也已整理好。

**先看下面的结果总览，再看需要的代码或数据。**

| 内容 | 从这里看 |
|---|---|
| 结果总览、比较图、原始数据 | [三轮实验](examples/controlled_20261023/README.md) |
| 多线程加速后的效率 | [线程效率表](examples/thread_efficiency_20261023/README.md) |
| Roofline 假设和图 | [Roofline 分析](examples/controlled_20261023/roofline_analysis/README.md) |
| 我的实验方法与结果段落，供合并进小组报告 | [英文报告草稿](docs/REPORT_CHEN_SECTION_DRAFT.md) |
| 正式程序使用 SIMD 的证据 | [GCC 汇编核对](docs/GCC_CODEGEN_EVIDENCE.md) |
| 跑代码、重建图表、查看机器信息 | [九版本测试](docs/INTEGRATED_BENCHMARK.md) · [重建统计和图](examples/controlled_20261023/README.md) · [机器记录](examples/machine_20261023/README.md) |

这台电脑上，L=4096、d=dv=64 时三条路线都是 4 线程比 8 线程快。小尺寸波动较大；roofline 的内存流量是模型估算。具体数据和限制都在结果总览里。

代码、数据和说明已在 `dev/chencaoyu-integration`，集中 review [PR #5](https://github.com/szwww2/ATTENTION_OPTIMIZATION_on_CPU/pull/5)。报告草稿已补上提案四个假设的结论和计时成本说明，另附正式 GCC 程序的汇编证据。本次只补文档和证据，原始实验数据不变。

这个 PR 包含 #4 的工作，仍保留草稿。请李检查基线和参考答案，唐检查 SIMD/OpenMP 的接入，崔检查分块路线和块大小说明；大家再一起定报告用图和最终版本。看结果不需要重新跑实验；需要检查时可以先用保存的数据重建图表。

<details>
<summary>以前的实验和详细说明</summary>

以下是阶段记录，本轮结果以上面的 10 月 23 日三轮实验为准。

[第一阶段示例](examples/scalar_20261020/README.md) · [首轮整合实验](examples/integrated_20261021/README.md) · [10 月 22 日两路线复测](examples/validation_20261022/README.md) · [结果字段](docs/SCALAR_BENCHMARK.md) · [分析方法](docs/RESULT_ANALYSIS.md) · [完整实验流程](docs/CONTROLLED_STUDY.md)

</details>

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
Blocked streaming is included in both the native tests (`--streaming`) and the
PyTorch API comparison. The three-batch study and its data checks are complete;
see [the collected results](examples/controlled_20261023/README.md).
The team still needs to review the interpretation and finish the report.

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
