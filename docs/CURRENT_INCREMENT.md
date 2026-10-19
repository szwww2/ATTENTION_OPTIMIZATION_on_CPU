# Current increment: scalar validation and timing

This is the first part of Chen Caoyu's harness integration work. It puts the existing
full-matrix and rowwise scalar kernels through one input, validation and timing flow.
It is not the complete project or the final performance study.

## What changed and why

- The full-matrix baseline now completes all scores, then all row softmax operations,
  then all PV outputs. It previously allocated L² scratch but finished each row at once.
  Rowwise uses the same row arithmetic; the phase-order correction is a separate commit
  for review. Previous measurements must remain separate.
- Both routes use the same Q/K/V buffers and independent FP64 reference. Each must pass
  before timing and after its final timed call. Warm-up and alternating route order
  precede raw samples, median/IQR, error and workspace records.
- The Python runner records source/binary hashes, build settings and machine metadata.
- Review also fixed the NumPy checker's missing C++ FP32 pass/fail gate and made the
  Bq=0 workspace helper agree with the kernel's one-row fallback. The benchmark CLI
  still rejects zero Bq. Both routes now receive the full numerical contract suite.

## Try it and inspect the evidence

See [SCALAR_BENCHMARK.md](SCALAR_BENCHMARK.md) for build commands, timer boundaries and CSV columns.
After building and running the listed tests:

```text
python tools/run_scalar_bench.py --out results/my_first_run
```

Inspect `my_first_run_environment.json` (status/exit code and environment), then
`my_first_run_summary.csv`, then the corresponding samples in `my_first_run_raw.csv`.
Use a new prefix each time. The small historical example is in
[`examples/scalar_20261020`](../examples/scalar_20261020); its recorded revision is
49cd7c6, not a claim of measurement on every later commit.

Local Windows/GCC 16.2.0 checks after the review fixes: 90 numerical checks covering
both routes; 448 workspace/block/order checks; 4 benchmark/runner unittest methods;
3 injected-failure methods; 1 NumPy checker method with a valid dump and three corrupt
outputs (finite error, NaN, infinity). Counts include assertions, not distinct shapes.
NumPy is only required for the independent cross-check and its regression test.
These are local checks, not GitHub CI or validation on Linux/macOS. Sanitizers were not run.

## What remains

Core work still includes integrating blocked streaming, SIMD and static OpenMP;
running the proposal's six shapes and required thread comparisons; freezing settings;
and interpreting results with implementation owners. PyTorch CPU and DRAM-level
roofline are conditional targets, subject to environment support in the final proposal.
Owners still supply their own tests, measurements, plots and explanations.

Current timings are native scalar measurements. Affinity and power mode are uncontrolled.
The example does not establish stable speedup, DRAM traffic or a cache bottleneck.
`workspace_bytes` is allocated score scratch, not traffic. Rowwise currently allocates
Bq×L floats (default Bq=64), computing one row at a time; it is not key-block streaming.

Please review the phase-order change, numerical checks and timer boundary before merging.

## 中文定位

这是陈曹玉任务的第一阶段：把已有两条标量算法放进同一套“验答案、计时、留证据”的工具。
这次补的是统一实验基础，不是替队友写完所有优化，也不是整项作业已完成。
先看环境记录是否成功，再看汇总，最后回到原始读数核对。完整流程和命令见上面的说明。
后续由各实现负责人提供新版本与证据，再接进同一套规则；核心实验、PyTorch 条件目标、
roofline 条件目标和最后报告仍要继续做。PR 是代码审查入口，目前为草稿，不是 Canvas 回执。
