# OpenMP parallel efficiency

This adds the efficiency requested in the proposal, using the existing [three-batch study](../controlled_20261023/README.md). It introduces no new performance measurements and leaves the archived raw data unchanged.

For each batch and route, S(T)=time(OpenMP T1)/time(OpenMP T). Efficiency is E(T)=S(T)/T. The CSV stores fractions; multiply by 100 to display percentages. Median/min/max summarize three batch efficiencies, not confidence intervals.

| Route, L4096 d=dv64 | T4 efficiency | T8 efficiency |
|---|---:|---:|
| Full matrix | 93.55% | 38.43% |
| Rowwise | 89.06% | 38.14% |
| Blocked streaming | 88.47% | 37.15% |

Efficiency is not measured CPU utilization. T8 uses mixed CPU classes here, so these numbers do not describe scaling over eight identical cores.

[efficiency_by_batch.csv](efficiency_by_batch.csv) contains 72 batch-level values; [efficiency_summary.csv](efficiency_summary.csv) contains 24 summaries, covering all three routes, L128/L4096 and T1/2/4/8. Source hashes and analysis details are in [analysis.json](analysis.json).

From the repository root, use a new output directory:

```sh
python tools/analyze_thread_efficiency.py --input examples/controlled_20261023 --out results/my_efficiency
```

The script first validates the full archived study using the existing checks, then computes efficiency from each within-batch speedup.

中文：这是“加速比÷线程数”，不是任务管理器里的CPU占用率。例如分块T4加速3.539倍，效率约为3.539÷4=88.47%。这张表直接由已有记录计算，不需要重新跑实验。
