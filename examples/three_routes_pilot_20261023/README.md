# 三路线接入试跑 · 2026-10-23

全矩阵、逐行、分块三条路线的普通版、SIMD 版和 OpenMP 版已接进同一套测试。
这轮完成 60 个配置、1,200 次计时，全部通过计时前后的 FP64 对照；
分析脚本重新核算了原始样本与统计值，43 份原始记录已与本地测量文件逐字节核对。

这是检查接入和记录流程的一轮试跑。前段测量与同机 WSL 的编译、正确性测试有重叠，
没有固定 CPU 核或后台负载；部分耗时波动明显，最大 IQR/中位数约 81.5%。
不要把这轮的加速倍数当作最终结论，也不能据此认定某种算法或电脑有问题。

## 看哪些文件

- [汇总表和波动](analysis/RESULTS.md)，[全部配置 CSV](analysis/comparison.csv)
- [单线程耗时](analysis/serial_latency.png)，[多线程耗时](analysis/thread_latency.png)，[临时空间](analysis/workspace.png)
- `*_raw.csv` 保存每次耗时，`*_summary.csv` 保存统计与误差，`*_environment.json` 保存版本和运行环境。

## 这轮设置

源版本 `8df4cdcb42d3f1826add971c8dc74b1d0b9b5ce4`，测量时工作树干净。
Windows / Intel Core Ultra 7 356H / GCC 16.2.0，16 物理核，AVX2；
自动向量化及 FMA contraction 关闭，`OMP_WAIT_POLICY=PASSIVE`。
每配置预热 10 次、计时 20 次；轮换执行顺序，输入、分配、参考答案和文件写入不计时。
本机电源方案读到“高性能”，未修改设置；亲和性未固定。

Bq=16、Bk=256 是本轮预先固定的值，尚未按最终机器重新调参。
Streaming 的完整 scratch 包含分数块、累积输出与在线 softmax 状态。
Streaming 记录每次计算的实际线程数；全矩阵和逐行沿用运行前探测，口径见环境文件。
图中的 scratch 是临时空间容量，不是实际 DRAM 流量。

## 运行

```text
python tools/build_integrated.py
python tools/run_project_grid.py --physical-cores 16 --streaming --key-block 256 --out results/new_three_route_run
python tools/analyze_results.py --input results/new_three_route_run --out results/new_three_route_analysis
```

运行前在自己的 shell 中设置 `OMP_WAIT_POLICY=PASSIVE`；核数使用该机器的真实物理核预算。
完整方法见 [统一测试说明](../../docs/INTEGRATED_BENCHMARK.md)。

下一步先检查运行条件，再按提案在 L=2048、d=dv=64、Bq=16 下比较 Bk=128/256/512，
冻结设置后独立复测。旧的两路线数据、队友各自的数据保持分开。
三路线 PyTorch API 对照及最终性能解释仍待补齐。
