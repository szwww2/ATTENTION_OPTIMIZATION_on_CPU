# GP8 三轮受控 CPU 实验 · 2026-10-23

本包计时对应源码 `f7b4b09a6d4243269401fccb14e61f5664f1d2e5`。先固定 `Bq=16`，用 3 轮调参比较 `Bk=128/256/512`，按预设规则选用 `Bk=128`，再完成三轮 native 测量、单轮 PyTorch API 对照和机器参考校准。

| 数据 | 范围 | 原始样本数 |
|---|---|---:|
| native | 3 轮 × 60 配置 × 20 次 | 3,600 |
| 调参 | 3 轮 × 3 个 Bk × 20 次 | 180 |
| API | 12 个尺寸/线程组合 × 4 路 × 20 次 | 960 |
| 校准 | 2 种负载 × 4 个线程数 × 7 次 | 56 |

先看 [线程延迟图](summary/thread_latency.png)、[配对加速比图](summary/thread_speedup.png) 和 [API 对照图](pytorch_analysis/pytorch_api.png)。查数字用 [三轮时间表](summary/timing_summary.csv)、[配对加速比表](summary/paired_speedup_summary.csv)；查设置用 [协议](protocol.json)、[冻结记录](frozen_settings.json) 和各目录的 environment/manifest JSON。

在 `L=4096, d=dv=64` 下，native 三轮中位数如下。加速比先在各轮内计算，再取三轮中位数；误差线是三轮的最小/最大值，不是置信区间。

| 路线 | SIMD 时间 / ms | OpenMP T4 时间 / ms | SIMD / scalar 加速比 | T4 / OpenMP T1 加速比 |
|---|---:|---:|---:|---:|
| Full matrix | 775.358 | 213.924 | 2.240× | 3.742× |
| Rowwise | 770.423 | 213.448 | 2.260× | 3.562× |
| Blocked streaming | 775.839 | 216.406 | 1.559× | 3.539× |

该尺寸下三条路线的 T8 都慢于 T4。T4 scratch 容量分别为 64 MiB、1 MiB 和 48.5 KiB；这是程序工作区容量，不是进程总内存或实际 DRAM 流量。API 对照中的 PyTorch SDPA 在 T4 为 12.776 ms，自研三路为 213.394–232.950 ms；该计时包含 Python 调用及分配，与上表 native 口径分开。API 只有一轮。

分块 scalar 在该尺寸下更快，但 PV 累加的循环顺序也不同，不能把差异单独归功于分块。SIMD 三路线耗时相近；softmax 的具体耗时占比尚未测量。

调参值 Bk128 为 296.628 ms，Bk512 为 298.378 ms，两者很接近；128 是按预设规则选中，不能据此声称显著更优或适合所有模式。

限制：180 个配置次中，32 个的 IQR/median 超过 10%，8 个超过 20%，最大约 58%；小输入尤其不宜按小差距排稳定名次。CPU ID 集合只限制进程可以在哪些 CPU 上运行，不保证独占 CPU 或逐 worker 绑定。T1/2/4 使用记录中的 efficiency class 1，T8 混合 class 1 和 0。未锁定或记录温度/频率，也没有持续背景负载监控，不能证明全程空闲或据此判定某段已受污染。[Roofline 图](roofline_analysis/README.md)只用 native 第 1 轮；流量是两种模型假设，不能据此报告实测峰值利用率或已确定的瓶颈。

重新分析时，在仓库根目录使用 Python 3，并先安装 `tools/requirements-plots.txt` 中的绘图依赖。三个输出目录必须尚不存在：

```shell
python examples/controlled_20261023/verify_package.py
python tools/summarize_controlled_study.py --input examples/controlled_20261023 --out results/reanalysis_20261023/native --cpu-class-note "T1/2/4: efficiency class 1; T8: mixed classes 1 and 0"
python tools/analyze_pytorch.py --input examples/controlled_20261023/pytorch --out results/reanalysis_20261023/pytorch
python tools/analyze_roofline.py --calibration examples/controlled_20261023/calibration --grid examples/controlled_20261023/native_1 --out results/reanalysis_20261023/roofline --thread-scaling-note "T1/2/4: efficiency class 1; T8: mixed classes 1 and 0; native batch 1 only"
```

以上命令读取保存的数据并重建统计与图，不启动新的性能测量，也不需要安装 PyTorch 或编译 C++。重新采集实验需要本仓库的运行器、编译器和相应依赖；原机器的时间不保证能在其他机器复现。导出范围和文件校验见 [EXPORT.md](EXPORT.md)。
