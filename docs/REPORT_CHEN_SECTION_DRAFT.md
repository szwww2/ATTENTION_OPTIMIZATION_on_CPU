# Experimental methods and results — Chen Caoyu

## Method

The shared harness compares full-matrix, rowwise and blocked-streaming attention in scalar, AVX2 SIMD and static OpenMP modes. All versions receive the same FP32 inputs. Before timing, each output is checked against the independent FP64 reference using the project contract: abs(O - O64) <= 1e-4 + 1e-3 * abs(O64). The final output is checked again after timing. These checks do not inspect every intermediate timed output.

Native timings cover the complete kernel call, including softmax, reductions, online-state updates and OpenMP overhead where present. Input generation, allocation, the reference calculation and validation are outside the timer. The scalar build disables automatic vectorization. The fixed seed is 20261020.

There is no packing stage or FP64 conversion of the input arrays inside the native timer. The SIMD helpers accept unaligned pointers, use separate FP32 multiply/add operations and reduce each dot product's lane sums. Softmax calls `expf`. In this build, that statically linked wrapper converts one float argument to double, calls `exp`, then converts the result back; the general finite-input path in `exp` uses x87 instructions. These operations are inside the timer, but their time share was not measured. [Static inspection of the measured GCC executable](GCC_CODEGEN_EVIDENCE.md) also confirms packed vector arithmetic in all three optimized routes and no FMA instructions in the executable. Shared helpers do not imply equal time per FLOP: loop shape, reductions, data access and work around the helpers still differ.

The experiment ran on one Windows laptop with a Core Ultra 7 356H, 16 physical cores and GCC 16.2.0. Windows reports CPU0–3 as efficiency class 1 and CPU4–15 as class 0. A T-thread process may run on CPU0 through CPU(T-1), so T8 uses mixed classes. This is an allowed CPU set, not exclusive access or individual worker binding. The recorded WMI cache totals are 24 MiB L2 and 18 MiB L3; they do not establish per-worker capacity or cache residency. See the [machine records](../examples/machine_20261023/README.md).

Bq was fixed at 16. Three tuning batches compared Bk=128, 256 and 512 at L=2048, d=dv=64, using scalar T1. The smallest median of the three batch medians selected Bk=128. It was then held fixed for every mode. Bk128 and Bk512 were close (296.628 and 298.378 ms), so this selection is not evidence of a general optimum.

Each main batch contains 60 configurations: six input shapes for scalar/SIMD and L=128/4096, d=dv=64 at T=1/2/4/8 for OpenMP. Each configuration has ten warm-ups and twenty recorded samples. Three batches use forward, reverse and rotated job orders. All 3,600 timings are retained. Speedups are calculated within each batch before taking the median and range across the three ratios. Ranges are descriptive, not confidence intervals.

## Native results

At L=4096, d=dv=64, the medians across the three batch medians are:

| Route | Scalar / ms | SIMD / ms | OpenMP T4 / ms | SIMD speedup over scalar | T4 speedup over OpenMP T1 |
|---|---:|---:|---:|---:|---:|
| Full matrix | 1776.387 | 775.358 | 213.924 | 2.240x | 3.742x |
| Rowwise | 1769.858 | 770.423 | 213.448 | 2.260x | 3.562x |
| Blocked streaming | 1239.406 | 775.839 | 216.406 | 1.559x | 3.539x |

The speedup columns summarize within-batch ratios; dividing the displayed aggregate times is not the calculation used for these columns. [Latency and paired-speedup plots](../examples/controlled_20261023/README.md) show the batch ranges.

Parallel efficiency is E(T)=S(T)/T, with S(T) measured against the same route's OpenMP T1 in the same batch. At this shape, median T4 efficiencies are 93.55%, 89.06% and 88.47% for full matrix, rowwise and blocked streaming. T8 efficiencies fall to 38.43%, 38.14% and 37.15%. These ratios are not measured CPU utilization. T8 was slower than T4 for all three routes in this experiment; mixed CPU classes and static work assignment are possible explanations, not isolated causes. [All 24 efficiency summaries](../examples/thread_efficiency_20261023/README.md) also include L=128.

Blocked scalar is faster, but its PV loop order and data reuse both differ from rowwise scalar. The result does not isolate the benefit of blocking alone. SIMD times are similar across the three routes. At T4, scratch capacities are 64 MiB, 1 MiB and 48.5 KiB respectively: smaller scratch did not produce a clear latency advantage in these SIMD/OpenMP results.

At L2048, increasing both d and dv from 32 to 64 raises SIMD latency by 10.1%, 6.2% and 8.7% for full matrix, rowwise and blocked streaming. The counted matrix work doubles, but total time does not. The x87 exponential path behind `expf`, reductions, normalization, dependencies and data access are candidate explanations; these two shapes do not identify their individual time shares or establish a unique bottleneck.

## Findings against the proposal hypotheses

| Hypothesis | Finding in this experiment |
|---|---|
| H1: streaming reduces workspace traffic | Workspace capacity is smaller. Actual DRAM traffic was not measured, so a traffic reduction remains model-dependent. The optimized blocked route did not show a clear latency advantage at L4096. |
| H2: blocking depends on tile size and cache behavior | Bk128/256/512 give 296.628/302.155/298.378 ms after the planned three-batch aggregation. Differences are below 2% and smaller than between-batch variation. This limited scalar tuning does not establish a general optimum or a cache cause. |
| H3: SIMD helps, with gains limited by other work | SIMD improves L4096 latency by 1.559–2.260x. Softmax and memory effects may limit total gains, but their contributions were not isolated. |
| H4: larger inputs scale better with query-block parallelism | T4 speedups are 2.302–2.747x at L128 and 3.539–3.742x at L4096. This supports the scaling trend. Scheduling and synchronization costs were not separately measured; T8 slowdowns do not identify a single cause. |

Rowwise already avoids the full score matrix. Blocking additionally permits K/V reuse across a query tile and needs less scratch, but reuse across the DRAM boundary depends on cache behavior. The current evidence does not turn those opportunities into a guaranteed speedup. No packed-kernel or fast-exponential extension is included in these results.

## PyTorch and roofline

A separate allocating API comparison measures the Python-facing custom routes against PyTorch CPU SDPA. At L4096, d=dv64, T4, PyTorch took 12.776 ms while the custom routes took 213.394–232.950 ms. This comparison includes allocation and dispatch and has only one batch. It must not be mixed with native kernel-only timings.

Effective throughput uses F_mat=2L²(d+dv), divided by complete native time. The numerator counts matrix multiply/add work, not every softmax or online-state operation. Reference workloads measure AVX2 multiply/add without FMA and sequential reads. They are workload references, not verified hardware peak ceilings.

The model plot uses native batch 1 and explicitly assumed DRAM traffic. For blocked streaming, the read-once scenario gives 5 MiB, while refilling K/V for each query block gives 515 MiB. The corresponding modeled arithmetic intensities are 819.2 and 7.953 FLOP/byte. These scenarios are neither measured traffic nor guaranteed bounds. Cache capacity alone cannot decide which scenario matches execution.

Thus the experiment directly demonstrates correctness checks, latency and scratch capacity for these implementations. It does not establish actual DRAM traffic or a unique bottleneck. In particular, the share of time spent in exp/softmax has not been profiled.

## Limitations and source data

Of the 180 native configuration instances, 32 have IQR/median above 10%, including eight above 20%; the maximum is about 58%. Small differences should not be treated as stable rankings. Temperature, frequency and continuous background load were not recorded. Results apply to this machine and build; timings from teammates' different machines are kept separate.

Measurements use source revision f7b4b09a6d4243269401fccb14e61f5664f1d2e5. The [archived data](../examples/controlled_20261023/README.md), [experiment protocol](CONTROLLED_STUDY.md), [API timing scope](PYTORCH_COMPARISON.md) and [roofline formulas](ROOFLINE_REFERENCE.md) provide the settings and reproduction steps.
