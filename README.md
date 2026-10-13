# Attention Optimisation on CPU

Memory-Performance Trade-offs of Blocked and Streaming Attention on a CPU

Welcome to CEG5206 team project, Team GP8! Group members are: Li Ruiqin, Cui Ziwei, Tang Zhenxiong, Chen Caoyu.
Deadline 13 Nov 2026, 23:59.

## Objectives

Study how optimisation choices affect the performance of scaled dot-product attention on a CPU, under one
numerical contract, one FP64 oracle and one harness:

- implement and validate three routes (full-matrix, row-wise, blocked streaming) in scalar, basic SIMD and
  SIMD + static OpenMP form;
- identify when blocking and online softmax cut memory traffic or improve locality;
- quantify what basic SIMD and OpenMP query-block parallelism actually buy;
- explain the bottlenecks from throughput and workspace measurements, with a roofline-style model where feasible.

A PyTorch CPU SDPA comparison is a target. A packed microkernel or a vectorised exponential is an optional
extension; at most one is taken, decided at the end of Week 4.

GPU kernels, multi-head fusion, training, causal
masking and KV-cache decoding are outside the project.

## Weeks

| Week | Planned outcome |
|---|---|
| 1 | Contract, CPU and compiler settings, FP64 oracle, full-matrix baseline, harness skeleton. PyTorch checked. |
| 2 | Row-wise and blocked-streaming kernels validated. Repeatable timing. Small CPU SDPA comparison if it runs. |
| 3 | Shared basic SIMD loops on all routes. Vectorisation verified. Bounded tile parameters picked. Result tables drafted. |
| 4 | Static OpenMP. Core comparisons and target evidence. Figures drafted. Decision on at most one extension. |
| 5 | Results and interpretation finished. Extension finished. Report cross-reviewed. Code and settings frozen. |
| 6 | Frozen results reproduced, defects fixed, report finished. Last half-week kept as buffer. |

## Team roles

- Li Ruiqin: scalar baselines, FP64 oracle, validation tests
- Cui Ziwei: blocked streaming
- Tang Zhenxiong: SIMD and OpenMP
- Chen Caoyu: harness, PyTorch comparison, roofline

Each owner supplies their own tests, measurements and figures; the report is written together.

## Use of generative AI

Generative AI was used for language clarity, discussion of the plan, and drafting parts of the baseline code
and tests.

The team owns the design, implementation, experiments and interpretation, and each member can
explain their own part.
