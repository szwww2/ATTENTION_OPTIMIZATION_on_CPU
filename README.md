# Attention Optimization on CPU

Memory-Performance Trade-offs of Blocked and Streaming Attention on a CPU

Welcome to CEG5206 team project, Team GP8! Group members are: Li Ruiqin, Cui Ziwei, Tang Zhenxiong, Chen Caoyu.
Deadline 13 Nov 2026, 23:59.

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
build\attn_tests.exe         # route 1 contract suite, has to end with SUITE PASSED
build\attn_blocks.exe        # route 2, block sizes and row orders, bit for bit
build\attn_bench.exe         # timing, L = 128 ... 2048
```

On Linux or macOS the binaries drop the .exe suffix.

The scalar build keeps -O3 -fno-tree-vectorize on purpose: if the compiler already vectorizes the
baseline there is nothing left for the SIMD level to be compared against.

## Blocked streaming: scalar, SIMD and OpenMP (Cui Ziwei)

Route 3 is implemented with explicit query/key blocks and online softmax.
Run `build/attn_streaming_tests` after `bash build.sh` (Windows: `.exe`).
See [the interface and verification guide](docs/STREAMING_CUIZIWEI.md) and
[the local scalar tile pilot](examples/streaming_cuiziwei_20261022/README.md).
SIMD and static OpenMP are now implemented for route 3. Build with
`python3 -B tools/build_streaming_optimized.py`, then run
`build/streaming_optimized/test`. See [optimized API/tests](docs/STREAMING_OPTIMIZED.md)
and [same-build measurements](examples/streaming_optimized_20261022/README.md).
Integration into the pending shared harness remains a teammate coordination task.
See [the team review checklist](docs/STREAMING_TEAM_HANDOFF.md) for integration
ownership, interface decisions and the nine-mode comparison matrix.

## Use of generative AI

Generative AI was used for language clarity, discussion of the plan, and drafting parts of the baseline code
and tests.

The team owns the design, implementation, experiments and interpretation, and each member can
explain their own part.
