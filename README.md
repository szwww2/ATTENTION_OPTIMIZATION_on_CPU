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
python tools/run_scalar_bench.py --out results/run1  # both scalar routes, L=128 and 512
```

On Linux or macOS the binaries drop the .exe suffix.

The small harness validates both routes before timing, alternates their order, and saves
raw samples, median/IQR, numerical error, workspace and environment information. It refuses
to overwrite earlier outputs. See [the run guide and code walkthrough](docs/SCALAR_BENCHMARK.md).
A small local example with raw samples and its limitations is in [examples/scalar_20261020](examples/scalar_20261020/README.md).
`python tests/test_benchmark.py build/attn_bench.exe` independently checks the CSV statistics
and invalid-argument handling (use `build/attn_bench` on Linux/macOS).
`python tests/test_contract_gate.py` builds a separate fault-injection binary to verify
that finite errors, NaN, and a failed final check cannot produce performance CSVs.

The full-matrix scalar route materializes the complete score matrix, normalizes all rows
in place to form P, then computes P*V. Rowwise uses the same row arithmetic but completes
each query row immediately. This phase-order correction changes the baseline; measurements
from older binaries should be kept separate.

The scalar build keeps -O3 -fno-tree-vectorize on purpose: if the compiler already vectorizes the
baseline there is nothing left for the SIMD level to be compared against.

## AI-use declaration

We used generative AI for language clarity, discussion of the plan, and drafting parts of
the baseline code.

For my benchmark contribution, I used OpenAI Codex to help implement the scalar phase-order
correction, benchmark harness, environment runner, tests and documentation, and to run
the recorded local checks. I am responsible for the content and quality of the submitted work.

## Current scalar increment

See [current scope and review guide](docs/CURRENT_INCREMENT.md) for what this increment completes, how to inspect its outputs, and what remains.
