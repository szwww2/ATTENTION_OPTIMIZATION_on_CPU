# Attention Optimization on CPU

Memory-Performance Trade-offs of Blocked and Streaming Attention on a CPU

Welcome to CEG5206 team project, Team GP8! Group members are: Li Ruiqin, Cui Ziwei, Tang Zhenxiong, Chen Caoyu.
Deadline 13 Nov 2026, 23:59.

## Final submission

| Item | Where |
|---|---|
| Report | [`report/`](report/) holds the LaTeX source; the PDF `report/GP8_Final_Report.pdf` is added with the final tag |
| Video (10-15 min) | link added with the final tag |
| Assessed version | release tag `v1.0-final` |

## What this project is

We compute scaled dot-product attention, O = softmax(QK^T / sqrt(d)) V, on a CPU in single precision
(one head, batch one, no mask) along three routes:

- **full matrix**: store the whole L x L score matrix, then softmax, then P V;
- **row-wise**: finish one query row at a time;
- **blocked streaming**: query/key tiles with an online softmax, so the score matrix is never stored.

Each route exists as scalar, AVX2 and AVX2 + static OpenMP code. The nine variants share one interface
(`include/attn.h`), one independent FP64 reference and one benchmark harness that checks every output
before and after timing.

## Headline results

Measured on one laptop (Intel Core Ultra 7 356H, Windows 11, GCC 16.2.0) on 23 October 2026:
three batches x 60 configurations x 20 timed calls = 3,600 timed calls.

- All 180 configuration instances meet the error contract |O - O64| <= 1e-4 + 1e-3 |O64|; the largest
  absolute error is 1.76e-7.
- AVX2 makes single-thread calls 1.25-2.26x faster.
- At L = 4096 blocked streaming needs 12,416 bytes of workspace instead of 64 MiB, yet with AVX2 the
  three routes run within 4.3% of each other at every shape: 85-89% of the AVX2 time does not grow with
  the head dimension, which points to the scalar per-score work (the exponential and the softmax
  passes), not to memory traffic.
- Four threads give 3.54-3.74x at L = 4096; eight threads are slower than four on this hybrid CPU.
- PyTorch's CPU `scaled_dot_product_attention` is 17-20x faster at L = 4096.

The archived study with raw samples, tables and plots is in
[`examples/controlled_20261023/`](examples/controlled_20261023/README.md) (that page is in Chinese).
The report states the limits of these measurements.

## Build, test, reproduce

Needs GCC with OpenMP on an AVX2-capable CPU and Python 3. PyTorch (CPU build) is needed only for the
API comparison.

```text
python -m pip install -r tools/requirements-plots.txt
python tools/build_integrated.py                        # nine variants, benchmark, seven test programs
build/attn_tests        build/attn_blocks       build/attn_simd_kernels_tests
build/attn_simd_tests   build/attn_omp_tests    build/attn_streaming_tests
build/attn_streaming_optimized_tests                    # add .exe on Windows
python tests/test_nine_mode_benchmark.py
python tools/run_scalar_bench.py --suite optimized --streaming --key-block 128 --mode all --threads 2 --L 128 --query-block 16 --warmup 1 --reps 3 --out results/nine_check
python examples/controlled_20261023/verify_package.py   # integrity of the archived study
python report/make_numbers.py                           # every number, table and figure of the report
```

The small benchmark run is a functional check, not a performance result. The full procedure, including
a new three-batch study, is in [docs/REPRODUCE_RESULTS.md](docs/REPRODUCE_RESULTS.md).

## Where to find things

| Content | Location |
|---|---|
| Results overview, comparison plots, raw data | [three-batch study](examples/controlled_20261023/README.md) |
| Parallel efficiency | [thread-efficiency tables](examples/thread_efficiency_20261023/README.md) |
| Roofline assumptions and plot | [roofline analysis](examples/controlled_20261023/roofline_analysis/README.md) |
| SIMD evidence for the measured binary | [GCC code generation](docs/GCC_CODEGEN_EVIDENCE.md) |
| Running the nine variants, machine record | [integrated benchmark](docs/INTEGRATED_BENCHMARK.md), [machine record](examples/machine_20261023/README.md), [environment](docs/ENVIRONMENT.md) |
| Checking code and results from a fresh copy | [reproduction guide](docs/REPRODUCE_RESULTS.md) |
| Notes used while assembling the report | [report material index](docs/REPORT_ASSEMBLY.md), [earlier experiment-section notes](docs/REPORT_CHEN_SECTION_DRAFT.md) |

The roofline pages under `examples/` use the first batch only; the report uses the median of the three
batches, so its throughput values at four threads are slightly lower.

<details>
<summary>Earlier experiments</summary>

These are stage records. The results that count are the 23 October three-batch study above.

[first-stage example](examples/scalar_20261020/README.md) ·
[first integrated run](examples/integrated_20261021/README.md) ·
[22 October two-route re-measurement](examples/validation_20261022/README.md) ·
[result fields](docs/SCALAR_BENCHMARK.md) ·
[analysis method](docs/RESULT_ANALYSIS.md) ·
[full experiment procedure](docs/CONTROLLED_STUDY.md)

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

## How we work

`main` is shared, so please be careful: work on a dev branch of your own -- `dev/liruiqin` as an
example -- and push there.

A pull request then gives the others a chance to look before the work lands on `main`. A quick fix
can still go straight to `main`.

Writing a weekly note is also useful for the project report.

## Build and run

The current comparison includes all nine implementations. On an AVX2-capable
CPU with Python 3 and GCC available:

```text
python tools/build_integrated.py
python tests/test_nine_mode_benchmark.py
python tools/run_scalar_bench.py --suite optimized --streaming --key-block 128 --mode all --threads 2 --L 128 --query-block 16 --out results/nine_check
```

The last command is a small end-to-end check, not a replacement for the recorded
study. It checks outputs, times all nine variants and saves CSVs plus environment
records. Use a new output prefix each time. The script name `run_scalar_bench.py`
is retained from the first stage; `--suite optimized --streaming` selects the
integrated executable. See [all test commands](docs/INTEGRATED_BENCHMARK.md)
and the [three-batch experiment procedure](docs/CONTROLLED_STUDY.md).

### Earlier scalar-only build

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

## Use of generative AI

Generative AI was used for language clarity, discussion of the plan, and drafting parts of the baseline code
and tests.

The team owns the design, implementation, experiments and interpretation, and each member can
explain their own part.

Chen Caoyu's integration code, benchmark and experiment runners, analysis tools and documentation were
produced with the AI coding assistants Codex (OpenAI) and Claude (Anthropic) and accepted after review
and the repository's tests. Claude also wrote the first version of the report and
`report/make_numbers.py`, ran the Linux build and tests logged in `report/test_logs/`, and made the video.

Each member's own statement is in Section 10 of the report.
