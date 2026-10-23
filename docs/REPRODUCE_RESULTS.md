# Reproducing the results

Start from `dev/chencaoyu-integration` while PR #5 is open. Run commands from the
repository root and use a new output directory each time.

## Rebuild tables and plots from the saved data

Use Python 3 and install the plot dependency:

```text
python -m pip install -r tools/requirements-plots.txt
python examples/controlled_20261023/verify_package.py
```

Then run the three analysis commands in the
[saved experiment README](../examples/controlled_20261023/README.md).
For parallel efficiency, also run:

```text
python tools/analyze_thread_efficiency.py --input examples/controlled_20261023 --out results/reanalysis_20261023/efficiency
```

Compare the generated CSVs with the saved `summary/`, `pytorch_analysis/`,
`roofline_analysis/` and `examples/thread_efficiency_20261023/` tables. This
checks calculations from the recorded samples. It does not collect new timings
and does not require PyTorch or a C++ compiler.

## Build and run a small example

Use GCC with OpenMP support and an AVX2-capable CPU. The recorded Windows build
uses GCC 16.2.0. Put `g++` on PATH, or set `CXX` to the compiler path.

```text
python tools/build_integrated.py
python tests/test_nine_mode_benchmark.py
python tools/run_scalar_bench.py --suite optimized --streaming --key-block 128 --mode all --threads 2 --L 128 --query-block 16 --warmup 1 --reps 3 --out results/nine_check
```

The last command should produce nine summary rows and 27 raw samples, with
`contract_pass=1` for every summary row. Check the environment record's status
and exit code as well. This is a functional example, not a performance study.
Run the seven C++ suites listed in [build and test](INTEGRATED_BENCHMARK.md#build-and-test)
for the numerical and boundary tests. On Windows, keep the output prefix ASCII
and relative, as above; a Chinese-named parent folder is supported.

## Check the final frozen version

Once the group chooses the final code and report results, record that commit
or tag and rerun the relevant build/tests and result reproduction. The proposal
uses one documented CPU; this can be Chen's machine. Four computers are not
required.

The archived timings belong to `f7b4b09a6d4243269401fccb14e61f5664f1d2e5`.
Later workspace-overflow checks do not relabel those measurements as timings of
the latest code. For a fresh performance study, follow the
[controlled protocol](CONTROLLED_STUDY.md), keep all batches in a new directory,
and use the [PyTorch](PYTORCH_COMPARISON.md) and
[roofline](ROOFLINE_REFERENCE.md) procedures for those separate experiments.
If code, timing scope or settings change, check and remeasure the affected
comparisons. Explain variation rather than expecting identical milliseconds.

## Checked on 24 October 2026

A fresh GitHub checkout of `838739f` built on the same Windows/GCC machine.
All seven C++ suites, the nine-mode tests and the small example passed.
Saved native, PyTorch, roofline and efficiency analyses were regenerated;
eight generated CSV tables matched the archived values field for field. This
checks the published files and instructions using installed dependencies; it
is not a second-machine
test or a new full performance run.

Git attributes preserve CSV and Python file bytes. This avoids false modified
files from legacy CRLF content and keeps the archived hashes unchanged.
