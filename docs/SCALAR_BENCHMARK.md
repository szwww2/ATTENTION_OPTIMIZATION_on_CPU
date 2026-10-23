# Scalar benchmark

Measures the full-matrix and rowwise scalar FP32 kernels using the same Q/K/V inputs and FP64 reference.

## Build and run

Requires a C++17 compiler and Python 3. Run from the repository root.

Windows:

```text
build.bat
build\attn_tests.exe
build\attn_blocks.exe
python tests/test_benchmark.py build/attn_bench.exe
python tests/test_contract_gate.py
python tools/run_scalar_bench.py --out results/run1
```

Linux/macOS:

```text
bash build.sh
./build/attn_tests
./build/attn_blocks
python tests/test_benchmark.py build/attn_bench
python tests/test_contract_gate.py
python tools/run_scalar_bench.py --out results/run1
```

Local validation was performed on Windows with GCC 16.2.0; Linux/macOS validation is pending.
The compiler defaults to `g++`; set `CXX` to use another compatible compiler. Rebuild after source changes.
The runner uses the Python standard library. The optional NumPy cross-check requires NumPy:

```text
python tests/test_crosscheck.py
```

Defaults: L=128 and 512, d=dv=64, Bq=64, seed=20261020, one thread, 10 warm-ups and 20 timed calls per route and shape.
Use a new output prefix for each run. Existing outputs are not overwritten.

Custom shape:

```text
python tools/run_scalar_bench.py --out results/odd --L 31 --d 31 --dv 33 --query-block 3 --warmup 2 --reps 7 --seed 42
```

`build/attn_bench --help` lists options and bounds. `--L` selects one length instead of the two defaults.
Direct binary execution produces the CSV files; the Python runner also records the environment.

## Measurement method

1. Generate one Q/K/V set and calculate the FP64 reference.
2. Allocate workspace and output buffers for each route.
3. Validate both routes, then warm up both.
4. Time complete kernel calls, alternating which route runs first in each round.
5. Validate each route's final output, then write the results.

Input generation, allocation, reference calculation, validation, sorting and file output are outside the timer.
Kernel resets remain inside it. The post-check covers the final output of each route, not every timed sample.
Validation failure stops the run before performance CSVs are written. Zero or non-finite timer samples are rejected.

The full-matrix route completes all scores, all row softmax operations, then PV. Rowwise completes each query row immediately.
Measurements from the earlier full-matrix implementation, which completed each row immediately, should be kept separate.

## Output files

| File | Contents |
| --- | --- |
| `<prefix>_raw.csv` | Every timed call, its route and execution order |
| `<prefix>_summary.csv` | Timing statistics, throughput, workspace and errors |
| `<prefix>_environment.json` | Run status, exit code, revision, source/binary hashes, build flags and machine details |

A successful run has environment `status: passed` and `exit_code: 0`. Failed or interrupted runs must not be included in performance results.
Use distinct prefixes for concurrent runs: overwrite checks are not a cross-process lock, and CSV writes are not atomic.
The environment records the whole working tree's dirty status. Hashes identify source and binary contents; they do not replace rebuilding.

| Column | Meaning |
| --- | --- |
| `route,L,d,dv,seed` | Implementation and input configuration |
| `query_block` | Effective Bq=min(requested Bq,L) for rowwise and full-matrix OpenMP; 0 for serial full-matrix |
| `sample,order,ms` | Zero-based sample index, position within the round (0..N-1 for N selected variants), and elapsed milliseconds |
| `warmup,reps` | Additional warm-up calls and timed calls per route |
| `median_ms,min_ms` | Median and minimum elapsed time |
| `q1_ms,q3_ms,iqr_ms` | Quartiles and Q3-Q1 |
| `gflops` | Matrix-operation count 2L²(d+dv) divided by median time, in GFLOP/s |
| `workspace_bytes` | Allocated score scratch, excluding input, output and reference buffers |
| `pre_*`, `post_*` | Maximum absolute and relative errors before/after timing |
| `contract_pass` | Both numerical checks passed |

Quartiles use linear interpolation at position `p*(n-1)` in sorted samples.
Group raw samples by route, mode, threads, L, d, dv, seed and query_block to reproduce each summary; each group contains `reps` samples. Scalar-only runs have two variants; integrated runs may have two, four or six.

The numerical check requires finite outputs and `abs(diff) <= 1e-4 + 1e-3*abs(ref)` for every element.
A large maximum relative error near a zero reference does not by itself mean this contract failed.

## Tests and limits

- `tests/test_benchmark.py`: CSV statistics, arguments, overwrite protection and environment records.
- `tests/test_contract_gate.py`: injected finite errors, NaN and a failed final check must not produce performance CSVs.
- `tests/test_crosscheck.py`: NumPy comparison and rejection of corrupt C++ FP32 output.

CPU affinity and power mode are not controlled by the harness. Report timing spread and repeat measurements before drawing speedup conclusions.
The GFLOP/s numerator counts the two matrix products; elapsed time also includes softmax.
Workspace capacity is not measured DRAM traffic. Rowwise allocates Bq×L floats, even though it computes one row at a time.

A recorded run is available in [examples/scalar_20261020](../examples/scalar_20261020/README.md).
