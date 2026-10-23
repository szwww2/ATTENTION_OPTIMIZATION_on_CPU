# PyTorch CPU comparison

Compare the full-matrix, rowwise and blocked-streaming implementations with PyTorch CPU
`scaled_dot_product_attention`. All four APIs receive the same contiguous FP32 Q/K/V
buffers and must pass the existing independent FP64 reference check.

```sh
python -m pip install numpy torch --index-url https://download.pytorch.org/whl/cpu
python tools/build_python_bridge.py
python tests/test_pytorch_api.py
python tools/compare_pytorch.py --L 512 --d 64 --dv 64 --threads 1 --out results/torch_small
python tools/run_pytorch_grid.py --physical-cores 16 --query-block 16 --key-block 256 --out results/torch_grid
```

Use the actual physical core count. The bridge build uses GCC; set `CXX` if it is
not on PATH. Keep the same `CXX` setting when running on Windows so GCC runtime
DLLs, if needed, can be found beside the compiler. Only the explicit SIMD translation units are compiled with AVX2. The measurement
records exact package versions, compiler flags, source hashes and library hash.

The timer covers each Python API call: output allocation, dispatch, C++ scratch
allocation/initialization/release, and computation. Inputs, tensor views, oracle,
checks and output release are outside. PyTorch allocates its output and any
internal temporary storage within its call. These are API costs; compare them
with one another, not with the allocation-free native benchmark timings.
Any packing inside the library call is included. The bridge does not convert
input dtype or layout; creating the shared FP32 tensor views is outside timing.

At T=1 the custom variants use SIMD; at T>1 they use SIMD plus OpenMP. Each case
uses 10 warmups and 20 measured calls per variant, rotating their order. The
saved raw samples reproduce median, Q1, Q3 and IQR. Each variant is checked before
timing and after its final timed call; inputs are checked for mutation.

The default Bq=16 and Bk=256 are experiment settings, not a claim of optimal tuning.
Freeze the selected values before collecting the comparison. Requested and effective
block sizes are recorded separately; effective sizes are clipped to L. The blocked
API allocates its complete score tile, online maximum/normalizer and output-accumulator
scratch within the call, using the same allocation/zero-initialization/release boundary
as full-matrix and rowwise. The C bridge retains its original six entry modes and adds
three blocked modes; this does not add extra measured variants to a single thread case.

After a controlled native study, pass its frozen settings instead of retyping Bk:

```sh
python tools/run_pytorch_grid.py --physical-cores 16 --cpu-order 0,1,2,3,4,5,6,7 --frozen-settings results/my_study/frozen_settings.json --out results/my_study/pytorch
```

This reads Bq/Bk from the study, checks the protocol hash and source revision,
and rejects a different block override or CPU plan. Without `--frozen-settings`,
the standalone defaults remain Bq=16 and Bk=256.

PyTorch uses inference mode, batch/head counts of one, no mask, no dropout and
no causal restriction. CPU profiler operator names record the selected SDPA
backend outside timing. A profiler backend record is not a timing sample.

Each thread setting starts a new process. PyTorch intra-op threads match the
requested count; inter-op threads are one. A C++ OpenMP team probe checks that
the requested team can start. Blocked streaming additionally checks its actual compute
team in every call and fails if it differs; full/row retain the preflight-only scope.
The two libraries still use different runtimes.
`OMP_WAIT_POLICY=PASSIVE` and `KMP_BLOCKTIME=0` reduce idle pool spinning. An optional
`--cpu-list` is applied before configuring the thread pools and is forwarded to every
case by the grid runner. Without it the runner leaves affinity unchanged. Requested
affinity and runtime conditions are recorded; these do not prove identical placement
of every worker or fixed clock frequency.
The grid also accepts `--cpu-order` instead of `--cpu-list`: each T case uses
the first T processor IDs, matching the native grid's policy. The analyzer verifies
the job's requested and observed process eligibility and rejects changes in recorded
power/affinity conditions. For a heterogeneous CPU selection, pass a descriptive
`--thread-scaling-note`; it is preserved in the analysis and printed below the plot.

Read `environment.json` first: only `status=passed`, `exit_code=0` is a successful
case. Then read `summary.csv` and `raw.csv`. Existing output directories are
refused. Failed checks cannot publish a successful summary. New records use schema 2,
with four variants per case; the analyzer still accepts historical two-custom-route
records without a schema field. These histories remain separate datasets.

API reference: [PyTorch SDPA](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html).
