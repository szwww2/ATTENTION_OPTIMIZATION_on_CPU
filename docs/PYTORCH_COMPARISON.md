# PyTorch CPU comparison

Compare the current full-matrix and rowwise implementations with PyTorch CPU
`scaled_dot_product_attention`. All three receive the same contiguous FP32 Q/K/V
buffers and must pass the existing independent FP64 reference check.

```sh
python -m pip install numpy torch --index-url https://download.pytorch.org/whl/cpu
python tools/build_python_bridge.py
python tests/test_pytorch_api.py
python tools/compare_pytorch.py --L 512 --d 64 --dv 64 --threads 1 --out results/torch_small
python tools/run_pytorch_grid.py --physical-cores 16 --out results/torch_grid
```

Use the actual physical core count. The bridge build uses GCC; set `CXX` if it is
not on PATH. Keep the same `CXX` setting when running on Windows so GCC runtime
DLLs, if needed, can be found beside the compiler. Only the explicit SIMD source is compiled with AVX2. The measurement
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

PyTorch uses inference mode, batch/head counts of one, no mask, no dropout and
no causal restriction. CPU profiler operator names record the selected SDPA
backend outside timing. A profiler backend record is not a timing sample.

Each thread setting starts a new process. PyTorch intra-op threads match the
requested count; inter-op threads are one. A C++ OpenMP team probe checks that
the requested team can start. The two libraries still use different runtimes.
`OMP_WAIT_POLICY=PASSIVE` and `KMP_BLOCKTIME=0` reduce idle pool spinning. Affinity
is not pinned and runtime settings do not prove identical core placement.

Read `environment.json` first: only `status=passed`, `exit_code=0` is a successful
case. Then read `summary.csv` and `raw.csv`. Existing output directories are
refused. Failed checks cannot publish a successful summary. Blocked streaming
will be added when its implementation is available.

API reference: [PyTorch SDPA](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html).
