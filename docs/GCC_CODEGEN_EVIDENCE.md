# SIMD evidence for the measured GCC build

The [assembly excerpts](evidence/gcc_measured_binary.txt) come from the executable used for the three-batch study. Its SHA256 matches `native_binary_sha256` in [the frozen settings](../examples/controlled_20261023/frozen_settings.json):

```text
ec79664567419e4bc0925d440b11cdc7c7a489b329133bff5847727527d71d96
```

The build uses GCC 16.2.0, `-O3 -fno-tree-vectorize -ffp-contract=off`, with `-mavx2` on explicit SIMD units and `-fopenmp` on OpenMP units. The recorded source revision is `f7b4b09a6d4243269401fccb14e61f5664f1d2e5`.

- Full-matrix score/value helpers, the rowwise SIMD helper and the blocked AVX2 tile contain 256-bit packed FP32 multiply/add instructions (`vmulps`, `vaddps`). Dot products reduce the lane partial sums before storing a score.
- The three inspected scalar computation functions contain scalar FP32 arithmetic and no packed FP32 add/multiply. Together with the recorded build flags, this supports keeping them as scalar baselines.
- The executable scan found no FMA-family instructions. This is a static code check, not a measurement of instruction execution counts or time shares. External DLL code is outside the scan; the `expf` and `exp` routines described below are inside this executable.
- SIMD helpers accept unaligned pointers and use unaligned loads/stores. There is no packing stage. The matrix loops use FP32 multiply/add operations; the independent FP64 reference runs outside native timing.

## The exponential call inside softmax

In this measured GCC/MinGW build, `expf` and `exp` are statically linked into the executable. `expf` converts its float argument to double (`cvtss2sd`), calls `exp`, then converts the returned double to float (`cvtsd2ss`). The general finite-input path in `exp` uses x87 instructions, including `frndint`, `f2xm1` and `fscale`; zero, non-finite and out-of-range inputs have separate paths. This describes a double-argument/result interface, not a claim that every internal x87 operation is rounded to FP64.

These per-call conversions and exponential evaluations are inside the native timer. They do not convert the full Q/K/V arrays to FP64, and they are separate from the FP64 oracle outside the timer. Static inspection does not show how much runtime they consume or prove that they are the main bottleneck. This library path is specific to the recorded build; other compilers or math libraries may use different implementations.

The [excerpts](evidence/gcc_measured_binary.txt) include the complete `expf` wrapper and the ordinary x87 calculation path in `exp`. To inspect those functions alone, use `--disassemble=expf` or `--disassemble=exp` with the command below.

To inspect the same executable, first compare its SHA256 with the frozen record, then run:

```shell
objdump -d -C --no-show-raw-insn build/attn_bench_all.exe
```

The excerpts are static evidence, not a new timing run. A rebuild may produce different addresses or instructions; inspect and identify that new binary separately. Tang's earlier [MSVC evidence](evidence/tang_avx2_assembly.txt) remains a separate build record.
