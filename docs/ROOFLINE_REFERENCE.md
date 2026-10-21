# Roofline reference and traffic scenarios

This experiment measures two reference workloads: independent AVX2 FP32
multiply/add chains and a 512 MiB sequential read. Run after other experiments,
with no benchmark running concurrently.

```sh
python tools/run_roofline_calibration.py --out results/roofline_calibration
python tools/analyze_roofline.py --calibration results/roofline_calibration --grid results/my_native_grid --out results/roofline_analysis
```

The arithmetic loop has eight independent vectors, eight lanes per vector,
one multiply and one add per iteration: 128 FLOPs per iteration per thread.
FMA contraction is disabled. Inspect the compiled arithmetic loop for eight
vector multiplies and eight adds. The read workload visits 512 MiB four times
per sample and checks the sum. Both workloads use two warmups, seven samples
and T=1/2/4/8. Record whether 512 MiB exceeds the tested machine's cache capacity.

The reference curve is `min(C, B*I)`, where C is measured arithmetic GFLOP/s
and B is measured read-stream payload GB/s. It is not a verified upper bound
for mixed attention reads/writes. Affinity and clock frequency are not fixed.
The arithmetic reference omits attention's repeated dot-product reductions,
exponentials, normalization and query-block scheduling. Its timing still includes
OpenMP team launch and the final checksum reduction.

Attention throughput uses `F = 2*L*L*(d+dv)` divided by complete native kernel
time. This counts matrix multiply/add work; it excludes softmax operations from
the numerator while retaining their execution time in the denominator.

## Traffic assumptions

There is no DRAM counter measurement in this experiment. `dram_bytes_counter`
is blank, not zero. The two scenarios below illustrate L=4096, d=dv=64 only.
They are neither measured values nor guaranteed lower/upper bounds.

| Scenario | Rowwise modeled bytes | Full-matrix modeled bytes |
|---|---|---|
| Q/K/V read once, cold output | `4*L*(2*d+3*dv)` | rowwise value + `20*L*L` |
| K/V refilled per query | `4*L*d + 4*L*L*(d+dv) + 8*L*dv` | rowwise value + `20*L*L` |

Both assume normal cold output stores (read-for-ownership plus writeback),
row scratch retained in cache, and the full score table evicted between phases.
For a score table of S=4L² bytes, its modeled traffic is 2S for initial stores,
2S for softmax read/writeback, and S for PV reads. This assumes repeated scans
within one softmax row stay in cache. It omits extra traffic beyond these stated
assumptions and is not inferred from workspace capacity.

The current Bq=16 groups rows for workspace/scheduling; it does not implement
K/V reuse across 16 queries. Increasing threads does not divide total FLOPs
or modeled traffic by T. Scalar strided accesses introduce additional cache-line
effects, so the scenario table covers SIMD/OpenMP only.

Warmup does not establish DRAM residency: alternating full and row calls can
evict each other's data. A fully cached row call can transfer less than the
first scenario assumes. The main figure therefore keeps reference curves and
measured attention throughput in separate panels. It does not assign automatic
memory-bound/compute-bound labels or percentage-of-peak scores.

The blocked implementation will need its own traffic model, including its
online softmax state and tile reuse, before a three-route conclusion is possible.

Model reference: [Williams, Waterman and Patterson, Roofline](https://escholarship.org/uc/item/5tz795vq).
