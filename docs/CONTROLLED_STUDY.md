# Controlled experiment

This run separates tile selection from the main comparison. All three routes use Bq=16. Only blocked streaming has Bk; choose from 128, 256 and 512 at L=2048, d=dv=64 using scalar T1. Run three tuning batches, select the lowest median of their medians, and break exact ties with the smaller Bk. Freeze the choice before the main grid. This is a common scalar-selected Bk, not an optimum for every mode.

After a successful build and tests, commit the code and use a new output directory:

```text
python tools/run_controlled_study.py --out results/my_study --physical-cores 16 --cpu-order 0,1,2,3,4,5,6,7
```

The CPU IDs above are an example for the recorded Windows laptop. Inspect your own topology before using them. Set OMP_WAIT_POLICY=PASSIVE and use the same compiler environment as the build. Do not edit or rebuild during collection. The runner stops if the frozen code, build or revision changes.

Three complete native batches use 10 warm-ups and 20 recorded repetitions. The job orders are forward, reverse and rotated. Each batch has 60 configurations and 1,200 timings when the core budget is at least eight. Keep every batch. Per-call variant order still cycles; this does not remove every cache/predecessor effect.

Serial uses the first listed CPU; T-thread jobs can run on the first T listed CPUs. This controls process eligibility, not individual worker placement or exclusive access. On the recorded laptop CPU0-3 have Windows efficiency class 1 and CPU4-15 class 0. T=8 therefore mixes classes. Do not describe it as scaling across eight identical cores. Power scheme and AC state are read back; clock speed and temperature are not fixed or measured.

`protocol.json` is written before tuning. `tuning/` keeps all candidates, `frozen_settings.json` keeps the selected Bk, and `native_1` to `native_3` keep the complete grids. Validate them with `tools/summarize_controlled_study.py`; its ratios are paired within each batch before aggregation. The range across three medians is descriptive, not a confidence interval.

PyTorch uses a separate allocating API measurement. Roofline traffic remains a stated model; workspace capacity is not measured DRAM traffic. See PYTORCH_COMPARISON.md and ROOFLINE_REFERENCE.md.
