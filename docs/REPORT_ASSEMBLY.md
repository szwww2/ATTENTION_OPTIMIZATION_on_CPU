# Putting the group report together

The shared experiment and Chen's results section are ready for review. The final
group report and video are still to be assembled. This is a guide to the current
material, not a record of a completed submission.

## Sections and source material

| Section | Material to use | Review still needed |
|---|---|---|
| Problem and course connection | Final proposal: the same attention calculation with different storage, SIMD and thread choices | Keep the CPU performance question and scope clear. |
| Full-matrix and rowwise | `src/full_matrix_scalar.cpp`, `src/rowwise_scalar.cpp`, FP64 reference and [interface notes](INTEGRATION.md) | Li: check the baseline and numerical contract. |
| Blocked streaming | [Algorithm and interface](STREAMING_CUIZIWEI.md), [SIMD/OpenMP implementation](STREAMING_OPTIMIZED.md) | Cui: check online softmax, scratch and tile interpretation. |
| SIMD and OpenMP | [Current integration changes](INTEGRATED_BENCHMARK.md#integration-changes), [measured GCC executable](GCC_CODEGEN_EVIDENCE.md) | Tang: check shared primitives, loop order and thread explanation. |
| Methods, results and limitations | [Chen's report section](REPORT_CHEN_SECTION_DRAFT.md), [collected results](../examples/controlled_20261023/README.md) | Chen and route owners: agree on the comparisons and conclusions. |
| Conclusion, references and contributions | Agreed findings and each member's actual work | Write together; each person checks their own contribution. |

These are review requests, not confirmed task assignments. Tang's
[earlier report draft](REPORT_TANG_SECTION_DRAFT.md) is retained as history.
Its normalization description, integration status and MSVC timings must not be
copied as facts about the current GCC experiment.

## Tables and figures already available

- Chen's section contains both proposal tables: L128 and L4096, with scalar T1,
  SIMD T1 and SIMD + OpenMP T8 for all three routes. It also keeps the T4 results,
  since T4 was faster on this machine.
- The [results directory](../examples/controlled_20261023/README.md) has latency,
  thread scaling and paired-speedup plots. Pick the figures that support the
  report; there is no need to include every intermediate plot.
- [Thread efficiency](../examples/thread_efficiency_20261023/README.md) uses
  within-batch speedup divided by the thread count, not measured CPU utilization.
- [PyTorch](PYTORCH_COMPARISON.md) uses an allocating API timing scope. Keep it
  separate from native kernel-only results.
- [Roofline](../examples/controlled_20261023/roofline_analysis/README.md) uses
  stated traffic scenarios and workload references. It does not measure DRAM
  traffic or identify a unique bottleneck.

## Decisions before the final version

1. Review PR #5 and agree on the implementation descriptions. Fix actual defects
   before freezing the code. If a change affects a measured path, rerun the
   affected tests and experiments; do not silently relabel the archived data.
2. Agree on the report figures and on whether to take an optional extension.
   The core comparison is available; a packed microkernel or fast exponential
   is not needed to complete it. No extension has been added here.
3. Combine the report, check references and contributions, record the video,
   then choose the final commit or tag and the person submitting.

The course Project Briefing (pp. 5–7) asks for one group PDF report, a 10–15 minute
video link, the public GitHub link and final commit/tag, plus contribution and
AI-use records in the submission. Keep these final records truthful to the work
actually done. The Canvas Team Project page checked on 24 October gives
**13 November 2026, 23:59** as the deadline. **20 November is the availability
end, not an extension of the deadline.** Check Canvas again before submitting.

Reviewing these results does not require everyone to rerun the full study.
Start with the report and saved data; use the build instructions when checking
code or reproducing a specific result. Runs from different computers should
remain separate experiments.
