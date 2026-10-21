# Analyze a saved benchmark grid

`tools/analyze_results.py` turns one completed project grid into a comparison table and three plots. It reads saved files and does not rerun attention. Requires Python 3.10 or newer.

```text
python -m pip install -r tools/requirements-plots.txt
python tools/analyze_results.py --input examples/integrated_20261021 --out results/analysis_first_run
```

Choose a new output directory each time. Use `--tables-only` to check and export the data without Matplotlib. Run the analysis tests with `python tests/test_result_analysis.py`.

## Outputs

| File | Contents |
|---|---|
| `comparison.csv` | All configurations, checked statistics, workspace and IQR/median |
| `serial_latency.png` | Scalar and SIMD time versus input length, split by d/dv |
| `thread_latency.png` | OpenMP time versus requested threads, split by input shape |
| `workspace.png` | Single-thread reserved scratch versus input length |
| `RESULTS.md` | Readable table, largest relative spreads and unavailable runs/routes |
| `analysis.json` | Counts, recorded environment, input hashes and analysis/plot code hashes |

Points are medians. Error bars run from Q1 to Q3 of the individual timings; they are not confidence intervals. The two serial axes use log scales. OpenMP points at different thread counts come from separate runs, not paired measurements.

## Checks before export

The analyzer requires the full current grid plan, including explicit unavailable entries. It checks successful run status, matching arguments, input dimensions, variants, thread records, sample counts, execution order, finite timings and the numerical pass flags. It then recomputes median, quartiles, minimum and GFLOP/s from raw timings and compares them with each summary. Source, binary and recorded environment must agree across the grid.

These checks validate the saved records. They do not rerun the FP64 comparison, prove that unrecorded machine conditions stayed fixed, or check every timed call's output. The benchmark checks correctness before timing and checks the last output after timing.

The supported plan is currently the six serial shapes plus the two OpenMP shapes from `run_project_grid.py`, with full-matrix and rowwise routes. Extend the analyzer's plan/variant/workspace rules when blocked streaming is integrated; it should not silently omit a new route.

Files are found next to the manifest using each job's id. Historical `prefix` paths are deliberately ignored so a copied example can be analyzed independently. Failed, incomplete or mixed-environment data stop the export. Unavailable jobs remain listed with their reasons. Existing output directories are never overwritten.

## Interpreting the first run

See [the plotted first-run results](../examples/integrated_20261021/analysis/README.md). CPU affinity and power mode were uncontrolled. Repeat the noisier configurations under documented conditions before final conclusions. A smaller workspace does not establish lower DRAM traffic, and the SIMD comparison includes a different PV loop order as well as vector instructions. Reported GFLOP/s uses the two matrix products' operation count over complete kernel time, including softmax and resets.
