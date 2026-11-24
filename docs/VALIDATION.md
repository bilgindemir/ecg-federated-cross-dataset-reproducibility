# Validation performed

This report separates software checks from a scientific rerun.

- Core CPU tests cover fixed split integrity; half-open beat boundaries and inclusive tau labels; model dimensions/38,305 parameters; four optimizer aggregation rules and BN buffer behavior; equal-domain weights; validation threshold optimum; 40-job canonical plan; D1 recording-specific sampling, stored window predictions and cache reuse; round-boundary resume equals uninterrupted execution for each of four methods.
- Preprocessing output on a synthetic native-rate 12-lead signal is byte-identical to the reference implementation.
- Model initialization and equal-client server aggregation are exactly equal to reference implementation for all 40 canonical configurations.
- Actual synthetic local training produces identical model states, exposure records and mean loss to the reference implementation for all four methods.
- Source metadata reconstruction yields all 21,463 retained record rows and the exact saved data ID `9d68265b9e8940c9c1b1`. PTB cohort and the fixed INCART assignments match the reference implementation.
- Reference verification confirms 40 completed canonical runs, 800 round histories, 80 domain rows, seed coverage, exact best-round/earliest-tie selection, macro-domain scores and equal aggregation. ROC-AUC mean/SD round to manuscript primary table values.

## Dense evaluation checks

`python scripts/verify_dense.py` checks all 40 original checkpoint hashes, all 80 validation thresholds, the exact stored INCART starts for FedAvg seed 0, its eight published metrics, and the class totals compatible with every saved dense row, and the complete eight-row ten-seed summary. With `--annotations-dir work/data/raw/incart` it also rebuilds the per-window beat counts from the public annotations. See the [coverage and limits of the historical evidence](TABLES.md#d1-sampling-provenance). A full independent inference pass for all 40 checkpoints and both domains is a separate computation.

Re-run `python -m pytest -q` and `python scripts/verify_reference.py` locally. After full canonical execution, run `python scripts/compare_results.py work/study/canonical_metrics.csv --output work/comparison.csv` to report metric deltas without an invented tolerance.

## Supplementary implementation checks

Additional tests cover observed-extremum plateaus, patient-removal masks, exact patient-vector permutation against brute-force ROC-AUC enumeration (including tied scores), bootstrap determinism, complete pairwise seed coverage, and the calibration-audit threshold rule. Core training and primary reference outputs remain unchanged. These software checks do not represent a new full-dataset execution of the supplementary analyses.
