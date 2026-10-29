# Validation performed

This report separates software checks from a scientific rerun.

- 15 CPU tests passed: fixed split integrity; half-open beat boundaries and inclusive tau labels; model dimensions/38,305 parameters; four optimizer aggregation rules and BN buffer behavior; equal-domain weights; validation threshold optimum; 40-job canonical plan; D1 conflict guard; round-boundary resume equals uninterrupted execution for each of four methods.
- Preprocessing output on a synthetic native-rate 12-lead signal is byte-identical to the recovered original engine.
- Model initialization and equal-client server aggregation are exactly equal to original source for all 40 canonical configurations.
- Actual synthetic local training produces identical model states, exposure records and mean loss to the original engine for all four methods.
- Source metadata reconstruction yields all 21,463 retained record rows and the exact saved data ID `9d68265b9e8940c9c1b1`. PTB cohort and the fixed INCART assignments match the original engine.
- Reference verification confirms 40 completed canonical runs, 800 round histories, 80 domain rows, seed coverage, exact best-round/earliest-tie selection, macro-domain scores and equal aggregation. ROC-AUC mean/SD round to manuscript primary table values.
- Python modules compile; all JSON experiment plans expand and validate. Source files are SHA-256 inventoried and archived unchanged.

## Not performed

Full waveform preprocessing, 40-run GPU retraining, D1 evaluation, privacy inference and cloud deployment were not executed. Original checkpoint payloads were not downloaded or verified; saved completion metadata alone is not a new scientific replication. Synthetic tensors and software-test results are never used as experimental reference results.

The observed historical environment and the packaging-test environment are separate JSON files. Refactoring creates a new code hash and new run IDs; the historical fingerprint is used only as provenance evidence.

Re-run `python -m pytest -q` and `python scripts/verify_reference.py` locally. After full canonical execution, run `python scripts/compare_results.py work/study/canonical_metrics.csv --output work/comparison.csv` to report metric deltas without an invented tolerance.
