# Supplementary reference outputs

These CSVs contain the numerical tables supplied with the completed supplementary analyses. Values retain the displayed decimal precision; they are not full-precision prediction exports. Repeated summary rows were deduplicated by scenario, removed patient and method. Machine-specific checkpoint paths were omitted; checkpoint and transfer-source SHA-256 values are retained.

The patient-score matrix, all 80 per-seed threshold rows, full signal-quality lead tables and full patient-removal per-seed rows were not available as complete displayed tables. The corresponding commands regenerate them from public data and trained checkpoints. No values have been filled in for unavailable outputs.

- `incart_patient_robustness_summary.csv`: full-test, seven leave-one-patient-out subsets, and joint removal of INC_1/INC_25.
- `mia_patient_uncertainty.csv`: mean seed AUC, seed SD, patient bootstrap intervals and exact patient-label permutation results.
- `all_pairwise_roc_auc.csv`: all six contrasts in each domain.
- `threshold_audit_summary.csv`: validation threshold range, fixed/tuned MCC and Brier score.
- `fedprox_gap*.csv`: validation/test comparison for μ=0 and μ=3 on three matched tuning seeds.
- `baseline_checkpoint_provenance.csv`, `baseline_seed_metrics.csv`, `local_only_incart_summary.csv`: baseline identities and held-out evaluations.
- `tau_050_patient_distribution.csv`: positive-label concentration across INCART test patients.
- `matched_timing*.csv`: three warm-up-controlled, five-round repetitions per execution mode.
- `i03_excerpt.json`: fixed 10-second labeling example.

These evaluations do not change the primary training protocol. The matched-work timing control is a single-device experiment; it does not provide the missing multi-region Flower deployment artifacts.
