# Manuscript tables and reproduction inputs

Table numbers follow the manuscript draft in which the fixed INCART split is Table 1 and multi-region FedAvg timing is Table 16. Run the commands in the [README](../README.md) from the repository root. The files below are the inputs for checking the reported values. The canonical run records additionally identify the exact Python source by its recorded fingerprint.

| Table | Analysis | Code or configuration | Reference or verification input |
|---|---|---|---|
| 1 | Fixed INCART assignment | `prepare`; `ecg_revision/data.py` | [Patient split](../manifests/incart_patient_split.csv), [split audit](../reference/manifest/split_audit.json) |
| 2 | D2 dataset split distribution | `prepare`; counts by domain and split | [Window counts](../reference/root/data_counts.csv) |
| 3 | D2 client distribution and class ratios | `prepare`; group the training index by client and label | [Record assignments](../reference/manifest/records.csv) with two windows per PTB-XL training record; [INCART training window counts](../reference/root/data_counts.csv) for C3 |
| 4 | D1 evaluation-set sizes | `dense`; `ecg_revision/dense.py` | [Domain and seed rows](../reference/study/d1_canonical_10seed_restart_safe_rows.csv). See the sampling detail below. |
| 5 | Architecture and training configuration | [Model](../ecg_revision/model.py), [canonical configuration](../configs/canonical.json) | [Canonical run records](../reference/runs) |
| 6 | Training hyperparameters | [Canonical configuration](../configs/canonical.json), [tuning configuration](../configs/tuning.json), [epsilon grid](../configs/epsilon.json) | [Frozen selections](../reference/study/selected_hyperparameters_after_epsilon.json) |
| 7–8 | Canonical PTB-XL and INCART performance | `train` and `evaluate` with [canonical configuration](../configs/canonical.json); pooled rows use [pooled configuration](../configs/pooled.json) | [Canonical seed rows](../reference/canonical_d2_rows.csv), [baseline summary](../reference/study/baseline_canonical_summary.csv), [frozen canonical source](../reference/source/canonical/ecg_revision) |
| 9 | All pairwise ROC-AUC contrasts | `python -m ecg_revision.supplementary all-pairs --root work` | [Paired contrasts](../reference/supplementary/all_pairwise_roc_auc.csv) |
| 10 | Validation-selected decision thresholds | `thresholds` with [canonical configuration](../configs/canonical.json) | [Threshold summary](../reference/study/decision_threshold_tuning_summary.csv), [seed rows and thresholds](../reference/study/decision_threshold_tuning_runs.csv) |
| 11 | Optimizer hyperparameter sensitivity | `train` and `summarize` with [tuning configuration](../configs/tuning.json) and [epsilon grid](../configs/epsilon.json) | Validation [tuning runs](../reference/study/tuning_all_runs.csv) and [epsilon runs](../reference/study/tuning_fedadam_epsilon_runs.csv) |
| 12 | Local-epoch sensitivity | `train` and `evaluate` with [local epochs configuration](../configs/local_epochs.json) | [D2 evaluation rows](../reference/study/all_evaluation_rows.csv). Select report-phase, equal-client, test D2 rows at `eval_tau=0.05`, seeds 0–2 and E ∈ {1, 3, 5}. |
| 13 | Aggregation weighting | `train` and `evaluate` with [weighting configuration](../configs/weighting.json) | [Weighting summary](../reference/study/weighting_sensitivity_summary.csv), [run settings](../reference/study/weighting_sensitivity_runs.csv) |
| 14 | Non-federated baselines | `train` and `evaluate` with [pooled](../configs/pooled.json) and [local/transfer](../configs/local_transfer.json) configurations | [Baseline summary](../reference/study/baseline_canonical_summary.csv), [evaluation rows](../reference/study/baseline_canonical_evaluation_rows.csv) |
| 15 | Historical same-Colab timing | [Local timing summarizer](../scripts/summarize_timing.py) calculates timings from newly generated histories. | The exact historical run identifiers, timing script, and elapsed-time logs for the reported values are unavailable. The [matched-work control](../reference/supplementary/matched_timing_summary.csv) represents a separate measurement. |
| 16 | Historical five-node Flower timing | Synchronous FedAvg on four clients and one server. | [Original client-export manifest](../reference/timing/table16_client_manifest.csv) identifies the four input shards by SHA-256 and accounts for 66,426 windows. The Flower source, launch configuration, dependency versions and timing/traffic logs for all three repetitions remain unavailable. |
| 17 | INCART membership-inference audit | `privacy` with [canonical configuration](../configs/canonical.json) | [Seed rows](../reference/study/mia_incart_canonical_control_rows.csv), [privacy summary](../reference/study/mia_incart_canonical_control_summary.csv) |

## Frozen canonical source

The 11 Python files in [the canonical source snapshot](../reference/source/canonical/ecg_revision) match the source fingerprint recorded in all 40 canonical run configurations and the [canonical protocol](../reference/study/primary_canonical_protocol.json). Run `python scripts/verify_reference.py` to verify those source bytes and the reference runs. The README commands use the maintained, modular implementation. The frozen source retains the original package bytes, including the SSH timing module. The SSH module and the historical Flower deployment use different transports.

To use the frozen source, run the following from the repository root after installing the dependencies. The launcher verifies the archived source and each run fingerprint. Training and evaluation also verify the generated record manifest and acquire data from the versioned PhysioNet URLs. Omit `--seed 0` to select all ten seeds for training or held-out evaluation.

```bash
python scripts/run_frozen_canonical.py plan
python scripts/run_frozen_canonical.py train --seed 0
python scripts/run_frozen_canonical.py evaluate --seed 0
```

## D1 sampling provenance

The evaluation program that writes the saved D1 row file invokes the original `dense_starts` routine. For capped INCART evaluation, the routine retains all stride-512 starts, fills the remaining quota without replacement from other starts using a recording-ID-derived seed, then sorts the starts. The manuscript draft describes a fixed seed 42 reset per recording and sampling from all starts. All 80 saved D1 rows agree exactly with the saved validation thresholds, but the saved rows contain neither start indices nor per-window predictions. Confirm the sampler history of any resumed rows before claiming exact reconstruction of those outputs.
