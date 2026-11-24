# Manuscript tables and reproduction inputs

Table numbers follow the manuscript draft in which the fixed INCART split is Table 1 and multi-region FedAvg timing is Table 16. Run the commands in the [README](../README.md) from the repository root. The files below are the inputs for checking the reported values. The canonical run records additionally identify the exact Python source by its recorded fingerprint.

| Table | Analysis | Code or configuration | Reference or verification input |
|---|---|---|---|
| 1 | Fixed INCART assignment | `prepare`; `ecg_revision/data.py` | [Patient split](../manifests/incart_patient_split.csv), [split audit](../reference/manifest/split_audit.json) |
| 2 | D2 dataset split distribution | `prepare`; counts by domain and split | [Window counts](../reference/root/data_counts.csv) |
| 3 | D2 client distribution and class ratios | `prepare`; group the training index by client and label | [Record assignments](../reference/manifest/records.csv) with two windows per PTB-XL training record; [INCART training window counts](../reference/root/data_counts.csv) for C3 |
| 4 | D1 evaluation-set sizes | `python scripts/verify_dense.py` | [Per-record INCART sampling manifest](../reference/dense/incart_sampling.csv) and [domain and seed rows](../reference/study/d1_canonical_10seed_restart_safe_rows.csv) |
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
| Supplementary S1 | D1 test performance across ten seeds | `python -m ecg_revision.cli dense --reference-checkpoints --root work` | [Ten-seed summary](../reference/study/d1_canonical_10seed_summary.csv), [seed-level metrics](../reference/study/d1_canonical_10seed_restart_safe_rows.csv) and [checkpoint hashes](../reference/canonical_checkpoints.csv) |

## Frozen canonical source

The 11 Python files in [the canonical source snapshot](../reference/source/canonical/ecg_revision) match the source fingerprint recorded in all 40 canonical run configurations and the [canonical protocol](../reference/study/primary_canonical_protocol.json). Run `python scripts/verify_reference.py` to verify those source bytes and the reference runs. The README commands use the maintained, modular implementation. The frozen source retains the original package bytes, including the SSH timing module. The SSH module and the historical Flower deployment use different transports.

To use the frozen source, run the following from the repository root after installing the dependencies. The launcher verifies the archived source and each run fingerprint. Training and evaluation also verify the generated record manifest and acquire data from the versioned PhysioNet URLs. Omit `--seed 0` to select all ten seeds for training or held-out evaluation.

```bash
python scripts/run_frozen_canonical.py plan
python scripts/run_frozen_canonical.py train --seed 0
python scripts/run_frozen_canonical.py evaluate --seed 0
# Score the published weights directly on D2
python scripts/run_frozen_canonical.py evaluate --reference-checkpoints --seed 0
```

## D1 sampling provenance

D2 uses complete 1,024-sample windows starting at `0, 512, 1024, …`. D1 considers every integer start from zero through `length − 1024`, inclusive, in the signal resampled to 200 Hz. PTB-XL uses every candidate. For INCART the evaluation program uses the following rule from the [fingerprint-matched source](../reference/source/canonical/ecg_revision/evaluation.py).

1. Retain every D2 start. Build the ascending list of remaining stride-one candidates.
2. Derive the random seed from the first eight hexadecimal digits of SHA-256 over the UTF-8 JSON string for the full recording ID, including its quotes (for example, `"INC_I01"`). Pass that integer to `numpy.random.default_rng`.
3. Draw the remaining positions without replacement to reach 16,384 per recording. Sort the union in ascending order. Apply the minimum-three-mapped-beat rule and the inclusive abnormal-fraction threshold of 0.05. Sampling precedes the eligibility filter.
4. Use the same positions across all checkpoints. For threshold-dependent metrics, use the saved D2 validation threshold for the matching method, seed and dataset.

### Evidence for the INCART sample

| Check on the 16 held-out recordings | Result |
|---|---|
| Exact saved starts, FedAvg seed 0 | All 262,144 positions match the sampler, including all 11,232 D2 positions. The [manifest](../reference/dense/incart_sampling.csv) gives the SHA-256 of the ordered starts for each recording as little-endian int64 values. |
| Beat annotations | Public INCART v1.0.0 annotations reproduce the total and abnormal beat counts at every saved start. All 262,144 windows are eligible and 123,948 are positive. |
| Alternative fixed-seed draw | Resetting `numpy.random.default_rng(42)` for each recording and drawing uniformly from all positions gives 124,260 positives and retains 304 D2 starts. The positive count is incompatible with the 40 saved INCART rows. |
| Window-level scores | The [FedAvg seed-0 INCART archive](../reference/dense/fedavg_seed0_incart.npz) contains the original start, total mapped beats, abnormal beats and predicted probability for every scored window, grouped by recording ID. These probabilities reproduce all eight saved metrics at the recorded threshold. |
| Coverage across methods and seeds | All 40 INCART rows agree with the reconstructed class totals and threshold-dependent metrics. All 80 domain thresholds match their D2 validation selections. For the other 39 INCART checkpoint evaluations, historical evidence is at the aggregate level. Exact per-window correspondence for those runs requires comparison to original window-level outputs. |

The public inputs support both an offline check and an [evaluation-only rerun](../README.md#window-stride-sensitivity-d1-and-d2). Each new prediction archive identifies the checkpoint, dataset, evaluator, domain and sampling rule. Use the recording offsets to associate every probability with its start in the resampled waveform. The [public INCART records and annotations](https://physionet.org/content/incartdb/1.0.0/) are the source for the independent label check.
