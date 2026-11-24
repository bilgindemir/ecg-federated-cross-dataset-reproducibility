# Reproducibility limitations

## Dense test-time sampling

The archived INCART starts and predictions support a direct window-level check for FedAvg seed 0. The other 39 INCART runs have historical aggregate rows with matching class counts and thresholds. Their published checkpoints support independent evaluation on the same deterministic positions. See the [sampling evidence](TABLES.md#d1-sampling-provenance).

## Multi-region systems timing

The [original client-export manifest](../reference/timing/table16_client_manifest.csv) identifies the earlier 66,426-window partition by shard counts and SHA-256 hashes. The multi-region Flower server/client deployment, complete timing and interface-traffic logs, and environment definitions are unavailable. The local timing summarizer reports execution-history measurements and an idealized parallel lower bound. The [manuscript table index](TABLES.md) records the separate evidence for each timing table.

## Hyperparameter search scope

The epsilon sensitivity configuration evaluates two epsilon values across six server learning rates and three tuning seeds (36 runs). The manuscript discussion emphasizes the comparison at server learning rate 1.0. Interpret selection in the context of the full configured search.

## Patient-level privacy score

The membership-inference audit averages window losses within each recording and then averages recordings equally within each patient. This recording-equal score can differ from pooling all windows belonging to a patient. The validation-versus-test control is reported separately from the training-versus-test comparison; neither provides a formal privacy guarantee.

## Evaluation schedule

Training loads only training and validation partitions. Test evaluation is performed separately after validation-based checkpoint selection. Per-round test predictions are not included. Validation histories support the reported round-wise validation curves.

## Environment and independent replication

Canonical run configurations record the scientific-package versions listed in the README. A complete transitive dependency lockfile, equivalent Colab image, GPU driver version, peak RAM, and peak GPU memory are unavailable. A complete independent public-dataset GPU rerun has not yet been completed. Implementation tests and reference-table checks are distinct from end-to-end experimental replication.

The 40 canonical validation-selected checkpoints are indexed by [run ID and SHA-256](../reference/canonical_checkpoints.csv). The training commands also support independent fitting from the public datasets.

## Numerical reporting

Use full-precision reference CSVs when comparing results. The pooled INCART ROC-AUC mean is approximately 0.844704, whereas the manuscript table prints 0.844. This rounding difference should not be interpreted as a change in the underlying predictions.

## Citation and license

Version v1.0.0 is archived at [Zenodo](https://doi.org/10.5281/zenodo.24514278). The additional dense-evaluation artifacts are identified by the Git commit containing them. The repository does not specify a separate code license. Dataset licenses and citation requirements remain those of the respective providers.

## Supplementary output precision and threshold conventions

The additional reference CSVs retain displayed precision. Complete patient-score matrices, full lead-level signal audits, and all per-seed calibration rows are not bundled. Regenerate them with the supplementary commands before performing further inference. Exact patient-label permutation and patient bootstrap intervals apply to the privacy analysis; they do not imply independent overlapping ECG windows.

The calibration audit uses a distinct-probability candidate sweep and higher-threshold equal-distance tie-break. The original threshold analysis additionally evaluates 0.5 and breaks an equal-distance tie toward the lower threshold. Both commands are documented explicitly; the audit does not overwrite existing D1 threshold choices.

Matched-work single-device timing results and their executable control are included. The separate multi-region Flower deployment remains unavailable.
