# Reproducibility limitations

## Dense test-time sampling

The manuscript describes an RNG initialized with seed 42 for each INCART recording, sampling without replacement from all valid stride-one start positions. The implementation in `ecg_revision/dense.py` includes all stride-512 starts and samples additional positions using a record-ID-derived seed. Reference D1 tables do not include the selected start positions needed to resolve this difference.

Dense evaluation therefore requires `--acknowledge-source-d1`. Results should be interpreted according to the implemented D2-inclusive sampling rule. The sampling procedure must be reconciled before claiming exact reproduction of the manuscript's D1 protocol.

## Multi-region systems timing

The multi-region Flower server/client deployment, complete timing logs, environment lockfile, and infrastructure definitions are unavailable in this release. The local timing summarizer reports execution-history measurements and an idealized parallel lower bound. It does not reproduce the manuscript's multi-region deployment or establish equivalent runtime on different hardware.

## Hyperparameter search scope

The epsilon sensitivity configuration evaluates two epsilon values across six server learning rates and three tuning seeds (36 runs). The manuscript discussion emphasizes the comparison at server learning rate 1.0. Interpret selection in the context of the full configured search.

## Patient-level privacy score

The membership-inference audit averages window losses within each recording and then averages recordings equally within each patient. This recording-equal score can differ from pooling all windows belonging to a patient. The validation-versus-test control is reported separately from the training-versus-test comparison; neither provides a formal privacy guarantee.

## Evaluation schedule

Training loads only training and validation partitions. Test evaluation is performed separately after validation-based checkpoint selection. Per-round test predictions are not included. Validation histories support the reported round-wise validation curves.

## Environment and independent replication

Canonical run configurations record the scientific-package versions listed in the README. A complete transitive dependency lockfile, equivalent Colab image, GPU driver version, peak RAM, and peak GPU memory are unavailable. A complete independent public-dataset GPU rerun has not yet been completed. Implementation tests and reference-table checks are distinct from end-to-end experimental replication.

Reference checkpoint payloads are not distributed with this release. Researchers can generate checkpoints using the documented training commands.

## Numerical reporting

Use full-precision reference CSVs when comparing results. The pooled INCART ROC-AUC mean is approximately 0.844704, whereas the manuscript table prints 0.844. This rounding difference should not be interpreted as a change in the underlying predictions.

## Citation and license

Final publication details, a persistent release identifier, and a code license have not yet been specified. Dataset licenses and citation requirements remain those of the respective providers.
