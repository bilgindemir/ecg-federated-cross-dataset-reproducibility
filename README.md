# Federated Learning for ECG Classification Under Cross-Dataset Heterogeneity

Reproducibility repository accompanying the manuscript **“A Comparative Study of Federated Learning Methods for ECG Classification Under Cross-Dataset Heterogeneity”** by B. Demir, E. Domazet, M. Gusev, D. Mechkaroska, and A. Hasan, prepared for IEEE Access.

This repository provides a common experimental framework for comparing FedAvg, FedAvgM, FedProx, and FedAdam on binary ECG classification using PTB-XL and INCART. It includes preprocessing, patient-disjoint dataset construction, model training, evaluation, supplementary analyses, and reference results.

The primary protocol comprises four logical clients, 20 communication rounds, and 10 matched random seeds. Three clients represent disjoint PTB-XL training folds; the fourth represents the INCART training partition. Checkpoint selection uses validation data exclusively.

## Reproducibility status

The repository includes automated implementation tests and verification of the reference results. A complete independent GPU rerun has not yet been completed. The dense-evaluation sampling discrepancy and unavailable multi-region deployment artifacts are described in the relevant sections below and in [reproducibility limitations](docs/OPEN_ISSUES.md).

## Repository layout

```text
ecg_revision/
  preprocessing.py    Native-rate filtering, PCHIP resampling, record normalization
  data.py             Labels, fixed split, WFDB acquisition, disk-backed D2 packs
  model.py            Three-block BatchNorm CNN
  training.py         Local AdamW, aggregation, validation selection, checkpoints
  metrics.py          ROC-AUC, average precision, confusion-derived metrics
  experiments.py      Explicit configuration expansion
  evaluation.py       D2 evaluation, validation thresholds, label sensitivity
  dense.py            Dense test-time segmentation analysis
  privacy.py          INCART patient-level membership/control analysis
  cli.py              Portable commands
configs/              Canonical and supplementary experiment specifications
manifests/            Fixed 75-record INCART patient/split assignment
reference/            Reference tables, protocols, configurations and histories
provenance/           Implementation-test and reference-verification reports
scripts/              Reference verification and timing-summary commands
tests/                Automated implementation and protocol tests
```

## Dataset acquisition

Use the versioned public sources:

- [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/), using `filename_hr` and native 500 Hz waveforms.
- [INCART v1.0.0](https://physionet.org/content/incartdb/1.0.0/), using native 257 Hz waveforms, `.atr` beat annotations, and `files-patients-diagnoses.txt`.

Respect the dataset providers' terms and citation instructions. Raw ECGs are not bundled in this repository. The preparation command downloads the metadata and required waveform files from those exact versioned PhysioNet URLs; it requires network access. No Google Drive account is required.

```bash
python -m ecg_revision.cli prepare --root work
```

Expected generated structure:

```text
work/
  data/
    metadata/
      ptbxl_database.csv
      scp_statements.csv
      incart_patients.txt             # downloaded files-patients-diagnoses.txt
    raw/
      ptb/records500/00000/..._hr.hea # plus .dat; other numbered folders follow
      incart/I01.hea                  # plus I01.dat, I01.atr, ... I75
    processed_native_pchip_v1/
      PTB_00001.npz                   # signal + annotation arrays; JSON hashes beside it
      INC_I01.npz
    manifests/9d68265b9e8940c9c1b1/
      records.csv
      split_audit.json
      provenance.json
    d2/9d68265b9e8940c9c1b1/{train,val,test}/
      X.npy                          # float32 [windows, 12, 1024], memory mapped
      index.csv
      coverage.csv
      complete.json
  study/
    runs/<new-run-id>/{config.json,history.json,last.pt,best.pt,complete.json}
    predictions/<prediction-id>/{p.npy,provenance.json}
    canonical_runs.csv
    canonical_metrics.csv
    canonical_summary.csv
```

For an existing download, place the files at the paths above before running `prepare`; non-empty files are reused. Keep PTB relative paths from `filename_hr`. A complete existing dataset extraction can also be linked into this layout. Do not substitute PTB `records100`, previously windowed caches, or old INCART train/test arrays. Use a fresh `--root` for a different protocol or code revision.

Preprocessing proceeds as follows: reorder leads to I, II, III, aVR, aVL, aVF, V1–V6; fourth-order Butterworth 0.5–40 Hz filtering at the native sampling rate using `filtfilt`; PCHIP interpolation to 200 Hz; per-lead, full-record z-score with denominator `std + 1e-8`; then 1,024-sample windows at stride 512. Trailing partial windows are dropped. Signals are represented as float32 arrays; resampled beat indices use nearest-integer rounding.

PTB labels: NORM-only maps to 0; any mapped non-NORM diagnostic superclass maps to 1; records without a mapped diagnostic superclass are excluded. The mapped cohort retains 21,388 recordings (411 exclusions). INCART labels use mapped AAMI-style beat groups: N/L/R/e/j are normal; S/V/F/Q groups are abnormal. Windows require at least three mapped beats and are positive when abnormal/total is **at least 0.05**. Beats fall in half-open `[start, start+1024)` windows. This is an operational binary target, not equivalent clinical labeling between domains.

## Installation and recorded environment

Create an isolated Python environment and install a hardware-appropriate PyTorch build following [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). The experiments used the Colab PyTorch environment documented below. A complete environment lockfile is unavailable.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install PyTorch for your platform first; preserve the existing Colab GPU build if using Colab.
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip install pytest
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m ecg_revision.cli environment --root work
python -m pytest -q
python scripts/verify_reference.py
```

Run these commands from the repository root; the supported installation is editable so the checked-in split manifest remains available. On Windows, activate `.venv\Scripts\activate` and set the environment variable using your shell's syntax.

All 40 saved canonical configs report the same environment:

| Item | Recorded value |
|---|---|
| Python | 3.13.16 |
| PyTorch | 2.11.0+cu130 |
| CUDA reported by PyTorch | 13.0 |
| NumPy / pandas | 2.1.3 / 2.2.3 |
| SciPy / scikit-learn | 1.16.3 / 1.6.1 |
| WFDB | 4.2.0 |
| GPU | Tesla T4 |
| CPU count / PyTorch threads | 2 / 1 |
| Platform | Linux 6.6.159+, x86_64, glibc 2.39 |

`requirements-observed.txt` lists the scientific-package versions recorded for the canonical experiments. `requirements.txt` specifies installation dependencies. Availability of an identical Colab image has not been verified. Requests, tqdm, soundfile, driver version, complete `pip freeze`, RAM, and Colab image identifier were not recorded. CLI execution uses one PyTorch CPU thread. Different software/hardware may produce numerical differences despite seeded deterministic operations.

## Fixed patient-disjoint split

[incart_patient_split.csv](manifests/incart_patient_split.csv) specifies the fixed assignment reported in manuscript Table 18. It contains **record, source patient ID, and split**, not just a seed. The loader reads this file and checks it against the source patient mapping; it does not regenerate random splits on every run.

| Split | Patients | Recordings | Eligible D2 windows |
|---|---:|---:|---:|
| INCART train | 19 | 44 | 30,888 |
| INCART validation | 6 | 15 | 10,523 |
| INCART test | 7 | 16 | 11,232 |
| PTB train | official folds 1–8 | 17,084 | 34,168 |
| PTB validation | official fold 9 | 2,146 | 4,292 |
| PTB test | official fold 10 | 2,158 | 4,316 |

The INCART split was constructed using sorted source patients and two `train_test_split` calls with random state 42: 60/40, then 50/50 of the held-out 40%. Patient groups and all their records stay together. PTB training folds form C0=1–3, C1=4–6, C2=7–8; C3 is INCART train. Client counts are 12,854, 12,788, 8,526, and 30,888 windows. Seven INCART validation candidate windows fail the minimum-beat rule. Splits do not change with experiment seeds.

## Canonical 10-seed experiment

Inspect before running:

```bash
python -m ecg_revision.cli plan --config configs/canonical.json
python -m ecg_revision.cli train --config configs/canonical.json --root work
python -m ecg_revision.cli evaluate --config configs/canonical.json --root work
python -m ecg_revision.cli paired --config configs/canonical.json --root work
```

This trains 4 methods × seeds 0–9 × 20 rounds, all four clients participating every round. Each client uses one local epoch, batch size 64, AdamW lr=0.001 and weight decay=0.0001, gradient-norm clipping at 1.0, BCE with logits, and exactly `n_client` inverse-class-frequency draws with replacement per epoch. AdamW state is reset for each federated client/round. Centralized/local baseline optimizer states persist across epochs.

The network has channels 32/64/128, kernels 7/5/3 with padding `kernel//2`, BatchNorm/ReLU/MaxPool2 after each convolution, global average pooling, and one linear logit: **38,305 trainable parameters**. Sigmoid is applied for prediction.

| Method | Frozen setting |
|---|---|
| FedAvg | direct equal-client average |
| FedAvgM | server lr 3.0; momentum 0.9 |
| FedProx | μ=3.0; local penalty μ/2 × squared distance from global parameters |
| FedAdam | server lr 1.0; β1=0.9; β2=0.999; ε=0.001; bias-corrected moments |

Floating BatchNorm buffers use the same averaging weights; integer batch counters use the maximum. Server momentum/Adam updates apply to trainable parameters, while buffers take their aggregated values. The FedAdam variant uses global-minus-client-average pseudo-gradients and bias correction, as implemented in this study.

After every round, the unweighted mean of PTB and INCART validation ROC-AUC selects the checkpoint; exact ties retain the earliest round. Training does not load test partitions. Held-out evaluation is a separate command. Primary threshold metrics use 0.5; PR-AUC is implemented as `average_precision_score`, not trapezoidal integration. Results are summarized over seeds with sample SD (`ddof=1`); paired differences use seed-level t intervals.

Re-running `train` resumes at a completed round or skips completed runs. Do not edit code mid-run: new source hashes deliberately create different run IDs. This package does not provide mid-minibatch resume.

## Supplementary analyses

All commands use the same prepared data. E=1/equal-weight matching runs reuse canonical identities when their configurations are identical.

### Validation-only hyperparameter and epsilon sweeps

```bash
python -m ecg_revision.cli train --config configs/tuning.json --root work
python -m ecg_revision.cli summarize --config configs/tuning.json --root work
python -m ecg_revision.cli train --config configs/epsilon.json --root work
python -m ecg_revision.cli summarize --config configs/epsilon.json --root work
```

Tuning seeds are 101, 202, 303. The hyperparameter grid is μ={0,0.0001,0.001,0.01,0.1,1,3,10}; FedAvgM/FedAdam server lr={0.001,0.01,0.1,1,3,10}. Initial FedAdam ε=1e-8. The epsilon analysis evaluates both 1e-8 and 0.001 with **all six** server rates (36 rows), although manuscript prose emphasizes the comparison at lr=1.0. The canonical configuration fixes the selected hyperparameters; these commands do not automatically replace it using new outcomes. Grid summaries use mean best validation AUC across tuning seeds, never test scores.

### Local computation and aggregation weighting

```bash
python -m ecg_revision.cli train --config configs/local_epochs.json --root work
python -m ecg_revision.cli evaluate --config configs/local_epochs.json --root work
python -m ecg_revision.cli train --config configs/weighting.json --root work
python -m ecg_revision.cli evaluate --config configs/weighting.json --root work
```

Both use seeds 0–2 and frozen optimizer settings. Local epochs are 1/3/5. Weightings are equal-client `(1/4,1/4,1/4,1/4)`, empirical client counts normalized to one, and equal-domain `(1/6,1/6,1/6,1/2)`.

### Pooled, local-only, and transfer baselines

```bash
python -m ecg_revision.cli train --config configs/pooled.json --root work
python -m ecg_revision.cli evaluate --config configs/pooled.json --root work
python -m ecg_revision.cli train --config configs/local_transfer.json --root work
python -m ecg_revision.cli evaluate --config configs/local_transfer.json --root work
```

Pooled uses 10 seeds and five epochs with macro-domain validation selection. Local-only/transfer use seeds 0–2 and five epochs per stage. PTB-only selects on PTB validation, INCART-only on INCART validation. Transfer starts from the matching PTB-only selected checkpoint, fine-tunes all parameters on INCART, and selects on INCART validation. These are not compute-matched FL comparisons.

### Decision threshold and INCART label sensitivity

```bash
python -m ecg_revision.cli thresholds --config configs/canonical.json --root work
python -m ecg_revision.cli labels --config configs/canonical.json --root work
```

Decision thresholds maximize validation MCC separately for each model/domain. Ties choose the value closest to 0.5, then the lower threshold. Test predictions are evaluated only after selection. Label sensitivity retains fixed models/predictions and changes INCART test labels for τ={0.01,0.05,0.10,0.20,0.30,0.50}; it does not retrain under alternate targets.

### Dense D1 sensitivity: unresolved sampling discrepancy

```bash
# Run the validation-threshold analysis before dense evaluation.
python -m ecg_revision.cli dense --config configs/canonical.json --root work --acknowledge-source-d1
```

PTB-XL is evaluated exhaustively at stride one, yielding 2,108,366 windows. INCART evaluation includes all 16 test recordings, with 16,384 candidate start positions per recording. D2 validation-selected decision thresholds are reused unchanged.

**Sampling limitation:** the current implementation includes all stride-512 starts and samples additional positions using a record-ID-derived seed. The manuscript describes seed-42 sampling from all candidate positions. This difference remains unresolved. The command therefore requires explicit acknowledgement and writes `*_dense_source_protocol.csv`. These outputs should be interpreted according to the implemented sampling rule; they do not establish replication of the manuscript's stated sampling procedure.

### Patient-level membership-inference audit

```bash
python -m ecg_revision.cli privacy --config configs/canonical.json --root work
```

For each canonical checkpoint: compute INCART stride-512 loss; average within recording, then equally across a patient's recordings; negate this loss as the membership score. Compare exposed training patients against test patients and validation against test as a distribution-shift control. The latter groups are both unseen during training. No low-FPR claim or formal privacy guarantee is produced. Each recording contributes equally to its patient-level loss score.

### Systems timing

```bash
python scripts/summarize_timing.py --root work
```

This command summarizes local execution histories and calculates `sum(max(client_train_s) + aggregation_s)` as an idealized synchronous parallel lower bound. Evaluation, network communication, and coordination overhead are excluded from that lower bound.

The multi-region Flower deployment is not executable from this release: deployment scripts, complete timing logs, an environment lockfile, and infrastructure definitions are unavailable. The reported distributed timing results are provided as manuscript context rather than expected runtimes for this package.

## Additional robustness and calibration analyses

The following commands evaluate fixed checkpoints or source signals. Results are written to `work/study/supplementary/`. They do not select new canonical hyperparameters or modify the patient split.

### INCART signal quality and patient-removal sensitivity

```bash
python -m ecg_revision.supplementary signal-quality --root work
python -m ecg_revision.supplementary patient-robustness --root work
```

Signal checks use native physical and digital waveforms. They report amplitude excursions above 10 mV, ADC-rail samples, observed digital extrema, and constant/extreme plateaus. A long plateau at an observed extremum can occur without reaching a theoretical ADC rail; these measures should be interpreted separately. The checks describe signal characteristics and do not introduce preprocessing exclusions.

Patient-removal sensitivity uses fixed canonical D2 predictions at threshold 0.5. It evaluates the full INCART test set, each of its seven leave-one-patient-out subsets, and joint removal of patients `INC_1` (I01/I02) and `INC_25` (I57/I58). These are post-hoc held-out evaluation subsets, without retraining. Single-class subsets return undefined ranking metrics and MCC.

### Patient-level privacy uncertainty

```bash
python -m ecg_revision.supplementary privacy-uncertainty --root work
```

This extends the privacy audit with patient-level inference. Each patient's complete ten-seed score vector stays together. The statistic is the mean ROC-AUC over matched seeds. Exact label permutation enumerates 657,800 assignments for train/test (19 versus 7 patients) and 1,716 assignments for validation/test (6 versus 7 patients). Stratified patient bootstrap intervals use 20,000 repetitions, retaining all ten scores when resampling a patient. RNG seeds are `1000 + method_index` for train/test and `2000 + method_index` for validation/test, in order FedAvg, FedAvgM, FedProx, FedAdam.

Seed SD and patient bootstrap intervals describe different uncertainty sources. The small patient cohort and distribution-shift control limit membership-specific interpretation. Exact permutation and bootstrap calculations can take substantial CPU time.

### All pairwise comparisons and calibration audit

```bash
python -m ecg_revision.supplementary all-pairs --root work
python -m ecg_revision.supplementary threshold-audit --root work
```

The pairwise command uses `canonical_metrics.csv` and reports all six optimizer contrasts per domain with paired 95% t intervals over the ten seeds. These intervals are not multiplicity-adjusted.

The calibration audit reports the mean, SD, minimum and maximum of validation-selected thresholds, fixed/tuned test MCC, and binary Brier score (mean squared probability error). It sweeps distinct validation probabilities using the rule `p >= threshold`; ties within `1e-12` choose the candidate closest to 0.5, with the higher candidate retained on an equal-distance tie. This candidate-only audit is separate from the existing `thresholds` command, whose sweep additionally includes 0.5 and uses a lower-threshold secondary tie-break. Existing D1 thresholds remain unchanged.

### FedProx validation–test comparison

```bash
python -m ecg_revision.cli train --config configs/fedprox_gap.json --root work
python -m ecg_revision.supplementary fedprox-gap --root work
```

The configuration specifies μ=0 and μ=3 for tuning seeds 101, 202 and 303 under the same 20-round, one-local-epoch protocol. Configurations identical to completed tuning runs reuse those checkpoints. The evaluation reports validation-selected checkpoint scores alongside held-out PTB, INCART and macro-domain ROC-AUC. Test results are descriptive and do not change hyperparameter selection.

### Label concentration and fixed ECG excerpt

```bash
python -m ecg_revision.supplementary label-distribution --root work
python -m ecg_revision.supplementary excerpt --root work
```

The label-distribution command counts INCART test positives by patient at τ=0.50 using fixed window eligibility. The reference output contains 125 positives among 11,232 windows; the two largest contributing patients account for 88.8% of those positives.

The excerpt command fixes INCART I03 to samples 18,758–20,757 at 200 Hz (93.79–103.79 s). It exports all 977 stride-one window labels and an excerpt manifest. Reference counts are 561 abnormal and 416 normal windows, with no ineligible windows. This labeling example is distinct from D1 held-out model evaluation; it does not resolve the D1 sampling limitation. Native-lead plateau diagnostics are available through `signal-quality` when selecting a lead for visualization.

### Baseline checkpoint provenance and matched-work timing

```bash
# Requires completed pooled and local_transfer experiments.
python -m ecg_revision.supplementary baseline-provenance --root work
python -m ecg_revision.supplementary matched-timing --root work
```

The baseline audit hashes all 19 selected checkpoints and verifies that each transfer model starts from a PTB-only checkpoint with the matching seed and data manifest.

The matched timing control performs three repetitions of five rounds, with seeds 900–902, batch size 64, and one local epoch. Each execution mode receives an unmeasured warm-up. It compares sequential four-client FedAvg against pooled work on one device, with 65,056 sampled examples per round in each mode. Local optimizer state resets per federated client/round and persists for the pooled control. Validation, testing and checkpoint writes are excluded from the measured training loop. Warm-up initializes runtime kernels but does not update the measured initial model. This is a work-controlled timing diagnostic, not a time-to-equal-quality comparison or a multi-region deployment.

Supplementary reference tables are available in [reference/supplementary](reference/supplementary/README.md). They retain the precision displayed in the supplied outputs. Complete prediction arrays and some full per-seed audit tables are unavailable; the commands above generate them during reproduction.

## Principal reference results

Reference results are summarized over 10 matched seeds as mean ± sample standard deviation. Threshold-dependent metrics use the fixed threshold of 0.5.

| Method | Domain | ROC-AUC | Average precision | MCC |
|---|---|---|---|---|
| FedAvg | PTB | 0.911090 ± 0.006267 | 0.941239 ± 0.004230 | 0.608682 ± 0.031410 |
| FedAvgM | PTB | 0.888224 ± 0.017782 | 0.921560 ± 0.010351 | 0.512753 ± 0.126184 |
| FedProx | PTB | 0.884733 ± 0.004693 | 0.920626 ± 0.003550 | 0.569796 ± 0.021428 |
| FedAdam | PTB | 0.852803 ± 0.016159 | 0.891350 ± 0.013280 | 0.188433 ± 0.158570 |
| FedAvg | INCART | 0.775582 ± 0.026698 | 0.704692 ± 0.028042 | 0.416319 ± 0.050305 |
| FedAvgM | INCART | 0.792354 ± 0.054442 | 0.731898 ± 0.086010 | 0.412303 ± 0.087229 |
| FedProx | INCART | 0.734369 ± 0.037462 | 0.651998 ± 0.037738 | 0.385428 ± 0.066813 |
| FedAdam | INCART | 0.821559 ± 0.024341 | 0.780323 ± 0.034766 | 0.199226 ± 0.179913 |

Pooled values and supplementary tables are in `reference/study/baseline_canonical_summary.csv` and the corresponding named CSVs. A manuscript rounding discrepancy exists for pooled INCART ROC-AUC: the table prints 0.844, whereas saved data/text give about 0.8447 (normally 0.845 at three decimals). Use full-precision saved values when checking.

`python scripts/verify_reference.py` checks 40 configurations, 800 history rows, selected rounds, 80 domain rows, seed coverage and primary ROC-AUC rounding. To verify a new run, first compare the data ID, counts, configuration and selected-round traces, then compare seed-level metrics and summary distributions. No unsupported numerical tolerance or bitwise GPU reproducibility guarantee is asserted.

## Computational requirements

Recorded canonical execution used a Tesla T4, two CPU cores and one PyTorch thread. Required host RAM, disk peak, GPU memory peak and full-study runtime were not recorded. The supplied hardware observations are not minimum requirements.

D2 alone contains 95,419 float32 windows × 12 × 1,024, or **4,690,034,688 bytes (~4.37 GiB)** of waveform payload (calculated, excluding NPY headers, metadata, raw files, full-record processed arrays, temporary files and checkpoints). Processing both sources also retains full-record arrays; allow additional disk space. D1 is streamed in batches of 256 rather than materializing all windows. Canonical training consists of 800 rounds / 3,200 client local epochs. Expanded sweeps, baselines, D1, and privacy add substantial work. CPU execution is supported but no wall-time prediction is established.

Manuscript historical timing: centralized five epochs 191.23 s, sequential FL 20 rounds 639.67 s, calculated parallel lower bound 182.88 s. Multi-region timing reports CPU clients with 2 vCPU/8 GB and a 2 vCPU/4 GB server, three 20-round runs averaging 2011.64 ± 38.27 s. These measurements use distinct execution protocols and hardware; they are not runtime targets for this package and should not be used to calculate cross-hardware speedups.

## Citation and licensing

When using this repository, cite the accompanying manuscript and the dataset publications linked above. Publication details and a persistent repository release identifier will be added when available.

A code license has not yet been specified. Dataset use remains subject to the respective PhysioNet license and citation requirements.

## Validation and limitations

Implementation checks cover split integrity, window labeling, model structure, aggregation, threshold selection, and deterministic round-boundary resume. Reference verification checks all 40 canonical runs, 800 round histories, and 80 held-out domain evaluations. See [validation details](docs/VALIDATION.md).

The current validation does not constitute a complete independent rerun of all experiments. Consult [reproducibility limitations](docs/OPEN_ISSUES.md) before interpreting dense-evaluation or systems-timing comparisons.
