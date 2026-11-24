"""Check saved D1 sampling and scores. Optionally rebuild labels from PhysioNet annotations."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_revision.common import file_hash
from ecg_revision.data import AAMI, FS, INC_URL, STRIDE, WIN, beat_counts, fetch, labels_from_counts
from ecg_revision.dense import INCART_CAP, REFERENCE, dense_starts, dense_summary, start_digest
from ecg_revision.metrics import METRICS, from_confusion, scores


def require(condition, message):
    if not condition:
        raise ValueError(message)


def annotation_counts(directory, row, starts, totals, abnormal):
    import wfdb

    record = row.rid.removeprefix("INC_")
    base = directory / record
    for ext in (".hea", ".atr"):
        fetch(INC_URL + record + ext, base.with_suffix(ext))
    header = wfdb.rdheader(str(base))
    require(round(header.sig_len * FS / header.fs) == row.samples_200hz, f"{record}: resampled length")
    ann = wfdb.rdann(str(base), "atr")
    valid = np.asarray([symbol in AAMI for symbol in ann.symbol])
    beats = np.rint(ann.sample[valid] * FS / header.fs).astype(np.int64)
    flags = [AAMI[s] != "N" for s in ann.symbol if s in AAMI]
    total, ab = beat_counts(beats, flags, starts)
    np.testing.assert_array_equal(total, totals, err_msg=f"{record}: mapped beat count")
    np.testing.assert_array_equal(ab, abnormal, err_msg=f"{record}: abnormal beat count")
    # Check the alternative described in the draft against the same annotations.
    other = np.random.default_rng(42).choice(row.candidate_starts, size=INCART_CAP, replace=False)
    other_labels = labels_from_counts(*beat_counts(beats, flags, other), tau=0.05)
    return (np.count_nonzero(other_labels >= 0), np.count_nonzero(other_labels == 1),
            np.count_nonzero(other % STRIDE == 0))


def verify_rows(frame, domain, n, positive, records, patients):
    rows = frame.xs(domain, level="domain")
    require((rows.n_windows == n).all() and (rows.n_records == records).all()
            and (rows.n_patients == patients).all(), f"{domain}: coverage")
    metrics = ["sensitivity", "ppv", "accuracy", "balanced_accuracy", "mcc", "f1"]
    for row in rows.itertuples():
        tp = round(row.sensitivity * positive)
        tn = round(n * row.accuracy) - tp
        confusion = [tn, n - positive - tn, positive - tp, tp]
        require(min(confusion) >= 0, f"{domain}: negative confusion count")
        rebuilt = from_confusion(confusion)
        np.testing.assert_allclose([rebuilt[m] for m in metrics], [getattr(row, m) for m in metrics],
                                   rtol=0, atol=1e-12, err_msg=f"{domain} {row.Index}: class support")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path,
                        help="Verify beat counts from INCART .hea/.atr files, downloading missing files here")
    args = parser.parse_args()
    dense = REFERENCE / "dense"
    manifest = pd.read_csv(dense / "incart_sampling.csv")
    test = pd.read_csv(REFERENCE / "manifest" / "records.csv").query("split == 'test'")
    incart = test.query("domain == 'INCART'").set_index("rid").patient
    require(manifest.set_index("rid").patient.sort_index().equals(incart.sort_index()), "INCART patient split")
    labels, probabilities, other = [], [], np.zeros(3, dtype=int)
    with np.load(dense / "fedavg_seed0_incart.npz", allow_pickle=False) as archive:
        for row in manifest.itertuples(index=False):
            start = archive[f"{row.rid}_start"]
            total, ab = archive[f"{row.rid}_total"], archive[f"{row.rid}_abnormal"]
            y = labels_from_counts(total, ab, 0.05)
            candidates = dense_starts(row.samples_200hz, row.rid, INCART_CAP)
            np.testing.assert_array_equal(candidates, start, err_msg=row.rid)
            d2 = np.arange(0, row.samples_200hz - WIN + 1, STRIDE)
            require(np.isin(d2, start).all() and len(d2) == row.d2_starts, f"{row.rid}: D2 coverage")
            require(row.candidate_starts == row.samples_200hz - WIN + 1 and
                    len(candidates) == row.sampled_starts == len(y) == row.eligible_windows and
                    (y >= 0).all() and y.sum() == row.positive_windows and
                    start_digest(start) == row.start_sha256, f"{row.rid}: window or label manifest")
            p = archive[f"{row.rid}_p"]
            require(len(p) == len(start), f"{row.rid}: prediction count")
            labels.append(y)
            probabilities.append(p)
            if args.annotations_dir:
                other += annotation_counts(args.annotations_dir, row, start, total, ab)

    y, p = np.concatenate(labels), np.concatenate(probabilities)
    study = REFERENCE / "study"
    keys = ["method", "seed", "domain"]
    rows = pd.read_csv(study / "d1_canonical_10seed_restart_safe_rows.csv").set_index(keys, verify_integrity=True)
    thresholds = pd.read_csv(study / "decision_threshold_tuning_runs.csv").set_index(keys, verify_integrity=True)
    checkpoints = pd.read_csv(REFERENCE / "canonical_checkpoints.csv")
    runs = pd.read_csv(study / "primary_canonical_runs.csv")
    expected_runs = runs.assign(run_id=runs.run_path.map(lambda path: Path(path).name))
    cols = ["method", "seed", "run_id", "best_round"]
    pd.testing.assert_frame_equal(checkpoints[cols], expected_runs[cols])
    expected_keys = {(r.method, r.seed, d) for r in checkpoints.itertuples() for d in ("PTB", "INCART")}
    require(set(rows.index) == set(thresholds.index) == expected_keys and len(expected_keys) == 80, "D1 run grid")
    np.testing.assert_array_equal(rows.decision_threshold, thresholds.threshold.reindex(rows.index))
    summary = pd.read_csv(study / "d1_canonical_10seed_summary.csv")
    pd.testing.assert_frame_equal(dense_summary(rows.reset_index()), summary, rtol=0, atol=1e-12)
    for r in checkpoints.itertuples():
        require(file_hash(REFERENCE / "runs" / r.run_id / "best.pt") == r.checkpoint_sha256,
                f"{r.method} seed {r.seed}: checkpoint SHA-256")

    saved = rows.loc[("fedavg", 0, "INCART")]
    rescored = scores(y, p, threshold=saved.decision_threshold)
    np.testing.assert_allclose([rescored[m] for m in METRICS], saved[METRICS].to_numpy(float),
                               rtol=0, atol=1e-12, err_msg="FedAvg seed 0 archived INCART probabilities")
    verify_rows(rows, "INCART", len(y), int(y.sum()), len(manifest), manifest.patient.nunique())
    ptb = test.query("domain == 'PTB'")
    per_record = 10 * FS - WIN + 1
    verify_rows(rows, "PTB", len(ptb) * per_record, int(ptb.label.sum()) * per_record, len(ptb), ptb.patient.nunique())
    print(f"INCART: {len(manifest)} recordings, {len(y):,} eligible windows, {int(y.sum()):,} positives, "
          f"all {manifest.d2_starts.sum():,} D2 starts retained.")
    print("FedAvg seed 0: all eight INCART metrics match the saved probabilities to 12 decimal places.")
    print("40 checkpoint hashes and 80 domain rows verified for thresholds, class counts and threshold metrics.")
    print("All eight ten-seed summary rows reproduce Supplementary Table S1's reference CSV.")
    if args.annotations_dir:
        print(f"PhysioNet beat counts match at every saved start. Seed-42 comparison: {other[0]:,} eligible, "
              f"{other[1]:,} positives, {other[2]:,} D2 starts retained.")


if __name__ == "__main__":
    main()
