"""Held-out stride-one evaluation of validation-selected ECG checkpoints."""
import json
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch

from .common import code_hash, file_hash, fingerprint
from .data import STRIDE, WIN, beat_counts, labels_from_counts
from .evaluation import checked_checkpoint
from .metrics import scores
from .training import CNN, device_name

REFERENCE = Path(__file__).resolve().parents[1] / "reference"
INCART_CAP = 16_384
DOMAINS = ("PTB", "INCART")


def dense_starts(length, rid, cap=None):
    candidates = np.arange(max(0, length - WIN + 1), dtype=np.int64)
    if cap is None or len(candidates) <= cap:
        return candidates
    d2 = candidates[::STRIDE]
    if cap < len(d2):
        raise ValueError("Cap must contain all D2 starts")
    rng = np.random.default_rng(int(fingerprint(rid)[:8], 16))
    pool = np.setdiff1d(candidates, d2, assume_unique=True)
    extra = rng.choice(pool, size=cap - len(d2), replace=False)
    return np.sort(np.concatenate((d2, extra)))


def start_digest(starts):
    """SHA-256 of sorted, zero-based sample positions as little-endian int64."""
    return sha256(np.asarray(starts, dtype="<i8").tobytes()).hexdigest()


@contextmanager
def atomic_output(path):
    """Publish a complete prediction archive or table in one rename."""
    with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
    try:
        yield temp
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def record_windows(row, length, beats, abnormal):
    if row.domain == "PTB":
        starts = dense_starts(length, row.rid)
        return starts, np.full(len(starts), row.label, dtype=np.int8)
    starts = dense_starts(length, row.rid, INCART_CAP)
    total, ab = beat_counts(beats, abnormal, starts)
    labels = labels_from_counts(total, ab, 0.05)
    eligible = labels >= 0
    return starts[eligible], labels[eligible]


@torch.inference_mode()
def write_predictions(path, identity, store, records, model, device, manifest):
    starts, labels, probabilities = [], [], []
    offsets = [0]
    for row in records.itertuples(index=False):
        x, beats, abnormal = store.signal(row)
        start, y = record_windows(row, x.shape[1], beats, abnormal)
        if manifest is not None and row.domain == "INCART":
            expected = manifest.loc[row.rid]
            if start_digest(start) != expected.start_sha256 or int(y.sum()) != expected.positive_windows:
                raise ValueError(f"INCART starts or class counts differ from the reference for {row.rid}")
        p = np.empty(len(start), dtype=np.float32)
        for i in range(0, len(start), 256):
            batch = start[i:i + 256]
            windows = np.stack([x[:, s:s + WIN] for s in batch])
            p[i:i + len(batch)] = model(torch.from_numpy(windows).to(device)).sigmoid().cpu().numpy()
        starts.append(start)
        labels.append(y)
        probabilities.append(p)
        offsets.append(offsets[-1] + len(start))

    with atomic_output(path) as temp:
        with temp.open("wb") as tmp:
            np.savez(tmp, provenance=json.dumps(identity, sort_keys=True),
                     record=records.rid.to_numpy(dtype=str), offset=np.asarray(offsets, dtype=np.int64),
                     start=np.concatenate(starts), y=np.concatenate(labels), p=np.concatenate(probabilities))


def evaluate(paths, store, threshold_table, output, acknowledge_source_sampling=None, *,
             domains=DOMAINS, sampling_csv=None):
    """Evaluate completed runs. The former acknowledgement argument is accepted for old callers."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = output.parent / "dense_predictions"
    cache.mkdir(exist_ok=True)
    manifest = pd.read_csv(sampling_csv).set_index("rid") if sampling_csv else None
    thresholds = threshold_table.set_index(["method", "seed", "domain"], verify_integrity=True)
    device = device_name()
    environment = dict(code_hash=code_hash(), torch=str(torch.__version__), numpy=np.__version__,
                       device=torch.cuda.get_device_name() if device == "cuda" else "cpu",
                       cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                       sampling_manifest_sha256=file_hash(sampling_csv) if sampling_csv else None)
    selected = store.select_records("test")
    rows = []
    for path in paths:
        checkpoint = checked_checkpoint(path, store)
        cfg = checkpoint["config"]
        model = CNN(cfg["architecture"]).to(device)
        model.load_state_dict(checkpoint["state"])
        model.eval()
        checkpoint_sha = file_hash(path)
        for domain in domains:
            records = selected.loc[selected.domain == domain]
            if records.empty:
                raise ValueError(f"No held-out records for {domain}")
            if domain == "INCART" and manifest is not None and set(records.rid) != set(manifest.index):
                raise ValueError("INCART recording set differs from the sampling manifest")
            threshold = float(thresholds.loc[(cfg["method"], cfg["seed"], domain), "threshold"])
            if not np.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("Invalid validation threshold")
            identity = dict(**environment, checkpoint_sha256=checkpoint_sha, data_id=store.data_id,
                            method=cfg["method"], seed=cfg["seed"], domain=domain,
                            sampling="record_hash_seed_D2_inclusive" if domain == "INCART" else "exhaustive_stride_one")
            archive = cache / (fingerprint(identity) + ".npz")
            if not archive.exists():
                write_predictions(archive, identity, store, records, model, device, manifest)
            with np.load(archive, allow_pickle=False) as data:
                if json.loads(data["provenance"].item()) != identity:
                    raise ValueError(f"Prediction provenance mismatch in {archive}")
                metrics = scores(data["y"], data["p"], threshold)
            rows.append(dict(method=cfg["method"], seed=cfg["seed"], domain=domain,
                             n_records=len(records), n_patients=records.patient.nunique(),
                             n_windows=int(metrics["n"]), sampling=identity["sampling"],
                             checkpoint_sha256=checkpoint_sha, predictions=f"dense_predictions/{archive.name}",
                             **metrics))

    frame = pd.DataFrame(rows)
    with atomic_output(output) as temp:
        frame.to_csv(temp, index=False)
    return frame


def dense_summary(frame):
    """Aggregate seed rows using the columns and sample SD of Supplementary Table S1."""
    stats = dict(n_seeds=("seed", "nunique"), n_records=("n_records", "min"),
                 n_patients=("n_patients", "min"), n_windows=("n_windows", "mean"))
    for metric in ("roc_auc", "average_precision", "mcc", "f1", "sensitivity", "ppv"):
        name = "pr_auc" if metric == "average_precision" else metric
        stats[name + "_mean"] = (metric, "mean")
        stats[name + "_sd"] = (metric, "std")
    return frame.groupby(["method", "domain"]).agg(**stats).reset_index()


def evaluate_reference(store, output_dir, method=None, seed=None, domain=None):
    checkpoints = pd.read_csv(REFERENCE / "canonical_checkpoints.csv")
    if method is not None:
        checkpoints = checkpoints.loc[checkpoints.method == method]
    if seed is not None:
        checkpoints = checkpoints.loc[checkpoints.seed == seed]
    if checkpoints.empty:
        raise ValueError("No canonical checkpoints match this method and seed")
    paths = []
    for row in checkpoints.itertuples(index=False):
        path = REFERENCE / "runs" / row.run_id / "best.pt"
        if file_hash(path) != row.checkpoint_sha256:
            raise ValueError(f"Canonical checkpoint SHA-256 mismatch for {row.method}, seed {row.seed}")
        paths.append(path)
    thresholds = pd.read_csv(REFERENCE / "study" / "decision_threshold_tuning_runs.csv")
    suffix = "_".join(x for x in (method, f"seed{seed}" if seed is not None else None, domain) if x)
    output = Path(output_dir) / ("canonical_dense_reference" + (f"_{suffix}" if suffix else "") + ".csv")
    frame = evaluate(paths, store, thresholds, output, domains=(domain,) if domain else DOMAINS,
                     sampling_csv=REFERENCE / "dense" / "incart_sampling.csv")
    if seed is None:
        with atomic_output(output.with_name(output.stem + "_summary.csv")) as temp:
            dense_summary(frame).to_csv(temp, index=False)
    return output
