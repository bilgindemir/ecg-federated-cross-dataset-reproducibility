from __future__ import annotations
import ast
import os
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from .common import VERSION, file_hash, fingerprint, read_json, write_json

PTB_URL = "https://physionet.org/files/ptb-xl/1.0.3/"
INC_URL = "https://physionet.org/files/incartdb/1.0.0/"
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 200
WIN = 1024
STRIDE = 512
MIN_BEATS = 3
# Matches supplied grouping; '/' additionally recognizes WFDB paced beats.
# INCART documentation says no pacemaker patients. Unknown symbols are logged.
AAMI = {**dict.fromkeys(["N", "L", "R", "e", "j"], "N"),
        **dict.fromkeys(["A", "a", "J", "S"], "S"),
        **dict.fromkeys(["V", "E"], "V"), "F": "F",
        **dict.fromkeys(["P", "/", "Q", "f"], "Q")}
_sessions = threading.local()

def fetch(url, destination):
    import requests
    destination = Path(destination)
    if destination.is_file() and destination.stat().st_size: return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(_sessions, "session"): _sessions.session = requests.Session()
    tmp = destination.with_name(destination.name + ".partial")
    for attempt in range(3):
        try:
            with _sessions.session.get(url, timeout=(20, 180), stream=True) as response:
                response.raise_for_status()
                with tmp.open("wb") as f:
                    for block in response.iter_content(256 * 1024): f.write(block)
            if not tmp.stat().st_size: raise IOError("Empty download")
            os.replace(tmp, destination); return destination
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt == 2: raise

def ptb_label(codes, superclass_map, unknown_policy="legacy_positive"):
    classes = sorted({superclass_map[c] for c in codes if c in superclass_map})
    if not classes:
        return (1 if unknown_policy == "legacy_positive" else -1), classes
    return int(any(c != "NORM" for c in classes)), classes

def parse_incart_patients(text):
    patient, mapping = None, {}
    for line in text.splitlines():
        match = re.match(r"patient\s+(\d+)", line.strip(), re.I)
        if match: patient = int(match.group(1))
        for record in re.findall(r"\bI\d{2}\b", line):
            if patient is None or record in mapping: raise ValueError("Invalid INCART patient mapping")
            mapping[record] = patient
    if len(mapping) != 75 or len(set(mapping.values())) != 32:
        raise ValueError("Expected INCART v1.0.0: 75 records and 32 source patients")
    return mapping

def incart_splits(mapping, policy, seed=42):
    units = sorted(mapping) if policy == "recording42" else sorted(set(mapping.values()))
    train, temporary = train_test_split(units, test_size=0.4, random_state=seed)
    val, test = train_test_split(temporary, test_size=0.5, random_state=seed)
    assignment = {u: name for name, us in [("train", train), ("val", val), ("test", test)] for u in us}
    result = {r: assignment[r if policy == "recording42" else p] for r, p in mapping.items()}
    return result

def split_audit(records):
    result = {}
    for domain, frame in records.groupby("domain"):
        groups = {s: set(frame.loc[frame.split == s, "patient"]) for s in ("train", "val", "test")}
        result[domain] = {
            "records": frame.groupby("split").size().to_dict(),
            "patients": {s: len(g) for s, g in groups.items()},
            "patient_overlap": {f"{a}_{b}": sorted(groups[a] & groups[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]},
            "test_records_with_training_patient": frame.loc[(frame.split == "test") & frame.patient.isin(groups["train"]), "rid"].tolist(),
        }
    return result

def preprocess_signal(x, fs_original):
    """Native-rate 0.5-40 Hz filter -> PCHIP -> record-wise z-score.

    PCHIP is algebraically the cubic-Hermite/Bezier implementation in the
    uploaded code. Preserve native filtering before downsampling.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != 12 or not np.isfinite(x).all(): raise ValueError("Invalid 12-lead input")
    b, a = butter(4, [0.5 / (fs_original / 2), 40 / (fs_original / 2)], btype="band")
    x = filtfilt(b, a, x, axis=1).astype(np.float32)
    old = np.arange(x.shape[1], dtype=np.float64) / fs_original
    new = np.arange(round(x.shape[1] * FS / fs_original), dtype=np.float64) / FS
    x = PchipInterpolator(old, x, axis=1)(new).astype(np.float32)
    x = ((x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    if not np.isfinite(x).all(): raise ValueError("Nonfinite preprocessed signal")
    return x

def beat_counts(samples, abnormal, starts, window=WIN):
    """Count mapped beats in half-open windows [start, start + window)."""
    samples = np.asarray(samples, np.int64); abnormal = np.asarray(abnormal, np.int64)
    starts = np.asarray(starts, np.int64)
    if len(samples) != len(abnormal) or np.any(np.diff(samples) < 0): raise ValueError("Invalid beat arrays")
    left = np.searchsorted(samples, starts, side="left")
    right = np.searchsorted(samples, starts + window, side="left")
    prefix = np.r_[0, np.cumsum(abnormal)]
    return (right - left).astype(np.int32), (prefix[right] - prefix[left]).astype(np.int32)

def labels_from_counts(total, abnormal, tau):
    total = np.asarray(total); abnormal = np.asarray(abnormal)
    if not 0 < tau <= 1: raise ValueError("tau must lie in (0,1]")
    # Inclusive boundary as in uploaded code; no change in eligibility across tau.
    y = (abnormal / np.maximum(total, 1) >= tau).astype(np.int8)
    y[total < MIN_BEATS] = -1
    return y

class Store:
    def __init__(self, root, legacy_root=None, split_policy="patient42", unknown_policy="exclude", workers=6):
        self.root = Path(root); self.legacy_root = Path(legacy_root) if legacy_root else None
        if unknown_policy not in ("exclude", "legacy_positive"): raise ValueError("Unknown PTB label policy")
        self.split_policy = split_policy; self.unknown_policy = unknown_policy; self.workers = workers
        self.meta = self.root / "metadata"; self.meta.mkdir(parents=True, exist_ok=True)
        self._records = None

    @property
    def records(self):
        if self._records is not None: return self._records
        if self.split_policy not in ("patient42", "recording42"): raise ValueError("Unknown split policy")
        for name in ("ptbxl_database.csv", "scp_statements.csv"): fetch(PTB_URL + name, self.meta / name)
        fetch(INC_URL + "files-patients-diagnoses.txt", self.meta / "incart_patients.txt")
        df = pd.read_csv(self.meta / "ptbxl_database.csv")
        scp = pd.read_csv(self.meta / "scp_statements.csv", index_col=0)
        mapping = scp.loc[scp.diagnostic_class.notna(), "diagnostic_class"].to_dict()
        rows, unknown, mixed = [], 0, 0
        for row in df.itertuples():
            label, classes = ptb_label(ast.literal_eval(row.scp_codes), mapping, self.unknown_policy)
            unknown += not classes; mixed += "NORM" in classes and len(classes) > 1
            if label < 0: continue
            fold = int(row.strat_fold)
            if pd.isna(row.patient_id): raise ValueError("Missing PTB patient ID")
            rows.append(dict(rid=f"PTB_{int(row.ecg_id):05d}", domain="PTB", patient=f"PTB_{int(row.patient_id)}",
                             split="train" if fold <= 8 else "val" if fold == 9 else "test",
                             client=(0 if fold <= 3 else 1 if fold <= 6 else 2) if fold <= 8 else -1,
                             label=label, raw_rel=row.filename_hr, native_fs=500))
        incmap = parse_incart_patients((self.meta / "incart_patients.txt").read_text())
        assignment = incart_splits(incmap, self.split_policy)
        for rid in sorted(incmap):
            rows.append(dict(rid="INC_" + rid, domain="INCART", patient=f"INC_{incmap[rid]}",
                             split=assignment[rid], client=3 if assignment[rid] == "train" else -1,
                             label=-1, raw_rel=rid, native_fs=257))
        self._records = pd.DataFrame(rows).sort_values(["domain", "rid"]).reset_index(drop=True)
        audit = split_audit(self._records)
        audit.update(ptb_unknown_records=int(unknown), ptb_mixed_norm_abnormal=int(mixed),
                     unknown_policy=self.unknown_policy, split_policy=self.split_policy)
        if self.split_policy == "patient42":
            for domain in ("PTB", "INCART"):
                if any(audit[domain]["patient_overlap"].values()): raise ValueError("Patient leakage")
        self._records.to_csv(self.manifest_dir / "records.csv", index=False)
        write_json(self.manifest_dir / "split_audit.json", audit)
        write_json(self.manifest_dir / "provenance.json", dict(version=VERSION, data_id=self.data_id,
            source_urls=[PTB_URL, INC_URL], fs=FS, win=WIN, stride=STRIDE, min_beats=MIN_BEATS,
            preprocessing="native-butter4-filtfilt-pchip-zscore-float32-canonical-leads",
            metadata_sha256={p.name: file_hash(p) for p in self.meta.glob("*") if p.is_file()},
            unknown_policy=self.unknown_policy, aami=AAMI))
        return self._records

    @property
    def data_id(self):
        if self._records is None: _ = self.records
        return fingerprint(dict(version=VERSION, records=self._records.to_dict("records"),
                                unknown_policy=self.unknown_policy, preprocessing="native-butter4-pchip-zscore-v1"))[:20]

    @property
    def manifest_dir(self):
        p = self.root / "manifests" / self.data_id; p.mkdir(parents=True, exist_ok=True); return p

    def select_records(self, scope):
        r = self.records
        if scope.startswith("client_"):
            return r[(r.split == "train") & (r.client == int(scope.split("_")[1]))]
        if scope not in ("train", "val", "test"): raise ValueError("Unknown scope")
        return r[r.split == scope]

    def raw_base(self, row):
        subdir = "ptb" if row.domain == "PTB" else "incart"
        if self.legacy_root:
            legacy = self.legacy_root / "data_200hz" / ("ptbxl_hr" if row.domain == "PTB" else "incart") / row.raw_rel
            exts = [".hea", ".dat"] + ([".atr"] if row.domain == "INCART" else [])
            if all(legacy.with_suffix(e).is_file() for e in exts): return legacy
        target = self.root / "raw" / subdir / row.raw_rel
        url = PTB_URL if row.domain == "PTB" else INC_URL
        for ext in ([".hea", ".dat", ".atr"] if row.domain == "INCART" else [".hea", ".dat"]):
            fetch(url + row.raw_rel + ext, target.with_suffix(ext))
        return target

    def processed_path(self, rid):
        return self.root / "processed_native_pchip_v1" / (rid + ".npz")

    def prepare_record(self, row):
        target = self.processed_path(row.rid)
        if target.exists(): return target
        import wfdb
        base = self.raw_base(row)
        record = wfdb.rdrecord(str(base))
        lead_names = [str(s).upper() for s in record.sig_name]
        if set(lead_names) != set(LEADS) or int(record.fs) != row.native_fs: raise ValueError(f"Unexpected header for {row.rid}")
        x = record.p_signal[:, [lead_names.index(lead) for lead in LEADS]].T.astype(np.float32)
        x = preprocess_signal(x, float(record.fs))
        beats = np.empty(0, np.int64); abnormal = np.empty(0, np.int8); unknown = []
        if row.domain == "INCART":
            annotation = wfdb.rdann(str(base), "atr")
            valid = np.array([s in AAMI for s in annotation.symbol])
            beats = np.rint(annotation.sample[valid] * FS / record.fs).astype(np.int64)
            abnormal = np.array([AAMI[s] != "N" for s in annotation.symbol if s in AAMI], np.int8)
            unknown = sorted(set(annotation.symbol) - set(AAMI))
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".partial")
        with temp.open("wb") as f: np.savez(f, x=x, beats=beats, abnormal=abnormal)
        os.replace(temp, target)
        write_json(target.with_suffix(".json"), dict(rid=row.rid, native_fs=record.fs, n_samples=int(x.shape[1]), leads=LEADS,
            source_hashes={e: file_hash(base.with_suffix(e)) for e in ([".hea", ".dat", ".atr"] if row.domain == "INCART" else [".hea", ".dat"])},
            ignored_annotation_symbols=unknown, processed_sha256=file_hash(target)))
        return target

    def signal(self, row):
        path = self.prepare_record(row)
        with np.load(path, allow_pickle=False) as z: return z["x"], z["beats"], z["abnormal"]

    def prepare(self, scopes=("train", "val", "test")):
        from tqdm.auto import tqdm
        frames = [self.select_records(s) for s in scopes]
        unique = pd.concat(frames).drop_duplicates("rid")
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for _ in tqdm(pool.map(self.prepare_record, unique.itertuples(index=False)), total=len(unique), desc="Prepare recordings"): pass
        for scope in scopes: self.pack(scope)
        return read_json(self.manifest_dir / "split_audit.json")

    def pack(self, scope, fast_root=None):
        """Build D2 contiguously on disk; dense windows are never materialized."""
        from tqdm.auto import tqdm
        directory = self.root / "d2" / self.data_id / scope
        directory.mkdir(parents=True, exist_ok=True)
        done = directory / "complete.json"
        if not done.exists():
            rows, parts, counts = self.select_records(scope), [], []
            for row in tqdm(rows.itertuples(index=False), total=len(rows), desc=f"Index {scope}"):
                x, beats, abnormal = self.signal(row)
                starts = np.arange(0, x.shape[1] - WIN + 1, STRIDE, dtype=np.int64)
                if row.domain == "INCART":
                    total, ab = beat_counts(beats, abnormal, starts)
                    y = labels_from_counts(total, ab, 0.05); keep = y >= 0
                else:
                    total = np.zeros(len(starts), np.int32); ab = total.copy()
                    y = np.full(len(starts), row.label, np.int8); keep = y >= 0
                parts.append(pd.DataFrame(dict(rid=row.rid, patient=row.patient, domain=row.domain,
                    client=row.client, start=starts[keep], total=total[keep], abnormal=ab[keep], y=y[keep])))
                counts.append(dict(rid=row.rid, candidates=len(starts), eligible=int(keep.sum())))
            index = pd.concat(parts, ignore_index=True)
            if index.empty: raise ValueError("No eligible D2 windows")
            tmp = directory / "X.partial.npy"
            X = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(index), 12, WIN))
            lookup = {r.rid: r for r in rows.itertuples(index=False)}
            for rid, group in tqdm(index.groupby("rid", sort=False), desc=f"Write {scope}"):
                x, _, _ = self.signal(lookup[rid])
                for offset in range(0, len(group), 256):
                    chunk = group.iloc[offset:offset+256]
                    X[chunk.index.to_numpy()] = np.stack([x[:, int(s):int(s)+WIN] for s in chunk.start])
            X.flush(); del X; os.replace(tmp, directory / "X.npy")
            index.to_csv(directory / "index.csv", index=False)
            pd.DataFrame(counts).to_csv(directory / "coverage.csv", index=False)
            write_json(done, dict(data_id=self.data_id, scope=scope, n=len(index),
                x_sha256=file_hash(directory / "X.npy"), index_sha256=file_hash(directory / "index.csv")))
        metadata = read_json(done)
        if metadata["data_id"] != self.data_id: raise ValueError("Wrong data provenance")
        if fast_root:
            local = Path(fast_root) / self.data_id / scope; local.mkdir(parents=True, exist_ok=True)
            if not (local / "complete.json").exists() or read_json(local / "complete.json") != metadata:
                for name in ("X.npy", "index.csv", "coverage.csv", "complete.json"): shutil.copy2(directory / name, local / name)
            directory = local
        return Pack(directory)

class Pack:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.X = np.load(self.directory / "X.npy", mmap_mode="r", allow_pickle=False)
        self.index = pd.read_csv(self.directory / "index.csv", dtype={"rid": str, "patient": str, "domain": str})
        self.metadata = read_json(self.directory / "complete.json")
        if len(self.X) != len(self.index): raise ValueError("Array/index mismatch")

    def view(self, mask=None, tau=0.05):
        indices = np.arange(len(self.index)) if mask is None else np.flatnonzero(np.asarray(mask))
        return View(self, indices, tau)

class View:
    def __init__(self, pack, indices, tau=0.05):
        self.pack = pack; self.indices = np.asarray(indices, np.int64); self.tau = tau
        self.frame = pack.index.iloc[self.indices].reset_index(drop=True)
        self.y = self.frame.y.to_numpy(np.int8).copy()
        inc = self.frame.domain.to_numpy() == "INCART"
        self.y[inc] = labels_from_counts(self.frame.total.to_numpy()[inc], self.frame.abnormal.to_numpy()[inc], tau)
        if len(self.y) == 0 or (self.y < 0).any(): raise ValueError("Empty view or invalid labels")

    def __len__(self): return len(self.indices)
    def subset(self, mask): return View(self.pack, self.indices[np.asarray(mask)], tau=self.tau)
