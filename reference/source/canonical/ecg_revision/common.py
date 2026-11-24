from __future__ import annotations
import hashlib
import json
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np

VERSION = "ecg-revision-2026-09-14-v2"

def clean_json(x):
    if isinstance(x, dict): return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray): return clean_json(x.tolist())
    if isinstance(x, np.generic): return clean_json(x.item())
    if isinstance(x, Path): return str(x)
    if isinstance(x, float) and not np.isfinite(x): return None
    return x

def fingerprint(x):
    return hashlib.sha256(json.dumps(clean_json(x), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def file_hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""): h.update(block)
    return h.hexdigest()

def code_hash():
    return fingerprint({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})

def write_json(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".partial")
    tmp.write_text(json.dumps(clean_json(obj), indent=2, sort_keys=True, allow_nan=False))
    os.replace(tmp, p)

def read_json(p): return json.loads(Path(p).read_text())

def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def clock():
    import torch
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return time.perf_counter()

def environment():
    import importlib.metadata as im
    import torch
    versions = {}
    for package in ("torch", "numpy", "scipy", "pandas", "scikit-learn", "wfdb"):
        try: versions[package] = im.version(package)
        except im.PackageNotFoundError: versions[package] = "missing"
    return dict(python=sys.version, platform=platform.platform(), packages=versions,
                cpu_count=os.cpu_count(), torch_threads=torch.get_num_threads(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                cuda=torch.version.cuda, code_hash=code_hash(), version=VERSION)

@dataclass(frozen=True)
class Config:
    method: str = "fedavg"
    seed: int = 0
    rounds: int = 20
    local_epochs: int = 1
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    clip: float = 1.0
    mu: float = 1e-3
    server_lr: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    server_momentum: float = 0.9
    weighting: str = "equal"
    architecture: str = "uploaded_gn"
    # sampled_epoch = exactly n_k balanced draws with replacement per epoch.
    # full_pass = one shuffled traversal without replacement per epoch.
    # fixed_steps = legacy-style batch budget; never described as a full epoch.
    schedule: str = "sampled_epoch"
    steps_per_epoch: int = 50
    imbalance: str = "sampler_only"
    selection: str = "macro_domain_auc"
    train_tau: float = 0.05
    unknown_policy: str = "exclude"
    split_policy: str = "patient42"
    phase: str = "report"

    def validate(self):
        assert self.method in ("fedavg", "fedavgm", "fedprox", "fedadam", "pooled", "ptb_only", "incart_only", "transfer")
        assert self.weighting in ("equal", "samples")
        assert self.schedule in ("sampled_epoch", "full_pass", "fixed_steps")
        assert self.imbalance in ("sampler_only", "pos_weight_only", "none", "legacy_double")
        assert self.selection in ("macro_domain_auc", "pooled_auc", "ptb_auc", "incart_auc")
        assert self.unknown_policy in ("exclude", "legacy_positive")
        assert self.split_policy in ("patient42", "recording42")
        assert self.architecture in ("uploaded_gn", "manuscript_bn")
        assert self.rounds > 0 and self.local_epochs > 0 and self.batch_size >= 2
        assert self.lr > 0 and self.server_lr > 0 and self.epsilon > 0 and self.mu >= 0
        if self.schedule == "full_pass":
            assert self.imbalance not in ("sampler_only", "legacy_double"), "Full traversal cannot use a replacement sampler."
        if self.schedule == "sampled_epoch":
            assert self.imbalance in ("sampler_only", "legacy_double"), "Use full_pass for a nonreplacement epoch."
        if self.phase in ("report", "tuning"):
            assert self.split_policy == "patient42", "Legacy recording splits are diagnostic only."
        return self

    def as_dict(self): return asdict(self)

    def canonical(self):
        d = self.as_dict()
        # Irrelevant parameters cannot create duplicate run fingerprints.
        if self.method != "fedprox":
            d["mu"] = 0.0
        # server_lr is meaningful for FedAdam and FedAvgM.
        if self.method not in ("fedadam", "fedavgm"):
            d.pop("server_lr")
        if self.method != "fedadam":
            for k in ("beta1", "beta2", "epsilon"):
                d.pop(k)
        if self.method != "fedavgm":
            d.pop("server_momentum")
        if self.schedule != "fixed_steps":
            d.pop("steps_per_epoch")
        return d
