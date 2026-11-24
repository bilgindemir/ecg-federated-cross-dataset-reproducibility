from __future__ import annotations
import os
import copy
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from .common import Config, clock, code_hash, environment, fingerprint, read_json, seed_all, write_json
from .metrics import scores

class CNN(nn.Module):
    def __init__(self, architecture="uploaded_gn"):
        super().__init__()
        if architecture == "uploaded_gn":
            self.net = nn.Sequential(
                nn.Conv1d(12, 64, 7, padding=3), nn.GroupNorm(8, 64), nn.ReLU(),
                nn.Conv1d(64, 64, 5, padding=2), nn.GroupNorm(8, 64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.ReLU(),
                nn.Conv1d(128, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
            self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64, 1))
        elif architecture == "manuscript_bn":
            # Explicit reconstruction of Table 4, not a recovered old model.
            layers = []; incoming = 12
            for outgoing, kernel in [(32, 7), (64, 5), (128, 3)]:
                layers += [nn.Conv1d(incoming, outgoing, kernel, padding=kernel//2), nn.BatchNorm1d(outgoing), nn.ReLU(), nn.MaxPool1d(2)]
                incoming = outgoing
            self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1)); self.head = nn.Linear(128, 1)
        else: raise ValueError("Unknown architecture")

    def forward(self, x): return self.head(self.net(x).squeeze(-1)).squeeze(-1)

def device_name(): return "cuda" if torch.cuda.is_available() else "cpu"
def state_of(model): return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def checkpoint_safe(value):
    """Convert checkpoint metadata to weights_only-safe primitives/tensors."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.array(value, copy=True))
    if isinstance(value, dict):
        return {checkpoint_safe(k): checkpoint_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [checkpoint_safe(v) for v in value]
    if isinstance(value, tuple):
        return tuple(checkpoint_safe(v) for v in value)
    if isinstance(value, set):
        return sorted(checkpoint_safe(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    return value

def save_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    torch.save(checkpoint_safe(value), tmp)
    os.replace(tmp, path)

def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Bare legacy state dict: use the explicit legacy import workflow first")
    model = CNN(checkpoint["config"]["architecture"])
    model.load_state_dict(checkpoint["state"], strict=True)
    return checkpoint

def aggregate_states(states, counts, weighting, old_state, cfg, moments=None):
    if len(states) != len(counts) or min(counts) <= 0: raise ValueError("Invalid client counts")
    alpha = np.asarray(counts if weighting == "samples" else np.ones(len(counts)), float)
    alpha /= alpha.sum()
    avg = {}
    for key, old in old_state.items():
        if old.is_floating_point():
            avg[key] = sum((st[key] * float(w) for st, w in zip(states, alpha)), torch.zeros_like(old))
        else:
            # BatchNorm counters are buffers, not floating model parameters.
            avg[key] = torch.stack([st[key] for st in states]).max(dim=0).values
    moments = copy.deepcopy(moments) if moments is not None else {"t": 0, "m": {}, "v": {}, "velocity": {}}
    parameter_names = set(dict(CNN(cfg.architecture).named_parameters()))
    result = {k: v.clone() for k, v in avg.items()}
    moments["t"] += 1
    for key in parameter_names:
        delta = avg[key] - old_state[key]
        if cfg.method == "fedavgm":
            velocity = cfg.server_momentum * moments["velocity"].get(key, torch.zeros_like(delta)) + delta
            moments["velocity"][key] = velocity
            result[key] = old_state[key] + cfg.server_lr * velocity
        elif cfg.method == "fedadam":
            # Bias-corrected Adam on global-minus-client-average pseudo-gradients,
            # matching the uploaded convention. Documented FedOpt variant.
            gradient = -delta
            m = cfg.beta1 * moments["m"].get(key, torch.zeros_like(delta)) + (1-cfg.beta1)*gradient
            v = cfg.beta2 * moments["v"].get(key, torch.zeros_like(delta)) + (1-cfg.beta2)*gradient.square()
            moments["m"][key], moments["v"][key] = m, v
            mh = m / (1-cfg.beta1**moments["t"]); vh = v / (1-cfg.beta2**moments["t"])
            result[key] = old_state[key] - cfg.server_lr * mh / (vh.sqrt() + cfg.epsilon)
    if not all(torch.isfinite(v).all() for v in result.values() if v.is_floating_point()):
        raise FloatingPointError("Nonfinite server update")
    return result, moments, alpha.tolist()

class TorchView(torch.utils.data.Dataset):
    def __init__(self, view): self.view = view
    def __len__(self): return len(self.view)
    def __getitem__(self, position):
        x = np.array(self.view.pack.X[self.view.indices[position]], dtype=np.float32, copy=True)
        return torch.from_numpy(x), torch.tensor(float(self.view.y[position])), int(position)

def sampling_indices(view, cfg, generator):
    n = len(view)
    draw_count = cfg.steps_per_epoch * cfg.batch_size if cfg.schedule == "fixed_steps" else n
    balanced = cfg.imbalance in ("sampler_only", "legacy_double")
    if balanced:
        count = np.bincount(view.y.astype(int), minlength=2)
        weights = torch.tensor(1 / np.maximum(count[view.y], 1), dtype=torch.float64)
        return torch.multinomial(weights, draw_count, replacement=True, generator=generator).tolist()
    if cfg.schedule == "fixed_steps":
        return torch.randint(n, (draw_count,), generator=generator).tolist()
    return torch.randperm(n, generator=generator).tolist()

def local_update(global_state, view, cfg, round_number, client_id, device=None, optimizer_state=None, return_optimizer=False):
    device = device or device_name()
    local_seed = int(cfg.seed * 1000003 + round_number * 1009 + client_id * 97) % (2**31-1)
    seed_all(local_seed)
    model = CNN(cfg.architecture).to(device); model.load_state_dict(global_state, strict=True); model.train()
    anchor = {name: p.detach().clone() for name, p in model.named_parameters()} if cfg.method == "fedprox" else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if optimizer_state is not None: optimizer.load_state_dict(optimizer_state)
    pos_weight = None
    if cfg.imbalance in ("pos_weight_only", "legacy_double"):
        count = np.bincount(view.y.astype(int), minlength=2)
        pos_weight = torch.tensor(float(count[0]/max(count[1], 1)), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    generator = torch.Generator().manual_seed(local_seed)
    dataset = TorchView(view); seen = np.zeros(len(view), dtype=bool)
    total_loss, examples, updates = 0.0, 0, 0
    started = clock()
    for epoch in range(cfg.local_epochs):
        order = sampling_indices(view, cfg, generator)
        loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, sampler=order, num_workers=0, drop_last=False)
        for xb, yb, positions in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            if anchor is not None:
                prox = sum((p-anchor[name]).square().sum() for name, p in model.named_parameters())
                loss = loss + .5*cfg.mu*prox
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite local loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip, error_if_nonfinite=True); optimizer.step()
            seen[positions.numpy()] = True
            total_loss += float(loss.detach())*len(yb); examples += len(yb); updates += 1
    state = state_of(model); elapsed = clock()-started
    records = sorted(set(view.frame.loc[seen, "rid"].astype(str)))
    stats = dict(client=client_id, n_eligible=len(view), examples=examples, updates=updates,
                 seen_windows=int(seen.sum()), train_s=elapsed, mean_loss=total_loss/max(examples, 1))
    persisted_optimizer = copy.deepcopy(optimizer.state_dict()) if return_optimizer else None
    if persisted_optimizer:
        for entry in persisted_optimizer["state"].values():
            for key, value in entry.items():
                if torch.is_tensor(value): entry[key] = value.detach().cpu()
    del optimizer, model
    if return_optimizer: return state, stats, records, persisted_optimizer
    return state, stats, records

@torch.inference_mode()
def predict(state, view, architecture="uploaded_gn", batch_size=256, device=None):
    device = device or device_name(); model = CNN(architecture).to(device)
    model.load_state_dict(state, strict=True); model.eval()
    out = np.empty(len(view), np.float32)
    for start in range(0, len(view), batch_size):
        ids = view.indices[start:start+batch_size]
        xb = torch.from_numpy(np.array(view.pack.X[ids], copy=True)).to(device)
        out[start:start+len(ids)] = model(xb).sigmoid().cpu().numpy()
    return out

def validation_score(state, val_view, cfg):
    p = predict(state, val_view, cfg.architecture, cfg.batch_size)
    out = {"pooled_auc": scores(val_view.y, p)["roc_auc"]}
    for name, label in [("ptb_auc", "PTB"), ("incart_auc", "INCART")]:
        mask = val_view.frame.domain.to_numpy() == label
        out[name] = scores(val_view.y[mask], p[mask])["roc_auc"] if mask.any() else np.nan
    out["macro_domain_auc"] = float((out["ptb_auc"]+out["incart_auc"])/2)
    score = out[cfg.selection]
    if not np.isfinite(score): raise ValueError("Selection metric undefined: inspect class coverage; do not choose a split using test performance")
    return float(score), out

def run_id(cfg, data_id):
    return fingerprint(dict(config=cfg.canonical(), data_id=data_id, code_hash=code_hash()))[:20]

def run(cfg, store, output_root, fast_root=None, stop_after=None, initial_checkpoint=None):
    """Round-boundary restart; test data are not loaded or inspected during fit."""
    cfg.validate()
    if cfg.split_policy != store.split_policy: raise ValueError("Configuration/split mismatch")
    data_id = store.data_id
    key = run_id(cfg, data_id)
    if initial_checkpoint:
        from .common import file_hash
        key = fingerprint([key, file_hash(initial_checkpoint)])[:20]
    path = Path(output_root) / "runs" / key; path.mkdir(parents=True, exist_ok=True)
    complete = path / "complete.json"
    if complete.exists(): return path
    train = store.pack("train", fast_root); val = store.pack("val", fast_root).view(tau=cfg.train_tau)
    if cfg.method in ("ptb_only", "incart_only", "transfer"):
        domain = "PTB" if cfg.method == "ptb_only" else "INCART"
        views = [train.view(train.index.domain.to_numpy() == domain, tau=cfg.train_tau)]
    elif cfg.method == "pooled": views = [train.view(tau=cfg.train_tau)]
    else: views = [train.view(train.index.client.to_numpy() == c, tau=cfg.train_tau) for c in range(4)]
    last = path / "last.pt"
    if last.exists():
        payload = torch.load(last, map_location="cpu", weights_only=True)
        if payload["code_hash"] != code_hash() or payload["data_id"] != data_id: raise ValueError("Resume provenance changed")
    else:
        seed_all(cfg.seed); initial = state_of(CNN(cfg.architecture))
        if initial_checkpoint:
            source = load_checkpoint(initial_checkpoint)
            if source["data_id"] != data_id or source["config"]["architecture"] != cfg.architecture: raise ValueError("Transfer source mismatch")
            if cfg.method != "transfer" or source["config"]["method"] != "ptb_only": raise ValueError("Transfer must start from PTB-only pretraining")
            initial = source["state"]
        payload = dict(state=initial, moments=None, completed_round=0, best_score=-float("inf"),
                       best_round=0, best_state=None, history=[], seen_records=[], best_seen_records=[],
                       config=cfg.as_dict(), data_id=data_id, code_hash=code_hash(), central_optimizer=None)
        if initial_checkpoint:
            payload["seen_records"] = source.get("seen_records", [])
        write_json(path / "config.json", dict(**cfg.as_dict(), data_id=data_id, run_id=key, environment=environment(),
            source_checkpoint=str(initial_checkpoint) if initial_checkpoint else None))
    for round_number in range(payload["completed_round"]+1, cfg.rounds+1):
        started = clock(); states, statistics, seen = [], [], set(payload["seen_records"])
        for client_id, view in enumerate(views):
            if len(views) == 1:
                state, stats, records, central_optimizer = local_update(payload["state"], view, cfg, round_number, client_id,
                    optimizer_state=payload.get("central_optimizer"), return_optimizer=True)
                payload["central_optimizer"] = central_optimizer
            else:
                state, stats, records = local_update(payload["state"], view, cfg, round_number, client_id)
            states.append(state); statistics.append(stats); seen.update(records)
        aggregate_start = clock()
        if len(states) == 1:
            new_state, moments, alpha = states[0], None, [1.0]
        else:
            new_state, moments, alpha = aggregate_states(states, [len(v) for v in views], cfg.weighting, payload["state"], cfg, payload["moments"])
        aggregate_end = clock()
        metric, validation = validation_score(new_state, val, cfg)
        finished = clock()
        history = dict(round=round_number, selection_score=metric, **validation, client_stats=statistics, alpha=alpha,
                       training_aggregation_s=aggregate_end-started, aggregation_s=aggregate_end-aggregate_start,
                       validation_s=finished-aggregate_end, round_without_checkpoint_io_s=finished-started)
        payload["history"].append(history)
        payload.update(state=new_state, moments=moments, completed_round=round_number, seen_records=sorted(seen))
        if metric > payload["best_score"]:
            payload.update(best_score=metric, best_round=round_number, best_state=copy.deepcopy(new_state), best_seen_records=sorted(seen))
            save_torch(path / "best.pt", dict(state=payload["best_state"], config=cfg.as_dict(), data_id=data_id,
                best_round=round_number, selection_score=metric, seen_records=sorted(seen), code_hash=code_hash(), run_id=key,
                origin="clean_patient_disjoint" if cfg.split_policy == "patient42" else "legacy_diagnostic"))
        save_torch(last, payload)
        write_json(path / "history.json", payload["history"])
        print(f"{key} {cfg.phase} {cfg.method} seed={cfg.seed} E={cfg.local_epochs} {cfg.weighting} round={round_number}/{cfg.rounds} val={metric:.4f}", flush=True)
        if stop_after and round_number >= stop_after and round_number < cfg.rounds: return path
    write_json(complete, dict(run_id=key, data_id=data_id, best_round=payload["best_round"],
        best_score=payload["best_score"], total_rounds=cfg.rounds, phase=cfg.phase, config=cfg.as_dict()))
    return path
