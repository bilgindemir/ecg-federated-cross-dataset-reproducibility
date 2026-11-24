from __future__ import annotations
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .common import clock, code_hash, file_hash, fingerprint, read_json, write_json
from .data import MIN_BEATS, WIN, STRIDE, beat_counts, labels_from_counts
from .metrics import METRICS, scores, cluster_confusions, cluster_ci, paired_cluster_delta
from .training import CNN, device_name, load_checkpoint, predict

def eval_directory(checkpoint_path, store, output_root):
    ckpt = load_checkpoint(checkpoint_path)
    if ckpt["data_id"] != store.data_id:
        raise ValueError("Checkpoint/data manifest mismatch. Never evaluate legacy models on a new patient split as if held out.")
    key = fingerprint([file_hash(checkpoint_path), store.data_id, code_hash()])[:20]
    path = Path(output_root) / "evaluations" / key; path.mkdir(parents=True, exist_ok=True)
    write_json(path / "provenance.json", dict(checkpoint=str(checkpoint_path), checkpoint_sha256=file_hash(checkpoint_path),
        data_id=store.data_id, origin=ckpt["origin"], config=ckpt["config"], best_round=ckpt.get("best_round"), code_hash=code_hash()))
    return ckpt, path

def tags(ckpt, store):
    cfg = ckpt["config"]
    return {**{k: cfg.get(k) for k in ["method", "seed", "weighting", "local_epochs", "schedule", "imbalance", "selection", "phase", "train_tau", "architecture", "rounds", "batch_size", "lr", "mu", "server_lr"]},
            "split_policy": store.split_policy, "data_id": store.data_id, "origin": ckpt["origin"], "run_id": ckpt.get("run_id")}

def evaluate_d2(checkpoint_path, store, output_root, taus=(.01, .05, .10), bootstraps=1000, fast_root=None):
    ckpt, path = eval_directory(checkpoint_path, store, output_root)
    pack = store.pack("test", fast_root); view = pack.view(tau=.05)
    pred_path = path / "d2_probabilities.npy"
    if pred_path.exists(): p = np.load(pred_path)
    else:
        p = predict(ckpt["state"], view, ckpt["config"]["architecture"])
        temporary = pred_path.with_suffix(".partial")
        with temporary.open("wb") as stream: np.save(stream, p)
        os.replace(temporary, pred_path)
    if len(p) != len(view): raise ValueError("Cached prediction length mismatch")
    output, per_record, changes = [], [], []
    for domain in ("PTB", "INCART"):
        domain_mask = view.frame.domain.to_numpy() == domain
        if not domain_mask.any(): continue
        for tau in taus if domain == "INCART" else [.05]:
            y = pack.view(tau=tau).y[domain_mask]; prob = p[domain_mask]
            frame = view.frame[domain_mask].reset_index(drop=True)
            conf = cluster_confusions(y, prob, frame.patient)
            row = dict(**tags(ckpt, store), domain=domain, representation="D2", eval_tau=tau,
                scope="test", n_records=frame.rid.nunique(), n_patients=frame.patient.nunique(),
                **scores(y, prob), **cluster_ci(conf, bootstraps))
            output.append(row)
            conf.to_csv(path / f"conf_D2_{domain}_{tau:g}.csv")
            for rid, rec in frame.groupby("rid", sort=False):
                idx = rec.index.to_numpy()
                per_record.append(dict(domain=domain, rid=rid, patient=rec.patient.iloc[0], eval_tau=tau,
                                       **scores(y[idx], prob[idx])))
            if domain == "INCART":
                baseline = pack.view(tau=.05).y[domain_mask]
                changes.append(dict(eval_tau=tau, changed_vs_005=int(np.count_nonzero(y != baseline)),
                    n=len(y), positives=int(y.sum()), prevalence=float(y.mean()),
                    distinct_beat_counts=sorted(map(int, frame.total.unique())),
                    interpretation="fixed-model test-label sensitivity, not training-label sensitivity"))
    pd.DataFrame(output).to_csv(path / "metrics_D2.csv", index=False)
    pd.DataFrame(per_record).to_csv(path / "per_record_D2.csv", index=False)
    write_json(path / "tau_label_changes.json", changes)
    return pd.DataFrame(output)

def dense_starts(length, rid, cap=None):
    candidates = np.arange(max(0, length-WIN+1), dtype=np.int64)
    if cap is None or len(candidates) <= cap: return candidates
    d2 = np.arange(0, length-WIN+1, STRIDE, dtype=np.int64)
    if cap < len(d2): raise ValueError("Dense sample cap must include every D2 position")
    rng = np.random.default_rng(int(fingerprint(rid)[:8], 16))
    pool = np.setdiff1d(candidates, d2, assume_unique=True)
    extra = rng.choice(pool, size=cap-len(d2), replace=False)
    return np.sort(np.r_[d2, extra])

@torch.inference_mode()
def evaluate_dense(checkpoint_path, store, output_root, cap_per_incart_record=None,
                   bootstraps=1000, batch_size=256, taus=(.01, .05, .10), record_limit=None):
    """All test records by default. cap=None is exhaustive D1, not a subsample.

    Only predictions and beat counts are persisted. A completed record is the
    restart unit. record_limit is a benchmark/debug option, marked incomplete.
    """
    from tqdm.auto import tqdm
    ckpt, directory = eval_directory(checkpoint_path, store, output_root)
    name = "D1_full" if cap_per_incart_record is None else f"D1_sampled_cap{cap_per_incart_record}"
    if record_limit is not None: name += f"_DEBUG_{record_limit}_records_per_domain"
    pred_dir = directory / name; pred_dir.mkdir(parents=True, exist_ok=True)
    all_records = store.select_records("test")
    records = all_records if record_limit is None else all_records.groupby("domain", group_keys=False).head(record_limit)
    device = device_name(); model = CNN(ckpt["config"]["architecture"]).to(device)
    model.load_state_dict(ckpt["state"], strict=True); model.eval()
    for row in tqdm(records.itertuples(index=False), total=len(records), desc=name):
        target = pred_dir / (row.rid + ".npz")
        if target.exists(): continue
        x, beats, abnormal = store.signal(row)
        starts = dense_starts(x.shape[1], row.rid, cap_per_incart_record if row.domain == "INCART" else None)
        candidates = max(0, x.shape[1]-WIN+1)
        total, ab = beat_counts(beats, abnormal, starts) if row.domain == "INCART" else (np.zeros(len(starts), np.int32), np.zeros(len(starts), np.int32))
        eligible = total >= MIN_BEATS if row.domain == "INCART" else np.ones(len(starts), bool)
        starts, total, ab = starts[eligible], total[eligible], ab[eligible]
        p = np.empty(len(starts), np.float32); started = clock()
        for i in range(0, len(starts), batch_size):
            batch = starts[i:i+batch_size]
            xb = torch.from_numpy(np.stack([x[:, int(s):int(s)+WIN] for s in batch])).to(device)
            p[i:i+len(batch)] = model(xb).sigmoid().cpu().numpy()
        elapsed = clock()-started
        temp = target.with_suffix(".partial")
        with temp.open("wb") as stream:
            np.savez(stream, p=p, start=starts, total=total, abnormal=ab,
                     candidates=np.int64(candidates), sampled_before_filter=np.int64(len(eligible)),
                     elapsed_s=np.float64(elapsed))
        os.replace(temp, target)
    del model
    result, record_rows, coverage = [], [], []
    for domain, selected in records.groupby("domain"):
        for tau in taus if domain == "INCART" else [.05]:
            labels, probabilities, group_conf = [], [], {}
            for row in selected.itertuples(index=False):
                with np.load(pred_dir / (row.rid + ".npz"), allow_pickle=False) as z:
                    p = z["p"]; y = labels_from_counts(z["total"], z["abnormal"], tau) if domain == "INCART" else np.full(len(p), row.label, np.int8)
                    if tau == .05:
                        coverage.append(dict(rid=row.rid, domain=domain, patient=row.patient, candidates=int(z["candidates"]),
                            sampled_before_filter=int(z["sampled_before_filter"]), eligible=len(p), inference_s=float(z["elapsed_s"])))
                if not len(y): continue
                m = scores(y, p)
                labels.append(y); probabilities.append(p)
                group_conf[row.patient] = group_conf.get(row.patient, np.zeros(4)) + np.array([m[k] for k in ["tn", "fp", "fn", "tp"]])
                record_rows.append(dict(rid=row.rid, patient=row.patient, domain=domain, eval_tau=tau, **m))
            if not labels: raise ValueError(f"No eligible {domain} windows")
            conf = pd.DataFrame.from_dict(group_conf, orient="index", columns=[0, 1, 2, 3]); conf.index.name="group"
            conf.to_csv(directory / f"conf_{name}_{domain}_{tau:g}.csv")
            m = scores(np.concatenate(labels), np.concatenate(probabilities))
            result.append(dict(**tags(ckpt, store), representation=name, eval_tau=tau, domain=domain,
                scope="test", n_records=sum(1 for r in record_rows if r["domain"]==domain and r["eval_tau"]==tau),
                n_patients=len(group_conf), all_test_records=record_limit is None,
                **m, **cluster_ci(conf, bootstraps)))
    pd.DataFrame(result).to_csv(directory / f"metrics_{name}.csv", index=False)
    pd.DataFrame(record_rows).to_csv(directory / f"per_record_{name}.csv", index=False)
    pd.DataFrame(coverage).to_csv(pred_dir / "coverage.csv", index=False)
    write_json(pred_dir / "complete.json", dict(data_id=store.data_id, source_records=len(records),
        expected_test_records=len(all_records), exhaustive=cap_per_incart_record is None and record_limit is None))
    d2_path = directory / "metrics_D2.csv"
    if d2_path.exists():
        d2 = pd.read_csv(d2_path); deltas = []
        for row in result:
            domain, tau = row["domain"], row["eval_tau"]
            baseline = d2[(d2.domain == domain) & np.isclose(d2.eval_tau, tau)]
            if len(baseline) != 1 or record_limit is not None: continue
            a = pd.read_csv(directory / f"conf_D2_{domain}_{tau:g}.csv", index_col=0)
            b = pd.read_csv(directory / f"conf_{name}_{domain}_{tau:g}.csv", index_col=0)
            delta = dict(**tags(ckpt, store), domain=domain, eval_tau=tau, representation=name,
                **{m+"_delta": row[m]-float(baseline.iloc[0][m]) for m in METRICS},
                **paired_cluster_delta(a, b, bootstraps))
            deltas.append(delta)
        pd.DataFrame(deltas).to_csv(directory / f"paired_delta_{name}_minus_D2.csv", index=False)
    return pd.DataFrame(result)

@torch.inference_mode()
def benchmark_dense(checkpoint_path, store, output_root, batch_size=256, batches=8):
    ckpt = load_checkpoint(checkpoint_path); device = device_name()
    model = CNN(ckpt["config"]["architecture"]).to(device); model.load_state_dict(ckpt["state"]); model.eval()
    out = []
    for domain, frame in store.select_records("test").groupby("domain"):
        row = next(frame.itertuples(index=False)); x, _, _ = store.signal(row)
        maximum = max(1, x.shape[1]-WIN+1)
        starts = np.arange(batch_size) % maximum
        batch = torch.from_numpy(np.stack([x[:, s:s+WIN] for s in starts])).to(device)
        for _ in range(2): model(batch)
        start = clock()
        for _ in range(batches): model(batch)
        elapsed = clock()-start
        speed = batches*batch_size/elapsed
        out.append(dict(domain=domain, windows_per_second=speed, batch_size=batch_size, device=device,
            approximate_candidates_per_model=len(frame)*maximum,
            optimistic_inference_hours_per_model=len(frame)*maximum/speed/3600,
            limitation="Estimate from one recording, warm GPU batches; excludes I/O, CPU metrics and interruption overhead"))
    write_json(Path(output_root) / "dense_benchmark.json", out)
    return pd.DataFrame(out)
