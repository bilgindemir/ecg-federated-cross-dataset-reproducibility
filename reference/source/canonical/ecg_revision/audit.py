from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .common import Config, file_hash, fingerprint, write_json
from .data import Store, incart_splits, parse_incart_patients, split_audit
from .training import CNN, save_torch

def audit_legacy_drive(legacy_root, store, output_root):
    """Read-only recovery audit. Does not execute old notebooks or edit caches."""
    legacy_root = Path(legacy_root); out = Path(output_root); out.mkdir(parents=True, exist_ok=True)
    inventory, tau_candidates, label_checks = [], [], []
    for subdirectory in ("checkpoints", "results", "cache/metadata"):
        directory = legacy_root / subdirectory
        if not directory.exists(): continue
        for p in sorted(directory.rglob("*")):
            if p.suffix.lower() not in (".pt", ".pth", ".csv", ".json", ".npz"): continue
            item = dict(path=str(p), bytes=p.stat().st_size)
            if p.suffix in (".pt", ".pth"):
                item["sha256"] = file_hash(p)
                try:
                    obj = torch.load(p, map_location="cpu", weights_only=True)
                    state = obj.get("state", obj.get("state_dict", obj)) if isinstance(obj, dict) else None
                    if not isinstance(state, dict): raise ValueError("Unsupported checkpoint layout")
                    model = CNN("uploaded_gn"); model.load_state_dict(state, strict=True)
                    item["architecture"] = "uploaded_gn_compatible"
                    item["embedded_provenance"] = bool(isinstance(obj, dict) and "config" in obj and "data_id" in obj)
                except Exception as exc: item["inspection_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            elif p.suffix == ".csv":
                try:
                    frame = pd.read_csv(p)
                    if any("tau" in c.lower() for c in frame.columns) or "tau" in p.name.lower():
                        tau_candidates.append(dict(path=str(p), columns=list(frame.columns), rows=len(frame),
                            verdict="Candidate only: confirm actual relabeling, checkpoint/data provenance, all requested taus and absolute metrics"))
                except Exception: pass
            inventory.append(item)
    records = store.records
    # Official PTB 10s/200Hz/1024/stride512 gives 2 complete windows per record.
    primary = legacy_root / "data_200hz_primary_stride512"
    for name, subset in [(f"client{c}", records[(records.domain=="PTB") & (records.client==c)]) for c in range(3)] + [
        ("val_ptb", records[(records.domain=="PTB") & (records.split=="val")]),
        ("test_ptb", records[(records.domain=="PTB") & (records.split=="test")])]:
        p = primary / f"y_{name}.npy"
        if not p.exists(): continue
        old = np.load(p, allow_pickle=False)
        expected = np.repeat(subset.sort_values("rid").label.to_numpy(np.int8), 2)
        item = dict(path=str(p), n_cached=len(old), n_expected=len(expected),
                    order_assumption="official ecg_id/filename order as in supplied primary builder")
        if len(old) == len(expected): item["different_labels"] = int(np.count_nonzero(old != expected))
        else: item["verdict"] = "Cannot compare: different windowing/cohort/order requires original record index"
        label_checks.append(item)
    mapping = parse_incart_patients((store.meta / "incart_patients.txt").read_text())
    assignment = incart_splits(mapping, "recording42")
    inc = pd.DataFrame([dict(rid=r, patient=f"INC_{p}", domain="INCART", split=assignment[r]) for r,p in mapping.items()])
    report = dict(legacy_root_exists=legacy_root.exists(), inventory=inventory, tau_candidates=tau_candidates,
        ptb_primary_label_checks=label_checks, reconstructed_recording42=split_audit(inc),
        new_data_id=store.data_id, automatic_reuse=False,
        note="Architecture compatibility is not evidence of training labels, seed, split, selection rule or complete five-seed provenance.")
    write_json(out / "legacy_audit.json", report)
    pd.DataFrame(inventory).to_csv(out / "legacy_inventory.csv", index=False)
    return report

def import_legacy_diagnostic(checkpoint_path, method, legacy_store, output_root, seed=None, provenance_note=""):
    """Label-corrected diagnostics on the ORIGINAL recording split only.

    Does not repair training leakage or turn old weights into patient-held-out
    models. Unknown seeds remain null. No automated manuscript reuse.
    """
    if legacy_store.split_policy != "recording42": raise ValueError("Use Store(..., split_policy='recording42') for recovered old models")
    if not provenance_note.strip(): raise ValueError("Describe known/unknown training provenance explicitly")
    if method not in ("fedavg", "fedavgm", "fedprox", "fedadam"): raise ValueError("Invalid method")
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = obj.get("state", obj.get("state_dict", obj))
    CNN("uploaded_gn").load_state_dict(state, strict=True)
    cfg = Config(method=method, split_policy="recording42", phase="legacy_diagnostic",
                 schedule="fixed_steps", imbalance="legacy_double", selection="pooled_auc",
                 unknown_policy=legacy_store.unknown_policy).as_dict()
    cfg["seed"] = seed
    key = fingerprint([file_hash(checkpoint_path), legacy_store.data_id, method])[:20]
    out = Path(output_root) / "legacy_imports" / key; out.mkdir(parents=True, exist_ok=True)
    save_torch(out / "best.pt", dict(state=state, config=cfg, data_id=legacy_store.data_id, run_id=key,
        origin="legacy_diagnostic_training_provenance_unverified", seen_records=[], best_round=None,
        provenance_note=provenance_note, configuration_status="inferred from supplied code except verified architecture; not a recovered training log"))
    write_json(out / "provenance.json", dict(source=str(checkpoint_path), source_sha256=file_hash(checkpoint_path),
        inferred_config=cfg, known_seed=seed, provenance_note=provenance_note,
        warning="Do not mix with new patient-disjoint results; no claim of unseen-patient generalization"))
    return out / "best.pt"
