from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, accuracy_score
from .common import read_json, write_json
from .evaluation import eval_directory, tags
from .training import load_checkpoint, predict

def patient_losses(frame, y, probabilities):
    prob = np.clip(np.asarray(probabilities, float), 1e-7, 1-1e-7)
    loss = -(np.asarray(y)*np.log(prob)+(1-np.asarray(y))*np.log1p(-prob))
    df = frame[["patient", "rid", "domain"]].copy()
    df["loss"] = loss; df["y"] = y
    per_record = df.groupby(["domain", "patient", "rid"]).agg(loss=("loss", "mean"), prevalence=("y", "mean")).reset_index()
    # Equal recording influence within a patient (not more weight to long Holters).
    patients = per_record.groupby(["domain", "patient"]).agg(loss=("loss", "mean"), prevalence=("prevalence", "mean"), records=("rid", "size")).reset_index()
    patients["score"] = -patients.loss
    patients["dominant_label"] = (patients.prevalence >= .5).astype(int)
    return patients

def make_attack_split(patients, seed=808):
    """Same domain, same dominant-label mix for members/nonmembers.

    Patient-level calibration/evaluation separation. INCART prevalence is
    continuous; matching this coarse stratum does not eliminate all case-mix
    confounding. Exact mean prevalences are included in the result.
    """
    rng = np.random.default_rng(seed); calibration, evaluation, exclusions = [], [], []
    for label, frame in patients.groupby("dominant_label"):
        a = frame[frame.member==1]; b = frame[frame.member==0]
        n = min(len(a), len(b))
        if n < 4:
            exclusions.append(dict(dominant_label=int(label), members=len(a), nonmembers=len(b), reason="Need >=4 in each membership group for a split"))
            continue
        for f in (a, b):
            chosen = f.iloc[rng.choice(len(f), size=n, replace=False)]
            ncal = n//2
            calibration.append(chosen.iloc[:ncal]); evaluation.append(chosen.iloc[ncal:])
    if not calibration: return None, None, exclusions
    cal, test = pd.concat(calibration, ignore_index=True), pd.concat(evaluation, ignore_index=True)
    assert not set(cal.patient) & set(test.patient)
    return cal, test, exclusions

def choose_threshold(calibration):
    fpr, tpr, thresholds = roc_curve(calibration.member, calibration.score, drop_intermediate=False)
    valid = np.isfinite(thresholds)
    acc = (tpr[valid]+1-fpr[valid])/2
    # Orientation is fixed: lower target loss predicts membership.
    return float(thresholds[valid][np.argmax(acc)])

def attack_scores(test, threshold):
    y = test.member.to_numpy(int); score = test.score.to_numpy(float); pred = score >= threshold
    fpr, tpr, _ = roc_curve(y, score, drop_intermediate=False)
    nnon = int(np.sum(y==0)); nmem = int(np.sum(y==1))
    tpr_fixed = float(pred[y==1].mean()); fpr_fixed = float(pred[y==0].mean())
    return dict(attack_auc=float(roc_auc_score(y, score)), attack_accuracy=float(accuracy_score(y, pred)),
        attack_balanced_accuracy=float(balanced_accuracy_score(y, pred)), calibrated_threshold=float(threshold),
        attack_advantage_at_threshold=tpr_fixed-fpr_fixed, tpr_at_threshold=tpr_fixed, fpr_at_threshold=fpr_fixed,
        tpr_at_fpr_001=float(tpr[fpr<=.01].max()) if nnon>=100 else None,
        low_fpr_note="empirical ROC ordinate; not a certified operating guarantee" if nnon>=100 else "Not reported: fewer than 100 nonmember patients",
        nonmember_patients=nnon, member_patients=nmem, fpr_resolution=1/nnon,
        member_mean_prevalence=float(test.loc[test.member==1,"prevalence"].mean()),
        nonmember_mean_prevalence=float(test.loc[test.member==0,"prevalence"].mean()))

def attack_ci(test, threshold, repetitions=1000, seed=809):
    rng = np.random.default_rng(seed); vals = []
    strata = list(test.groupby(["member", "dominant_label"]))
    for _ in range(repetitions):
        resample = pd.concat([frame.iloc[rng.integers(len(frame), size=len(frame))] for _,frame in strata])
        vals.append(roc_auc_score(resample.member, resample.score))
    return dict(attack_auc_ci_low=float(np.quantile(vals, .025)), attack_auc_ci_high=float(np.quantile(vals, .975)),
        ci_unit="patient, stratified by membership and dominant label", bootstrap_repetitions=repetitions)

def membership_audit(checkpoint_path, store, output_root, fast_root=None, bootstraps=1000):
    ckpt, directory = eval_directory(checkpoint_path, store, output_root)
    if store.split_policy != "patient42" or ckpt["origin"] != "clean_patient_disjoint":
        raise ValueError("MIA requires clean patient-disjoint provenance; old training exposure is unknown")
    exposed = set(ckpt.get("seen_records", []))
    if not exposed: raise ValueError("Missing actual training-exposure ledger")
    patient_path = directory / "mia_patient_scores.csv"
    if patient_path.exists(): all_patients = pd.read_csv(patient_path)
    else:
        parts = []
        for scope, membership in [("train", 1), ("test", 0)]:
            pack = store.pack(scope, fast_root)
            mask = pack.index.rid.isin(exposed).to_numpy() if membership else np.ones(len(pack.index), bool)
            view = pack.view(mask, tau=ckpt["config"]["train_tau"])
            p = predict(ckpt["state"], view, ckpt["config"]["architecture"])
            part = patient_losses(view.frame, view.y, p); part["member"] = membership; parts.append(part)
        all_patients = pd.concat(parts, ignore_index=True)
        if all_patients.patient.duplicated().any(): raise ValueError("Patient leakage in MIA construction")
        all_patients.to_csv(patient_path, index=False)
    results = []
    for domain, candidates in all_patients.groupby("domain"):
        cal, test, exclusions = make_attack_split(candidates)
        if cal is None:
            results.append(dict(**tags(ckpt, store), domain=domain, status="insufficient_patients", exclusions=exclusions)); continue
        threshold = choose_threshold(cal)
        result = dict(**tags(ckpt, store), domain=domain, status="completed", attack="patient-level negative-loss threshold",
            **attack_scores(test, threshold), **attack_ci(test, threshold, bootstraps),
            calibration_patients=len(cal), exclusions=exclusions, random_auc=0.5,
            threat_model="Black-box final selected global model; labeled ECGs; attacker knows membership of disjoint auxiliary calibration patients",
            limitation="A weak, representative attack. Near-chance results do not establish privacy or resistance to stronger/white-box attacks. No DP or secure aggregation.")
        # Null control: shuffle membership ONLY within matching label strata.
        rng = np.random.default_rng(810); null = []
        for _ in range(200):
            shuffled = test.member.to_numpy().copy()
            for _, part in test.groupby("dominant_label"):
                pos = test.index.get_indexer(part.index); shuffled[pos] = rng.permutation(shuffled[pos])
            null.append(roc_auc_score(shuffled, test.score))
        result["permutation_null_auc_mean"] = float(np.mean(null))
        cal.to_csv(directory / f"mia_{domain}_calibration.csv", index=False)
        test.to_csv(directory / f"mia_{domain}_evaluation.csv", index=False)
        results.append(result)
    write_json(directory / "membership_attack.json", results)
    pd.DataFrame([{k:v for k,v in r.items() if k != "exclusions"} for r in results]).to_csv(directory / "membership_attack.csv", index=False)
    return pd.DataFrame(results)
