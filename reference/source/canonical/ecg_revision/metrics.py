import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

METRICS = ["roc_auc", "average_precision", "mcc", "balanced_accuracy", "sensitivity", "ppv", "f1", "accuracy"]

def from_confusion(counts):
    tn, fp, fn, tp = map(float, counts)
    divide = lambda a, b: a / b if b else float("nan")
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, n=tn+fp+fn+tp,
                mcc=(tp*tn-fp*fn)/np.sqrt(den) if den else 0.0,
                balanced_accuracy=(divide(tp, tp+fn)+divide(tn, tn+fp))/2,
                sensitivity=divide(tp, tp+fn), ppv=divide(tp, tp+fp),
                f1=divide(2*tp, 2*tp+fp+fn), accuracy=divide(tp+tn, tn+fp+fn+tp),
                prevalence=divide(tp+fn, tn+fp+fn+tp))

def scores(y, p, threshold=0.5):
    y = np.asarray(y, np.int8); p = np.asarray(p, np.float64)
    if len(y) != len(p) or not len(y) or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError("Invalid predictions/labels")
    if np.any((p < 0) | (p > 1)): raise ValueError("Probabilities outside [0,1]")
    pred = (p >= threshold).astype(np.int8)
    conf = np.bincount(2*y + pred, minlength=4)
    out = from_confusion(conf)
    out.update(roc_auc=roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
               average_precision=average_precision_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
               decision_threshold=threshold)
    return out

def cluster_confusions(y, p, groups):
    frame = pd.DataFrame(dict(group=np.asarray(groups).astype(str), cell=2*np.asarray(y, int)+(np.asarray(p)>=0.5)))
    return pd.crosstab(frame.group, frame.cell).reindex(columns=[0, 1, 2, 3], fill_value=0)

def cluster_ci(confusions, repetitions=1000, seed=991):
    """Patient-cluster percentile intervals for pooled confusion-derived metrics.

    Ranking metrics retain exact point estimates; these intervals do not claim
    ROC-AUC/AP coverage. Windows are never independently bootstrapped.
    """
    matrix = np.asarray(confusions, float); n = len(matrix)
    if n < 2: return {}
    rng = np.random.default_rng(seed)
    replicates = [from_confusion(rng.multinomial(n, np.full(n, 1/n)) @ matrix) for _ in range(repetitions)]
    out = {}
    for metric in METRICS[2:]:
        vals = np.array([r[metric] for r in replicates]); vals = vals[np.isfinite(vals)]
        out[metric + "_ci_low"] = float(np.quantile(vals, .025)) if len(vals) else np.nan
        out[metric + "_ci_high"] = float(np.quantile(vals, .975)) if len(vals) else np.nan
        out[metric + "_bootstrap_valid"] = int(len(vals))
    return out

def paired_cluster_delta(a, b, repetitions=1000, seed=991):
    """D1-D2 using identical resampled patient clusters in both representations."""
    groups = sorted(set(a.index) | set(b.index))
    aa = a.reindex(groups, fill_value=0).to_numpy(float)
    bb = b.reindex(groups, fill_value=0).to_numpy(float)
    if len(groups) < 2: return {}
    rng = np.random.default_rng(seed); values = {m: [] for m in METRICS[2:]}
    for _ in range(repetitions):
        weight = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)))
        ma, mb = from_confusion(weight @ aa), from_confusion(weight @ bb)
        for m in values: values[m].append(mb[m]-ma[m])
    out = {}
    for m, vals in values.items():
        vals = np.asarray(vals); vals = vals[np.isfinite(vals)]
        out[m+"_delta_ci_low"] = np.quantile(vals, .025) if len(vals) else np.nan
        out[m+"_delta_ci_high"] = np.quantile(vals, .975) if len(vals) else np.nan
        out[m+"_bootstrap_valid"] = len(vals)
    return out
