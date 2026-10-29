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
