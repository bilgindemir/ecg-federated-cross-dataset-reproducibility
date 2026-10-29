import numpy as np
from .metrics import scores

def best_mcc_threshold(y,p):
    y=np.asarray(y,dtype=np.int8); p=np.asarray(p,dtype=np.float64)
    if len(y)!=len(p) or len(y)==0: raise ValueError("Invalid validation arrays")
    order=np.argsort(-p,kind="mergesort"); ps=p[order]; ys=y[order]
    tp=np.cumsum(ys==1); fp=np.cumsum(ys==0)
    # evaluate only after the last observation sharing each probability value
    ends=np.r_[np.flatnonzero(ps[:-1] != ps[1:]), len(ps)-1]
    tps=tp[ends].astype(float); fps=fp[ends].astype(float)
    pos=float((y==1).sum()); neg=float((y==0).sum())
    fns=pos-tps; tns=neg-fps
    den=(tps+fps)*(tps+fns)*(tns+fps)*(tns+fns)
    mcc=np.zeros_like(tps,dtype=float)
    valid=den>0
    mcc[valid]=(tps[valid]*tns[valid]-fps[valid]*fns[valid])/np.sqrt(den[valid])
    thresholds=ps[ends]
    # Also evaluate the conventional 0.5 threshold explicitly.
    base=scores(y,p,threshold=0.5)["mcc"]
    thresholds=np.r_[thresholds,0.5]; mcc=np.r_[mcc,base]
    best=np.nanmax(mcc)
    idx=np.flatnonzero(np.isclose(mcc,best,rtol=0,atol=1e-12))
    # deterministic tie-break: closest to 0.5, then lower threshold
    chosen=idx[np.lexsort((thresholds[idx],np.abs(thresholds[idx]-0.5)))[0]]
    return float(thresholds[chosen]), float(mcc[chosen])
