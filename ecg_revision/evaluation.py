"""Held-out evaluation, preserving source metric and threshold conventions."""
from pathlib import Path
import numpy as np
import pandas as pd
from .common import file_hash, fingerprint, code_hash, write_json
from .training import load_checkpoint, predict
from .metrics import scores
from .thresholds import best_mcc_threshold

def checked_checkpoint(path, store):
    ckpt=load_checkpoint(path)
    if ckpt['data_id'] != store.data_id:
        raise ValueError('Checkpoint/data mismatch; legacy checkpoints cannot be treated as held out')
    return ckpt

def probabilities(path, store, root, scope='test'):
    ckpt=checked_checkpoint(path,store)
    view=store.pack(scope).view(tau=.05)
    key=fingerprint([file_hash(path),store.data_id,code_hash(),scope])[:20]
    target=Path(root)/'predictions'/key
    target.mkdir(parents=True,exist_ok=True)
    dest=target/'p.npy'
    if dest.exists(): p=np.load(dest,allow_pickle=False)
    else:
        p=predict(ckpt['state'],view,ckpt['config']['architecture'])
        tmp=dest.with_suffix('.partial')
        with tmp.open('wb') as f: np.save(f,p)
        tmp.replace(dest)
        write_json(target/'provenance.json',dict(checkpoint_sha256=file_hash(path),data_id=store.data_id,code_hash=code_hash(),scope=scope))
    if len(p)!=len(view): raise ValueError('Prediction/index mismatch')
    return ckpt,view,p

def evaluate(paths,store,root):
    rows=[]
    for path in paths:
        c,v,p=probabilities(path,store,root)
        for domain in ('PTB','INCART'):
            mask=v.frame.domain.to_numpy()==domain
            rows.append(dict(run_path=str(Path(path).parent),**{k:c['config'][k] for k in ('method','seed','local_epochs','weighting')},domain=domain,**scores(v.y[mask],p[mask])))
    return pd.DataFrame(rows)

def thresholds(paths,store,root):
    rows=[]
    for path in paths:
        c,v,p=probabilities(path,store,root,'val')
        _,t,pt=probabilities(path,store,root,'test')
        for domain in ('PTB','INCART'):
            vm=v.frame.domain.to_numpy()==domain;tm=t.frame.domain.to_numpy()==domain
            threshold,val_mcc=best_mcc_threshold(v.y[vm],p[vm])
            fixed=scores(t.y[tm],pt[tm],threshold=.5);tuned=scores(t.y[tm],pt[tm],threshold=threshold)
            rows.append(dict(method=c['config']['method'],seed=c['config']['seed'],domain=domain,threshold=threshold,validation_mcc=val_mcc,**{'fixed_'+k:x for k,x in fixed.items()},**{'tuned_'+k:x for k,x in tuned.items()}))
    return pd.DataFrame(rows)

def label_sensitivity(paths,store,root,taus):
    rows=[]
    for path in paths:
        c,v,p=probabilities(path,store,root)
        mask=v.frame.domain.to_numpy()=='INCART'
        for tau in taus:
            y=v.pack.view(tau=tau).y[mask]
            rows.append(dict(method=c['config']['method'],seed=c['config']['seed'],tau=tau,changed_vs_005=int(np.count_nonzero(y!=v.y[mask])),**scores(y,p[mask])))
    return pd.DataFrame(rows)

def summary(frame,groups):
    metrics=[k for k in ('roc_auc','average_precision','mcc','sensitivity','ppv','f1','accuracy','balanced_accuracy') if k in frame]
    return frame.groupby(groups)[metrics].agg(['mean','std','count'])
