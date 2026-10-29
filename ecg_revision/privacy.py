"""INCART patient-level audit from final notebook cell 34."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .evaluation import probabilities

def patient_scores(view,p,scope,exposed):
    frame=view.frame.copy();frame['p']=p
    frame=frame[frame.domain=='INCART']
    if scope=='train':
        if not exposed:raise ValueError('Missing checkpoint training exposure ledger')
        frame=frame[frame.rid.isin(exposed)]
    prob=np.clip(frame.p.to_numpy(dtype=float),1e-7,1-1e-7);y=frame.y.to_numpy()
    frame['loss']=-(y*np.log(prob)+(1-y)*np.log1p(-prob))
    # Recovered code averages windows within recording, then recordings within patient.
    rec=frame.groupby(['patient','rid']).loss.mean().reset_index()
    return -rec.groupby('patient').loss.mean()

def evaluate(paths,store,root):
    rows=[]
    for path in paths:
        groups={}
        for scope in ('train','val','test'):
            c,v,p=probabilities(path,store,root,scope)
            groups[scope]=patient_scores(v,p,scope,c.get('seen_records',[]))
        def auc(a,b):return roc_auc_score(np.r_[np.ones(len(a)),np.zeros(len(b))],np.r_[a,b])
        attack=auc(groups['train'],groups['test']);control=auc(groups['val'],groups['test'])
        rows.append(dict(method=c['config']['method'],seed=c['config']['seed'],mia_auc=attack,control_auc=control,auc_above_control=attack-control,n_members=len(groups['train']),n_validation=len(groups['val']),n_test=len(groups['test'])))
    return pd.DataFrame(rows)
