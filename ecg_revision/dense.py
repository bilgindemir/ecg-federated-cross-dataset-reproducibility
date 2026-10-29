"""Dense test-time ECG segmentation. Manuscript sampling conflict must be acknowledged."""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from .common import fingerprint
from .data import WIN, STRIDE, MIN_BEATS, beat_counts, labels_from_counts
from .training import CNN,device_name
from .evaluation import checked_checkpoint
from .metrics import scores

def dense_starts(length,rid,cap=None):
    # D2-inclusive sampling; see the documented manuscript sampling limitation.
    candidates=np.arange(max(0,length-WIN+1),dtype=np.int64)
    if cap is None or len(candidates)<=cap:return candidates
    d2=np.arange(0,length-WIN+1,STRIDE,dtype=np.int64)
    if cap<len(d2):raise ValueError('Cap must contain all D2 starts')
    rng=np.random.default_rng(int(fingerprint(rid)[:8],16))
    pool=np.setdiff1d(candidates,d2,assume_unique=True)
    return np.sort(np.r_[d2,rng.choice(pool,size=cap-len(d2),replace=False)])

@torch.inference_mode()
def evaluate(paths,store,threshold_table,output,acknowledge_source_sampling=False):
    if not acknowledge_source_sampling:
        raise ValueError('Unresolved D1 sampling conflict: see docs/OPEN_ISSUES.md. Use --acknowledge-source-d1 only for the implemented sampling protocol.')
    rows=[];device=device_name()
    for path in paths:
        c=checked_checkpoint(path,store);cfg=c['config'];model=CNN(cfg['architecture']).to(device);model.load_state_dict(c['state']);model.eval()
        for domain in ('PTB','INCART'):
            selected=store.select_records('test');selected=selected[selected.domain==domain]
            hit=threshold_table[(threshold_table.method==cfg['method'])&(threshold_table.seed==cfg['seed'])&(threshold_table.domain==domain)]
            if len(hit)!=1:raise ValueError('Missing/duplicate validation threshold')
            threshold=float(hit.iloc[0].threshold);ys=[];ps=[]
            for row in selected.itertuples(index=False):
                x,b,a=store.signal(row);starts=dense_starts(x.shape[1],row.rid,16384 if domain=='INCART' else None)
                if domain=='INCART':
                    total,ab=beat_counts(b,a,starts);keep=total>=MIN_BEATS;starts=starts[keep];y=labels_from_counts(total[keep],ab[keep],.05)
                else:y=np.full(len(starts),row.label,np.int8)
                p=np.empty(len(starts),np.float32)
                for i in range(0,len(starts),256):
                    batch=starts[i:i+256];xb=torch.from_numpy(np.stack([x[:,int(s):int(s)+WIN] for s in batch])).to(device)
                    p[i:i+len(batch)]=model(xb).sigmoid().cpu().numpy()
                ys.append(y);ps.append(p)
            rows.append(dict(method=cfg['method'],seed=cfg['seed'],domain=domain,n_records=len(selected),n_patients=selected.patient.nunique(),sampling='record_hash_seed_D2_inclusive',**scores(np.concatenate(ys),np.concatenate(ps),threshold=threshold)))
            pd.DataFrame(rows).to_csv(output,index=False)
        del model
    return pd.DataFrame(rows)
