import json
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import torch
from ecg_revision.common import Config
from ecg_revision.data import incart_splits,labels_from_counts,beat_counts,Pack,Store
from ecg_revision.model import CNN
from ecg_revision.training import aggregate_states,state_of,run,load_checkpoint
from ecg_revision.thresholds import best_mcc_threshold
from ecg_revision.metrics import scores
from ecg_revision.experiments import jobs
from ecg_revision.dense import dense_starts,evaluate as dense_evaluate

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)

def test_frozen_manifest():
    m=pd.read_csv(ROOT/'manifests/incart_patient_split.csv')
    assert len(m)==75 and m.record.nunique()==75 and m.patient_id.nunique()==32
    assert m.groupby('split').size().to_dict()=={'train':44,'val':15,'test':16}
    assert m.groupby('split').patient_id.nunique().to_dict()=={'train':19,'val':6,'test':7}
    assert m.groupby('patient_id').split.nunique().max()==1
    assert incart_splits(dict(zip(m.record,m.patient_id)))==dict(zip(m.record,m.split))
    with pytest.raises(ValueError):incart_splits(dict(zip(m.record,m.patient_id)),policy='recording42')

def test_labels_and_half_open_boundary():
    total,ab=beat_counts([0,1,1023,1024],[0,1,0,1],[0,1024])
    assert total.tolist()==[3,1] and ab.tolist()==[1,1]
    assert labels_from_counts([20,20,2],[1,0,2],.05).tolist()==[1,0,-1]

def test_model():
    model=CNN();assert sum(p.numel() for p in model.parameters())==38305
    assert model(torch.zeros(2,12,1024)).shape==(2,)

@pytest.mark.parametrize('method',['fedavg','fedprox','fedavgm','fedadam'])
def test_aggregation_and_bn_buffers(method):
    cfg=Config(method=method,server_lr=3 if method=='fedavgm' else 1,epsilon=.001)
    state=state_of(CNN());states=[]
    for n in [1,2,3,4]:states.append({k:v+n for k,v in state.items()})
    out,_,alpha=aggregate_states(states,[1,2,3,4],'equal',state,cfg)
    assert alpha==[.25]*4
    for k,v in state.items():
        if not v.is_floating_point():assert torch.equal(out[k],v+4)
        elif 'running_' in k:assert torch.allclose(out[k],v+2.5)
    key=next(iter(dict(CNN().named_parameters())))
    delta=7.5 if method=='fedavgm' else (2.5/(2.5+.001) if method=='fedadam' else 2.5)
    assert torch.allclose(out[key],state[key]+delta,atol=1e-6)

def test_domain_weighting():
    cfg=Config(weighting='domain_equal');cfg.validate();s=state_of(CNN())
    out,_,a=aggregate_states([s]*4,[100,50,20,1],'domain_equal',s,cfg)
    assert np.allclose(a,[1/6,1/6,1/6,.5])

def test_thresholds_against_brute_force():
    y=np.array([0,1,1,0,1,0]);p=np.array([.1,.3,.3,.4,.8,.9])
    threshold,mcc=best_mcc_threshold(y,p)
    assert mcc==max(scores(y,p,t)['mcc'] for t in np.r_[np.unique(p),.5])
    assert scores(y,p,threshold)['mcc']==mcc

def test_canonical_plan():
    js=jobs(json.loads((ROOT/'configs/canonical.json').read_text()))
    assert len(js)==40 and {j.seed for j in js}==set(range(10))
    assert all(j.batch_size==64 and j.rounds==20 and j.architecture=='manuscript_bn' for j in js)
    assert {j.mu for j in js if j.method=='fedprox'}=={3}
    assert {j.epsilon for j in js if j.method=='fedadam'}=={.001}

def test_d1_conflict_guard():
    starts=dense_starts(360000,'INC_I01',16384)
    assert len(starts)==16384 and len(np.unique(starts))==16384
    assert set(np.arange(0,360000-1024+1,512))<=set(starts)
    with pytest.raises(ValueError,match='conflict'):dense_evaluate([],None,None,None)

class TinyStore:
    split_policy='patient42'
    data_id='synthetic-test-only'
    def __init__(self,root):
        self.packs={}
        rng=np.random.default_rng(100)
        for scope in ['train','val']:
            p=root/scope;p.mkdir(parents=True)
            n=32;np.save(p/'X.npy',rng.normal(size=(n,12,1024)).astype('float32'))
            idx=pd.DataFrame(dict(rid=[f'R{i}' for i in range(n)],patient=[f'P{i}' for i in range(n)],domain=['PTB']*24+['INCART']*8,client=np.repeat(range(4),8),start=0,total=20,abnormal=np.tile([0,2],16),y=np.tile([0,1],16)))
            idx.to_csv(p/'index.csv',index=False);(p/'complete.json').write_text(json.dumps({'data_id':self.data_id}))
            self.packs[scope]=Pack(p)
    def pack(self,scope,fast_root=None):return self.packs[scope]

@pytest.mark.parametrize('method',['fedavg','fedavgm','fedprox','fedadam'])
def test_resume_equals_uninterrupted(tmp_path,method):
    store=TinyStore(tmp_path/'data');cfg=Config(method=method,rounds=2,batch_size=4,server_lr=.01,mu=.1)
    full=run(cfg,store,tmp_path/'full')
    resumed=run(cfg,store,tmp_path/'resume',stop_after=1)
    assert not (resumed/'complete.json').exists()
    run(cfg,store,tmp_path/'resume')
    a=torch.load(full/'last.pt',weights_only=True);b=torch.load(resumed/'last.pt',weights_only=True)
    assert a['best_round']==b['best_round']
    for k in a['state']:assert torch.equal(a['state'][k],b['state'][k])
    assert load_checkpoint(resumed/'best.pt')['data_id']==store.data_id
