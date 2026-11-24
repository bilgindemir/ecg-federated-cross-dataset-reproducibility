import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from ecg_revision.supplementary import longest_run,longest_constant_run,robustness_rows,paired_contrasts
from ecg_revision.uncertainty import exact_patient_permutation,mean_seed_auc,patient_bootstrap_ci
from ecg_revision.thresholds import select_audit_threshold

def test_plateaus():
    assert longest_run([1,1,0,1,1,1])==3
    assert longest_run([])==0
    assert longest_constant_run([2,2,1,1,1,3])==(3,1)

def test_patient_subsets_are_evaluation_only():
    f=pd.DataFrame({'patient':['INC_1','INC_1','INC_25','INC_25','INC_10','INC_10']})
    y=np.array([0,1,0,1,0,1]);p=np.array([.2,.7,.1,.8,.3,.9])
    rows=robustness_rows(f,y,p)
    assert len(rows)==5
    assert rows[-1]['n']==2 and rows[0]['n']==6
    assert rows[0]['roc_auc']==roc_auc_score(y,p)
    assert len(y)==6

def test_exact_patient_permutation_against_brute_force():
    a=np.array([[.2,.4],[.9,.5]]);b=np.array([[.1,.4],[.7,.3],[.8,.6]])
    x=np.vstack([a,b]);observed=mean_seed_auc(a,b)[0];null=[]
    for indices in itertools.combinations(range(5),2):
        keep=np.zeros(5,bool);keep[list(indices)]=True
        null.append(mean_seed_auc(x[keep],x[~keep])[0])
    result=exact_patient_permutation(a,b)
    assert np.isclose(result['auc'],observed)
    assert result['permutations']==10
    assert result['exact_p_greater']==np.mean(np.array(null)>=observed-1e-15)
    assert result['exact_p_two_sided']==np.mean(abs(np.array(null)-.5)>=abs(observed-.5)-1e-15)
    assert patient_bootstrap_ci(a,b,20,12)==patient_bootstrap_ci(a,b,20,12)

def test_pairwise_seed_coverage():
    methods=['fedavg','fedavgm','fedprox','fedadam']
    f=pd.DataFrame([dict(method=m,seed=s,domain=d,roc_auc=.6+i*.02+s*.001) for d in ['PTB','INCART'] for i,m in enumerate(methods) for s in range(10)])
    result=paired_contrasts(f)
    assert len(result)==12 and np.allclose(result.mean_difference.iloc[::6],-.02)
    import pytest
    with pytest.raises(ValueError):paired_contrasts(f.iloc[:-1])

def test_threshold_audit_equal_distance_tie():
    # Candidate-only sweep prefers first (higher) probability on equal distance.
    threshold,mcc=select_audit_threshold(np.array([0,1,0,1]),np.array([.1,.9,.2,.8]))
    assert threshold==.8 and mcc==1

def test_supplied_pairwise_outputs_match_canonical_rows():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    actual=paired_contrasts(pd.read_csv(root/'reference/canonical_d2_rows.csv'))
    ref=pd.read_csv(root/'reference/supplementary/all_pairwise_roc_auc.csv')
    display={'fedavg':'FedAvg','fedavgm':'FedAvgM','fedprox':'FedProx','fedadam':'FedAdam'}
    for row in actual.itertuples():
        contrast=f'{display[row.method_a]} -- {display[row.method_b]}'
        domain='PTB-XL' if row.domain=='PTB' else row.domain
        hit=ref[(ref.domain==domain)&(ref.contrast==contrast)]
        assert len(hit)==1
        assert np.isclose(row.mean_difference,hit.iloc[0].delta_roc_auc,atol=5.1e-7,rtol=0)
