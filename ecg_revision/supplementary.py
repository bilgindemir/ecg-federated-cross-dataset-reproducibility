"""Supplementary evaluations of fixed checkpoints and source signals."""
from pathlib import Path
import argparse
import itertools
import json
import numpy as np
import pandas as pd
from scipy.stats import t
from .metrics import scores

METHODS=('fedavg','fedavgm','fedprox','fedadam')

def longest_run(mask):
    edges=np.diff(np.r_[False,np.asarray(mask,dtype=bool),False].astype(np.int8))
    lengths=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
    return int(lengths.max()) if len(lengths) else 0

def longest_constant_run(x):
    x=np.asarray(x)
    if not len(x):return 0,np.nan
    idx=np.flatnonzero(np.r_[True,x[1:]!=x[:-1],True]);lengths=np.diff(idx);k=np.argmax(lengths)
    return int(lengths[k]),int(x[idx[k]])

def robustness_rows(frame,y,p):
    patients=sorted(frame.patient.unique());masks=[('FULL','none',np.ones(len(y),bool))]
    masks += [('LOPO',patient,frame.patient.to_numpy()!=patient) for patient in patients]
    masks += [('REMOVE_CLIPPED_PATIENTS','INC_1+INC_25',~frame.patient.isin(['INC_1','INC_25']).to_numpy())]
    rows=[]
    for scenario,removed,keep in masks:
        yy,pp=np.asarray(y)[keep],np.asarray(p)[keep]
        metric=scores(yy,pp) if len(yy) else dict(roc_auc=np.nan,average_precision=np.nan,mcc=np.nan,n=0)
        if len(np.unique(yy))<2:
            for key in ('roc_auc','average_precision','mcc'):metric[key]=np.nan
        rows.append(dict(scenario=scenario,removed_patient=removed,positives=int(yy.sum()),**metric))
    return rows

def paired_contrasts(frame):
    rows=[]
    if frame.duplicated(['method','seed','domain']).any():raise ValueError('Duplicate evaluation rows')
    for domain in ('PTB','INCART'):
        wide=frame[frame.domain==domain].pivot(index='seed',columns='method',values='roc_auc').sort_index()
        if list(wide.index)!=list(range(10)) or set(wide.columns)!=set(METHODS) or wide.isna().any().any():raise ValueError('Expected four methods and ten matched seeds per domain')
        for a,b in itertools.combinations(METHODS,2):
            d=wide[a]-wide[b];half=t.ppf(.975,9)*d.std(ddof=1)/np.sqrt(10)
            rows.append(dict(domain=domain,method_a=a,method_b=b,n=10,mean_difference=d.mean(),paired_sd=d.std(ddof=1),ci_low=d.mean()-half,ci_high=d.mean()+half))
    return pd.DataFrame(rows)

def signal_quality(store,out):
    import wfdb
    all_rows=[];details=[]
    for row in store.records[store.records.domain=='INCART'].itertuples(index=False):
        base=store.raw_base(row);physical=wfdb.rdrecord(str(base),physical=True);digital=wfdb.rdrecord(str(base),physical=False)
        for j,lead in enumerate(physical.sig_name):
            x=physical.p_signal[:,j];d=digital.d_signal[:,j];res=int(digital.adc_res[j]);zero=int(digital.adc_zero[j]);lo=zero-2**(res-1);hi=zero+2**(res-1)-1
            rail=(d<=lo)|(d>=hi);large=np.abs(x)>10
            all_rows.append(dict(rid=row.raw_rel,patient=row.patient,split=row.split,lead=lead,max_abs_mV=float(np.nanmax(np.abs(x))),samples_abs_gt_10mV=int(large.sum()),fraction_abs_gt_10mV=float(large.mean()),adc_rail_samples=int(rail.sum()),adc_rail_fraction=float(rail.mean()),longest_adc_rail_run=longest_run(rail)))
            if row.split=='test' or row.raw_rel=='I03':
                n,value=longest_constant_run(d);minimum=int(d.min());maximum=int(d.max())
                details.append(dict(rid=row.raw_rel,lead=lead,fmt=digital.fmt[j],adc_res=res,adc_gain=float(digital.adc_gain[j]),baseline=int(digital.baseline[j]),adc_zero=zero,physical_min_mV=float(np.nanmin(x)),physical_max_mV=float(np.nanmax(x)),max_abs_mV=float(np.nanmax(np.abs(x))),fraction_abs_gt_10mV=float(large.mean()),observed_digital_min=minimum,observed_digital_max=maximum,theoretical_digital_min=-2**(res-1),theoretical_digital_max=2**(res-1)-1,samples_at_observed_min=int((d==minimum).sum()),samples_at_observed_max=int((d==maximum).sum()),longest_run_at_min=longest_run(d==minimum),longest_run_at_max=longest_run(d==maximum),longest_extreme_run=longest_run((d==minimum)|(d==maximum)),longest_constant_run=n,constant_run_value=value))
    lead=pd.DataFrame(all_rows);lead.to_csv(out/'incart_signal_quality_by_lead.csv',index=False)
    rec=lead.groupby(['rid','patient','split'],as_index=False).agg(max_abs_mV=('max_abs_mV','max'),leads_gt_10mV=('samples_abs_gt_10mV',lambda x:int((x>0).sum())),samples_gt_10mV=('samples_abs_gt_10mV','sum'),leads_at_adc_rail=('adc_rail_samples',lambda x:int((x>0).sum())),adc_rail_samples=('adc_rail_samples','sum'),longest_adc_rail_run=('longest_adc_rail_run','max'))
    rec['quality_flag']=(rec.samples_gt_10mV>0)|(rec.adc_rail_samples>0);rec.to_csv(out/'incart_signal_quality.csv',index=False)
    pd.DataFrame(details).to_csv(out/'incart_extreme_plateau.csv',index=False)

def excerpt(store,out):
    from .data import WIN,FS,beat_counts,labels_from_counts
    row=next(store.records[store.records.rid=='INC_I03'].itertuples(index=False));x,beats,abnormal=store.signal(row)
    start=18758;length=10*FS;starts=np.arange(start,start+length-WIN+1);total,ab=beat_counts(beats,abnormal,starts);y=labels_from_counts(total,ab,.05)
    pd.DataFrame(dict(start_sample=starts,total_beats=total,abnormal_beats=ab,label=y)).to_csv(out/'i03_excerpt_windows.csv',index=False)
    result=dict(record='I03',fs=FS,start_sample=start,start_seconds=start/FS,end_seconds=(start+length)/FS,window_samples=WIN,stride=1,tau=.05,candidates=len(starts),eligible=int((y>=0).sum()),abnormal=int((y==1).sum()),normal=int((y==0).sum()))
    (out/'i03_excerpt.json').write_text(json.dumps(result,indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['signal-quality','patient-robustness','privacy-uncertainty','all-pairs','threshold-audit','label-distribution','excerpt','fedprox-gap','baseline-provenance','matched-timing'])
    parser.add_argument('--root',default='work');args=parser.parse_args()
    root=Path(args.root).resolve();study=root/'study';out=study/'supplementary';out.mkdir(parents=True,exist_ok=True)
    if args.action=='all-pairs':
        paired_contrasts(pd.read_csv(study/'canonical_metrics.csv')).to_csv(out/'all_pairwise_roc_auc.csv',index=False);return
    import torch
    torch.set_num_threads(1)
    from .data import Store
    from .evaluation import probabilities,checked_checkpoint
    from .common import read_json,write_json,file_hash
    store=Store(root/'data')
    if args.action=='signal-quality':signal_quality(store,out);return
    if args.action=='excerpt':excerpt(store,out);return
    if args.action=='matched-timing':
        from .timing import local_controls
        spec=read_json(Path(__file__).parent.parent/'configs/canonical.json');cfg=dict(spec['base']);timing=read_json(Path(__file__).parent.parent/'configs/local_timing.json');repeats=timing.pop('repeats');cfg.update(timing)
        settings=out/'local_timing_settings.json';write_json(settings,dict(experiment=cfg,store=dict(root=str(root/'data'))))
        frame=local_controls(settings,out/'matched_timing',repeats)
        frame['seconds_per_round']=frame.total_s/cfg['rounds'];frame['examples_per_round']=frame.examples/cfg['rounds'];frame['examples_per_second']=frame.examples/frame.total_s
        frame.groupby('kind')[['total_s','seconds_per_round','examples_per_round','examples_per_second']].agg(['mean','std']).to_csv(out/'matched_timing_summary.csv');return
    if args.action=='label-distribution':
        view=store.pack('test').view(tau=.50);frame=view.frame.copy();frame['y']=view.y;frame=frame[frame.domain=='INCART'];result=frame.groupby('patient').y.agg(windows='size',positive='sum').reset_index();result['negative']=result.windows-result.positive;result['positive_fraction']=result.positive/result.windows;result.to_csv(out/'tau_050_patient_distribution.csv',index=False);return
    name='fedprox_gap' if args.action=='fedprox-gap' else 'local_transfer' if args.action=='baseline-provenance' else 'canonical'
    runs=pd.read_csv(study/(name+'_runs.csv'))
    if args.action=='baseline-provenance':
        runs=pd.concat([runs,pd.read_csv(study/'pooled_runs.csv')],ignore_index=True)
        expected={(m,s) for m in ('ptb_only','incart_only','transfer') for s in range(3)} | {('pooled',s) for s in range(10)}
        if set(zip(runs.method,runs.seed))!=expected:raise ValueError('Expected 19 baseline runs')
    if runs.duplicated(['method','seed']).any() and name!='fedprox_gap':raise ValueError('Ambiguous checkpoint assignments')
    paths=[Path(p)/'best.pt' for p in runs.run_path]
    if name=='canonical':
        expected={(m,s) for m in METHODS for s in range(10)}
        if set(zip(runs.method,runs.seed))!=expected:raise ValueError('Expected exactly 40 canonical runs')
    if args.action=='patient-robustness':
        result=[]
        for path in paths:
            c,v,p=probabilities(path,store,study);mask=v.frame.domain.eq('INCART').to_numpy();frame=v.frame[mask].reset_index(drop=True)
            result.extend(dict(method=c['config']['method'],seed=c['config']['seed'],checkpoint_sha256=file_hash(path),**row) for row in robustness_rows(frame,v.y[mask],p[mask]))
        frame=pd.DataFrame(result);frame.to_csv(out/'incart_patient_robustness.csv',index=False);frame.groupby(['scenario','removed_patient','method'])[['roc_auc','average_precision','mcc','n','positives']].agg(['mean','std']).to_csv(out/'incart_patient_robustness_summary.csv')
    elif args.action=='threshold-audit':
        from .thresholds import select_audit_threshold
        result=[]
        for path in paths:
            c,v,p=probabilities(path,store,study,'val');_,test,pt=probabilities(path,store,study)
            for domain in ('PTB','INCART'):
                vm=v.frame.domain.eq(domain).to_numpy();tm=test.frame.domain.eq(domain).to_numpy();threshold,mcc=select_audit_threshold(v.y[vm],p[vm]);y=test.y[tm];prob=pt[tm]
                result.append(dict(method=c['config']['method'],seed=c['config']['seed'],domain=domain,selected_threshold=threshold,validation_mcc=mcc,test_mcc_fixed_05=scores(y,prob)['mcc'],test_mcc_val_tuned=scores(y,prob,threshold)['mcc'],test_brier=float(np.mean((prob.astype(float)-y)**2))))
        frame=pd.DataFrame(result);frame.to_csv(out/'threshold_audit.csv',index=False);frame.groupby(['method','domain'])[['selected_threshold','test_mcc_fixed_05','test_mcc_val_tuned','test_brier']].agg(['mean','std','min','max']).to_csv(out/'threshold_audit_summary.csv')
    elif args.action=='privacy-uncertainty':
        from .privacy import patient_scores
        from .uncertainty import exact_patient_permutation,patient_bootstrap_ci,mean_seed_auc
        records=[]
        for path in paths:
            for split in ('train','val','test'):
                c,v,p=probabilities(path,store,study,split);pts=patient_scores(v,p,split,c.get('seen_records',[]))
                records.extend(dict(method=c['config']['method'],seed=c['config']['seed'],split=split,patient=patient,score=float(score)) for patient,score in pts.items())
        frame=pd.DataFrame(records);frame.to_csv(out/'mia_patient_scores.csv',index=False);result=[]
        for index,method in enumerate(METHODS):
            matrices={}
            for split,n in [('train',19),('val',6),('test',7)]:
                matrix=frame[(frame.method==method)&(frame.split==split)].pivot(index='patient',columns='seed',values='score').sort_index()
                if matrix.shape!=(n,10) or list(matrix.columns)!=list(range(10)) or matrix.isna().any().any():raise ValueError('Incomplete patient/seed score matrix')
                matrices[split]=matrix.to_numpy()
            row=dict(method=method)
            for split,prefix,seed in [('train','train_test',1000+index),('val','val_test',2000+index)]:
                a,b=matrices[split],matrices['test'];auc,values=mean_seed_auc(a,b);perm=exact_patient_permutation(a,b);low,high=patient_bootstrap_ci(a,b,seed=seed)
                row.update({prefix+'_auc':auc,prefix+'_seed_sd':values.std(ddof=1),prefix+'_patient_ci_low':low,prefix+'_patient_ci_high':high,**{prefix+'_'+k:v for k,v in perm.items() if k!='auc'}})
            result.append(row)
        pd.DataFrame(result).to_csv(out/'mia_patient_uncertainty.csv',index=False)
    elif args.action=='fedprox-gap':
        result=[]
        if len(runs)!=6 or set(zip(runs.mu,runs.seed))!={(mu,seed) for mu in (0,3) for seed in (101,202,303)}:raise ValueError('Expected six matched FedProx checkpoints')
        for path in paths:
            c,v,p=probabilities(path,store,study);aucs={d:scores(v.y[v.frame.domain.eq(d)],p[v.frame.domain.eq(d)])['roc_auc'] for d in ('PTB','INCART')}
            result.append(dict(mu=c['config']['mu'],seed=c['config']['seed'],best_round=c['best_round'],macro_validation_auc=c['selection_score'],ptb_test_auc=aucs['PTB'],incart_test_auc=aucs['INCART'],macro_test_auc=(aucs['PTB']+aucs['INCART'])/2))
        frame=pd.DataFrame(result);frame.to_csv(out/'fedprox_gap.csv',index=False);frame.groupby('mu')[['macro_validation_auc','ptb_test_auc','incart_test_auc','macro_test_auc']].agg(['mean','std']).to_csv(out/'fedprox_gap_summary.csv')
    elif args.action=='baseline-provenance':
        records=[]
        for path in paths:
            c=checked_checkpoint(path,store);cfg=read_json(path.parent/'config.json');source=cfg.get('source_checkpoint');row=dict(method=c['config']['method'],seed=c['config']['seed'],checkpoint_sha256=file_hash(path),source_checkpoint_sha256=file_hash(source) if source else None)
            if c['config']['method']=='transfer':
                if not source:raise ValueError('Missing transfer source')
                origin=checked_checkpoint(source,store)
                if origin['config']['method']!='ptb_only' or origin['config']['seed']!=row['seed']:raise ValueError('Transfer source method/seed mismatch')
            records.append(row)
        pd.DataFrame(records).to_csv(out/'baseline_checkpoint_provenance.csv',index=False)

if __name__=='__main__':main()
