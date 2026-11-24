"""Portable commands; no Colab, Drive, or cloud infrastructure required."""
import argparse
import os
from pathlib import Path
from dataclasses import replace
import json
import pandas as pd
from .common import read_json,write_json,environment
from .data import Store
from .experiments import jobs
from .training import run
from . import evaluation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','plan','train','evaluate','thresholds','labels','dense','privacy','summarize','environment','paired'])
    p.add_argument('--config',default='configs/canonical.json')
    p.add_argument('--root',default='work')
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--reference-checkpoints', action='store_true',
                   help='Run dense evaluation with the 40 published checkpoints and their D2 validation thresholds')
    p.add_argument('--method', choices=('fedavg', 'fedavgm', 'fedprox', 'fedadam'),
                   help='Select a method with dense --reference-checkpoints')
    p.add_argument('--seed', type=int, choices=range(10), help='Select a seed with dense --reference-checkpoints')
    p.add_argument('--domain', choices=('PTB', 'INCART'), help='Select a domain with dense --reference-checkpoints')
    p.add_argument('--acknowledge-source-d1', action='store_true', help=argparse.SUPPRESS)
    args=p.parse_args();
    if args.reference_checkpoints and args.action != 'dense':
        p.error('--reference-checkpoints applies to dense evaluation')
    if any(x is not None for x in (args.method, args.seed, args.domain)) and not args.reference_checkpoints:
        p.error('--method, --seed and --domain select from the published canonical checkpoints')
    import torch
    torch.set_num_threads(1)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    spec=read_json(args.config);name=spec['name'];out=root/'study';out.mkdir(exist_ok=True)
    if args.action=='environment':write_json(root/'environment.json',environment());print(environment());return
    if args.action=='plan':
        plan=[x.as_dict() for x in jobs(spec)];print(json.dumps(plan,indent=2));print(f'{len(plan)} runs');return
    store=Store(root/'data',workers=args.workers)
    if args.reference_checkpoints:
        from .dense import evaluate_reference
        if spec != read_json(Path(__file__).resolve().parents[1] / 'configs' / 'canonical.json'):
            p.error('--reference-checkpoints uses configs/canonical.json')
        print(evaluate_reference(store, out, method=args.method, seed=args.seed, domain=args.domain))
        return
    if args.action=='prepare':
        audit=store.prepare();print(json.dumps(audit,indent=2));return
    table=out/(name+'_runs.csv')
    if args.action=='train':
        rows=[];sources={}
        for cfg in jobs(spec):
            initial=None
            if cfg.method=='transfer':
                initial=sources.get(cfg.seed)
                if initial is None:raise ValueError('PTB-only training must precede transfer for each seed')
            path=run(cfg,store,out,initial_checkpoint=initial)
            if cfg.method=='ptb_only':sources[cfg.seed]=path/'best.pt'
            done=read_json(path/'complete.json')
            rows.append(dict(run_path=str(path),**cfg.as_dict(),best_round=done['best_round'],best_validation=done['best_score']))
            pd.DataFrame(rows).to_csv(table,index=False)
        write_json(out/(name+'_protocol.json'),dict(spec=spec,data_id=store.data_id,selection='validation only',environment=environment()))
        return
    if not table.exists():raise ValueError(f'Train this configuration first: missing {table}')
    runs=pd.read_csv(table);paths=[Path(x)/'best.pt' for x in runs.run_path]
    if len(runs)!=len(jobs(spec)):raise ValueError('Incomplete experiment run table')
    if args.action=='evaluate':
        frame=evaluation.evaluate(paths,store,out)
        frame.to_csv(out/(name+'_metrics.csv'),index=False)
        evaluation.summary(frame,['method','domain','local_epochs','weighting']).to_csv(out/(name+'_summary.csv'))
    elif args.action=='thresholds':
        evaluation.thresholds(paths,store,out).to_csv(out/(name+'_thresholds.csv'),index=False)
    elif args.action=='labels':
        evaluation.label_sensitivity(paths,store,out,spec.get('label_taus',[.01,.05,.1,.2,.3,.5])).to_csv(out/(name+'_label_sensitivity.csv'),index=False)
    elif args.action=='dense':
        from .dense import evaluate
        thresholds=pd.read_csv(out/(name+'_thresholds.csv'))
        evaluate(paths,store,thresholds,out/(name+'_dense_source_protocol.csv'))
    elif args.action=='privacy':
        from .privacy import evaluate
        evaluate(paths,store,out).to_csv(out/(name+'_privacy.csv'),index=False)
    elif args.action=='paired':
        from scipy.stats import t
        import numpy as np
        frame=pd.read_csv(out/(name+'_metrics.csv'));result=[]
        for domain in ('PTB','INCART'):
            wide=frame[frame.domain==domain].pivot(index='seed',columns='method',values='roc_auc')
            for a,b in ([('fedavg','fedavgm'),('fedavg','fedprox'),('fedavg','fedadam')] if domain=='PTB' else [('fedadam','fedavg'),('fedadam','fedavgm'),('fedadam','fedprox')]):
                d=(wide[a]-wide[b]).dropna();half=t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
                result.append(dict(domain=domain,method_a=a,method_b=b,n=len(d),mean_difference=d.mean(),ci_low=d.mean()-half,ci_high=d.mean()+half))
        pd.DataFrame(result).to_csv(out/(name+'_paired.csv'),index=False)
    elif args.action=='summarize':
        # Validation-only grid ranking; no test metrics loaded here.
        cols=['method','mu','server_lr','epsilon','lr']
        result=runs.groupby(cols,dropna=False).best_validation.agg(['mean','std','count']).reset_index()
        result.sort_values('mean',ascending=False).to_csv(out/(name+'_validation_summary.csv'),index=False)

if __name__=='__main__':main()
