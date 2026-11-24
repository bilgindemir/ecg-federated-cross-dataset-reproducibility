"""Single-device, matched-work timing controls."""
from pathlib import Path
from dataclasses import replace
import pandas as pd
from .common import Config,read_json,write_json,seed_all,clock,environment
from .data import Store
from .training import CNN,state_of,local_update,aggregate_states

def local_controls(settings_path, output_dir, repeats=3):
    """Measured sequential FL and pooled-work controls on the coordinator.

    Multiple-GPU distributed time versus one-device time is a resource contrast;
    it is not an algorithmic speedup or a time-to-equal-quality comparison.
    """
    settings=read_json(settings_path); cfg=Config(**settings["experiment"]); cfg.validate()
    store=Store(**settings["store"]); store.prepare(("train",)); pack=store.pack("train",settings.get("fast_root"))
    clients=[pack.view(pack.index.client.to_numpy()==c,tau=cfg.train_tau) for c in range(4)]
    pooled=pack.view(tau=cfg.train_tau); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for kind in ("sequential_fl", "pooled_work_control"):
        for repeat in range(repeats):
            trial=replace(cfg,seed=cfg.seed+repeat,method="pooled" if kind=="pooled_work_control" else cfg.method)
            seed_all(trial.seed); state=state_of(CNN(trial.architecture)); moments=None; central_optimizer=None
            views=[pooled] if kind=="pooled_work_control" else clients
            for c,view in enumerate(views): local_update(state,view,trial,0,c)
            start=clock(); examples=0
            for r in range(1,trial.rounds+1):
                updates=[]
                for c,view in enumerate(views):
                    if kind=="pooled_work_control":
                        updated,stats,_,central_optimizer=local_update(state,view,trial,r,c,optimizer_state=central_optimizer,return_optimizer=True)
                    else:updated,stats,_=local_update(state,view,trial,r,c)
                    updates.append(updated);examples+=stats["examples"]
                if len(updates)==1:state=updates[0]
                else:state,moments,_=aggregate_states(updates,[len(v) for v in views],trial.weighting,state,trial,moments)
            rows.append(dict(kind=kind,repetition=repeat,total_s=clock()-start,examples=examples))
            pd.DataFrame(rows).to_csv(out/"local_controls.csv",index=False)
    write_json(out/"local_control_environment.json",environment())
    return pd.DataFrame(rows)
