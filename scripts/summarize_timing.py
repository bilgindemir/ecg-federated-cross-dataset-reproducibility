"""Summarize newly executed local histories; not historical Table 15 replication."""
import argparse,json
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--root',default='work');a=p.parse_args();root=Path(a.root)/'study';rows=[]
for name in ('canonical','pooled'):
    file=root/(name+'_runs.csv')
    if not file.exists():continue
    for run in pd.read_csv(file).itertuples():
        history=json.loads((Path(run.run_path)/'history.json').read_text())
        rows.append(dict(experiment=name,method=run.method,seed=run.seed,rounds=len(history),
            sequential_without_checkpoint_io_s=sum(h['round_without_checkpoint_io_s'] for h in history),
            local_training_plus_aggregation_s=sum(h['training_aggregation_s'] for h in history),
            parallel_lower_bound_s=sum(max(c['train_s'] for c in h['client_stats'])+h['aggregation_s'] for h in history)))
if not rows:raise SystemExit('Run canonical and/or pooled training first; no timing results invented.')
out=root/'local_timing_summary.csv';pd.DataFrame(rows).to_csv(out,index=False);print(out)
print('Sequential value excludes checkpoint I/O; parallel value excludes evaluation and communication. Not a replication of historical manuscript timing.')
