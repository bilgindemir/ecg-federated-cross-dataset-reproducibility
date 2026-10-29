"""Report numerical differences from original D2 outputs, without invented tolerance."""
import argparse
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('metrics_csv');p.add_argument('--output',default='comparison.csv');a=p.parse_args()
root=Path(__file__).resolve().parents[1]
x=pd.read_csv(a.metrics_csv);r=pd.read_csv(root/'reference/canonical_d2_rows.csv');keys=['method','seed','domain']
if len(x)!=80 or x.duplicated(keys).any():raise ValueError('Expected exactly 80 unique canonical domain rows')
metrics=['roc_auc','average_precision','mcc','sensitivity','ppv','f1']
m=x[keys+metrics].merge(r[keys+metrics],on=keys,suffixes=('_new','_reference'),validate='one_to_one')
if len(m)!=80:raise ValueError('Seed/method/domain coverage differs')
for metric in metrics:m[metric+'_delta']=m[metric+'_new']-m[metric+'_reference']
m.to_csv(a.output,index=False);print(m[[k+'_delta' for k in metrics]].abs().max().to_string())
print('Differences reported, not judged against an unsupported scientific tolerance.')
