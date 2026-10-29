"""Verify saved canonical coverage, settings, history selection, and summary statistics.
Does not claim to rerun model training or validate checkpoint bytes.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
reference=ROOT/'reference'
rows=pd.read_csv(reference/'canonical_d2_rows.csv')
runs=pd.read_csv(reference/'study/primary_canonical_runs.csv')
assert len(rows)==80 and len(runs)==40
assert rows.groupby(['method','domain']).seed.apply(set).apply(lambda s:s==set(range(10))).all()
assert not rows.duplicated(['method','seed','domain']).any()
assert set(rows.data_id)=={'9d68265b9e8940c9c1b1'}
for row in runs.itertuples():
    run=reference/'runs'/Path(row.run_path).name
    cfg=json.loads((run/'config.json').read_text());history=json.loads((run/'history.json').read_text());done=json.loads((run/'complete.json').read_text())
    assert len(history)==20
    assert cfg['architecture']=='manuscript_bn' and cfg['batch_size']==64
    assert cfg['split_policy']=='patient42' and cfg['unknown_policy']=='exclude'
    assert cfg['weighting']=='equal' and cfg['local_epochs']==1
    assert cfg['environment']['code_hash']=='0b9f67ac5d30270b791053942be0ec09af31fafe6da585af5b9d4b5620a27cf1'
    best=max(history,key=lambda h:h['selection_score'])
    assert best['round']==done['best_round']==row.best_round
    assert np.isclose(best['selection_score'],row.best_validation)
    assert all(np.isclose(h['selection_score'],(h['ptb_auc']+h['incart_auc'])/2) for h in history)
    assert all(np.allclose(h['alpha'],[.25]*4) for h in history)
summary=rows.groupby(['method','domain'])[['roc_auc','average_precision','mcc']].agg(['mean','std'])
expected={('fedavg','PTB'):(.911,.006),('fedavg','INCART'):(.776,.027),('fedavgm','PTB'):(.888,.018),('fedavgm','INCART'):(.792,.054),('fedprox','PTB'):(.885,.005),('fedprox','INCART'):(.734,.037),('fedadam','PTB'):(.853,.016),('fedadam','INCART'):(.822,.024)}
for key,pair in expected.items():
    assert tuple(summary.loc[key,'roc_auc'].round(3))==pair
print('PASS: 40 complete canonical runs; 800 round histories; 80 held-out domain rows; rounded manuscript ROC-AUC values match.')
print(summary.to_string())
