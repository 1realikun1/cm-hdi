"""Retrospective EO spatial analysis and matched 160/320 m CM-HDI sensitivity."""
from pathlib import Path
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[name]='1'
import json, hashlib
import numpy as np
import pandas as pd
import run_revision_sensitivities as rev

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'outputs/CMHDI_EO_spatial_resolution_20261002'
OUT.mkdir(parents=True,exist_ok=True)
SEED=20261002
def save(d,name): d.to_csv(OUT/name,index=False,float_format='%.15g')
def main():
    feature160=Path(r'E:\BeyondSurfaceHDI\data_work\p5_features_overview160\nationwide331_features_overview160.csv')
    inputs=[HERE/'input_snapshots/Indonesia505.csv',HERE/'D84_Indonesia505_predictions.csv',HERE/'D73_current_no_EO_predictions.csv',feature160]
    protocol=dict(retrospective=True,seed=SEED,inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
       Indonesia505='Same 505 units, corrected labels, original leave-one-province-out and five group inner folds; remove all EO candidate sets, select R6/R_S10 with original penalties and adopt seven-term constrained synthesis.',
       spatial='Delta_i = abs(noEO-y) - abs(fullEO-y), positive means EO helps. Main samples: China240, USA3104, Indonesia443. Full505 is a separate paired sensitivity; never splice predictions across fits.',
       subgroups='Before new outcomes: official excluded62 vs included443; within each main country density and urbanization terciles from inputs; 20000 paired province/state bootstrap resamples. Descriptive intervals, no new confirmatory significance claims.',
       resolution='China: replace only 64 population-weighted EO means with existing 160m features; preserve all units, targets, input libraries, inner/outer splits, penalties and synthesis. Reselect on inner folds only. All 24 outer folds scored on the same240 units. Native10m remains distinct; 160m is an intermediate overview check, not proof of native equivalence.')
    (OUT/'D99_EO_spatial_resolution_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    f=pd.read_csv(HERE/'input_snapshots/Indonesia505.csv',dtype={'unit':str,'group':str})
    models=rev.load_inputs('Indonesia')[2];splits=rev.base.make_group_splits(f,'unit','group')
    full=pd.read_csv(HERE/'D84_Indonesia505_predictions.csv',dtype={'unit':str,'group':str}).set_index('unit')
    rows=[];choices=[]
    for k,outer in enumerate(splits):
        tr,z,te,zt,sel=rev.select_components(f,outer,models,['R6','R_S10'])
        rule,_=rev.synth.fit_rule(z,tr.y_hdi.to_numpy(),tr.group.to_numpy(),'P4_constrained_interactions')
        p=rev.synth.predict(zt,rule['weights'],'P4_constrained_interactions')
        for i,r in enumerate(te.itertuples()):
            old=full.loc[r.unit];assert abs(old.y_true-r.y_hdi)<1e-12 and old.group==r.group
            rows.append(dict(unit=r.unit,group=r.group,main_status=old.main_status,y_true=r.y_hdi,pred_full=old.pred_CM,pred_no_EO=p[i]))
        choices += [dict(country='Indonesia505',**x) for x in sel]
        print(f'Indonesia505 noEO {k+1}/{len(splits)}',flush=True)
    d=pd.DataFrame(rows);d['delta_abs_error']=abs(d.pred_no_EO-d.y_true)-abs(d.pred_full-d.y_true)
    save(d,'D93_Indonesia505_paired_EO_predictions.csv');save(pd.DataFrame(choices),'D94_Indonesia505_no_EO_selection.csv')
    matrix,splits,models,_=rev.load_inputs('China');a=pd.read_csv(feature160,dtype={'city_code':str}).set_index('city_code')
    for j in range(64): matrix[f'feature_ae320_A{j:02d}_popmean']=matrix.unit.map(a[f'ae160_A{j:02d}_popmean'])
    assert matrix.notna().all().all()
    save(matrix,'China160_input_snapshot.csv')
    rows=[];choices=[]
    for k,outer in enumerate(splits):
        tr,z,te,zt,sel=rev.select_components(matrix,outer,models,list(models))
        rule,_=rev.synth.fit_rule(z,tr.y_hdi.to_numpy(),tr.group.to_numpy(),'P4_constrained_interactions')
        p=rev.synth.predict(zt,rule['weights'],'P4_constrained_interactions')
        rows += [dict(unit=r.unit,group=r.group,y_true=r.y_hdi,pred_160=p[i],pred_H=zt[i,0],pred_E=zt[i,1],pred_I=zt[i,2]) for i,r in enumerate(te.itertuples())]
        choices += sel
        print(f'China160 {k+1}/{len(splits)}',flush=True)
    save(pd.DataFrame(rows),'D97_China160_CM_predictions.csv');save(pd.DataFrame(choices),'D98_China160_selection.csv')
if __name__=='__main__': main()
