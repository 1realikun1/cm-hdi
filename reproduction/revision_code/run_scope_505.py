"""Retrospective full-coverage Indonesia sensitivity with adopted CM-HDI rule."""
from pathlib import Path
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[name]='1'
import json,hashlib
import numpy as np
import pandas as pd
import run_revision_sensitivities as rev

HERE=Path(__file__).resolve().parent
ROOT=Path(r'E:\BeyondSurfaceHDI')
if __name__=='__main__':
    freeze=ROOT/'data_raw/indonesia_validation/frozen_experiment_2020/indonesia_adm2_model_matrix_2020.csv'
    labels_path=Path(r'C:\Users\华为\Documents\Codex\2026-09-22\new-chat\outputs\indonesia_corrected_labels_2020_v1\corrected_model_matrix_2020.csv')
    cohort_path=rev.SUPP/'D04_Indonesia_cohort_514.csv'
    snapshot=HERE/'input_snapshots/Indonesia505.csv'
    portable=snapshot.exists() and not freeze.exists()
    frame=pd.read_csv(snapshot,dtype={'unit':str,'group':str}) if portable else pd.read_csv(freeze,dtype={'adm2_code':str,'adm1_code':str}).rename(columns={'adm2_code':'unit','adm1_code':'group'})
    labels=frame.set_index('unit') if portable else pd.read_csv(labels_path,dtype={'adm2_code':str}).set_index('adm2_code')
    target_map={target:target for target in rev.base.COMPONENTS+['y_hdi']}
    for target,col in target_map.items(): frame[target]=frame.unit.map(labels[col])
    cohort=pd.read_csv(cohort_path,dtype={'adm2_code':str}).set_index('adm2_code')
    frame['main_status']=frame.unit.map(cohort.main_status)
    assert len(frame)==505 and frame.unit.nunique()==505 and frame.group.nunique()==34
    assert frame.main_status.eq('included_443').sum()==443
    adopted=pd.read_csv(rev.SUPP/'D56_constrained_aggregation_predictions.csv',dtype={'unit':str})
    adopted=adopted.loc[(adopted.country=='Indonesia')&(adopted.method=='P4_constrained_interactions')].set_index('unit')
    check=np.max(abs(frame.set_index('unit').loc[adopted.index,'y_hdi']-adopted.y_true));assert check<1e-9
    current=rev.load_inputs('Indonesia')[0].set_index('unit')
    for target in rev.base.COMPONENTS+['y_hdi']:
        assert np.max(abs(frame.set_index('unit').loc[current.index,target]-current[target]))<1e-9
    protocol=dict(retrospective=True,n=505,groups=34,primary_cohort_changed=False,
      selection='All five original input sets, original five Ridge penalties, raw group-equal component MAE; adopted seven-term synthesis and original shrinkage grid.',
      scope='Coverage-eligible 505 units; report all505, included443, official-scope-excluded62 from the same nested evaluation.',
      labels='Existing corrected 505-unit BPS matrix with documented 30 administrative label swaps, verified against all current 443 targets.',
      hashes=({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [freeze,labels_path,cohort_path]} if not portable else {}))
    if not portable:
        (HERE/'scope_505_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rev.save(frame,'input_snapshots/Indonesia505.csv')
    models=rev.load_inputs('Indonesia')[2]
    splits=rev.base.make_group_splits(frame,'unit','group');preds=[];choices=[]
    for k,outer in enumerate(splits):
        train,z,test,zt,selection=rev.select_components(frame,outer,models,list(models))
        rule,_=rev.synth.fit_rule(z,train.y_hdi.to_numpy(),train.group.to_numpy(),'P4_constrained_interactions')
        prediction=rev.synth.predict(zt,rule['weights'],'P4_constrained_interactions');geom=np.cbrt(np.prod(zt,axis=1))
        choices += [dict(country='Indonesia505',**x) for x in selection]
        preds += [dict(unit=r.unit,group=r.group,main_status=r.main_status,y_true=r.y_hdi,pred_CM=prediction[i],pred_G=geom[i],
                       pred_H=zt[i,0],pred_E=zt[i,1],pred_I=zt[i,2]) for i,r in enumerate(test.itertuples())]
        rev.save(pd.DataFrame(preds),'D84_Indonesia505_predictions.csv')
        print(f'Indonesia505: fold {k+1}/{len(splits)} complete',flush=True)
    d=pd.DataFrame(preds);metrics=[]
    for scope,part in [('all505',d),('included443',d.loc[d.main_status.eq('included_443')]),('excluded62',d.loc[~d.main_status.eq('included_443')])]:
        for method,col in [('CM','pred_CM'),('G','pred_G')]: metrics.append(dict(scope=scope,method=method,n=len(part),**rev.summary(part.y_true,part[col])))
    rev.save(pd.DataFrame(metrics),'D85_Indonesia505_metrics.csv');rev.save(pd.DataFrame(choices),'D86_Indonesia505_selection.csv')
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)
