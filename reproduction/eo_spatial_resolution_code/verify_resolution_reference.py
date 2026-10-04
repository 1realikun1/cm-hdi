"""Reconstruct all320m folds to ensure160m changes only raster resolution."""
import run_eo_spatial_resolution as cfg
import run_revision_sensitivities as rev
import numpy as np,pandas as pd,json
matrix,splits,models,_=rev.load_inputs('China')
old=pd.read_csv(cfg.HERE/'D73_current_no_EO_predictions.csv',dtype={'unit':str}).query("country=='China'").set_index('unit')
diff=[]
for k,outer in enumerate(splits):
    tr,z,te,zt,sel=rev.select_components(matrix,outer,models,list(models))
    rule,_=rev.synth.fit_rule(z,tr.y_hdi.to_numpy(),tr.group.to_numpy(),'P4_constrained_interactions')
    p=rev.synth.predict(zt,rule['weights'],'P4_constrained_interactions')
    diff.extend(p-old.loc[te.unit,'pred_full'].to_numpy())
    print(f'320m reference reconstruction {k+1}/{len(splits)}',flush=True)
result=dict(n=len(diff),max_abs_prediction_difference=float(np.max(abs(np.array(diff)))),matched=bool(np.max(abs(np.array(diff)))<1e-8))
(cfg.OUT/'resolution_reference_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
assert result['matched'];print(result)
