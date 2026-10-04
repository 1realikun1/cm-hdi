"""Reproduce cross-setting EO contrasts from archived outer predictions.

Positive contrasts favour EO. Inference conditions on existing predictions.
The 36-test Holm family covers three countries, two contrasts, six metrics.
No model is fitted and no causal effect is estimated.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from eo_incremental_inference import paired_cluster_inference, holm_adjust, model_metrics, N_BOOT, SEED

DATA = Path(__file__).resolve().parent
CONTRASTS = {'all_tabular':('R_S10','B_RS74'),'population_economy':('R6','B_R70')}
SIZES = {'China':(240,24),'Indonesia':(443,34),'USA':(3104,51)}

def main():
    rows = pd.read_csv(DATA/'D40_CN_ID_fixed_input_predictions.csv',dtype={'unit':str,'province':str})
    us = pd.read_csv(DATA/'D18_USA_100m_HDI_predictions.csv',dtype={'adm2_code':str,'adm1_code':str})
    us = us.loc[us.prediction_group.eq('fixed_direct')].rename(columns={'adm2_code':'unit','adm1_code':'province','y_true_hdi':'observed','y_pred_hdi':'predicted'})
    us['country'] = 'USA'
    rows = pd.concat([rows,us[['country','unit','province','model','observed','predicted']]],ignore_index=True)
    records, summary = [], []
    for country,(n,ngroups) in SIZES.items():
        cr = rows.loc[rows.country.eq(country)]
        for contrast,(traditional,eo) in CONTRASTS.items():
            pair = cr.loc[cr.model.isin([traditional,eo])].pivot(index=['unit','province'],columns='model',values=['observed','predicted'])
            assert len(pair)==n and pair.index.get_level_values('province').nunique()==ngroups
            y=pair['observed',traditional].to_numpy(float)
            assert np.allclose(y,pair['observed',eo],atol=1e-12)
            p0=pair['predicted',traditional].to_numpy(float); p1=pair['predicted',eo].to_numpy(float)
            groups=pair.index.get_level_values('province').to_numpy(str)
            m0=model_metrics(y,p0); m1=model_metrics(y,p1)
            summary.append(dict(country=country,contrast=contrast,traditional_model=traditional,EO_model=eo,n=n,groups=ngroups,p5_traditional_pct=100*m0['P5'],p5_EO_pct=100*m1['P5'],p5_gain_pp=100*(m1['P5']-m0['P5']),p8_traditional_pct=100*m0['P8'],p8_EO_pct=100*m1['P8'],mae_traditional=m0['MAE'],mae_EO=m1['MAE'],mae_gain=m0['MAE']-m1['MAE'],rmse_traditional=m0['RMSE'],rmse_EO=m1['RMSE'],inference_available=True))
            for metric in ('P3','P5','P8','P10','MAE_gain','MSE_gain'):
                r=paired_cluster_inference(country+':'+contrast,metric,y,p0,p1,groups)
                for old,new in [('leave_one_state_out_min','leave_one_group_out_min'),('leave_one_state_out_max','leave_one_group_out_max'),('states_positive','groups_positive'),('states_zero','groups_zero'),('states_negative','groups_negative'),('n_counties','n_units')]: r[new]=r.pop(old)
                r['country']=country; r['contrast']=contrast
                records.append(r)
    adjusted=holm_adjust([r['signflip_p_two_sided'] for r in records])
    for r,p in zip(records,adjusted): r['holm_p_36_tests']=p
    inference=pd.DataFrame(records)
    inference.to_csv(DATA/'D41_cross_country_EO_paired_inference.csv',index=False,float_format='%.17g')
    pd.DataFrame(summary).to_csv(DATA/'D39_cross_country_EO_fixed_input_summary.csv',index=False,float_format='%.17g')
    audit={'scope':'Conditional incremental prediction, not causal identification','inputs':{name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['D40_CN_ID_fixed_input_predictions.csv','D18_USA_100m_HDI_predictions.csv']},'contrasts':CONTRASTS,'cohorts':SIZES,'replicates':N_BOOT,'seed':SEED,'multiplicity':'One 36-test Holm family: 3 countries x 2 fixed-input contrasts x 6 metrics','positive_contrasts':'EO passing rate minus traditional passing rate; traditional absolute/squared error minus EO error','assumptions':'Group sign symmetry/exchangeability; fixed archived outer predictions; no resampled refitting','D37_status':'Preserved historical USA-only 12-test analysis; D41 is the joint cross-setting analysis'}
    (DATA/'D42_cross_country_EO_inference_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(inference.loc[inference.contrast.eq('all_tabular')&inference.metric.isin(['P5','MAE_gain']),['country','metric','estimate','bootstrap_95_lo','bootstrap_95_hi','signflip_p_two_sided','holm_p_36_tests','groups_positive','n_outer_groups']].to_string(index=False))

if __name__=='__main__': main()
