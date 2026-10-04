"""Component-level extension of archived D70; frozen China models, no reselection."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

import sys
PROJECT = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
OUT = PROJECT/'component_importance_recomputed'
SELECTION = PROJECT/'china_frozen_component_selection.csv'
if not SELECTION.exists():
    SELECTION = PROJECT/'input_snapshots/china_frozen_component_selection.csv'
def data_file(name):
    candidates=[PROJECT/name, PROJECT/'supplementary_data'/name]
    return next(p for p in candidates if p.exists())
DATA = PROJECT
TARGETS = {'H': 'lifeindex', 'E': 'eduindex', 'I': 'incomeindex'}
KEYS = {'H': 'y_health_index', 'E': 'y_education_index', 'I': 'y_income_index'}
WC = ['weight_G', 'weight_H', 'weight_E', 'weight_I', 'weight_sqrt_HE', 'weight_sqrt_HI', 'weight_sqrt_EI']

def synthesis(z, w):
    h, e, i = z.T
    return np.column_stack([np.cbrt(h*e*i), h, e, i, np.sqrt(h*e), np.sqrt(h*i), np.sqrt(e*i)]) @ w

def main():
    d = pd.read_csv(data_file('D49_China_neural_input_matrix.csv'), dtype={'city_code':str, 'province_code':str}).set_index('city_code', drop=False)
    splits = json.loads((data_file('D50_China_neural_splits.json')).read_text())['splits']
    features = [c for c in d if c.startswith('feature_')]
    blocks = {
        'EO': [c for c in features if 'ae320_' in c],
        'Demography': [c for c in features if 'control_' in c],
        'Economy': [c for c in features if 'economic_' in c],
        'Healthcare': [c for c in features if 'health_' in c],
    }
    assert sorted(sum(blocks.values(), [])) == sorted(features)
    p, eo, ec, med = [blocks[b] for b in ['Demography', 'EO', 'Economy', 'Healthcare']]
    models = {'B69':p+eo, 'R6':p+ec, 'B_R70':p+eo+ec, 'R_S10':p+ec+med, 'B_RS74':p+eo+ec+med}
    sel = pd.read_csv(SELECTION, dtype={'outer_test_province':str})
    weights = pd.read_csv(data_file('D57_constrained_aggregation_coefficients.csv'), dtype={'outer_group':str})
    weights = weights.loc[weights.country.eq('China') & weights.method.eq('P4_constrained_interactions')].set_index('outer_group')
    reference = pd.read_csv(data_file('D56_constrained_aggregation_predictions.csv'), dtype={'unit':str})
    reference = reference.loc[reference.country.eq('China') & reference.method.eq('P4_constrained_interactions')].set_index('unit')
    records, predictions, configs = [], [], []
    max_component_difference = max_total_difference = 0.
    with threadpool_limits(limits=1):
        for split in splits:
            group = str(split['outer_test_province'])
            train = d.loc[list(map(str, split['outer_train_city_codes']))]
            test = d.loc[list(map(str, split['outer_test_city_codes']))]
            selections = sel.loc[sel.outer_test_province.eq(group) & sel.method.eq('baseline_C')]
            fitted = []
            for target, label in TARGETS.items():
                r = selections.loc[selections.target.eq(KEYS[target])].iloc[0]
                cols = models[r.model]
                x = train[cols].to_numpy(float)
                keep = np.isfinite(x).all(0) & (x.std(0)>1e-12)
                cols = list(np.array(cols)[keep])
                scale = StandardScaler().fit(x[:,keep])
                est = Ridge(alpha=float(r.alpha), solver='svd').fit(scale.transform(x[:,keep]), train[label].to_numpy(float))
                fitted.append((est, scale, cols))
                configs.append(dict(group=group, target=target, model=r.model, alpha=float(r.alpha), effective_features=len(cols)))
            def predict(frame):
                return np.clip(np.column_stack([est.predict(scale.transform(frame[cols].to_numpy(float))) for est,scale,cols in fitted]), 0, 1)
            z = predict(test)
            w = weights.loc[group, WC].to_numpy(float)
            total = synthesis(z, w)
            truth_z = test[list(TARGETS.values())].to_numpy(float)
            truth_total = test.hdi.to_numpy(float)
            max_component_difference = max(max_component_difference, float(np.abs(z-reference.loc[test.city_code, ['pred_H','pred_E','pred_I']].to_numpy(float)).max()))
            max_total_difference = max(max_total_difference, float(np.abs(total-reference.loc[test.city_code, 'y_pred'].to_numpy(float)).max()))
            baseline_error = np.abs(np.column_stack([z,total]) - np.column_stack([truth_z,truth_total]))
            for j, unit in enumerate(test.city_code):
                predictions.append(dict(group=group, unit=unit, pred_H=z[j,0], pred_E=z[j,1], pred_I=z[j,2], pred_HDI=total[j]))
            rng = np.random.default_rng(20260930+int(group))
            for block, cols in blocks.items():
                for repeat in range(30):
                    pert = test.copy()
                    order = rng.permutation(len(test))
                    pert.loc[:,cols] = test[cols].to_numpy()[order]
                    zp = predict(pert)
                    yp = synthesis(zp,w)
                    delta = np.abs(np.column_stack([zp,yp])-np.column_stack([truth_z,truth_total]))-baseline_error
                    for j,unit in enumerate(test.city_code):
                        for k,target in enumerate(['H','E','I','HDI']):
                            records.append(dict(group=group,unit=unit,block=block,repeat=repeat,target=target,delta_absolute_error=float(delta[j,k]),permutable=len(test)>1))
    assert len(predictions)==240 and len(records)==240*4*30*4
    assert max_component_difference<1e-9, max_component_difference
    assert max_total_difference<1e-9, max_total_difference
    raw = pd.DataFrame(records)
    old = pd.read_csv(data_file('D70_heldout_group_permutation_repeats.csv'),dtype={'group':str,'unit':str})
    check = raw.loc[raw.target.eq('HDI')].merge(old,on=['group','unit','block','repeat'],validate='one_to_one',suffixes=('_new','_archive'))
    assert len(check)==len(old)==28800
    max_permutation_difference = float(np.abs(check.delta_absolute_error_new-check.delta_absolute_error_archive).max())
    assert max_permutation_difference<1e-9, max_permutation_difference
    means = raw.groupby(['target','block','group','unit'],as_index=False).delta_absolute_error.mean()
    summary = []
    rng = np.random.default_rng(20261002)
    # Province bootstrap preserves within-province dependence; region-equal MAE.
    for target in ['H','E','I','HDI']:
        subset = means.loc[means.target.eq(target)]
        avg = subset.groupby('block').delta_absolute_error.mean()
        total_increment = float(avg.sum())
        group_sum = subset.groupby(['group','block']).delta_absolute_error.sum().unstack('block').loc[:,avg.index]
        group_n = subset.drop_duplicates(['group','unit']).groupby('group').size().loc[group_sum.index]
        ix = rng.integers(0,len(group_sum),size=(10000,len(group_sum)))
        boot = group_sum.to_numpy()[ix].sum(axis=1)/group_n.to_numpy()[ix].sum(axis=1)[:,None]
        boot_share = boot/boot.sum(axis=1)[:,None]*100
        for k,(block,value) in enumerate(avg.items()):
            summary.append(dict(target=target,block=block,mean_MAE_increase=float(value),normalized_share_pct=float(value/total_increment*100),
                ci95_MAE_low=float(np.quantile(boot[:,k],.025)),ci95_MAE_high=float(np.quantile(boot[:,k],.975)),
                ci95_share_low=float(np.quantile(boot_share[:,k],.025)),ci95_share_high=float(np.quantile(boot_share[:,k],.975)),
                n_regions=240,n_groups=24,permutation_repeats=30))
    summary = pd.DataFrame(summary)
    old_summary = pd.read_csv(data_file('D69_heldout_group_permutation_summary.csv'))
    control = summary.loc[summary.target.eq('HDI')].merge(old_summary,on='block',suffixes=('_new','_archive'))
    max_summary_difference = float(np.abs(control.mean_MAE_increase_new-control.mean_MAE_increase_archive).max())
    assert max_summary_difference<1e-9
    OUT.mkdir(parents=True,exist_ok=True)
    raw.to_csv(OUT/'component_permutation_repeats.csv',index=False)
    means.to_csv(OUT/'component_region_importance.csv',index=False)
    summary.to_csv(OUT/'component_importance_summary.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'reproduced_predictions.csv',index=False)
    pd.DataFrame(configs).to_csv(OUT/'frozen_component_configurations.csv',index=False)
    audit = dict(status='PASS',scope='China: 240 held-out cities, 24 provinces',method='Joint block permutation within each held-out province; 30 repeats; frozen selected Ridge configurations; component predictions clipped to [0,1]; region-equal MAE',
        normalization='Signed mean MAE increment for a block divided by the sum of the four signed block increments for each target. Negative estimates are retained.',
        bootstrap='10000 province-cluster bootstrap draws, seed 20261002; ratios recomputed in each draw',
        source_data=str(DATA),source_selection=str(SELECTION),
        max_component_prediction_difference=max_component_difference,max_HDI_prediction_difference=max_total_difference,
        max_D70_permutation_difference=max_permutation_difference,max_D69_summary_difference=max_summary_difference)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2,ensure_ascii=True))
    print(summary.to_string(index=False))
    print(summary.pivot(index='target',columns='block',values='normalized_share_pct').to_string())

if __name__=='__main__':
    main()
