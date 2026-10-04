"""Second-stage constrained aggregation experiment.

P3 is a simplex-weighted geometric mean.
P4 is a simplex mixture of the geometric mean, the three components, and
the three pairwise geometric means.  Both are monotone, idempotent, and
internal (bounded by the component minimum and maximum) by construction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_aggregation_experiment as base

SEED=20260930
PENALTIES=[0.0,1e-4,1e-3,1e-2]


def bases(z,method):
    z=np.clip(np.asarray(z,float),base.EPS,1)
    h,e,i=z.T
    if method=='P3_weighted_geometric':
        return np.log(z),['log_H','log_E','log_I']
    g=np.cbrt(h*e*i)
    return np.column_stack([g,h,e,i,np.sqrt(h*e),np.sqrt(h*i),np.sqrt(e*i)]),[
        'G','H','E','I','sqrt_HE','sqrt_HI','sqrt_EI']


def predict(z,w,method):
    x,_=bases(z,method)
    if method=='P3_weighted_geometric': return np.exp(x@w)
    return x@w


def objective(w,z,y,groups,method,penalty,anchor):
    # A differentiable approximation to group-equal MAE avoids SLSQP stalls
    # at the absolute-loss kink.  Exact MAE is still used for model selection
    # and all reported evaluation metrics.
    residual=np.asarray(y,float)-predict(z,w,method)
    group_array=np.asarray(groups)
    unique,counts=np.unique(group_array,return_counts=True)
    count_map=dict(zip(unique,counts))
    sample_weight=np.array([1.0/count_map[g] for g in group_array],float)
    sample_weight/=sample_weight.sum()
    smooth_abs=np.sqrt(residual*residual+1e-10)
    return float(np.sum(sample_weight*smooth_abs))+penalty*float(np.sum((w-anchor)**2))


def objective_gradient(w,z,y,groups,method,penalty,anchor):
    x,_=bases(z,method)
    prediction=predict(z,w,method)
    residual=np.asarray(y,float)-prediction
    group_array=np.asarray(groups)
    unique,counts=np.unique(group_array,return_counts=True)
    count_map=dict(zip(unique,counts))
    sample_weight=np.array([1.0/count_map[g] for g in group_array],float)
    sample_weight/=sample_weight.sum()
    denominator=np.sqrt(residual*residual+1e-10)
    derivative_prediction=(prediction[:,None]*x if method=='P3_weighted_geometric' else x)
    gradient=np.sum((sample_weight*(-residual/denominator))[:,None]*derivative_prediction,axis=0)
    return gradient+2.0*penalty*(w-anchor)


def solve(z,y,groups,method,penalty):
    _,names=bases(z,method); k=len(names)
    anchor=np.full(3,1/3) if method=='P3_weighted_geometric' else np.array([1.,0,0,0,0,0,0])
    # Deterministic starts: equal weights for the weighted geometric mean and
    # the original geometric mean for the interaction mixture.  The latter is
    # a convex simplex problem after the smooth-L1 approximation.
    starts=[anchor]
    results=[]
    for start in starts:
        result=minimize(objective,start,args=(z,y,groups,method,penalty,anchor),
            jac=objective_gradient,method='SLSQP',bounds=[(0.,1.)]*k,
            constraints={'type':'eq','fun':lambda w:np.sum(w)-1,'jac':lambda w:np.ones_like(w)},
            options={'maxiter':5000,'ftol':1e-11})
        feasible=(np.all(np.isfinite(result.x)) and np.min(result.x)>=-1e-7 and
                  np.max(result.x)<=1+1e-7 and abs(np.sum(result.x)-1)<=1e-6)
        if feasible and np.isfinite(result.fun): results.append(result)
    if not results: raise RuntimeError('Optimizer produced no finite feasible solution')
    result=min(results,key=lambda r:r.fun)
    w=np.clip(result.x,0,1); w=w/w.sum()
    return w,names,float(result.fun)


def fit_rule(z,y,groups,method):
    folds=list(GroupKFold(n_splits=min(5,len(np.unique(groups)))).split(z,groups=groups))
    rows=[]
    for penalty in PENALTIES:
        p=np.full(len(y),np.nan)
        for tr,va in folds:
            w,_,_=solve(z[tr],y[tr],groups[tr],method,penalty)
            p[va]=predict(z[va],w,method)
        rows.append({'penalty':penalty,'group_equal_MAE':base.group_equal_mae(y,p,groups)})
    tuning=pd.DataFrame(rows); best=tuning.group_equal_MAE.min()
    penalty=float(tuning.loc[tuning.group_equal_MAE<=best+1e-12,'penalty'].min())
    w,names,obj=solve(z,y,groups,method,penalty)
    return {'weights':w,'names':names,'penalty':penalty,'objective':obj},tuning


def main():
    old=pd.read_csv(HERE/'outer_predictions.csv',dtype={'unit':str,'group':str})
    p0=old.loc[old.method.eq('P0_geometric')].copy()
    predictions=[p0]; coefficients=[]; tuning_rows=[]
    audit={'status':'RUNNING','seed':SEED,'penalty_grid':PENALTIES,
        'methods':{
            'P3_weighted_geometric':'H^wH E^wE I^wI; w>=0, sum(w)=1',
            'P4_constrained_interactions':'simplex mixture of G,H,E,I,sqrt(HE),sqrt(HI),sqrt(EI)'},
        'properties':'Both candidates are monotone, idempotent and internal by construction.'}
    for country in ['China','USA','Indonesia']:
        matrix,splits,models,fixed,outer_components,sources,counts=base.load_country(country)
        outer_wide=outer_components.pivot(index=['unit','group'],columns='target',values='pred').reset_index()
        matrix_index=matrix.set_index('unit')
        country_rows=[]
        for number,outer in enumerate(splits,1):
            train,z_train,selected=base.crossfit_outer_training(matrix,outer,models,fixed,country)
            fitted={}
            for method in ['P3_weighted_geometric','P4_constrained_interactions']:
                rule,tuning=fit_rule(z_train,train.y_hdi.to_numpy(float),train.group.to_numpy(),method)
                fitted[method]=rule
                for r in tuning.itertuples():
                    tuning_rows.append({'country':country,'outer_group':outer['outer_group'],'method':method,
                        'penalty':r.penalty,'inner_group_equal_MAE':r.group_equal_MAE,
                        'selected':abs(r.penalty-rule['penalty'])<1e-15})
                row={'country':country,'record_type':'outer_fold','outer_group':outer['outer_group'],
                     'method':method,'penalty':rule['penalty'],'weight_sum':rule['weights'].sum(),
                     'minimum_weight':rule['weights'].min()}
                row.update({f'weight_{n}':v for n,v in zip(rule['names'],rule['weights'])})
                coefficients.append(row)
            test=outer_wide.loc[outer_wide.group.astype(str).eq(str(outer['outer_group']))].sort_values('unit')
            z_test=test[base.COMPONENTS].to_numpy(float)
            y=matrix_index.loc[test.unit,'y_hdi'].to_numpy(float)
            for method,rule in fitted.items():
                values=predict(z_test,rule['weights'],method)
                for unit,group,truth,value,h,e,i in zip(test.unit,test.group,y,values,*z_test.T):
                    country_rows.append({'country':country,'unit':unit,'group':group,'outer_group':outer['outer_group'],
                        'method':method,'y_true':truth,'y_pred':value,'pred_H':h,'pred_E':e,'pred_I':i})
            print(f'{country} constrained outer {number:02d}/{len(splits)} {outer["outer_group"]} complete',flush=True)
        predictions.append(pd.DataFrame(country_rows))
        # Post-evaluation deployment weights on country-wide archived OOF components.
        z=outer_wide[base.COMPONENTS].to_numpy(float); y=matrix_index.loc[outer_wide.unit,'y_hdi'].to_numpy(float)
        groups=outer_wide.group.to_numpy()
        for method in ['P3_weighted_geometric','P4_constrained_interactions']:
            rule,_=fit_rule(z,y,groups,method)
            row={'country':country,'record_type':'post_evaluation_deployment','outer_group':'ALL',
                 'method':method,'penalty':rule['penalty'],'weight_sum':rule['weights'].sum(),
                 'minimum_weight':rule['weights'].min()}
            row.update({f'weight_{n}':v for n,v in zip(rule['names'],rule['weights'])})
            coefficients.append(row)
    pred=pd.concat(predictions,ignore_index=True)
    metrics=[]
    for country in ['China','USA','Indonesia']:
        metrics.extend(base.metric_rows(country,pred.loc[pred.country.eq(country)]))
    metrics=pd.DataFrame(metrics)
    rng=np.random.default_rng(SEED); inference=[]
    for country in ['China','USA','Indonesia']:
        d=pred.loc[pred.country.eq(country)]
        for candidate in ['P3_weighted_geometric','P4_constrained_interactions']:
            inference.extend(base.cluster_inference(country,d,candidate,rng))
    inference=pd.DataFrame(inference); inference['holm_p_12_tests']=base.holm(inference.group_signflip_p)
    coeff=pd.DataFrame(coefficients); tuning=pd.DataFrame(tuning_rows)
    # Numerical property audit on every held-out prediction.
    candidates=pred.loc[~pred.method.eq('P0_geometric')].copy()
    candidates['component_min']=candidates[['pred_H','pred_E','pred_I']].min(axis=1)
    candidates['component_max']=candidates[['pred_H','pred_E','pred_I']].max(axis=1)
    tol=1e-10
    prop=candidates.groupby(['country','method']).apply(lambda d:pd.Series({
        'n':len(d),'below_component_min':int((d.y_pred<d.component_min-tol).sum()),
        'above_component_max':int((d.y_pred>d.component_max+tol).sum()),
        'finite':bool(np.isfinite(d.y_pred).all())}),include_groups=False).reset_index()
    pred.to_csv(HERE/'constrained_outer_predictions.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    metrics.to_csv(HERE/'constrained_metrics.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    inference.to_csv(HERE/'constrained_inference.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    coeff.to_csv(HERE/'constrained_coefficients.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    tuning.to_csv(HERE/'constrained_tuning.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    prop.to_csv(HERE/'constrained_property_audit.csv',index=False,encoding='utf-8-sig')
    checks={'each_method_counts':{
                f'{country}|{method}': int(count)
                for (country, method), count in pred.groupby(['country','method']).size().items()
            },
            'all_weights_simplex':bool((abs(coeff.weight_sum-1)<1e-9).all() and (coeff.minimum_weight>=-1e-12).all()),
            'all_predictions_internal':bool((prop.below_component_min.eq(0)&prop.above_component_max.eq(0)).all()),
            'all_predictions_finite':bool(prop.finite.all())}
    audit.update({'status':'PASS' if all([checks['all_weights_simplex'],checks['all_predictions_internal'],checks['all_predictions_finite']]) else 'FAIL',
        'outer_test_independence':'Rules fitted only from outer-training cross-fitted C predictions',
        'bootstrap_replicates':base.BOOTSTRAP_REPS,'signflip_replicates':base.SIGNFLIP_REPS,
        'multiplicity':'Holm over 3 countries x 2 candidates x 2 losses','checks':checks,
        'interpretation':'Weights are predictive aggregation parameters, not welfare weights.'})
    (HERE/'constrained_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    print('\nCONSTRAINED METRICS\n'+metrics.to_string(index=False)); print('\nCONSTRAINED INFERENCE\n'+inference.to_string(index=False))
    print('\nDEPLOYMENT WEIGHTS\n'+coeff.loc[coeff.record_type.eq('post_evaluation_deployment')].to_string(index=False)); print('\nPROPERTY AUDIT\n'+prop.to_string(index=False))


if __name__=='__main__': main()
