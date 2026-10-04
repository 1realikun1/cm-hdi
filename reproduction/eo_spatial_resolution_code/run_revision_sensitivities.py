"""Retrospective, fixed-protocol sensitivities addressing the CM-HDI audit.

No new candidate or protocol is chosen from the reported outer scores.
All outputs are written beside this script. Original experiment files are read only.
"""
from pathlib import Path
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import sys, json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
OLD=HERE if (HERE/'run_aggregation_experiment.py').exists() else Path(r'C:\Users\华为\Documents\Codex\2026-09-27\new-chat\outputs\HDI_aggregation_experiment_20260928')
SUPP=HERE.parent if (HERE.parent/'D56_constrained_aggregation_predictions.csv').exists() else Path(r'C:\Users\华为\Documents\Codex\2026-09-28\new-chat\CMHDI_Fig2_code_data_package_20260930\data\supplementary_data')
sys.path.insert(0,str(OLD))
import run_aggregation_experiment as base
import run_constrained_aggregation as synth
base.SUPP=SUPP
SEED=20261001
POWER_GRID=[-2.,-1.,-.5,0.,.5,1.,2.]

def save(d,name):
    d.to_csv(HERE/name,index=False,float_format='%.15g',encoding='utf-8')

def summary(y,p):
    y=np.asarray(y);p=np.asarray(p)
    return dict(MAE=float(np.mean(abs(y-p))),RMSE=float(np.sqrt(np.mean((y-p)**2))),
                Spearman=float(spearmanr(y,p).statistic),bias=float(np.mean(p-y)))

def raw_fit(tr,va,columns,alpha):
    x=tr[columns].to_numpy(float);xv=va[columns].to_numpy(float)
    keep=np.ptp(x,axis=0)>0
    scaler=StandardScaler().fit(x[:,keep])
    model=Ridge(alpha=alpha,solver='svd').fit(scaler.transform(x[:,keep]),tr[base.COMPONENTS].to_numpy(float))
    return model.predict(scaler.transform(xv[:,keep]))

def select_components(matrix,outer,models,allowed):
    ix=matrix.set_index('unit',drop=False);tr=ix.loc[outer['outer_train_ids']]
    vals={}
    for model in allowed:
        for alpha in base.core.ALPHAS:
            pred=pd.DataFrame(index=tr.unit,columns=base.COMPONENTS,dtype=float)
            for inner in outer['inner_folds']:
                fit=ix.loc[inner['train_ids']];va=ix.loc[inner['validation_ids']]
                assert not set(fit.group)&set(va.group)
                pred.loc[va.unit]=raw_fit(fit,va,models[model],alpha)
            assert pred.notna().all().all()
            vals[(model,float(alpha))]=pred.loc[tr.unit].to_numpy(float)
    z=np.empty((len(tr),3));choices=[];test=ix.loc[outer['outer_test_ids']];zt=np.empty((len(test),3))
    for j,target in enumerate(base.COMPONENTS):
        scores={k:base.group_equal_mae(tr[target].to_numpy(),v[:,j],tr.group.to_numpy()) for k,v in vals.items()}
        best=min(scores.values())
        eligible=[k for k,v in scores.items() if v<=best+1e-12]
        model,alpha=min(eligible,key=lambda k:(len(models[k[0]]),allowed.index(k[0]),-k[1]))
        z[:,j]=np.clip(vals[(model,alpha)][:,j],0,1)
        zt[:,j]=np.clip(raw_fit(tr,test,models[model],alpha)[:,j],0,1)
        choices.append(dict(outer_group=outer['outer_group'],target=target,model=model,alpha=alpha,inner_group_equal_MAE=scores[(model,alpha)]))
    return tr,z,test,zt,choices

def diagnostic(z,country,group):
    b,_=synth.bases(z,'P4_constrained_interactions')
    bc=(b-b.mean(0))/b.std(0)
    contrast=b[:,1:]-b[:,[0]]
    return dict(country=country,outer_group=group,n=len(z),centered_standardized_condition_number=float(np.linalg.cond(bc)),
                simplex_contrast_condition_number=float(np.linalg.cond(contrast)),simplex_contrast_rank=int(np.linalg.matrix_rank(contrast)))

def power_mean(z,p):
    z=np.clip(z,base.EPS,1)
    return np.exp(np.log(z).mean(1)) if p==0 else np.mean(z**p,axis=1)**(1/p)

def paired(frame):
    rng=np.random.default_rng(SEED);rows=[]
    for country,d in frame.groupby('country',sort=False):
        y=d.y_true.to_numpy();a=abs(d.pred_no_EO.to_numpy()-y)-abs(d.pred_full.to_numpy()-y)
        gs=d.group.to_numpy();groups=np.unique(gs)
        n=np.array([sum(gs==g) for g in groups]);s=np.array([sum(a[gs==g]) for g in groups])
        idx=rng.integers(0,len(groups),(20000,len(groups)))
        boot=s[idx].sum(1)/n[idx].sum(1)
        observed=a.mean();signs=rng.choice([-1.,1.],size=(100000,len(groups)))
        p=(1+np.count_nonzero(abs(signs@s/n.sum())>=abs(observed)-1e-15))/100001
        rows.append(dict(country=country,gain_no_EO_minus_full=observed,ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),p=p,n=len(d),groups=len(groups)))
    result=pd.DataFrame(rows);order=np.argsort(result.p.to_numpy());adj=np.empty(len(result));running=0.
    for i,j in enumerate(order):
        running=max(running,(len(result)-i)*result.p.iloc[j]);adj[j]=min(1.,running)
    result['holm_three_MAE_tests']=adj
    old_tests=pd.read_csv(SUPP/'D41_cross_country_EO_paired_inference.csv')
    all_p=np.r_[old_tests.signflip_p_two_sided.to_numpy(),result.p.to_numpy()]
    order=np.argsort(all_p);joint=np.empty(len(all_p));running=0.
    for i,j in enumerate(order):
        running=max(running,(len(all_p)-i)*all_p[j]);joint[j]=min(1.,running)
    result['holm_39_EO_tests']=joint[-len(result):]
    return result

def load_inputs(country):
    payload_path=HERE/'input_snapshots'/f'{country}_configuration.json'
    if payload_path.exists():
        payload=json.loads(payload_path.read_text(encoding='utf-8'))
        matrix=pd.read_csv(HERE/'input_snapshots'/f'{country}.csv',dtype={'unit':str,'group':str})
        fixed={(r['outer_group'],r['target']):(r['model'],r['alpha']) for r in payload['fixed']}
        return matrix,payload['splits'],payload['models'],fixed
    matrix,splits,models,fixed,_,sources,_=base.load_country(country)
    (HERE/'input_snapshots').mkdir(exist_ok=True)
    save(matrix,'input_snapshots/'+country+'.csv')
    payload=dict(splits=splits,models=models,fixed=[dict(outer_group=k[0],target=k[1],model=v[0],alpha=v[1]) for k,v in fixed.items()],
                 sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    payload_path.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    return matrix,splits,models,fixed

def local_sensitivities():
    archived=pd.read_csv(SUPP/'D56_constrained_aggregation_predictions.csv',dtype={'unit':str,'group':str})
    predrows=[];choice_rows=[];diag=[];powerrows=[];powertuning=[];weightrows=[]
    for country in ['China','USA','Indonesia']:
        matrix,splits,models,fixed=load_inputs(country)
        full=archived.loc[(archived.country==country)&(archived.method=='P4_constrained_interactions')].set_index('unit')
        for k,outer in enumerate(splits):
            tr,z,test,zt,choices=select_components(matrix,outer,models,['R6','R_S10'])
            fitted,_=synth.fit_rule(z,tr.y_hdi.to_numpy(),tr.group.to_numpy(),'P4_constrained_interactions')
            weightrows.append(dict(country=country,outer_group=outer['outer_group'],penalty=fitted['penalty'],
                                   **dict(zip(fitted['names'],fitted['weights']))))
            pn=synth.predict(zt,fitted['weights'],'P4_constrained_interactions')
            for i,r in enumerate(test.itertuples()):
                old=full.loc[r.unit]
                assert abs(old.y_true-r.y_hdi)<1e-9
                predrows.append(dict(country=country,unit=r.unit,group=r.group,y_true=r.y_hdi,pred_full=old.y_pred,pred_no_EO=pn[i],
                  pred_H_no_EO=zt[i,0],pred_E_no_EO=zt[i,1],pred_I_no_EO=zt[i,2],pred_G_no_EO=np.cbrt(np.prod(zt[i]))))
            choice_rows += [dict(country=country,**c) for c in choices]
            train,zfull,_=base.crossfit_outer_training(matrix,outer,models,fixed,country)
            diag.append(diagnostic(zfull,country,outer['outer_group']))
            scores={p:base.group_equal_mae(train.y_hdi.to_numpy(),power_mean(zfull,p),train.group.to_numpy()) for p in POWER_GRID}
            best=min(scores.values());p=min([p for p,s in scores.items() if s<=best+1e-12],key=lambda p:(abs(p),p))
            for exponent,score in scores.items():
                powertuning.append(dict(country=country,outer_group=outer['outer_group'],power=exponent,group_equal_MAE=score,selected=exponent==p))
            original=archived.loc[(archived.country==country)&(archived.method=='P0_geometric')].set_index('unit').loc[test.unit]
            zout=original[['pred_H','pred_E','pred_I']].to_numpy(float)
            pp=power_mean(zout,p)
            powerrows += [dict(country=country,unit=r.unit,group=r.group,y_true=r.y_hdi,y_pred=pp[i],power=p) for i,r in enumerate(test.itertuples())]
            print(f'{country}: fold {k+1}/{len(splits)} complete',flush=True)
            save(pd.DataFrame(predrows),'D73_current_no_EO_predictions.csv')
    preds=pd.DataFrame(predrows);save(preds,'D73_current_no_EO_predictions.csv');save(pd.DataFrame(choice_rows),'D76_current_no_EO_selection.csv')
    metrics=[]
    for country,d in preds.groupby('country',sort=False):
        for setting,col in [('full','pred_full'),('no_EO','pred_no_EO')]:
            metrics.append(dict(country=country,setting=setting,n=len(d),**summary(d.y_true,d[col])))
    save(pd.DataFrame(metrics),'D74_current_no_EO_metrics.csv');save(paired(preds),'D75_current_no_EO_inference.csv')
    save(pd.DataFrame(weightrows),'D87_current_no_EO_synthesis_weights.csv')
    save(pd.DataFrame(diag),'D77_fold_basis_diagnostics.csv')
    save(pd.DataFrame(powerrows),'D78_power_mean_predictions.csv');save(pd.DataFrame(powertuning),'D79_power_mean_tuning.csv')
    pm=pd.DataFrame(powerrows)
    save(pd.DataFrame([dict(country=c,n=len(d),**summary(d.y_true,d.y_pred)) for c,d in pm.groupby('country')]),'D80_power_mean_metrics.csv')

def few_shot():
    oldroot=Path(r'C:\Users\华为\Documents\Codex\2026-09-28\new-chat\CMHDI_Indonesia_zero_shot_20260930')
    source=HERE/'input_snapshots'/'initial_transfer_predictions.csv'
    if not source.exists():
        source=oldroot/'indonesia_zero_shot_predictions.csv'
    d=pd.read_csv(source,dtype={'unit':str,'group':str})
    d=d.loc[d.method.eq('geometric_pooled')].sort_values('unit').reset_index(drop=True)
    (HERE/'input_snapshots').mkdir(exist_ok=True)
    save(d,'input_snapshots/initial_transfer_predictions.csv')
    rng=np.random.default_rng(SEED);rows=[];anchors=[]
    # Anchors never enter the evaluation set. All methods share each draw's remainder.
    for k in [1,5,10]:
        for rep in range(1000):
            ai=rng.choice(len(d),k,replace=False);mask=np.ones(len(d),bool);mask[ai]=False
            tr=d.iloc[ai];te=d.loc[mask];offset=float(np.median(tr.y_true-tr.y_pred))
            for method,p in [('uncalibrated',te.y_pred.to_numpy()),('offset',np.clip(te.y_pred.to_numpy()+offset,0,1))]:
                rows.append(dict(k=k,replicate=rep,method=method,n_test=len(te),offset=offset,**summary(te.y_true,p)))
            anchors += [dict(k=k,replicate=replicate,unit=d.iloc[j].unit) for replicate in [rep] for j in ai]
    records=pd.DataFrame(rows);save(records,'D81_few_shot_replicates.csv');save(pd.DataFrame(anchors),'D82_few_shot_anchors.csv')
    summaries=[]
    for (k,method),part in records.groupby(['k','method']):
        row=dict(k=k,method=method,repetitions=len(part),n_test=int(part.n_test.iloc[0]))
        for metric in ['MAE','Spearman','bias']:
            row[metric+'_median']=part[metric].median();row[metric+'_q025']=part[metric].quantile(.025);row[metric+'_q975']=part[metric].quantile(.975)
        summaries.append(row)
    save(pd.DataFrame(summaries),'D83_few_shot_summary.csv')

if __name__=='__main__':
    protocol=dict(status='FROZEN_BEFORE_NEW_OUTER_SCORES',seed=SEED,retrospective=True,
      EO='Remove all EO-containing candidate sets; reselect three Ridge configurations using the same five grouped inner folds, raw group-equal MAE, five alpha values, then refit the current seven-term synthesis. Compare with archived full-system predictions at current USA 100 m resolution.',
      power_mean=dict(grid=POWER_GRID,selection='outer-training cross-fitted group-equal MAE; ties prefer exponent closest to zero'),
      few_shot=dict(k=[1,5,10],repetitions=1000,method='median signed-residual offset; clip to [0,1]; score only non-anchor units; paired uncalibrated comparator on identical remainder'),
      inference='20,000 cluster bootstraps; 100,000 group sign flips; Holm over three current no-EO MAE tests and a conservative sensitivity combining those with the 36 archived EO tests',
      note='Neither repeated few-shot draw quantiles nor retrospective sensitivities are independent confirmatory confidence intervals.')
    (HERE/'revision_sensitivity_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    few_shot()
    local_sensitivities()
    print('ALL SENSITIVITIES COMPLETE',flush=True)
