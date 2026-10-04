from pathlib import Path
import sys,json,importlib.util
import numpy as np
import pandas as pd
import run_eo_spatial_resolution as cfg
import run_revision_sensitivities as rev
sys.path.insert(0,r'E:\BeyondSurfaceHDI\.geodeps')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from shapely.geometry import mapping
OUT=cfg.OUT;HERE=cfg.HERE
main=pd.read_csv(HERE/'D73_current_no_EO_predictions.csv',dtype={'unit':str,'group':str})
main['scope']='main';main['delta_abs_error']=abs(main.pred_no_EO-main.y_true)-abs(main.pred_full-main.y_true)
full=pd.read_csv(OUT/'D93_Indonesia505_paired_EO_predictions.csv',dtype={'unit':str,'group':str})
full['country']='Indonesia';full['scope']='full505'
allrows=[];summaries=[];rng=np.random.default_rng(cfg.SEED)
def bootstrap(d):
    g=d.groupby('group').delta_abs_error.agg(['sum','count']);ix=rng.integers(0,len(g),(20000,len(g)))
    b=g['sum'].to_numpy()[ix].sum(1)/g['count'].to_numpy()[ix].sum(1)
    return np.quantile(b,[.025,.975])
def metric(d,country,scope,family,stratum,cutlo=np.nan,cuthi=np.nan):
    ci=bootstrap(d);m0=np.mean(abs(d.pred_no_EO-d.y_true));m1=np.mean(abs(d.pred_full-d.y_true))
    return dict(country=country,scope=scope,family=family,stratum=stratum,n=len(d),groups=d.group.nunique(),
      MAE_full=m1,MAE_no_EO=m0,gain=m0-m1,EO_reduction_pct=100*(m0-m1)/m0,removal_increase_pct=100*(m0-m1)/m1,
      ci_low=ci[0],ci_high=ci[1],positive_unit_pct=100*np.mean(d.delta_abs_error>0),cut_low=cutlo,cut_high=cuthi)
for c,d in main.groupby('country',sort=False):
    f=rev.load_inputs(c)[0].set_index('unit')
    prefix='feature_' if c=='China' else ''
    d=d.copy();d['log_population_density']=d.unit.map(f[prefix+'control_log1p_population_density'])
    d['urbanization_rate']=d.unit.map(f[prefix+'control_urbanization_rate'])
    assert d[['log_population_density','urbanization_rate']].notna().all().all()
    summaries.append(metric(d,c,'main','all','all'))
    for variable in ['log_population_density','urbanization_rate']:
        # Numerical terciles, retain tied values in the same stratum.
        cuts=d[variable].quantile([1/3,2/3]).to_numpy();bins=np.r_[-np.inf,np.unique(cuts),np.inf]
        d[variable+'_stratum']=pd.cut(d[variable],bins=bins,labels=['low','middle','high'][:len(bins)-1],include_lowest=True).astype(str)
        for s,p in d.groupby(variable+'_stratum',sort=False):
            index=['low','middle','high'].index(s)
            summaries.append(metric(p,c,'main',variable,s,bins[index],bins[index+1]))
    allrows.append(d)
summaries.append(metric(full,'Indonesia','full505','all','all'))
for s,d in full.groupby('main_status',sort=False): summaries.append(metric(d,'Indonesia','full505','official_scope',s))
# Province-paired bootstrap heterogeneity, retain both class denominators.
t=full.groupby(['group','main_status']).delta_abs_error.agg(['sum','count']).unstack(fill_value=0)
ix=rng.integers(0,len(t),(20000,len(t)));bs=[]
for s in ['excluded_underdeveloped_62','included_443']:
    if s not in t['sum'].columns: s=next(k for k in t['sum'].columns if (k!='included_443')==(s!='included_443'))
    den=t['count'][s].to_numpy()[ix].sum(1);bs.append(np.divide(t['sum'][s].to_numpy()[ix].sum(1),den,out=np.full(len(ix),np.nan),where=den>0))
diff=bs[0]-bs[1];ci=np.nanquantile(diff,[.025,.975]);obs=full.groupby('main_status').delta_abs_error.mean()
hetero=dict(gain62_minus_gain443=float(obs.loc[obs.index!='included_443'].iloc[0]-obs['included_443']),ci_low=float(ci[0]),ci_high=float(ci[1]),valid_bootstrap_replicates=int(np.isfinite(diff).sum()),interpretation='Descriptive, retrospective; same province draws for both groups; no new multiplicity-adjusted significance claim.')
(OUT/'Indonesia505_heterogeneity.json').write_text(json.dumps(hetero,indent=2),encoding='utf-8')
cfg.save(pd.concat(allrows,ignore_index=True),'D95_main_EO_regional_gains.csv');cfg.save(pd.DataFrame(summaries),'D96_EO_spatial_subgroup_metrics.csv')
r160=pd.read_csv(OUT/'D97_China160_CM_predictions.csv',dtype={'unit':str,'group':str})
res=main[main.country=='China'].merge(r160,on=['unit','group','y_true'],validate='one_to_one')
assert len(res)==240
res['delta_abs_error']=abs(res.pred_160-res.y_true)-abs(res.pred_full-res.y_true)
ci=bootstrap(res);stats={name:rev.summary(res.y_true,res[col]) for name,col in [('320m','pred_full'),('160m','pred_160')]}
stats.update(MAE160_minus320=float(res.delta_abs_error.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),prediction_abs_difference_median=float(abs(res.pred_160-res.pred_full).median()),prediction_abs_difference_p95=float(abs(res.pred_160-res.pred_full).quantile(.95)))
(OUT/'D102_China160_320_sensitivity_summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
cfg.save(res,'China160_320_paired_predictions.csv')
# Reuse published boundary/projection conventions; export plot geometry for portability.
path=Path(r'C:\Users\华为\Documents\Codex\2026-09-28\new-chat\redraw_figures_4_8.py')
spec=importlib.util.spec_from_file_location('priorfig',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cn,us,ind=m.boundaries();bounds={'China':{str(k):v for k,v in cn.items()},'USA':us,'Indonesia':ind}
centers={'China':(105,35),'USA':(-99,39),'Indonesia':(118,-3)}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'pdf.fonttype':42,'svg.fonttype':'none'})
norm=TwoSlopeNorm(vmin=-.03,vcenter=0,vmax=.03);cmap=plt.get_cmap('RdBu')
def draw(ax,country,d,title,full=False):
    val=dict(zip(d.unit,d.delta_abs_error));bb=bounds[country]
    if country=='USA':
        lower=[(k,g) for k,g in bb.items() if k[:2] not in {'02','15'}]
        m.draw_map(ax,lower,val,centers[country],cmap,norm)
        for prefix,rect,ctr in [('02',[.015,.01,.24,.26],(-153,64)),('15',[.27,.01,.15,.18],(-157,20))]:
            inset=ax.inset_axes(rect);m.draw_map(inset,[(k,g) for k,g in bb.items() if k[:2]==prefix],val,ctr,cmap,norm)
    else: m.draw_map(ax,list(bb.items()),val,centers[country],cmap,norm)
    ax.set_title(title,loc='left',fontsize=17)
    if full:
        from pyproj import CRS,Transformer
        from shapely.ops import transform
        tx=Transformer.from_crs('EPSG:4326',CRS.from_proj4('+proj=laea +lat_0=-3 +lon_0=118 +datum=WGS84 +units=m +no_defs'),always_xy=True)
        for key in d.loc[d.main_status!='included_443','unit']:
            geo=transform(tx.transform,bb[key]);parts=[geo] if geo.geom_type=='Polygon' else geo.geoms
            for g in parts:
                x,y=g.exterior.xy;ax.plot(x,y,color='#333333',lw=.45)
    ax.text(.5,-.075,f'n={len(d)}; EO helps {100*(d.delta_abs_error>0).mean():.1f}%',ha='center',transform=ax.transAxes,fontsize=13)
def finish(fig,name):
    for extension in ['png','pdf','svg']: fig.savefig(OUT/(name+'.'+extension),dpi=300,bbox_inches='tight',pad_inches=.1)
    plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(14,5.2));fig.subplots_adjust(bottom=.26,top=.88,wspace=.10)
for ax,c,title in zip(axes,['China','USA','Indonesia'],['a  Chinese Mainland','b  United States','c  Indonesia (443 regions)']): draw(ax,c,main[main.country==c],title)
cax=fig.add_axes([.29,.145,.42,.025]);fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',extend='both',label='Absolute-error increment after EO removal (HDI units)')
fig.text(.5,-.055,'Blue: EO helps     Red: EO hurts     Grey: outside scored cohort',ha='center',fontsize=14)
finish(fig,'eo_gain_map')
fig,ax=plt.subplots(figsize=(12,4.6));fig.subplots_adjust(bottom=.24)
draw(ax,'Indonesia',full,'Indonesia (505 regions; 62 designated regencies outlined)',True)
cax=fig.add_axes([.29,.12,.42,.025]);fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',extend='both',label='No-EO absolute error minus full-system absolute error')
finish(fig,'eo_gain_indonesia505')
geofeatures=[]
for c,bb in bounds.items():
    for key,g in bb.items(): geofeatures.append(dict(type='Feature',properties=dict(country=c,unit=key),geometry=mapping(g.simplify(.015,preserve_topology=True))))
(OUT/'eo_map_boundaries.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=geofeatures)),encoding='utf-8')
print(pd.DataFrame(summaries).to_string(index=False));print(json.dumps(hetero));print(json.dumps(stats))
