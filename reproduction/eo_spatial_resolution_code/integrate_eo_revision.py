from pathlib import Path
import re
import json,shutil,hashlib,zipfile
import numpy as np,pandas as pd
import run_eo_spatial_resolution as cfg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=cfg.OUT;HERE=cfg.HERE;OLD=OUT.parent/'CMHDI_revision_20261001'
for name in ['main.tex','supplement.tex','hdi_references.bib','reference-style.tex']:
    shutil.copy2(OLD/name,OUT/name)
main=(OUT/'main.tex').read_text(encoding='utf-8')
addition=r'''The native AlphaEarth product has 10-m pixels; the 320-m representation used here is a data-extraction choice, not a display resolution. For a quantized axis value $q\ne-128$, decoding uses $\operatorname{sign}(q)(q/127.5)^2$. Official overview vectors are approximately unit normalized before population sampling. Averaging native vectors and renormalizing their average are different operators, so an overview-based regional mean cannot be called a native-pixel mean. Section~\ref{sec:eo-resolution} evaluates this distinction through a local 10/320-m measurement check and a matched Chinese 160/320-m prediction sensitivity.'''
anchor="Five demographic controls describe total population"
assert main.count(anchor)==1;main=main.replace(anchor,addition+'\n\n'+anchor)
block=r'''\subsection{Spatial heterogeneity of EO gains}\label{sec:eo-spatial}
For each held-out region, define $\Delta_i=|\widehat y_i^{\mathrm{noEO}}-y_i|-|\widehat y_i^{\mathrm{full}}-y_i|$. Positive values mean that permitting EO in the component library reduces absolute error; negative values mean it increases error. Figure~\ref{fig:eo-gain} maps the adopted-system comparison from Table~\ref{tab:eo-system}. EO reduces error in 57.1\% of Chinese cities, 63.6\% of United States counties and 56.2\% of Indonesian main-cohort units. All scored regions are shown; the common colour scale saturates beyond $\pm0.03$ HDI units, with exact values retained in D95. Area on the map is not the weighting used in MAE.

The gains vary with observed regional characteristics. In China, the lowest urbanization third has mean gain 0.003471, compared with 0.000217 in the highest third. The corresponding United States gains are 0.006279 and 0.003890, while Indonesian main-cohort gains are 0.000571 and 0.004030. Population-density summaries likewise differ across settings. These descriptive strata use input-defined within-country thirds and preserve tied values; country-specific urbanization definitions prevent a common physical interpretation. They show where the evaluated models benefit, without identifying a causal EO mechanism or demonstrating performance where labels are unavailable. Supplementary Section~\ref{sec:s-eo-spatial} reports every stratum and its group-bootstrap interval.

\begin{figure}[H]
\centering\includegraphics[width=\linewidth]{eo_gain_map.pdf}
\caption{Regional absolute-error changes after EO removal from the adopted component-library search. Blue denotes lower error with EO, red higher error with EO and grey regions outside the scored cohort. Alaska and Hawaii appear in separate insets.}
\label{fig:eo-gain}
\end{figure}

A separate, matched 505-unit Indonesian analysis includes the 62 designated underdeveloped regencies in both full and no-EO training. Full/no-EO MAEs are 0.026266/0.028350 overall and 0.021062/0.023915 in its 443-unit subset. In the 62-regency subset, however, full/no-EO MAEs are 0.063454/0.060035: the mean gain is $-0.003419$ (descriptive province-bootstrap 95\% interval $[-0.008841,0.005043]$), and only 43.5\% of units benefit. The difference in mean gain between the 62 and 443 classes is $-0.006272$ ($[-0.012145,0.002849]$). These intervals condition on the paired outer predictions and are not new multiplicity-adjusted confirmatory tests. The results do not support stronger EO gains in the most disadvantaged class. This 505-unit refit is kept separate from the original 443-unit fits; nine coverage-ineligible units remain unassessed.

\subsection{Sensitivity to EO representation resolution}\label{sec:eo-resolution}
In a previously cached 10.24-km square Chinese pilot block centred near $117.493^\circ$E, $32.993^\circ$N, all 1,048,576 valid native 10-m pixels form 1,024 aligned 320-m cells. Comparing each cell's decoded native-vector arithmetic mean with its official overview gives median vector norms of 0.94718 and 1.00028, respectively, despite median cosine similarity 0.999937. Median Euclidean differences are 0.05424 before native-mean renormalization and 0.01144 after it. This local measurement check demonstrates the normalization distinction; it neither reproduces population-weighted administrative extraction at 10 m nor represents national spatial heterogeneity.

The separate predictive sensitivity replaces only the 64 Chinese EO means with existing population-weighted 160-m overview means. All 280 training-pool cities, 240 scored cities, targets, candidate libraries, penalties, grouped inner and outer partitions and synthesis rules are retained; configurations are reselected within training partitions. CM-HDI MAEs are 0.012129 at 160 m and 0.012062 at 320 m. The paired difference, 160 minus 320, is 0.000067 (descriptive province-bootstrap 95\% interval $[-0.000348,0.000518]$); median and 95th-percentile absolute prediction changes are 0.000652 and 0.003962. This check finds little aggregate MAE change across these two overview scales, but does not establish equivalence or validate native 10-m prediction. No corresponding 160-m rerun is claimed for the United States or Indonesia. Supplementary Section~\ref{sec:s-eo-resolution} supplies the protocol, paired predictions and measurement diagnostics.

'''
assert main.count(r'\section{Discussion}')==1
block=block.replace(r'Supplementary Section~\ref{sec:s-eo-spatial}',r'Supplementary Section~S10').replace(r'Supplementary Section~\ref{sec:s-eo-resolution}',r'Supplementary Section~S11')
main=main.replace(r'\section{Discussion}',block+r'\section{Discussion}')
# Add explicit practical limits without changing older conclusions.
phrase='EO may capture settlement form, infrastructure, land use or other conditions imperfectly represented by tabular predictors'
assert main.count(phrase)==1
main=main.replace(phrase,'The spatial removal analysis reveals heterogeneous gains, and the full-coverage Indonesian comparison does not show an advantage in the 62 designated underdeveloped regencies. The resolution sensitivity supports reporting the actual overview extraction scale explicitly, while native-scale model performance remains untested.\n\n'+phrase)
(OUT/'main.tex').write_text(main,encoding='utf-8')
# Diagnostic plot, fully reproducible from released small tables.
d=pd.read_csv(OUT/'D100_native10_overview320_cells.csv');res=pd.read_csv(OUT/'China160_320_paired_predictions.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axs=plt.subplots(1,3,figsize=(12,3.7));fig.subplots_adjust(wspace=.36,bottom=.23)
axs[0].scatter(d.native_mean_norm,d.overview_norm,s=5,alpha=.4,color='#3077a5');axs[0].plot([.82,1.02],[.82,1.02],color='grey',lw=.8)
axs[0].set(xlabel='Native cell-mean norm (10 m)',ylabel='Overview vector norm (320 m)',title='a  Native vs overview')
axs[1].hist(d.raw_L2_difference,bins=30,alpha=.6,label='Raw native mean',color='#bd4d55');axs[1].hist(d.renormalized_mean_L2_difference,bins=20,alpha=.7,label='Renormalized native mean',color='#3077a5')
axs[1].set(xlabel='L2 difference from overview',ylabel='320 m cells',title='b  Normalization difference');axs[1].legend(fontsize=10)
axs[2].scatter(res.pred_full,res.pred_160,s=9,color='#3077a5',alpha=.6);axs[2].plot([.6,.92],[.6,.92],color='grey',lw=.8)
axs[2].set(xlabel='Prediction at 320 m',ylabel='Prediction at 160 m',title='c  240 held-out cities')
for ax in axs: ax.spines[['top','right']].set_visible(False)
for ext in ['png','pdf','svg']: fig.savefig(OUT/('eo_resolution_sensitivity.'+ext),dpi=300,bbox_inches='tight')
plt.close(fig)
supp=(OUT/'supplement.tex').read_text(encoding='utf-8')
supp=supp.replace(r'\usepackage[labelsep=period]{caption}',r'\usepackage[labelsep=period]{caption}'+'\n'+r'\usepackage{tocloft}'+'\n'+r'\setlength{\cftsecnumwidth}{3em}'+'\n'+r'\setlength{\cftsubsecnumwidth}{3.5em}')
groups=pd.read_csv(OUT/'D96_EO_spatial_subgroup_metrics.csv')
table=[]
for c in ['China','USA','Indonesia']:
    for family,label in [('log_population_density','Density'),('urbanization_rate','Urbanization')]:
        for s in ['low','middle','high']:
            r=groups.query('country==@c and scope=="main" and family==@family and stratum==@s').iloc[0]
            table.append(f"{c} & {label}: {s} & {int(r.n)} & {r.MAE_full:.6f} & {r.MAE_no_EO:.6f} & {r.gain:.6f} & [{r.ci_low:.6f},{r.ci_high:.6f}] \\\\")
new=r'''\section{Spatial heterogeneity of EO removal}\label{sec:s-eo-spatial}
This retrospective analysis uses every adopted-system paired prediction in D73. Region-level gain is $\Delta_i=|\widehat y_i^{\mathrm{noEO}}-y_i|-|\widehat y_i^{\mathrm{full}}-y_i|$. D95 retains exact regional gains and strata, including negative values; the common main-map scale saturates at $\pm0.03$. Density and urbanization thirds are specified from the observed input distributions within each scored country, before subgroup outcomes are calculated. Values on a tercile boundary are kept together, explaining unequal United States urbanization stratum sizes when many values are zero. These strata use log population density and each country's existing urbanization control. No outcome, error or newly selected threshold defines a stratum.

Table~\ref{tab:s-eo-strata} reports all six strata in every country. Region-equal average gains and pointwise percentile intervals resample complete provinces or states 20,000 times, with seed 20261002. The intervals condition on the outer predictions and omit model-retraining uncertainty. They are descriptive exploratory summaries, without subgroup hypothesis tests or a claim of cross-country physical comparability. The earlier main-cohort multiplicity correction remains in force for its specified tests. The mean gain is not the share of units benefiting or an area-weighted benefit.

\begin{table}[htbp]
\caption{Input-defined regional strata in the original main cohorts. Positive gain means lower absolute error with EO.}
\label{tab:s-eo-strata}\centering\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{llrrrrl}
\toprule Setting & Stratum & $N$ & Full MAE & No-EO MAE & Gain & Pointwise 95\% interval \\
\midrule
'''+'\n'.join(table)+r'''
\bottomrule\end{tabular}
\end{table}

\paragraph{Matched 505-unit analysis.} The full comparator is the adopted corrected-label 505-unit nested evaluation in D84, not the original main443 fit. The no-EO run uses exactly the same units, labels, outer leave-one-province-out partitions and five grouped inner folds. All EO-containing candidates are deleted; R6 and R\_S10 remain, with the original five penalties and tie rules. Raw group-equal component MAE selects configurations; cross-fitted clipped training components fit the original constrained seven-term synthesis with its existing penalty grid. Thus the same training-cohort change applies to both sides of this contrast. D93--D94 retain paired predictions and 102 component selections.

\begin{table}[htbp]
\caption{EO removal in the same 505-unit Indonesian nested experiment. All subset values come from these 505-unit fits.}
\label{tab:s-eo-505}\centering\small
\begin{tabular}{lrrrrl}
\toprule Scope & $N$ & Full MAE & No-EO MAE & Gain & Pointwise 95\% interval \\
\midrule
All eligible & 505 & 0.026266 & 0.028350 & 0.002083 & [0.000307,0.003840] \\
Main-scope class & 443 & 0.021062 & 0.023915 & 0.002853 & [0.001433,0.004314] \\
Designated underdeveloped & 62 & 0.063454 & 0.060035 & $-0.003419$ & [$-0.008841$,0.005043] \\
\bottomrule\end{tabular}
\end{table}

The 62-regency class occurs in 11 provinces. EO reduces individual absolute error in 43.5\% of that class, versus 57.3\% of the 443-unit class within the same experiment. Relative to no EO, the mean-error reduction is $-5.70\%$ in the 62 class and $+11.93\%$ in the 443 class. A paired province bootstrap uses the same draws for both classes: the difference in mean gains, 62 minus 443, is $-0.006272$ with interval $[-0.012145,0.002849]$; all 20,000 replicates contain both classes. This is an exploratory heterogeneity estimate, not proof of an adverse effect or a confirmatory difference. The point estimates do not support stronger EO benefits in the designated class. Nine coverage-ineligible units remain outside this evaluation.

\begin{figure}[htbp]
\centering\includegraphics[width=\linewidth]{eo_gain_indonesia505.pdf}
\caption{EO gains in the matched 505-unit Indonesian sensitivity. Dark outlines identify all 62 designated underdeveloped regencies; grey units are coverage-ineligible. Predictions are not spliced with those of the main443 experiment.}
\end{figure}

\section{EO representation and resolution sensitivity}\label{sec:s-eo-resolution}
The native annual AlphaEarth embedding has 64 axes at 10 m \citep{googleaef,brown2025}. The pipeline reads the official 320-m COG overview (IFD page5) and decodes each valid int8 value with $\operatorname{sign}(q)(q/127.5)^2$; $q=-128$ is NoData. The overview contains approximately unit-length vectors. China and United States positive-population 100-m cell centres, and Indonesian 1-km centres, query those 320-m vectors before population-weighted administrative aggregation. Sampling at 100 m does not restore native 10-m information. Neither an overview pixel nor the final regional mean is represented as a physical land-cover variable.

For native vectors $\mathbf e_j$ in a coarse cell, an arithmetic mean $\mathbf m=K^{-1}\sum_j\mathbf e_j$ generally has norm below one. A normalized direction $\mathbf m/\|\mathbf m\|$ has different magnitude and may have different population-weighted administrative aggregation; nonlinear decoding, quantization, sampling support and boundary assignment also matter. The small check below directly compares the available vectors rather than assuming equality or attributing all differences to a specific undocumented COG generation algorithm.

\paragraph{Native10/overview320 measurement check.} The complete pre-existing cached native block(4,4) in the 2020 UTM50N COG, centred at $117.493^\circ$E,$32.993^\circ$N, spans10.24 km per side. Its64 planar bands have all page0 segments present. The frozen integer checksum is $-103283021$, matching the earlier range-read audit. All1,048,576 pixels are valid; each aligned32-by32 native window defines one of1,024 overview cells. All cells are used, without sampling by outcomes or reconstruction quality. D100 retains cell-level comparisons; D101 records source hashes, validity, location and quantiles. Small native cell means and overview vectors are released for recalculation; regenerating them from raw native pixels requires the source COG or existing cache.

Native cell-mean and overview norms have medians0.947177 and1.000283, respectively; their 5th--95th percentile ranges are0.896257--0.990934 and0.996988--1.003612. Median cosine similarity is0.999937. Median L2 differences before and after native-mean renormalization are0.054244 and0.011436. High directional agreement therefore does not imply equality of the vectors being averaged. This is a measurement check in one previously selected pilot block; it is not a representative national sample, a test of population-weighted native administrative features or a native10-m model-performance experiment.

\paragraph{Matched160/320-m predictive sensitivity.} Existing160-m population-weighted means are joined to the frozen280-city Chinese matrix by city code. Only the64 EO columns are replaced. All other inputs and labels, the40 development-city roles,24 outer province folds,5 grouped inner folds,5 input candidates,5 Ridge penalties, tie rules, clipping and7-term synthesis are identical to the320-m experiment. Input sets and penalties are reselected using training partitions only. An independent reconstruction of every320-m outer fold verifies numerical agreement with the adopted full comparator. D97--D99 retain160-m predictions, selections and the protocol; D102 records paired summary scores.

The240 scored cities give160-m MAE0.012129, RMSE0.015492 and Spearman0.908587, versus320-m0.012062,0.015573 and0.906178. The paired MAE difference is0.0000673, with20,000-replicate province-bootstrap interval$[-0.0003484,0.0005183]$. Median and95th-percentile absolute prediction changes are0.0006522 and0.0039615. These findings describe modest aggregate sensitivity between two overview scales under this Chinese regional task. No equivalence margin was defined, and a near-zero mean-error change can conceal regional changes. The original320-m results remain primary. Native10-m predictive performance and United States/Indonesian160-m sensitivities have not been evaluated.

\begin{figure}[htbp]
\centering\includegraphics[width=\linewidth]{eo_resolution_sensitivity.pdf}
\caption{Resolution diagnostics: native10-m arithmetic means versus official320-m vectors in one pilot block(a--b), and Chinese held-out predictions after matched160-m replacement(c). The two checks use different supports and do not jointly constitute a national native10-m accuracy test.}
\end{figure}

'''
# Insert immediately before final data dictionary, preserve S1--S10 references.
new=re.sub(r'([a-z])(\d)',r'\1 \2',new)
new=re.sub(r'(\d)(m|km)(?=[ ,.;])',r'\1 \2',new)
new=new.replace('eo_gain_indonesia 505.pdf','eo_gain_indonesia505.pdf')
new=new.replace('brown 2025','brown2025')
new=new.replace('32-by 32','32-by-32').replace('block(4,4)','block (4,4)').replace('UTM50N','UTM zone 50N').replace('block(a--b)','block (a--b)').replace('replacement(c)','replacement (c)')
new=re.sub(r'([A-Za-z]),(?=[\w$])',r'\1, ',new)
new=re.sub(r'(\.\d+),(?=\d+\.)',r'\1, ',new)
anchor=next(line for line in supp.splitlines() if line.startswith(r'\section{') and ('dictionary' in line.lower() or 'data files' in line.lower()))
supp=supp.replace(anchor,new+anchor)
dictionary=r'''D93--94 & Full505 paired EO removal & Corrected-label paired predictions and no-EO component selections for the same505-unit nested experiment. \\
D95--96 & Spatial EO gains and strata & Every main-cohort regional gain, input-defined terciles and all subgroup scores and descriptive cluster intervals. \\
D97--99 & Chinese160-m sensitivity & Matched held-out predictions, inner selection decisions and frozen retrospective analysis protocol. \\
D100--101 & Native10/overview320 measurement check & All1,024 cell diagnostics, source hashes, validity and summary quantiles in one cached pilot block. \\
D102 & Chinese160/320-m paired sensitivity & Continuous scores, paired MAE difference interval and regional prediction-change quantiles. \\
'''
assert supp.count('D92 &')==1
line=next(x for x in supp.splitlines() if x.startswith('D92 &'))
supp=supp.replace(line,line+'\n'+dictionary)
(OUT/'supplement.tex').write_text(supp,encoding='utf-8')
# Preserve the complete previous archive; extend it with auditable new records.
data=OUT/'supplementary_data';shutil.copytree(OLD/'supplementary_data',data,dirs_exist_ok=True)
code=data/'eo_spatial_resolution_code';code.mkdir(exist_ok=True)
for f in OUT.iterdir():
    if f.is_file() and f.suffix in {'.csv','.json','.geojson'}: shutil.copy2(f,data/f.name)
for name in ['run_eo_spatial_resolution.py','native_resolution_check.py','summarize_eo_spatial.py','verify_resolution_reference.py','integrate_eo_revision.py']:
    shutil.copy2(HERE/name,code/name)
for name in ['run_revision_sensitivities.py','run_aggregation_experiment.py','run_constrained_aggregation.py','run_indonesia_abc_2020.py','run_usa_c_validation_suite.py']:
    shutil.copy2(data/'revision_code'/name,code/name)
shutil.copytree(data/'revision_code/input_snapshots',code/'input_snapshots',dirs_exist_ok=True)
for name in ['D73_current_no_EO_predictions.csv','D84_Indonesia505_predictions.csv']:
    shutil.copy2(data/name,code/name)
(code/'README.txt').write_text('Recalculation: D93--D102 and the small native cell-mean/vector tables reproduce published metrics and resolution diagnostics. The nested rerun scripts use the frozen input snapshots and original candidate definitions. Geographic rebuilding and native-pixel aggregation require the E:/BeyondSurfaceHDI sources identified in the protocols; plot code exports eo_map_boundaries.geojson for map portability. Newly generated files go to the script-defined output directory. All analyses are retrospective; no new subgroup significance claim is made.',encoding='utf-8')
notes='''本次完成第①和③项。①主文新增区域EO增益地图，补充密度/城市化分层和印尼505配对EO移除实验。62县的增益点估计为负，区间跨零，未宣称欠发达县增益更大。③明确320m是实际提取尺度，新增全中国160/320m匹配预测敏感性及一个缓存样区的原生10/320m向量测量对照。未将局部样区或160m测试表述为全国原生10m性能验证。所有新增实验为回顾性敏感性；主队列和原来的图7浮动修复保留。数据记录扩展到D102。'''
(OUT/'修改说明.txt').write_text(notes,encoding='utf-8')
print('Sources and archive prepared:',OUT)
