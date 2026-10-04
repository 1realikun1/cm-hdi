"""Add the audited China block-importance comparison; no archive packaging."""
from pathlib import Path
import shutil
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT/'outputs/CMHDI_EO_spatial_resolution_20261002'
RESULT = ROOT/'outputs/CMHDI_component_importance_20261002'
BACKUP = ROOT/'work/cmhdi_revision/before_component_importance_20261002'
BACKUP.mkdir(exist_ok=True)
for name in ['main.tex','supplement.tex']:
    if not (BACKUP/name).exists():
        shutil.copy2(PAPER/name,BACKUP/name)

summary = pd.read_csv(RESULT/'component_importance_summary.csv')
colors = {'EO':'#F7A1C4','Demography':'#7FB3F0','Economy':'#81D4FA','Healthcare':'#B0BEC5'}
order = list(colors)
plt.rcParams.update({'font.family':'Arial','font.size':10,'text.color':'#263445',
    'axes.labelcolor':'#263445','xtick.color':'#263445','ytick.color':'#263445',
    'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(1,4,figsize=(8.3,3.25),sharex=True,sharey=True)
fig.subplots_adjust(left=.135,right=.988,bottom=.205,top=.77,wspace=.18)
for ax,target,title in zip(axes,['H','E','I','HDI'],['(a) Health','(b) Education','(c) Income','(d) HDI']):
    vals = summary.loc[summary.target.eq(target)].set_index('block').loc[order,'normalized_share_pct'].to_numpy()
    ax.set_axisbelow(True)
    ax.grid(axis='x',color='#E8F0F8',lw=.65)
    ax.axvline(0,color='#B0BEC5',lw=.85,zorder=2)
    ax.barh(np.arange(4),vals,color=[colors[b] for b in order],height=.57,zorder=3)
    for y,v in enumerate(vals):
        ax.text(max(v,0)+2.1,y,f'{v:.1f}%',va='center',fontsize=9.4,
                color='#B64F7F' if y==0 else '#263445',weight='bold' if y==0 else 'normal')
    ax.set_title(title,loc='left',fontsize=10.4,fontweight='bold',pad=12)
    ax.set_xlim(-5,100)
    ax.set_xticks([0,25,50,75,100])
    ax.tick_params(axis='x',labelsize=8.5,length=3,color='#B0BEC5')
    ax.tick_params(axis='y',length=0,labelsize=10)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#B0BEC5')
    ax.spines['bottom'].set_linewidth(.6)
    ax.set_yticks(range(4),['EO','Demography','Economy','Healthcare'])
    ax.set_ylim(3.6,-.65)
fig.text(.135,.945,'Relative predictive importance',fontsize=12.2,fontweight='bold',ha='left')
fig.text(.135,.865,'Chinese Mainland · 240 held-out cities',fontsize=9.6,ha='left',color='#617386')
fig.text(.56,.045,'Normalized block-permutation MAE increment (%)',ha='center',fontsize=10)
for ext in ['pdf','png','svg']:
    fig.savefig(PAPER/f'eo_component_importance.{ext}',dpi=360,facecolor='white')
plt.close(fig)

main = (PAPER/'main.tex').read_text(encoding='utf-8')
old = "Chinese held-out block permutations increase composite MAE by 0.007505 for economy, 0.006516 for demography, 0.000802 for EO and 0.000061 for healthcare, averaged across 30 within-province permutations. Healthcare's descriptive province-bootstrap interval spans zero. The EO perturbation affects the fitted component pipeline despite its small direct-Ridge contrast. The Supplement also reports standardized Ridge coefficients. These diagnostics describe fitted associations and reliance under the specified shuffling scheme."
new = r"""Chinese held-out block permutations increase composite MAE by 0.007505 for economy, 0.006516 for demography, 0.000802 for EO and 0.000061 for healthcare, averaged across 30 within-province permutations. To compare the four input blocks within each prediction target, we divide each mean MAE increment by the sum of the four signed increments. These normalized predictive-importance shares describe reliance of the fitted models under this shuffling scheme, rather than HDI synthesis weights or percentages of accuracy improvement.

Figure~\ref{fig:eo-component-importance} compares the shares for the 240 held-out Chinese cities. EO accounts for 55.4\% in health and 13.9\% in education; its mean MAE increments are 0.001111 and 0.001579, respectively. Demography accounts for 36.6\% in health and 75.3\% in education. Income relies mainly on economy (78.1\%) and demography (22.8\%); its selected models omit EO, giving an EO share of zero. At the composite level, economy and demography account for 50.4\% and 43.8\%, compared with 5.4\% for EO and 0.4\% for healthcare. The health share is less precisely estimated because its total permutation increment is small. Supplementary Section~S8 reports the signed increments and province-bootstrap intervals, including the near-zero healthcare effects.

\begin{figure}[htbp]
\centering\includegraphics[width=\linewidth]{eo_component_importance.pdf}
\caption{Relative block-permutation importance for Chinese component and HDI predictions.}
\label{fig:eo-component-importance}
\end{figure}"""
assert main.count(old)==1
main = main.replace(old,new)
old_disc = 'Economic and demographic blocks have the largest Chinese permutation effects; these conditional predictive diagnostics do not identify causal contributions.'
new_disc = 'EO has its largest relative permutation importance in Chinese health prediction, while demography leads education and economy leads income. Economy and demography dominate the composite diagnostic. The shares are normalized separately for each target, and composite errors also reflect nonlinear synthesis and offsetting component errors; component shares therefore do not average to the HDI share. These fitted-model diagnostics do not identify causal contributions.'
assert main.count(old_disc)==1
main = main.replace(old_disc,new_disc)
(PAPER/'main.tex').write_text(main,encoding='utf-8')

supp = (PAPER/'supplement.tex').read_text(encoding='utf-8')
anchor = '\\section{Supplementary figures}'
assert supp.count(anchor)==1
addition = r"""\subsection{Relative importance by component}
The component analysis uses the same frozen Chinese models and within-province block permutations as D70. It records the health, education and income errors as well as the composite error for each perturbed input matrix. For target $t\in\{H,E,I,\mathrm{HDI}\}$ and block $b$, define
\begin{equation}
 d_{b,t}=\frac{1}{240\times30}\sum_{i=1}^{240}\sum_{r=1}^{30}
 \left(\left|\widehat z_{i,t}^{(b,r)}-z_{i,t}\right|-\left|\widehat z_{i,t}-z_{i,t}\right|\right),
 \qquad s_{b,t}=100\frac{d_{b,t}}{\sum_{b'}d_{b',t}}.
\end{equation}
Here $z_{i,\mathrm{HDI}}=y_i$ and the composite prediction uses the archived fold-specific synthesis weights. A shared row permutation moves all variables in a block together. Blocks have 64 EO dimensions, five demographic variables, one economic variable and four healthcare variables. The model configurations, component clipping and synthesis remain fixed. Negative increments are retained, so normalized shares are signed estimates and can fall outside $[0,100]$. Normalization is performed separately for each target and describes predictive reliance under this perturbation, not causal attribution or synthesis weights.

Table~\ref{tab:component-importance} gives the increments and normalized shares. The 10,000 province-cluster bootstrap draws use seed 20261002 and retain region-equal weights; shares are recalculated in every draw. The health EO share has a wide descriptive 95\% interval (15.7--102.2\%), reflecting variation in both the EO increment and the small normalization denominator. The corresponding intervals are 5.5--23.1\% for education and 2.1--9.2\% for HDI. Income's EO increment is identically zero because its frozen selected configurations exclude EO. Income's healthcare increment is slightly negative and its interval spans zero. The intervals condition on the fitted models and are unadjusted.

\begin{table}[htbp]
\caption{Chinese block-permutation increments and normalized importance shares.}
\label{tab:component-importance}\centering\small
\begin{tabular}{llrrr}
\toprule Target & Input block & MAE increment & 95\% interval & Share (\%) \\
\midrule
"""
for ti,target in enumerate(['H','E','I','HDI']):
    label = {'H':'Health','E':'Education','I':'Income','HDI':'HDI'}[target]
    for bi,block in enumerate(order):
        r = summary.loc[summary.target.eq(target) & summary.block.eq(block)].iloc[0]
        interval = f"[{r.ci95_MAE_low:.6f}, {r.ci95_MAE_high:.6f}]"
        addition += f"{label if bi==0 else ''} & {block} & ${r.mean_MAE_increase:.6f}$ & ${interval}$ & ${r.normalized_share_pct:.1f}$ \\\\\n"
    if ti<3:
        addition += '\\addlinespace\n'
addition += r"""\bottomrule
\end{tabular}
\end{table}

The refits reproduce archived component and composite predictions within $9\times10^{-13}$ and all 28,800 D70 composite permutation changes within $1.2\times10^{-12}$. D103 retains the summary and bootstrap intervals, D104 the 115,200 component/composite permutation records, and D105 the reproduction audit. The figure and analysis code are supplied as individual files.

"""
supp = supp.replace(anchor,addition+anchor)
dict_anchor = 'D102 & Chinese160/320-m paired sensitivity & Continuous scores, paired MAE difference interval and regional prediction-change quantiles. \\\\'
assert supp.count(dict_anchor)==1
supp = supp.replace(dict_anchor,dict_anchor+'\nD103--105 & Component block importance & Chinese health, education, income and composite signed permutation increments, normalized shares, bootstrap intervals, all repeats and reproduction audit. \\\\')
(PAPER/'supplement.tex').write_text(supp,encoding='utf-8')
for src,dst in [('component_importance_summary.csv','D103_China_component_importance_summary.csv'),('component_permutation_repeats.csv','D104_China_component_permutation_repeats.csv'),('audit.json','D105_China_component_importance_audit.json')]:
    shutil.copy2(RESULT/src,PAPER/'supplementary_data'/dst)
for name in ['compute_component_permutation.py','add_component_importance.py']:
    shutil.copy2(Path(__file__).parent/name,PAPER/'supplementary_data/revision_code'/name)
canonical = Path(r'C:\Users\华为\Documents\Codex\2026-09-28\new-chat\CMHDI_manuscript')
for name in ['main.tex','supplement.tex','eo_component_importance.pdf','eo_component_importance.png','eo_component_importance.svg']:
    shutil.copy2(PAPER/name,canonical/name)
print(json.dumps({'figure':'eo_component_importance.pdf','main_updated':True,'supplement_updated':True,'data':['D103','D104','D105'],'archives_created':False}))
