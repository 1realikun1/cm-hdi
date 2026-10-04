"""Publication figure: signed bubble matrix plus EO bootstrap estimates."""
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT/'outputs/CMHDI_EO_spatial_resolution_20261002'
DATA = PAPER/'supplementary_data/D103_China_component_importance_summary.csv'
if not DATA.exists():
    PAPER = Path(__file__).resolve().parents[2]
    DATA = PAPER/'supplementary_data/D103_China_component_importance_summary.csv'
summary = pd.read_csv(DATA)
colors = {'EO':'#F7A1C4','Demography':'#7FB3F0','Economy':'#81D4FA','Healthcare':'#B0BEC5'}
targets = ['H','E','I','HDI']
names = ['Health','Education','Income','HDI']
plt.rcParams.update({'font.family':'Arial','font.size':10,'text.color':'#263445',
    'axes.labelcolor':'#263445','xtick.color':'#263445','ytick.color':'#263445',
    'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig = plt.figure(figsize=(8.9,4.5))
matrix = fig.add_axes([.135,.24,.445,.58])
forest = fig.add_axes([.72,.24,.25,.58])
matrix.axhspan(2.54,3.5,color='#FCE4EC',alpha=.48,zorder=0)
for y in range(5):
    matrix.axhline(y-.5,color='#E8F0F8',lw=.65,zorder=0)
for x in range(5):
    matrix.axvline(x-.5,color='#E8F0F8',lw=.65,zorder=0)
area_scale = 10.5
for row,block in enumerate(colors):
    y = 3-row
    for x,target in enumerate(targets):
        v = float(summary.loc[summary.target.eq(target) & summary.block.eq(block),'normalized_share_pct'].iloc[0])
        if v>0:
            matrix.scatter(x,y+.10,s=v*area_scale,c=colors[block],edgecolors='white',linewidth=.9,zorder=3)
        elif v<0:
            matrix.scatter(x,y+.10,s=abs(v)*area_scale,marker='s',facecolors='none',edgecolors='#617386',linewidth=1.15,zorder=3)
        else:
            matrix.plot(x,y+.10,marker='+',color='#B0BEC5',ms=6,mew=1.0,zorder=3)
        matrix.text(x,y-.29,f'{v:.1f}%',ha='center',va='center',fontsize=9.5,
                    color='#B64F7F' if block=='EO' else '#263445',fontweight='bold' if block=='EO' else 'normal')
matrix.set_xlim(-.5,3.5)
matrix.set_ylim(-.5,3.5)
matrix.set_xticks(range(4),names,fontsize=10)
matrix.xaxis.tick_top()
matrix.tick_params(axis='x',length=0,pad=10)
matrix.set_yticks([3,2,1,0],list(colors),fontsize=10)
matrix.tick_params(axis='y',length=0,pad=9)
for spine in matrix.spines.values():
    spine.set_visible(False)
fig.text(.135,.935,'(a) Input-block comparison',fontsize=11.2,fontweight='bold')
fig.text(.72,.935,'(b) EO uncertainty',fontsize=11.2,fontweight='bold')
eo = summary.loc[summary.block.eq('EO')].set_index('target').loc[targets]
forest.set_axisbelow(True)
forest.grid(axis='x',color='#E8F0F8',lw=.65)
forest.axvline(0,color='#B0BEC5',lw=.8)
for y,(_,r) in zip([3,2,1,0],eo.iterrows()):
    v,lo,hi = r[['normalized_share_pct','ci95_share_low','ci95_share_high']]
    forest.plot([lo,hi],[y,y],color='#D47FA5',lw=1.7,zorder=2)
    forest.plot([lo,lo],[y-.065,y+.065],color='#D47FA5',lw=1.1)
    forest.plot([hi,hi],[y-.065,y+.065],color='#D47FA5',lw=1.1)
    forest.scatter(v,y,s=38,c='#F7A1C4',edgecolor='#B64F7F',linewidth=.65,zorder=3)
    forest.text(v,y+.22,f'{v:.1f}%',ha='center',va='bottom',fontsize=9.1,color='#B64F7F')
forest.set_xlim(-4,110)
forest.set_ylim(-.5,3.5)
forest.set_yticks([3,2,1,0],names,fontsize=9.5)
forest.tick_params(axis='y',length=0,pad=6)
forest.set_xticks([0,25,50,75,100])
forest.tick_params(axis='x',labelsize=9,length=3,color='#B0BEC5')
forest.set_xlabel('EO share (%)',fontsize=10,labelpad=7)
forest.spines['left'].set_visible(False)
forest.spines['bottom'].set_color('#B0BEC5')
forest.spines['bottom'].set_linewidth(.6)
legend_ax = fig.add_axes([.135,.055,.445,.09])
legend_ax.set_xlim(0,1);legend_ax.set_ylim(0,1);legend_ax.axis('off')
legend_ax.text(0,.57,'Share',fontsize=9.2,va='center')
for x,value in zip([.20,.37,.58],[5,25,75]):
    legend_ax.scatter(x,.57,s=value*area_scale,c='#D0E4F7',edgecolors='white',linewidth=.8)
    legend_ax.text(x+.065,.57,f'{value}%',fontsize=8.9,va='center')
legend_ax.scatter(.85,.57,s=26,marker='s',facecolors='none',edgecolors='#617386',linewidth=1)
legend_ax.text(.878,.57,'Negative',fontsize=8.9,va='center')
fig.text(.72,.065,'95% province-bootstrap intervals',fontsize=8.6,color='#617386')
for ext in ['pdf','png','svg']:
    fig.savefig(PAPER/f'eo_component_importance.{ext}',dpi=360,facecolor='white')
plt.close(fig)
canonical = Path(r'C:\Users\华为\Documents\Codex\2026-09-28\new-chat\CMHDI_manuscript')
if canonical.exists() and PAPER!=canonical:
    for ext in ['pdf','png','svg']:
        shutil.copy2(PAPER/f'eo_component_importance.{ext}',canonical/f'eo_component_importance.{ext}')
print(str(PAPER/'eo_component_importance.pdf'))
