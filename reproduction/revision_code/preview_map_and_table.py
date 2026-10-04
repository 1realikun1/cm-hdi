"""Separate untitled China map and conventional importance table."""
from pathlib import Path
import sys,json
sys.path.insert(0,r'E:\BeyondSurfaceHDI\.geodeps')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch,Patch
from matplotlib.collections import PatchCollection
from shapely.geometry import shape
from shapely.geometry.polygon import orient
from shapely.ops import transform
from pyproj import CRS,Transformer

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/CMHDI_component_importance_20261002'
PAPER=ROOT/'outputs/CMHDI_EO_spatial_resolution_20261002'
regional=pd.read_csv(OUT/'component_region_importance.csv',dtype={'unit':str})
summary=pd.read_csv(OUT/'component_importance_summary.csv')
city=regional.loc[regional.target.eq('HDI') & regional.block.eq('EO')].set_index('unit').delta_absolute_error.to_dict()
features=[f for f in json.loads((PAPER/'eo_map_boundaries.geojson').read_text(encoding='utf-8'))['features'] if f['properties']['country']=='China']
assert len(city)==240
ink='#263445';slate='#B0BEC5'
plt.rcParams.update({'font.family':'Arial','font.size':11,'text.color':ink,'axes.labelcolor':ink,
    'xtick.color':ink,'ytick.color':ink,'pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(8.0,6.05),facecolor='white')
ax=fig.add_axes([.02,.165,.96,.81]);ax.set_aspect('equal');ax.axis('off')
tx=Transformer.from_crs('EPSG:4326',CRS.from_proj4('+proj=laea +lat_0=35 +lon_0=105 +datum=WGS84 +units=m +no_defs'),always_xy=True)
cmap=LinearSegmentedColormap.from_list('EO_blue_pink',[
    (0,'#467FB8'),(.20,'#7FB3F0'),(.40,'#CDEBFA'),(.5,'#F9FAFC'),
    (.60,'#FFD6E7'),(.80,'#F7A1C4'),(1,'#B64F7F')],N=2048)
norm=TwoSlopeNorm(vmin=-.020,vcenter=0,vmax=.020)
assert max(abs(v) for v in city.values())<.020
patches=[];faces=[];bounds=[]
for f in features:
    unit=str(f['properties']['unit'])
    g=transform(tx.transform,shape(f['geometry'])).simplify(900,preserve_topology=True)
    bounds.append(g.bounds)
    for poly in ([g] if g.geom_type=='Polygon' else g.geoms):
        if poly.geom_type!='Polygon':continue
        poly=orient(poly,sign=1);coords=[];codes=[]
        for ring in [poly.exterior,*poly.interiors]:
            arr=np.asarray(ring.coords);coords.extend(arr)
            codes.extend([MPath.MOVETO]+[MPath.LINETO]*(len(arr)-2)+[MPath.CLOSEPOLY])
        patches.append(PathPatch(MPath(coords,codes)))
        faces.append(cmap(norm(city[unit])) if unit in city else '#E4E8EB')
ax.add_collection(PatchCollection(patches,facecolors=faces,edgecolor='white',linewidth=.40))
bb=np.asarray(bounds);xmin,ymin=bb[:,:2].min(0);xmax,ymax=bb[:,2:].max(0)
ax.set_xlim(xmin-70000,xmax+70000);ax.set_ylim(ymin-70000,ymax+70000)
cax=fig.add_axes([.175,.112,.58,.015])
cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal')
cb.outline.set_visible(False)
cb.set_ticks([-.02,-.01,0,.01,.02],labels=['−0.020','−0.010','0.000','0.010','0.020'])
cb.ax.tick_params(labelsize=10,length=2.4,color=slate)
cb.set_label('EO permutation effect (HDI units)',fontsize=11,labelpad=6)
fig.legend(handles=[Patch(facecolor='#E4E8EB',label='Not evaluated')],loc='lower left',
           bbox_to_anchor=(.785,.073),frameon=False,fontsize=9.5,handlelength=1.2)
for ext in ['png','pdf','svg']:
    fig.savefig(OUT/f'eo_china_map_only.{ext}',dpi=400,facecolor='white')
plt.close(fig)

targets=['H','E','I','HDI'];blocks=['EO','Demography','Economy','Healthcare']
table=summary.pivot(index='target',columns='block',values='normalized_share_pct').loc[targets,blocks]
table.index=['Health','Education','Income','Overall HDI']
table.to_csv(OUT/'importance_share_table.csv',index_label='Prediction target')
fig,ax=plt.subplots(figsize=(8.0,2.3),facecolor='white')
ax.axis('off')
labels=['Prediction target','EO (%)','Population (%)','Economy (%)','Healthcare (%)']
rows=[[name]+[f'{v:.1f}' for v in row] for name,row in table.iterrows()]
tab=ax.table(cellText=rows,colLabels=labels,colWidths=[.25,.15,.20,.20,.20],
             cellLoc='right',colLoc='right',bbox=[0,.09,1,.87])
tab.auto_set_font_size(False);tab.set_fontsize(11)
for (r,c),cell in tab.get_celld().items():
    cell.set_edgecolor('white');cell.set_linewidth(0);cell.PAD=.06
    cell.get_text().set_color(ink)
    if c==0:
        cell.get_text().set_ha('left')
    if r==0:
        cell.get_text().set_weight('bold');cell.visible_edges='BT';cell.set_edgecolor(ink);cell.set_linewidth(.8)
    elif r==len(rows):
        cell.visible_edges='B';cell.set_edgecolor(ink);cell.set_linewidth(.8)
    if c==1:
        cell.get_text().set_weight('bold')
fig.subplots_adjust(left=.025,right=.975,bottom=.02,top=.99)
for ext in ['png','pdf','svg']:
    fig.savefig(OUT/f'importance_share_table.{ext}',dpi=360,facecolor='white')
plt.close(fig)
tex=r'''\begin{table}[htbp]
\caption{Normalized block-permutation importance in the Chinese Mainland.}
\label{tab:importance-shares}\centering\small
\begin{tabular}{lrrrr}
\toprule Prediction target & EO (\%) & Population (\%) & Economy (\%) & Healthcare (\%) \\
\midrule
'''
for name,row in table.iterrows():
    tex += name+' & '+ ' & '.join(f'{v:.1f}' for v in row)+' \\\\\n'
tex+=r'''\bottomrule
\end{tabular}
\end{table}
'''
(OUT/'importance_share_table.tex').write_text(tex,encoding='utf-8')
(OUT/'map_and_table_audit.json').write_text(json.dumps({'mapped_cities':240,'title_removed':True,
    'panel_b_removed':True,'table_values':'Original signed normalized shares; no negative values truncated',
    'map_scale':[-.02,.02],'clipped_values':0,'manuscript_updated':False},indent=2),encoding='utf-8')
print('Untitled map and separate table saved.')
