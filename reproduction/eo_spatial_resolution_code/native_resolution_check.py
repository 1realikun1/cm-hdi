"""Measurement-only 10/320m comparison in the complete pre-existing pilot block."""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,r'E:\BeyondSurfaceHDI\.geodeps')
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
import run_eo_spatial_resolution as cfg
root=Path(r'E:\BeyondSurfaceHDI')
cache=Path(r'C:\Users\华为\Documents\Codex\2026-09-15\new-chat\alphaearth_sparse_cache\x192t6tua5vez7fgz-0000000000-0000000000')
metadata=cache/'materialized_blocks.json'
protocol={'retrospective':True,'sample':'Entire previously cached page0 block (4,4),1024x1024 native pixels; all1024 aligned320m cells; no outcome-based sampling.',
 'scope':'Single10240m square pilot tile; measurement sensitivity only, no prediction accuracy claim or national representativeness.',
 'comparison':'Dequantize native10m pixels, arithmetic-average each32x32 cell; compare direct official320m vector, including norms, cosine and renormalized-mean residual.'}
(cfg.OUT/'native_check_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
decode=lambda q:np.sign(q.astype(float))*(q.astype(float)/127.5)**2
with rasterio.open(next(cache.glob('*.tiff'))) as ds:
    q=ds.read(window=Window(4096,4096,1024,1024))
    assert q.shape==(64,1024,1024) and np.all(q!=-128)
    assert int(q.sum())==-103283021
    transform=ds.transform;crs=str(ds.crs)
    lo,la=Transformer.from_crs(ds.crs,'EPSG:4326',always_xy=True).transform(*ds.xy(4608,4608))
mean=decode(q).reshape(64,32,32,32,32).mean(axis=(2,4)).transpose(1,2,0).reshape(-1,64)
overview=root/'data_work/alphaearth_overview320/x192t6tua5vez7fgz-0000000000-0000000000.overview320.npy'
o=decode(np.load(overview)[:,128:160,128:160]).transpose(1,2,0).reshape(-1,64)
assert o.shape==mean.shape and np.isfinite(o).all()
n=np.linalg.norm(mean,axis=1);on=np.linalg.norm(o,axis=1);unit=mean/n[:,None]
d=pd.DataFrame({'cell_row':np.repeat(np.arange(128,160),32),'cell_col':np.tile(np.arange(128,160),32),
 'native_mean_norm':n,'overview_norm':on,'raw_L2_difference':np.linalg.norm(o-mean,axis=1),
 'cosine_similarity':np.sum(o*unit,axis=1)/on,'renormalized_mean_L2_difference':np.linalg.norm(o-unit,axis=1)})
cfg.save(d,'D100_native10_overview320_cells.csv')
summary=dict(protocol,source_metadata=str(metadata),source_metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
 overview_sha256=hashlib.sha256(overview.read_bytes()).hexdigest(),native_integer_checksum=int(q.sum()),native_resolution_m=10,
 native_pixel_count=1024**2,overview_cell_count=len(d),crs=crs,center_longitude=lo,center_latitude=la,
 statistics={c:dict(mean=float(d[c].mean()),median=float(d[c].median()),p05=float(d[c].quantile(.05)),p95=float(d[c].quantile(.95))) for c in d.columns[2:]})
(cfg.OUT/'D101_native10_overview320_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
cfg.save(pd.DataFrame(mean,columns=[f'native_mean_A{i:02d}' for i in range(64)]),'native10_cell_means.csv')
cfg.save(pd.DataFrame(o,columns=[f'overview_A{i:02d}' for i in range(64)]),'overview320_cell_vectors.csv')
print(json.dumps(summary,indent=2))
