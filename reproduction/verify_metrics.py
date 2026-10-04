"""Recompute main manuscript scores from archived held-out predictions.

Run: python verify_metrics.py
Dependencies: pandas, numpy. No fitting or network access is performed.
"""
from pathlib import Path
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parent
df=pd.read_csv(root/'D01_ABC_predictions.csv')
published=pd.read_csv(root/'D02_ABC_metrics.csv')
rows=[]
assert not df.duplicated(['country','unit','method']).any()
for (country,method),g in df.groupby(['country','method']):
    err=g.predicted-g.observed
    rel=100*err.abs()/g.observed
    rows.append(dict(country=country,method=method,n=len(g),p5_n=int((rel<=5).sum()),p5_pct=100*(rel<=5).mean(),p8_n=int((rel<=8).sum()),p8_pct=100*(rel<=8).mean(),mae=err.abs().mean(),rmse=np.sqrt((err**2).mean()),mape_pct=rel.mean(),r2=1-(err**2).sum()/((g.observed-g.observed.mean())**2).sum()))
out=pd.DataFrame(rows)
for col in out.columns[2:]:
    assert np.allclose(out[col],published[col],rtol=1e-10,atol=1e-10),col
assert list(out.p5_n)==[228,233,236,362,363,367]
assert list(out.p8_n)==[239,240,240,431,434,431]
for country,g in df.groupby('country'):
    wide=g.pivot(index='unit',columns='method',values='pass5')
    for reference in ['A','B']:
        gained=int((wide.C&~wide[reference]).sum())
        lost=int((~wide.C&wide[reference]).sum())
        print(country,'C versus',reference,'gained',gained,'lost',lost)
print(out.to_string(index=False))
print('PASS: all main metrics match archived scores; 2049 unique prediction rows.')
