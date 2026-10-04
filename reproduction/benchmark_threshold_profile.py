"""Recompute the Chinese 10-model tolerance profile from existing predictions."""
from pathlib import Path
import pandas as pd
import numpy as np

DATA=Path(__file__).resolve().parent
def main():
    other=pd.read_csv(DATA/'D05_China_additional_benchmark_predictions.csv',dtype={'adm2_code':str,'adm1_code':str})
    other=other.rename(columns={'method':'model','predicted_hdi':'predicted','hdi':'observed','adm2_code':'unit','adm1_code':'province'})
    abc=pd.read_csv(DATA/'D01_ABC_predictions.csv',dtype={'unit':str,'province':str})
    abc=abc.loc[abc.country.eq('China')].rename(columns={'method':'model'})
    rows=pd.concat([other[['model','unit','province','observed','predicted']],abc[['model','unit','province','observed','predicted']]])
    assert not rows.duplicated(['model','unit']).any()
    results=[]
    for model,g in rows.groupby('model',sort=True):
        assert len(g)==240 and g.province.nunique()==24
        error=abs(g.predicted-g.observed); rel=error/g.observed
        r=dict(model=model,n=240,MAE=error.mean(),RMSE=np.sqrt((error**2).mean()))
        for t in (3,5,8,10): r[f'P{t}_n']=int((rel<=t/100).sum()); r[f'P{t}_pct']=100*(rel<=t/100).mean()
        results.append(r)
    profile=pd.DataFrame(results)
    assert len(profile)==10
    profile.to_csv(DATA/'D43_China_complete_threshold_profile.csv',index=False,float_format='%.17g')
    print(profile[['model','P3_pct','P5_pct','P8_pct','P10_pct','MAE','RMSE']].to_string(index=False))
if __name__=='__main__': main()
