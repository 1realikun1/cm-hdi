from pathlib import Path
import csv, math

ROOT = Path(__file__).resolve().parent

def metrics(rows):
    y = [float(r["observed"]) for r in rows]
    p = [float(r["predicted"]) for r in rows]
    n = len(y)
    ae = [abs(a-b) for a,b in zip(y,p)]
    se = [(a-b)**2 for a,b in zip(y,p)]
    rel = [x/a for x,a in zip(ae,y)]
    mean_y = sum(y)/n
    return {
        "n": n,
        "P3_pct": 100*sum(x<=0.03 for x in rel)/n,
        "P5_pct": 100*sum(x<=0.05 for x in rel)/n,
        "P8_pct": 100*sum(x<=0.08 for x in rel)/n,
        "P10_pct": 100*sum(x<=0.10 for x in rel)/n,
        "MAE": sum(ae)/n,
        "RMSE": math.sqrt(sum(se)/n),
        "MAPE_pct": 100*sum(rel)/n,
        "R2": 1-sum(se)/sum((v-mean_y)**2 for v in y),
    }

with (ROOT/"D46_C_no_EO_predictions.csv").open(encoding="utf-8", newline="") as f:
    predictions = list(csv.DictReader(f))
with (ROOT/"D44_C_no_EO_metrics.csv").open(encoding="utf-8-sig", newline="") as f:
    expected = list(csv.DictReader(f))

aliases = {"China": "Chinese Mainland", "USA (1 km)": "United States"}
for row in expected:
    country = aliases.get(row["country"], row["country"])
    subset = [r for r in predictions if r["country"] == country and r["strategy"] == row["strategy"]]
    got = metrics(subset)
    for key in ("n","P3_pct","P5_pct","P8_pct","P10_pct","MAE","RMSE","MAPE_pct","R2"):
        want = float(row[key])
        if not math.isclose(float(got[key]), want, rel_tol=1e-11, abs_tol=1e-12):
            raise AssertionError((country, row["strategy"], key, got[key], want))
print("PASS: D44 metrics exactly reproduced from D46 predictions")
