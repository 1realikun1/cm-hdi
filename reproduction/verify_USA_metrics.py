"""Recalculate the United States 100-m A/B/C scores from frozen outer predictions.

Run from any directory with Python 3; this script requires no external packages.
It does not fit a model or access the network.
"""

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GROUPS = {"A_direct": "A", "B_common_components": "B", "C_dimension_specific": "C"}


def read_csv(name):
    with (ROOT / name).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


predictions = read_csv("D18_USA_100m_HDI_predictions.csv")
expected = {
    row["method"]: row
    for row in read_csv("D17_ABC_metrics_with_USA.csv")
    if row["country"] == "USA"
}

for group, method in GROUPS.items():
    rows = [row for row in predictions if row["prediction_group"] == group]
    assert len(rows) == 3104, (group, len(rows))
    assert len({row["adm2_code"] for row in rows}) == len(rows), group
    truth = [float(row["y_true_hdi"]) for row in rows]
    pred = [float(row["y_pred_hdi"]) for row in rows]
    errors = [abs(p - y) for y, p in zip(truth, pred)]
    relative = [100 * e / y for e, y in zip(errors, truth)]
    mean_y = sum(truth) / len(rows)
    squared = [e * e for e in errors]
    scores = {
        "n": len(rows),
        "p5_n": sum(value <= 5 + 1e-12 for value in relative),
        "p8_n": sum(value <= 8 + 1e-12 for value in relative),
        "mae": sum(errors) / len(rows),
        "rmse": math.sqrt(sum(squared) / len(rows)),
        "mape_pct": sum(relative) / len(rows),
        "r2": 1 - sum(squared) / sum((y - mean_y) ** 2 for y in truth),
    }
    scores["p5_pct"] = 100 * scores["p5_n"] / scores["n"]
    scores["p8_pct"] = 100 * scores["p8_n"] / scores["n"]
    for key, actual in scores.items():
        target = float(expected[method][key])
        assert math.isclose(actual, target, rel_tol=0, abs_tol=1e-9), (method, key, actual, target)
    print(method, scores)

print("PASS: United States A/B/C 100-m scores match D17")
