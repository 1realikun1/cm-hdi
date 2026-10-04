"""Paired group-level inference for strategy C versus shared-input B.

Uses only archived out-of-fold predictions. The bootstrap conditions on these
predictions and does not refit the models or account for earlier model search.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA = SCRIPT_DIR if (SCRIPT_DIR / "D01_ABC_predictions.csv").exists() else (
    SCRIPT_DIR.parent / "outputs" / "JAG_manuscript_US_uncertainty_20260927" / "supplementary_data"
)
OUT = DATA / "D33_paired_C_vs_B_inference.csv"
SEED = 20260927
B = 20_000


def _seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return SEED + int.from_bytes(digest[:4], "big")


def _read_china_indonesia() -> list[pd.DataFrame]:
    all_rows = pd.read_csv(DATA / "D01_ABC_predictions.csv", dtype={"unit": str, "province": str})
    result = []
    for name in ("China", "Indonesia"):
        rows = all_rows.loc[(all_rows.country == name) & all_rows.method.isin(["B", "C"])].copy()
        wide = rows.pivot(index=["unit", "province"], columns="method", values=["observed", "predicted"])
        assert len(wide) in (240, 443)
        assert np.allclose(wide[("observed", "B")], wide[("observed", "C")], atol=1e-12)
        frame = pd.DataFrame({
            "country": name,
            "unit": wide.index.get_level_values("unit"),
            "group": wide.index.get_level_values("province"),
            "observed": wide[("observed", "B")].to_numpy(float),
            "B": wide[("predicted", "B")].to_numpy(float),
            "C": wide[("predicted", "C")].to_numpy(float),
        })
        assert frame.unit.is_unique
        result.append(frame)
    return result


def _read_us() -> pd.DataFrame:
    rows = pd.read_csv(DATA / "D18_USA_100m_HDI_predictions.csv", dtype={"adm1_code": str, "adm2_code": str})
    rows = rows.loc[rows.prediction_group.isin(["B_common_components", "C_dimension_specific"])].copy()
    rows["method"] = rows.prediction_group.map({"B_common_components": "B", "C_dimension_specific": "C"})
    wide = rows.pivot(index=["adm2_code", "adm1_code"], columns="method", values=["y_true_hdi", "y_pred_hdi"])
    assert len(wide) == 3104
    assert np.allclose(wide[("y_true_hdi", "B")], wide[("y_true_hdi", "C")], atol=1e-12)
    return pd.DataFrame({
        "country": "United States",
        "unit": wide.index.get_level_values("adm2_code"),
        "group": wide.index.get_level_values("adm1_code"),
        "observed": wide[("y_true_hdi", "B")].to_numpy(float),
        "B": wide[("y_pred_hdi", "B")].to_numpy(float),
        "C": wide[("y_pred_hdi", "C")].to_numpy(float),
    })


def _holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adjusted = np.empty(len(pvals))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def main() -> None:
    frames = _read_china_indonesia() + [_read_us()]
    records = []
    for frame in frames:
        country = str(frame.country.iloc[0])
        y = frame.observed.to_numpy(float)
        b = frame.B.to_numpy(float)
        c = frame.C.to_numpy(float)
        groups, inverse = np.unique(frame.group.to_numpy(str), return_inverse=True)
        counts = np.bincount(inverse).astype(float)
        for metric in ("P5_gain_pp", "MAE_gain"):
            if metric == "P5_gain_pp":
                delta = 100 * ((np.abs(c - y) / y <= 0.05).astype(float) - (np.abs(b - y) / y <= 0.05).astype(float))
            else:
                delta = np.abs(b - y) - np.abs(c - y)
            group_sums = np.bincount(inverse, weights=delta)
            estimate = float(delta.mean())
            leave_one_group_out = (delta.sum() - group_sums) / (len(frame) - counts)
            rng = np.random.default_rng(_seed(f"{country}:{metric}:bootstrap"))
            indices = rng.integers(0, len(groups), size=(B, len(groups)))
            boot = group_sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
            ci_low, ci_high = np.quantile(boot, [0.025, 0.975]).tolist()

            # The paired cluster sign-flip test assumes group-level contrasts
            # are exchangeable around zero under the null. Model predictions
            # and eligibility are kept fixed within each replicate.
            rng = np.random.default_rng(_seed(f"{country}:{metric}:signflip"))
            signs = rng.choice(np.array([-1.0, 1.0]), size=(B, len(groups)))
            null = signs @ group_sums / len(frame)
            p_two_sided = (1 + int(np.count_nonzero(np.abs(null) >= abs(estimate) - 1e-15))) / (B + 1)
            records.append({
                "country": country,
                "metric": metric,
                "contrast": "C_minus_B" if metric == "P5_gain_pp" else "B_MAE_minus_C_MAE",
                "estimate": estimate,
                "bootstrap_95_lo": ci_low,
                "bootstrap_95_hi": ci_high,
                "leave_one_group_out_min": float(leave_one_group_out.min()),
                "leave_one_group_out_max": float(leave_one_group_out.max()),
                "signflip_p_two_sided": p_two_sided,
                "n_units": len(frame),
                "n_outer_groups": len(groups),
                "replicates": B,
                "seed": SEED,
            })
    adjusted = _holm([r["signflip_p_two_sided"] for r in records])
    for record, p_holm in zip(records, adjusted):
        record["holm_p_six_tests"] = p_holm
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(pd.DataFrame(records).to_string(index=False, float_format=lambda x: f"{x:.8g}"))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
