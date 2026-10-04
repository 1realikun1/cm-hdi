"""Paired inference for the incremental predictive value of EO in the US data.

The analysis compares fixed-input, out-of-fold county predictions archived in
D18.  Each contrast adds the 64-dimensional AlphaEarth block while retaining
the same counties, outer leave-one-state-out folds, estimator family, and
non-EO input blocks.  It estimates predictive association only; it is not a
causal-effect analysis.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "D18_USA_100m_HDI_predictions.csv").exists():
    ROOT = SCRIPT_DIR.parent
    DATA = SCRIPT_DIR / "D18_USA_100m_HDI_predictions.csv"
    OUT_DIR = SCRIPT_DIR
    METRICS_NAME = "D36_USA_EO_fixed_input_metrics.csv"
    INFERENCE_NAME = "D37_USA_EO_paired_inference.csv"
    AUDIT_NAME = "D38_USA_EO_incremental_audit.json"
else:
    ROOT = SCRIPT_DIR.parent
    DATA = (
        ROOT
        / "outputs"
        / "JAG_manuscript_US_uncertainty_20260927"
        / "supplementary_data"
        / "D18_USA_100m_HDI_predictions.csv"
    )
    OUT_DIR = ROOT / "work" / "us_eo_incremental_value"
    METRICS_NAME = "US_EO_fixed_input_metrics.csv"
    INFERENCE_NAME = "US_EO_paired_cluster_inference.csv"
    AUDIT_NAME = "US_EO_incremental_value_audit.json"
SEED = 20260928
N_BOOT = 20_000

CONTRASTS = {
    "EO_added_to_population_economy": ("R6", "B_R70"),
    "EO_added_to_all_traditional_blocks": ("R_S10", "B_RS74"),
}


def seeded_rng(label: str) -> np.random.Generator:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return np.random.default_rng(SEED + int.from_bytes(digest[:4], "big"))


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def model_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = pred - y
    abs_error = np.abs(residual)
    relative_error = abs_error / y
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    result = {
        "MAE": float(abs_error.mean()),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "R2": 1.0 - ss_res / ss_tot,
    }
    for threshold in (0.03, 0.05, 0.08, 0.10):
        result[f"P{int(threshold * 100)}"] = float(np.mean(relative_error <= threshold))
    return result


def metric_delta(
    metric: str, y: np.ndarray, traditional: np.ndarray, eo_added: np.ndarray
) -> np.ndarray:
    if metric.startswith("P"):
        threshold = int(metric[1:]) / 100
        return 100 * (
            (np.abs(eo_added - y) / y <= threshold).astype(float)
            - (np.abs(traditional - y) / y <= threshold).astype(float)
        )
    if metric == "MAE_gain":
        return np.abs(traditional - y) - np.abs(eo_added - y)
    if metric == "MSE_gain":
        return (traditional - y) ** 2 - (eo_added - y) ** 2
    raise ValueError(metric)


def paired_cluster_inference(
    contrast: str,
    metric: str,
    y: np.ndarray,
    traditional: np.ndarray,
    eo_added: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float | int | str]:
    delta = metric_delta(metric, y, traditional, eo_added)
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    group_counts = np.bincount(inverse).astype(float)
    group_sums = np.bincount(inverse, weights=delta)
    estimate = float(delta.mean())

    rng = seeded_rng(f"{contrast}:{metric}:bootstrap")
    indices = rng.integers(0, len(unique_groups), size=(N_BOOT, len(unique_groups)))
    bootstrap = group_sums[indices].sum(axis=1) / group_counts[indices].sum(axis=1)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])

    rng = seeded_rng(f"{contrast}:{metric}:signflip")
    signs = rng.choice(np.array([-1.0, 1.0]), size=(N_BOOT, len(unique_groups)))
    null = signs @ group_sums / len(y)
    p_value = (1 + int(np.count_nonzero(np.abs(null) >= abs(estimate) - 1e-15))) / (
        N_BOOT + 1
    )

    leave_one_out = (delta.sum() - group_sums) / (len(y) - group_counts)
    state_mean = group_sums / group_counts
    return {
        "contrast": contrast,
        "metric": metric,
        "estimate": estimate,
        "bootstrap_95_lo": float(ci_low),
        "bootstrap_95_hi": float(ci_high),
        "signflip_p_two_sided": float(p_value),
        "leave_one_state_out_min": float(leave_one_out.min()),
        "leave_one_state_out_max": float(leave_one_out.max()),
        "states_positive": int(np.count_nonzero(state_mean > 0)),
        "states_zero": int(np.count_nonzero(state_mean == 0)),
        "states_negative": int(np.count_nonzero(state_mean < 0)),
        "n_counties": int(len(y)),
        "n_outer_groups": int(len(unique_groups)),
        "replicates": N_BOOT,
        "seed": SEED,
    }


def main() -> None:
    rows = pd.read_csv(DATA, dtype={"adm1_code": str, "adm2_code": str})
    rows = rows.loc[rows["prediction_group"].eq("fixed_direct")].copy()
    assert rows["adm1_code"].nunique() == 51

    metric_records: list[dict[str, float | int | str]] = []
    inference_records: list[dict[str, float | int | str]] = []
    for contrast, (traditional_name, eo_name) in CONTRASTS.items():
        pair = rows.loc[rows["model"].isin([traditional_name, eo_name])].pivot(
            index=["adm2_code", "adm1_code"],
            columns="model",
            values=["y_true_hdi", "y_pred_hdi"],
        )
        assert len(pair) == 3104
        y_traditional = pair[("y_true_hdi", traditional_name)].to_numpy(float)
        y_eo = pair[("y_true_hdi", eo_name)].to_numpy(float)
        assert np.allclose(y_traditional, y_eo, atol=1e-12)
        y = y_traditional
        traditional = pair[("y_pred_hdi", traditional_name)].to_numpy(float)
        eo_added = pair[("y_pred_hdi", eo_name)].to_numpy(float)
        groups = pair.index.get_level_values("adm1_code").to_numpy(str)

        for model_role, model_name, prediction in (
            ("traditional", traditional_name, traditional),
            ("traditional_plus_EO", eo_name, eo_added),
        ):
            metrics = model_metrics(y, prediction)
            metric_records.append(
                {
                    "contrast": contrast,
                    "model_role": model_role,
                    "model": model_name,
                    "n_counties": len(y),
                    **metrics,
                }
            )

        for metric in ("P3", "P5", "P8", "P10", "MAE_gain", "MSE_gain"):
            inference_records.append(
                paired_cluster_inference(
                    contrast, metric, y, traditional, eo_added, groups
                )
            )

    adjusted = holm_adjust(
        [float(record["signflip_p_two_sided"]) for record in inference_records]
    )
    for record, p_adjusted in zip(inference_records, adjusted):
        record["holm_p_12_tests"] = p_adjusted

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(metric_records)
    inference_frame = pd.DataFrame(inference_records)
    metrics_frame.to_csv(OUT_DIR / METRICS_NAME, index=False)
    inference_frame.to_csv(OUT_DIR / INFERENCE_NAME, index=False)
    audit = {
        "analysis_scope": "incremental predictive value; not causal identification",
        "source": str(DATA.relative_to(ROOT)),
        "contrasts": CONTRASTS,
        "outer_group": "US state or District of Columbia",
        "bootstrap_replicates": N_BOOT,
        "seed": SEED,
        "conditions": [
            "same 3,104 counties",
            "same archived 100-m out-of-fold prediction design",
            "fixed direct input sets",
            "paired county errors with state-cluster resampling",
            "inference conditions on archived predictions and does not refit models",
        ],
    }
    (OUT_DIR / AUDIT_NAME).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(metrics_frame.to_string(index=False, float_format=lambda x: f"{x:.8g}"))
    print()
    print(inference_frame.to_string(index=False, float_format=lambda x: f"{x:.8g}"))
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
