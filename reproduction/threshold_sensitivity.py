"""Threshold sensitivity and continuous-error inference for A/B/C HDI predictions.

The script reads archived out-of-fold predictions only. It does not refit any
model. Outputs are D34 (P3/P5/P8/P10), D35 (paired C-versus-B MAE/RMSE
inference), and the updated supplementary cumulative-error figure (Fig. S1).
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA = Path(__file__).resolve().parent
ROOT = DATA.parent
EDITABLE = ROOT / "editable_figures"
SEED = 20260927
REPLICATES = 20_000
THRESHOLDS = (3.0, 5.0, 8.0, 10.0)
COUNTRIES = ("China", "Indonesia", "United States")
METHODS = ("A", "B", "C")
COUNTRY_LABEL = {
    "China": "Chinese Mainland",
    "Indonesia": "Indonesia",
    "United States": "United States",
}
COLORS = {"A": "#467FB8", "B": "#258AAD", "C": "#B64F7F"}
LIGHT = {"A": "#7FB3F0", "B": "#81D4FA", "C": "#F7A1C4"}
LINESTYLES = {"A": "--", "B": "-.", "C": "-"}


def deterministic_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return SEED + int.from_bytes(digest[:4], "big")


def read_predictions() -> pd.DataFrame:
    base = pd.read_csv(DATA / "D01_ABC_predictions.csv", dtype={"unit": str, "province": str})
    base = base.rename(
        columns={"province": "group", "observed": "truth", "predicted": "prediction"}
    )[["country", "unit", "group", "method", "truth", "prediction"]]

    usa = pd.read_csv(
        DATA / "D18_USA_100m_HDI_predictions.csv",
        dtype={"adm1_code": str, "adm2_code": str},
    )
    mapping = {
        "A_direct": "A",
        "B_common_components": "B",
        "C_dimension_specific": "C",
    }
    usa = usa.loc[usa.prediction_group.isin(mapping)].copy()
    usa["country"] = "United States"
    usa["method"] = usa.prediction_group.map(mapping)
    usa = usa.rename(
        columns={
            "adm2_code": "unit",
            "adm1_code": "group",
            "y_true_hdi": "truth",
            "y_pred_hdi": "prediction",
        }
    )[["country", "unit", "group", "method", "truth", "prediction"]]

    out = pd.concat([base, usa], ignore_index=True)
    out["absolute_relative_error_pct"] = (
        100.0 * (out.prediction - out.truth).abs() / out.truth
    )
    expected = {
        ("China", "A"): 240,
        ("China", "B"): 240,
        ("China", "C"): 240,
        ("Indonesia", "A"): 443,
        ("Indonesia", "B"): 443,
        ("Indonesia", "C"): 443,
        ("United States", "A"): 3104,
        ("United States", "B"): 3104,
        ("United States", "C"): 3104,
    }
    assert out.groupby(["country", "method"]).size().to_dict() == expected
    assert np.isfinite(out[["truth", "prediction", "absolute_relative_error_pct"]]).all().all()
    return out


def write_threshold_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for country in COUNTRIES:
        for threshold in THRESHOLDS:
            values: list[dict[str, object]] = []
            for method in METHODS:
                frame = predictions.loc[
                    (predictions.country == country) & (predictions.method == method)
                ]
                count = int((frame.absolute_relative_error_pct <= threshold + 1e-12).sum())
                values.append(
                    {
                        "country": country,
                        "method": method,
                        "n": len(frame),
                        "threshold_pct": threshold,
                        "within_threshold_n": count,
                        "within_threshold_pct": 100.0 * count / len(frame),
                    }
                )
            rates = np.array([float(item["within_threshold_pct"]) for item in values])
            order = pd.Series(-rates).rank(method="min").astype(int).to_numpy()
            best = float(rates.max())
            for item, rank in zip(values, order):
                item["rank"] = int(rank)
                item["best_at_threshold"] = bool(
                    np.isclose(float(item["within_threshold_pct"]), best)
                )
                rows.append(item)
    table = pd.DataFrame(rows)
    table.to_csv(DATA / "D34_threshold_sensitivity.csv", index=False)
    return table


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_continuous_inference(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for country in COUNTRIES:
        rows = predictions.loc[
            (predictions.country == country) & predictions.method.isin(["B", "C"])
        ]
        wide = rows.pivot(
            index=["unit", "group"], columns="method", values=["truth", "prediction"]
        )
        assert np.allclose(wide[("truth", "B")], wide[("truth", "C")], atol=1e-12)
        truth = wide[("truth", "B")].to_numpy(float)
        pred_b = wide[("prediction", "B")].to_numpy(float)
        pred_c = wide[("prediction", "C")].to_numpy(float)
        groups, inverse = np.unique(wide.index.get_level_values("group"), return_inverse=True)
        counts = np.bincount(inverse).astype(float)

        abs_b, abs_c = np.abs(pred_b - truth), np.abs(pred_c - truth)
        sq_b, sq_c = (pred_b - truth) ** 2, (pred_c - truth) ** 2
        for metric in ("MAE", "RMSE"):
            seed_metric = "MAE_gain" if metric == "MAE" else "RMSE_gain"
            if metric == "MAE":
                unit_delta = abs_b - abs_c
                estimate = float(abs_b.mean() - abs_c.mean())
                sums_b = np.bincount(inverse, weights=abs_b)
                sums_c = np.bincount(inverse, weights=abs_c)
                test_basis = "absolute_error_difference"
            else:
                unit_delta = sq_b - sq_c
                estimate = float(np.sqrt(sq_b.mean()) - np.sqrt(sq_c.mean()))
                sums_b = np.bincount(inverse, weights=sq_b)
                sums_c = np.bincount(inverse, weights=sq_c)
                test_basis = "squared_error_difference"

            group_delta = np.bincount(inverse, weights=unit_delta)
            rng = np.random.default_rng(deterministic_seed(f"{country}:{seed_metric}:bootstrap"))
            sampled = rng.integers(0, len(groups), size=(REPLICATES, len(groups)))
            bootstrap_n = counts[sampled].sum(axis=1)
            if metric == "MAE":
                bootstrap = (
                    sums_b[sampled].sum(axis=1) - sums_c[sampled].sum(axis=1)
                ) / bootstrap_n
            else:
                bootstrap = np.sqrt(sums_b[sampled].sum(axis=1) / bootstrap_n) - np.sqrt(
                    sums_c[sampled].sum(axis=1) / bootstrap_n
                )
            ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975]).tolist()

            total_b, total_c = sums_b.sum(), sums_c.sum()
            loo_n = len(truth) - counts
            if metric == "MAE":
                loo = (total_b - sums_b) / loo_n - (total_c - sums_c) / loo_n
            else:
                loo = np.sqrt((total_b - sums_b) / loo_n) - np.sqrt(
                    (total_c - sums_c) / loo_n
                )

            rng = np.random.default_rng(deterministic_seed(f"{country}:{seed_metric}:signflip"))
            signs = rng.choice(np.array([-1.0, 1.0]), size=(REPLICATES, len(groups)))
            observed_test_statistic = float(unit_delta.mean())
            null = signs @ group_delta / len(truth)
            p_value = (1 + int(np.count_nonzero(np.abs(null) >= abs(observed_test_statistic) - 1e-15))) / (
                REPLICATES + 1
            )
            records.append(
                {
                    "country": country,
                    "metric": metric,
                    "contrast": f"B_{metric}_minus_C_{metric}",
                    "estimate": estimate,
                    "bootstrap_95_lo": float(ci_low),
                    "bootstrap_95_hi": float(ci_high),
                    "leave_one_group_out_min": float(loo.min()),
                    "leave_one_group_out_max": float(loo.max()),
                    "signflip_test_basis": test_basis,
                    "signflip_p_two_sided": p_value,
                    "n_units": len(truth),
                    "n_outer_groups": len(groups),
                    "replicates": REPLICATES,
                    "seed": SEED,
                }
            )

    adjusted = holm_adjust([float(row["signflip_p_two_sided"]) for row in records])
    for row, p_adjusted in zip(records, adjusted):
        row["holm_p_six_continuous_tests"] = p_adjusted
    table = pd.DataFrame(records)
    table.to_csv(DATA / "D35_continuous_metric_inference.csv", index=False)
    return table


def make_figure(predictions: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#263445",
            "text.color": "#263445",
            "xtick.color": "#263445",
            "ytick.color": "#263445",
            "axes.edgecolor": "#B0BEC5",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True, sharey=True)
    for panel, (axis, country) in enumerate(zip(axes, COUNTRIES)):
        for method in METHODS:
            errors = np.sort(
                predictions.loc[
                    (predictions.country == country) & (predictions.method == method),
                    "absolute_relative_error_pct",
                ].to_numpy()
            )
            x = np.r_[0.0, errors, 15.0]
            y = np.r_[0.0, np.arange(1, len(errors) + 1) / len(errors) * 100.0, 100.0]
            axis.step(
                x,
                y,
                where="post",
                color=COLORS[method],
                linewidth=2.3 if method == "C" else 1.3,
                linestyle=LINESTYLES[method],
                label="CM-HDI" if method == "C" else f"Method {method}",
            )
            for threshold in THRESHOLDS:
                rate = float(np.mean(errors <= threshold + 1e-12) * 100.0)
                axis.scatter(
                    threshold,
                    rate,
                    s=36 if method == "C" else 17,
                    marker="D" if method == "C" else "o",
                    facecolor=LIGHT[method],
                    edgecolor=COLORS[method],
                    linewidth=0.8,
                    zorder=4,
                )
        for threshold in THRESHOLDS:
            is_primary = threshold in (5.0, 8.0)
            axis.axvline(
                threshold,
                color="#263445" if is_primary else "#9AA7B4",
                linewidth=1.0 if is_primary else 0.8,
                linestyle="--" if threshold in (3.0, 5.0) else ":",
                zorder=0,
            )
            axis.text(
                threshold,
                102.0,
                f"{threshold:g}%",
                ha="center",
                color="#263445" if is_primary else "#6E7B87",
                weight="bold" if is_primary else "normal",
                fontsize=7.6,
            )
        axis.set_xlim(0, 15)
        axis.set_ylim(0, 106)
        axis.set_title(f"{'abc'[panel]}  {COUNTRY_LABEL[country]}", loc="left")
        axis.set_xlabel("Absolute relative HDI error threshold (%)")
        axis.grid(axis="y", color="#E8F0F8", linewidth=0.8, zorder=0)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Share of held-out areas within threshold (%)")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.savefig(ROOT / "figS1.png", bbox_inches="tight", pad_inches=0.12)
    EDITABLE.mkdir(exist_ok=True)
    fig.savefig(EDITABLE / "figS1.svg", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    predictions = read_predictions()
    threshold_table = write_threshold_table(predictions)
    inference_table = paired_continuous_inference(predictions)
    make_figure(predictions)
    print(threshold_table.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print()
    print(inference_table.to_string(index=False, float_format=lambda value: f"{value:.8g}"))
    print(f"Wrote {DATA / 'D34_threshold_sensitivity.csv'}")
    print(f"Wrote {DATA / 'D35_continuous_metric_inference.csv'}")
    print(f"Wrote {ROOT / 'figS1.png'}")


if __name__ == "__main__":
    main()
