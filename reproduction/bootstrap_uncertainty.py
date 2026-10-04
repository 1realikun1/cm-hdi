"""Quantify uncertainty from archived out-of-fold HDI predictions.

The script performs two related analyses for strategy C in the Chinese Mainland,
Indonesia and the United States:

1. A nonparametric outer-group cluster bootstrap (20,000 resamples) provides
   percentile confidence intervals for aggregate performance metrics.
2. A group-excluded, two-stage residual bootstrap provides a 95% empirical
   prediction interval for each administrative unit. Residuals are calibrated
   within country-specific predicted-HDI quintiles. For a target unit, its
   complete province/state group is excluded; each draw samples another group
   and then one residual from that group and quintile.

The procedure uses held-out predictions only. It does not refit a model and its
intervals quantify predictive variation represented by the archived spatial
validation residuals, not uncertainty in the source HDI labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED = 20260927
N_BOOT = 20_000
N_STRATA = 5
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def load_predictions() -> pd.DataFrame:
    d01 = pd.read_csv(ROOT / "D01_ABC_predictions.csv", dtype=str)
    d01 = d01.loc[d01["method"].eq("C")].copy()
    d01["y_true"] = pd.to_numeric(d01["observed"])
    d01["y_pred"] = pd.to_numeric(d01["predicted"])
    d01["unit_id"] = d01["unit"].astype(str)
    d01["outer_group"] = d01["province"].astype(str)
    d01["country"] = d01["country"].replace({"China": "Chinese Mainland"})
    d01 = d01[["country", "unit_id", "outer_group", "y_true", "y_pred"]]

    d18 = pd.read_csv(ROOT / "D18_USA_100m_HDI_predictions.csv", dtype=str)
    d18 = d18.loc[d18["prediction_group"].eq("C_dimension_specific")].copy()
    d18["y_true"] = pd.to_numeric(d18["y_true_hdi"])
    d18["y_pred"] = pd.to_numeric(d18["y_pred_hdi"])
    d18["unit_id"] = d18["adm2_code"].astype(str).str.zfill(5)
    d18["outer_group"] = d18["outer_test_province"].astype(str).str.zfill(2)
    d18["country"] = "United States"
    d18 = d18[["country", "unit_id", "outer_group", "y_true", "y_pred"]]

    out = pd.concat([d01, d18], ignore_index=True)
    if out.duplicated(["country", "unit_id"]).any():
        raise ValueError("Duplicate strategy-C unit predictions detected")
    expected = {"Chinese Mainland": 240, "Indonesia": 443, "United States": 3104}
    observed = out.groupby("country").size().to_dict()
    if observed != expected:
        raise ValueError(f"Unexpected country counts: {observed}")
    out["residual"] = out["y_true"] - out["y_pred"]
    out["absolute_error"] = out["residual"].abs()
    out["absolute_relative_error_pct"] = out["absolute_error"] / out["y_true"] * 100
    return out


def metric_values(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    err = y - p
    abs_err = np.abs(err)
    rel = abs_err / y
    return {
        "P5_pct": float(np.mean(rel <= 0.05) * 100),
        "P8_pct": float(np.mean(rel <= 0.08) * 100),
        "MAE": float(np.mean(abs_err)),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE_pct": float(np.mean(rel) * 100),
        "R2": float(1 - np.sum(err**2) / np.sum((y - np.mean(y)) ** 2)),
    }


def cluster_bootstrap_metrics(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    groups = sorted(df["outer_group"].unique())
    group_index = {g: j for j, g in enumerate(groups)}
    cols = ["n", "pass5", "pass8", "sum_abs", "sum_sq", "sum_rel", "sum_y", "sum_y2"]
    stats = np.zeros((len(groups), len(cols)), dtype=float)
    for group, part in df.groupby("outer_group", sort=True):
        j = group_index[group]
        y = part["y_true"].to_numpy(float)
        p = part["y_pred"].to_numpy(float)
        err = y - p
        abs_err = np.abs(err)
        rel = abs_err / y
        stats[j] = [
            len(part),
            np.sum(rel <= 0.05),
            np.sum(rel <= 0.08),
            np.sum(abs_err),
            np.sum(err**2),
            np.sum(rel),
            np.sum(y),
            np.sum(y**2),
        ]

    counts = rng.multinomial(len(groups), np.repeat(1 / len(groups), len(groups)), size=N_BOOT)
    totals = counts @ stats
    n = totals[:, 0]
    samples = {
        "P5_pct": totals[:, 1] / n * 100,
        "P8_pct": totals[:, 2] / n * 100,
        "MAE": totals[:, 3] / n,
        "RMSE": np.sqrt(totals[:, 4] / n),
        "MAPE_pct": totals[:, 5] / n * 100,
        "R2": 1 - totals[:, 4] / (totals[:, 7] - totals[:, 6] ** 2 / n),
    }
    point = metric_values(df["y_true"].to_numpy(float), df["y_pred"].to_numpy(float))
    result = {"n": len(df), "outer_groups": len(groups)}
    for metric, values in samples.items():
        result[metric] = point[metric]
        result[f"{metric}_lo"] = float(np.quantile(values, 0.025))
        result[f"{metric}_hi"] = float(np.quantile(values, 0.975))
    return result


def add_prediction_intervals(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for country, country_df in df.groupby("country", sort=False):
        work = country_df.copy()
        # Rank-based qcut avoids failures if predictions contain repeated values.
        work["predicted_hdi_quintile"] = pd.qcut(
            work["y_pred"].rank(method="first"), N_STRATA, labels=False
        ).astype(int) + 1
        limits: dict[tuple[int, str], tuple[float, float, float, float, int, int, float]] = {}
        for stratum in range(1, N_STRATA + 1):
            stratum_df = work.loc[work["predicted_hdi_quintile"].eq(stratum)]
            by_group = {
                str(group): part["residual"].to_numpy(float)
                for group, part in stratum_df.groupby("outer_group", sort=True)
            }
            for target_group in sorted(work["outer_group"].unique()):
                pools = [values for group, values in by_group.items() if group != str(target_group)]
                if len(pools) < 5:
                    raise ValueError(f"Too few calibration groups for {country}, Q{stratum}, {target_group}")
                chosen_groups = rng.integers(0, len(pools), size=N_BOOT)
                draws = np.empty(N_BOOT, dtype=float)
                for group_index in np.unique(chosen_groups):
                    positions = np.flatnonzero(chosen_groups == group_index)
                    pool = pools[group_index]
                    draws[positions] = pool[rng.integers(0, len(pool), size=len(positions))]
                calibration_n = int(sum(len(pool) for pool in pools))
                # Finite-sample correction for a nominal 95% predictive residual radius.
                quantile_level = min(1.0, np.ceil((calibration_n + 1) * 0.95) / calibration_n)
                center = float(np.median(draws))
                radius = float(np.quantile(np.abs(draws - center), quantile_level, method="higher"))
                limits[(stratum, str(target_group))] = (
                    center - radius,
                    center + radius,
                    center,
                    radius,
                    calibration_n,
                    int(len(pools)),
                    quantile_level,
                )

        rows = []
        for row in work.itertuples(index=False):
            lo_resid, hi_resid, center_resid, radius_resid, calibration_n, calibration_groups, quantile_level = limits[
                (int(row.predicted_hdi_quintile), str(row.outer_group))
            ]
            lower = max(0.0, float(row.y_pred) + lo_resid)
            upper = min(1.0, float(row.y_pred) + hi_resid)
            rows.append(
                {
                    **row._asdict(),
                    "bootstrap_residual_lower": lo_resid,
                    "bootstrap_residual_upper": hi_resid,
                    "bootstrap_residual_center": center_resid,
                    "bootstrap_residual_radius": radius_resid,
                    "prediction_interval_lower": lower,
                    "prediction_interval_upper": upper,
                    "prediction_interval_width": upper - lower,
                    "prediction_interval_covered": bool(lower <= row.y_true <= upper),
                    "calibration_units": calibration_n,
                    "calibration_groups": calibration_groups,
                    "finite_sample_quantile": quantile_level,
                }
            )
        pieces.append(pd.DataFrame(rows))
    return pd.concat(pieces, ignore_index=True)


def write_figure(intervals: pd.DataFrame, summary: pd.DataFrame) -> None:
    order = ["Chinese Mainland", "United States", "Indonesia"]
    # Shared manuscript palette: proposed-system summaries in pink, with the
    # two external settings distinguished by the companion blue tones.
    colors = {
        "Chinese Mainland": "#F7A1C4",
        "United States": "#7FB3F0",
        "Indonesia": "#81D4FA",
    }
    labels = {"Chinese Mainland": "Chinese Mainland", "Indonesia": "Indonesia", "United States": "United States"}
    y = np.arange(len(order))[::-1]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#263445",
        "text.color": "#263445",
        "xtick.color": "#263445",
        "ytick.color": "#263445",
        "axes.edgecolor": "#B0BEC5",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), gridspec_kw={"width_ratios": [1.05, 1.05, 1.45]})

    for ax, metric, xlabel, panel in [
        (axes[0], "P5_pct", "Within 5% relative error (%)", "a"),
        (axes[1], "MAE", "Mean absolute error", "b"),
    ]:
        for yi, country in zip(y, order):
            row = summary.loc[summary["country"].eq(country)].iloc[0]
            value = row[metric]
            lo = row[f"{metric}_lo"]
            hi = row[f"{metric}_hi"]
            ax.errorbar(
                value,
                yi,
                xerr=np.array([[value - lo], [hi - value]]),
                fmt="o",
                color=colors[country],
                ecolor=colors[country],
                capsize=3,
                markersize=6,
                linewidth=1.6,
            )
        ax.set_yticks(y, [labels[c] for c in order])
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#E8F0F8", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"({panel}) 95% cluster-bootstrap intervals", loc="left", fontweight="bold")

    ax = axes[2]
    data = [intervals.loc[intervals["country"].eq(c), "prediction_interval_width"].to_numpy(float) for c in order]
    parts = ax.violinplot(data, positions=y, orientation="horizontal", showmeans=False, showmedians=False, showextrema=False)
    for body, country in zip(parts["bodies"], order):
        body.set_facecolor(colors[country])
        body.set_edgecolor("white")
        body.set_alpha(0.68)
    for yi, country, values in zip(y, order, data):
        q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        coverage = intervals.loc[intervals["country"].eq(country), "prediction_interval_covered"].mean() * 100
        ax.plot([q1, q3], [yi, yi], color="#263445", linewidth=4, solid_capstyle="butt")
        ax.plot(med, yi, marker="o", color="white", markeredgecolor="#263445", markersize=5, zorder=3)
        ax.text(q3 + 0.003, yi, f"coverage {coverage:.1f}%", va="center", fontsize=8)
    ax.set_yticks(y, [labels[c] for c in order])
    ax.set_xlabel("Width of 95% empirical prediction interval")
    ax.grid(axis="x", color="#E8F0F8", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("(c) Region-level uncertainty distribution", loc="left", fontweight="bold")
    ax.set_xlim(left=0)

    fig.suptitle("Uncertainty in held-out CM-HDI predictions", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(PROJECT / "fig8.png", dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(PROJECT / "editable_figures" / "fig8_uncertainty.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(SEED)
    predictions = load_predictions()
    intervals = add_prediction_intervals(predictions, rng)

    rows = []
    for country, part in predictions.groupby("country", sort=False):
        metric_summary = cluster_bootstrap_metrics(part, rng)
        ip = intervals.loc[intervals["country"].eq(country)]
        widths = ip["prediction_interval_width"].to_numpy(float)
        metric_summary.update(
            {
                "country": country,
                "prediction_interval_coverage_pct": float(ip["prediction_interval_covered"].mean() * 100),
                "prediction_interval_median_width": float(np.median(widths)),
                "prediction_interval_q25_width": float(np.quantile(widths, 0.25)),
                "prediction_interval_q75_width": float(np.quantile(widths, 0.75)),
                "prediction_interval_mean_width": float(np.mean(widths)),
            }
        )
        rows.append(metric_summary)
    summary = pd.DataFrame(rows)

    interval_columns = [
        "country",
        "unit_id",
        "outer_group",
        "y_true",
        "y_pred",
        "residual",
        "predicted_hdi_quintile",
        "bootstrap_residual_lower",
        "bootstrap_residual_upper",
        "bootstrap_residual_center",
        "bootstrap_residual_radius",
        "prediction_interval_lower",
        "prediction_interval_upper",
        "prediction_interval_width",
        "prediction_interval_covered",
        "calibration_units",
        "calibration_groups",
        "finite_sample_quantile",
    ]
    intervals[interval_columns].to_csv(ROOT / "D30_prediction_uncertainty.csv", index=False, float_format="%.12g")
    summary.to_csv(ROOT / "D31_uncertainty_summary.csv", index=False, float_format="%.12g")
    config = {
        "analysis": "spatially grouped bootstrap uncertainty for held-out CM-HDI predictions",
        "seed": SEED,
        "bootstrap_resamples": N_BOOT,
        "interval_level": 0.95,
        "residual_strata": N_STRATA,
        "aggregate_interval": "percentile outer-group cluster bootstrap",
        "prediction_interval": "group-excluded two-stage residual bootstrap within predicted-HDI quintile",
        "scope": "predictive variation in archived held-out residuals; source-label uncertainty excluded",
        "input_files": ["D01_ABC_predictions.csv", "D18_USA_100m_HDI_predictions.csv"],
        "output_files": ["D30_prediction_uncertainty.csv", "D31_uncertainty_summary.csv"],
    }
    (ROOT / "D32_uncertainty_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_figure(intervals, summary)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
