"""Compare CM-HDI with the matched conventional HDI geometric mean.

Both methods use the same spatially held-out component predictions in D56.  The
only difference is the component-to-composite mapping: P0 uses the fixed
geometric mean, whereas P4 uses the learned constrained seven-term synthesis.
Inference conditions on the archived outer predictions and does not refit the
component models.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "D56_constrained_aggregation_predictions.csv"
METRICS_OUT = HERE / "D60_traditional_geometric_metrics.csv"
INFERENCE_OUT = HERE / "D61_traditional_geometric_inference.csv"
AUDIT_OUT = HERE / "D62_traditional_geometric_audit.json"
SEED = 20260930
REPLICATES = 20_000
METHODS = ("P0_geometric", "P4_constrained_interactions")


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return SEED + int.from_bytes(digest[:4], "big")


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def metric_row(country: str, method: str, frame: pd.DataFrame) -> dict[str, object]:
    y = frame.y_true.to_numpy(float)
    pred = frame.y_pred.to_numpy(float)
    error = pred - y
    absolute = np.abs(error)
    relative = absolute / y
    return {
        "country": country,
        "method": method,
        "n": len(frame),
        "P5_n": int(np.count_nonzero(relative <= 0.05)),
        "P5_pct": 100 * float(np.mean(relative <= 0.05)),
        "P8_n": int(np.count_nonzero(relative <= 0.08)),
        "P8_pct": 100 * float(np.mean(relative <= 0.08)),
        "MAE": float(np.mean(absolute)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAPE_pct": 100 * float(np.mean(relative)),
        "R2": float(1 - np.sum(error**2) / np.sum((y - y.mean()) ** 2)),
    }


def main() -> None:
    source = pd.read_csv(SOURCE, dtype={"unit": str, "group": str, "outer_group": str})
    source = source.loc[source.method.isin(METHODS)].copy()
    assert not source.empty

    metric_records: list[dict[str, object]] = []
    inference_records: list[dict[str, object]] = []

    for country, country_rows in source.groupby("country", sort=False):
        display_country = "United States" if country == "USA" else str(country)
        for method, method_rows in country_rows.groupby("method", sort=False):
            metric_records.append(metric_row(display_country, str(method), method_rows))

        wide = country_rows.pivot(
            index=["unit", "outer_group"], columns="method", values=["y_true", "y_pred"]
        )
        assert np.allclose(
            wide[("y_true", METHODS[0])], wide[("y_true", METHODS[1])], atol=1e-12
        )
        y = wide[("y_true", METHODS[0])].to_numpy(float)
        traditional = wide[("y_pred", METHODS[0])].to_numpy(float)
        cmhdi = wide[("y_pred", METHODS[1])].to_numpy(float)
        groups, inverse = np.unique(wide.index.get_level_values("outer_group"), return_inverse=True)
        group_counts = np.bincount(inverse).astype(float)

        contrasts = {
            "P5_gain_pp": 100 * (
                (np.abs(cmhdi - y) / y <= 0.05).astype(float)
                - (np.abs(traditional - y) / y <= 0.05).astype(float)
            ),
            "MAE_gain": np.abs(traditional - y) - np.abs(cmhdi - y),
        }
        for metric, delta in contrasts.items():
            group_sums = np.bincount(inverse, weights=delta)
            estimate = float(delta.mean())
            rng = np.random.default_rng(stable_seed(f"{country}:{metric}:bootstrap"))
            draw = rng.integers(0, len(groups), size=(REPLICATES, len(groups)))
            bootstrap = group_sums[draw].sum(axis=1) / group_counts[draw].sum(axis=1)
            ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
            leave_one_group_out = (delta.sum() - group_sums) / (len(delta) - group_counts)

            rng = np.random.default_rng(stable_seed(f"{country}:{metric}:signflip"))
            signs = rng.choice(np.array([-1.0, 1.0]), size=(REPLICATES, len(groups)))
            null = signs @ group_sums / len(delta)
            p_value = (1 + int(np.count_nonzero(np.abs(null) >= abs(estimate) - 1e-15))) / (
                REPLICATES + 1
            )
            inference_records.append(
                {
                    "country": display_country,
                    "metric": metric,
                    "contrast": (
                        "CM-HDI_minus_traditional_pp"
                        if metric == "P5_gain_pp"
                        else "traditional_MAE_minus_CM-HDI_MAE"
                    ),
                    "estimate": estimate,
                    "bootstrap_95_lo": float(ci_low),
                    "bootstrap_95_hi": float(ci_high),
                    "leave_one_group_out_min": float(leave_one_group_out.min()),
                    "leave_one_group_out_max": float(leave_one_group_out.max()),
                    "signflip_p_two_sided": p_value,
                    "n_units": len(delta),
                    "n_outer_groups": len(groups),
                    "replicates": REPLICATES,
                    "seed": SEED,
                }
            )

    adjusted = holm([float(row["signflip_p_two_sided"]) for row in inference_records])
    for row, p_adjusted in zip(inference_records, adjusted):
        row["holm_p_six_tests"] = p_adjusted

    pd.DataFrame(metric_records).to_csv(METRICS_OUT, index=False)
    with INFERENCE_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inference_records[0]))
        writer.writeheader()
        writer.writerows(inference_records)

    audit = {
        "status": "PASS",
        "source": SOURCE.name,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "comparison": "matched fixed geometric mean versus constrained seven-term synthesis",
        "component_predictions": "identical within country, unit and outer fold",
        "bootstrap_replicates": REPLICATES,
        "seed": SEED,
        "multiplicity": "Holm correction over 3 settings x 2 metrics",
        "interpretation": (
            "This isolates the final synthesis conditional on archived held-out component "
            "predictions; it does not compare observed official components with predicted components."
        ),
        "outputs": [METRICS_OUT.name, INFERENCE_OUT.name],
    }
    AUDIT_OUT.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(pd.DataFrame(metric_records).to_string(index=False))
    print(pd.DataFrame(inference_records).to_string(index=False))


if __name__ == "__main__":
    main()
