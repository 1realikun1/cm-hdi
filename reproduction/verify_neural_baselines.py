"""Independent consistency checks for D49--D55 neural baseline files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
THRESHOLDS = (3, 5, 8, 10)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    error = pred - y
    result = {
        "n": len(y),
        "mae": np.mean(np.abs(error)),
        "rmse": np.sqrt(np.mean(error**2)),
        "mape_pct": 100 * np.mean(np.abs(error) / y),
        "r2": 1 - np.sum(error**2) / np.sum((y - y.mean()) ** 2),
    }
    for threshold in THRESHOLDS:
        passed = np.abs(error) / y <= threshold / 100
        result[f"p{threshold}_n"] = int(passed.sum())
        result[f"p{threshold}_pct"] = 100 * passed.mean()
    return result


def main() -> None:
    x = pd.read_csv(HERE / "D49_China_neural_input_matrix.csv", dtype={"city_code": str})
    splits = json.loads((HERE / "D50_China_neural_splits.json").read_text(encoding="utf-8"))["splits"]
    pred = pd.read_csv(
        HERE / "D51_China_neural_predictions.csv",
        dtype={"city_code": str, "province_code": str},
    )
    tuning = pd.read_csv(HERE / "D52_China_neural_tuning.csv")
    reported = pd.read_csv(HERE / "D53_China_neural_metrics.csv").set_index("model")
    inference = pd.read_csv(HERE / "D54_China_neural_inference.csv")
    audit = json.loads((HERE / "D55_China_neural_audit.json").read_text(encoding="utf-8"))

    assert len(x) == 280 and x.city_code.is_unique
    assert sum(column.startswith("feature_") for column in x) == 74
    assert x.role.value_counts().to_dict() == {
        "nationwide_score": 240,
        "development_train_only": 40,
    }
    assert len(splits) == 24
    outer_test = [str(city) for split in splits for city in split["outer_test_city_codes"]]
    assert len(outer_test) == len(set(outer_test)) == 240
    for split in splits:
        train = set(map(str, split["outer_train_city_codes"]))
        test = set(map(str, split["outer_test_city_codes"]))
        assert not train & test and len(split["inner_folds"]) == 5
        validation = []
        for inner in split["inner_folds"]:
            inner_train = set(map(str, inner["train_city_codes"]))
            inner_valid = set(map(str, inner["validation_city_codes"]))
            assert not inner_train & inner_valid and inner_train | inner_valid == train
            validation.extend(inner_valid)
        assert len(validation) == len(set(validation)) == len(train)

    expected_models = {
        "Component-adaptive Ridge (C)", "FT-Transformer", "Residual MLP"
    }
    assert set(pred.model) == expected_models and len(pred) == 720
    assert pred.groupby("model").size().eq(240).all()
    assert pred.groupby("model").city_code.nunique().eq(240).all()
    wide_y = pred.pivot(index="city_code", columns="model", values="observed_hdi")
    assert np.max(np.ptp(wide_y.to_numpy(float), axis=1)) < 1e-12
    assert set(wide_y.index) == set(outer_test)

    for model, group in pred.groupby("model"):
        recomputed = metrics(
            group.observed_hdi.to_numpy(float), group.predicted_hdi.to_numpy(float)
        )
        for key, value in recomputed.items():
            assert np.isclose(float(reported.loc[model, key]), value, atol=1e-12), (model, key)

    assert len(tuning) == 576
    assert tuning.record_type.value_counts().to_dict() == {"inner_fold": 480, "pooled_inner": 96}
    selected = tuning.loc[tuning.selected.astype(str).str.lower().eq("true")]
    assert len(selected) == 48
    assert selected.groupby(["outer_test_province", "family"]).size().eq(1).all()
    assert len(inference) == 12 and inference.n_units.eq(240).all()
    assert inference.n_provinces.eq(24).all()
    assert np.isfinite(inference[["estimate", "bootstrap_95_lo", "bootstrap_95_hi"]]).all().all()

    for filename, digest in audit["sha256_inputs"].items():
        assert sha256(HERE / filename) == digest
    for filename, digest in audit["sha256_outputs"].items():
        assert sha256(HERE / filename) == digest
    assert len(audit["selected_outer_models"]) == 48
    print("PASS: D49--D55 neural baselines are internally consistent and reproducible.")


if __name__ == "__main__":
    main()
