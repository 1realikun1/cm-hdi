"""Run the frozen Indonesia 2020 five baselines and A/B/C experiment."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(r"E:\BeyondSurfaceHDI")
FREEZE = ROOT / "data_raw" / "indonesia_validation" / "frozen_experiment_2020"
OUT = ROOT / "runs" / "indonesia_abc_2020"
MATRIX = FREEZE / "indonesia_adm2_model_matrix_2020.csv"
SPLITS = FREEZE / "province_splits_2020.json"
MODELS_FILE = FREEZE / "model_input_combinations_2020.json"
FREEZE_CHECK = FREEZE / "freeze_validation_2020.json"
PROTOCOL = "indonesia_hdi_2020_abc_lopo_v1"
RUN_PROTOCOL = "indonesia_hdi_2020_abc_lopo_run_v1"
TARGETS = ["y_hdi", "y_health_index", "y_education_index", "y_income_index"]
COMPONENTS = ["y_health_index", "y_education_index", "y_income_index"]
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def province_equal_mae(y, pred, provinces) -> float:
    values = []
    for province in sorted(set(provinces)):
        take = np.asarray(provinces) == province
        values.append(mean_absolute_error(np.asarray(y)[take], np.asarray(pred)[take]))
    return float(np.mean(values))


def keep_columns(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].to_numpy(float)
    sd = np.std(values, axis=0, ddof=0)
    keep = np.flatnonzero(np.isfinite(values).all(axis=0) & np.isfinite(sd) & (sd > 1e-12))
    if not len(keep):
        raise RuntimeError("No finite nonconstant predictor remains")
    return keep


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, columns: list[str],
                target: str, alpha: float):
    keep = keep_columns(train, columns)
    x_train = train[columns].to_numpy(float)[:, keep]
    x_test = test[columns].to_numpy(float)[:, keep]
    scaler = StandardScaler().fit(x_train)
    fitted = Ridge(alpha=alpha, fit_intercept=True, solver="svd").fit(
        scaler.transform(x_train), train[target].to_numpy(float)
    )
    return np.asarray(fitted.predict(scaler.transform(x_test)), float), len(keep)


def choose_alpha(rows: pd.DataFrame) -> tuple[float, float]:
    best = float(rows.province_equal_MAE.min())
    eligible = rows.loc[rows.province_equal_MAE <= best + 1e-12, "alpha"].astype(float)
    return float(eligible.max()), best


def choose_model(candidates: list[tuple[str, float]], dimensions: dict[str, int],
                 model_order: list[str]) -> tuple[str, float]:
    best = min(score for _, score in candidates)
    eligible = [(name, score) for name, score in candidates if score <= best + 1e-12]
    chosen = min(eligible, key=lambda x: (dimensions[x[0]], model_order.index(x[0])))
    return chosen[0], best


def validate_frozen_inputs() -> tuple[pd.DataFrame, list[dict], dict, list[str]]:
    check = json.loads(FREEZE_CHECK.read_text(encoding="utf-8"))
    if check.get("status") != "PASS" or check.get("protocol") != PROTOCOL:
        raise RuntimeError("Frozen input validation is not PASS for the expected protocol")
    for name, expected in check["outputs"].items():
        path = FREEZE / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Frozen file hash mismatch: {name}")
    model_payload = json.loads(MODELS_FILE.read_text(encoding="utf-8"))
    models = model_payload["models"]
    model_order = model_payload["model_order"]
    if {name: len(columns) for name, columns in models.items()} != {
        "B69": 69, "R6": 6, "B_R70": 70, "R_S10": 10, "B_RS74": 74
    }:
        raise RuntimeError("Frozen model dimensions changed")
    matrix = pd.read_csv(MATRIX, encoding="utf-8-sig", dtype={"adm1_code": str, "adm2_code": str})
    split_payload = json.loads(SPLITS.read_text(encoding="utf-8"))
    splits = split_payload["splits"]
    if len(matrix) != 505 or matrix.adm2_code.nunique() != 505 or matrix.adm1_code.nunique() != 34:
        raise RuntimeError("Frozen model matrix cohort changed")
    if not np.isfinite(matrix[TARGETS + sum((models[x] for x in model_order), [])].to_numpy(float)).all():
        raise RuntimeError("Frozen matrix contains nonfinite model values")
    return matrix, splits, models, model_order


def tune_outer(matrix: pd.DataFrame, outer: dict, models: dict, model_order: list[str]):
    accum = {
        (target, model, alpha): {"y": [], "pred": [], "province": []}
        for target in TARGETS for model in model_order for alpha in ALPHAS
    }
    fold_rows = []
    for inner in outer["inner_folds"]:
        train = matrix.loc[matrix.adm2_code.isin(inner["train_adm2_codes"])].sort_values("adm2_code")
        valid = matrix.loc[matrix.adm2_code.isin(inner["validation_adm2_codes"])].sort_values("adm2_code")
        if set(train.adm1_code) & set(valid.adm1_code):
            raise RuntimeError("Province leakage in inner fold")
        y_train = train[TARGETS].to_numpy(float)
        y_valid = valid[TARGETS].to_numpy(float)
        provinces = valid.adm1_code.to_numpy()
        for model in model_order:
            columns = models[model]
            keep = keep_columns(train, columns)
            scaler = StandardScaler().fit(train[columns].to_numpy(float)[:, keep])
            x_train = scaler.transform(train[columns].to_numpy(float)[:, keep])
            x_valid = scaler.transform(valid[columns].to_numpy(float)[:, keep])
            for alpha in ALPHAS:
                fitted = Ridge(alpha=alpha, fit_intercept=True, solver="svd").fit(x_train, y_train)
                prediction = np.asarray(fitted.predict(x_valid), float)
                for target_index, target in enumerate(TARGETS):
                    key = (target, model, alpha)
                    accum[key]["y"].extend(y_valid[:, target_index].tolist())
                    accum[key]["pred"].extend(prediction[:, target_index].tolist())
                    accum[key]["province"].extend(provinces.tolist())
                    fold_rows.append({
                        "outer_test_province": outer["outer_test_province"],
                        "inner_fold": inner["fold"],
                        "target": target,
                        "model": model,
                        "alpha": alpha,
                        "validation_cities": len(valid),
                        "validation_provinces": valid.adm1_code.nunique(),
                        "kept_predictors": len(keep),
                        "fold_city_equal_MAE": mean_absolute_error(y_valid[:, target_index], prediction[:, target_index]),
                    })
    tuning = []
    for (target, model, alpha), values in accum.items():
        tuning.append({
            "outer_test_province": outer["outer_test_province"],
            "target": target,
            "model": model,
            "alpha": alpha,
            "province_equal_MAE": province_equal_mae(values["y"], values["pred"], values["province"]),
            "city_equal_MAE": mean_absolute_error(values["y"], values["pred"]),
            "inner_validation_cities": len(values["y"]),
            "inner_validation_provinces": len(set(values["province"])),
        })
    tuning = pd.DataFrame(tuning)
    selected = {}
    for target in TARGETS:
        for model in model_order:
            subset = tuning.loc[(tuning.target == target) & (tuning.model == model)]
            selected[(target, model)] = choose_alpha(subset)
    tuning["alpha_selected_within_target_model"] = False
    for (target, model), (alpha, _) in selected.items():
        tuning.loc[(tuning.target == target) & (tuning.model == model) & (tuning.alpha == alpha),
                   "alpha_selected_within_target_model"] = True
    return tuning, pd.DataFrame(fold_rows), selected


def run_outer(matrix, outer, models, model_order):
    outer_code = outer["outer_test_province"]
    train = matrix.loc[matrix.adm2_code.isin(outer["outer_train_adm2_codes"])].sort_values("adm2_code")
    test = matrix.loc[matrix.adm2_code.isin(outer["outer_test_adm2_codes"])].sort_values("adm2_code")
    if set(train.adm1_code) & set(test.adm1_code):
        raise RuntimeError("Province leakage in outer fold")
    tuning, fold_metrics, selected = tune_outer(matrix, outer, models, model_order)
    dimensions = {name: len(columns) for name, columns in models.items()}

    a_model, a_score = choose_model(
        [(model, selected[("y_hdi", model)][1]) for model in model_order], dimensions, model_order
    )
    b_scores = {
        model: float(np.mean([selected[(target, model)][1] for target in COMPONENTS]))
        for model in model_order
    }
    b_model, b_score = choose_model(list(b_scores.items()), dimensions, model_order)
    c_models = {
        target: choose_model(
            [(model, selected[(target, model)][1]) for model in model_order], dimensions, model_order
        ) for target in COMPONENTS
    }

    selection_rows = []
    for model in model_order:
        alpha, score = selected[("y_hdi", model)]
        selection_rows.append({
            "outer_test_province": outer_code, "selection_group": "fixed_direct",
            "selection_target": "y_hdi", "selected_model": model, "selected_alpha": alpha,
            "inner_province_equal_MAE": score, "group_selection_score": score,
            "selected_by_group": model == a_model,
        })
    selection_rows.append({
        "outer_test_province": outer_code, "selection_group": "A_direct",
        "selection_target": "y_hdi", "selected_model": a_model,
        "selected_alpha": selected[("y_hdi", a_model)][0],
        "inner_province_equal_MAE": selected[("y_hdi", a_model)][1],
        "group_selection_score": a_score, "selected_by_group": True,
    })
    for target in COMPONENTS:
        selection_rows.append({
            "outer_test_province": outer_code, "selection_group": "B_common_components",
            "selection_target": target, "selected_model": b_model,
            "selected_alpha": selected[(target, b_model)][0],
            "inner_province_equal_MAE": selected[(target, b_model)][1],
            "group_selection_score": b_score, "selected_by_group": True,
        })
    for target in COMPONENTS:
        c_model, c_score = c_models[target]
        selection_rows.append({
            "outer_test_province": outer_code, "selection_group": "C_dimension_specific",
            "selection_target": target, "selected_model": c_model,
            "selected_alpha": selected[(target, c_model)][0],
            "inner_province_equal_MAE": selected[(target, c_model)][1],
            "group_selection_score": c_score, "selected_by_group": True,
        })

    # Mark selection stages in the complete alpha-grid record.
    tuning["A_information_combination_selected"] = tuning.model.eq(a_model) & tuning.target.eq("y_hdi")
    tuning["B_common_information_combination_selected"] = tuning.model.eq(b_model) & tuning.target.isin(COMPONENTS)
    tuning["C_component_information_combination_selected"] = False
    for target, (model, _) in c_models.items():
        tuning.loc[(tuning.target == target) & (tuning.model == model),
                   "C_component_information_combination_selected"] = True

    fixed_predictions = {}
    prediction_rows = []
    component_rows = []
    for model in model_order:
        alpha = selected[("y_hdi", model)][0]
        pred, kept = fit_predict(train, test, models[model], "y_hdi", alpha)
        fixed_predictions[model] = pred
        for row, value in zip(test.itertuples(index=False), pred):
            prediction_rows.append({
                "adm1_code": row.adm1_code, "adm2_code": row.adm2_code,
                "outer_test_province": outer_code, "prediction_group": "fixed_direct",
                "model": model, "selected_alpha": alpha, "kept_predictors": kept,
                "y_true_hdi": row.y_hdi, "y_pred_hdi": value,
                "residual": row.y_hdi - value, "absolute_error": abs(row.y_hdi - value),
            })
    a_pred = fixed_predictions[a_model]
    for row, value in zip(test.itertuples(index=False), a_pred):
        prediction_rows.append({
            "adm1_code": row.adm1_code, "adm2_code": row.adm2_code,
            "outer_test_province": outer_code, "prediction_group": "A_direct",
            "model": a_model, "selected_alpha": selected[("y_hdi", a_model)][0],
            "kept_predictors": dimensions[a_model], "y_true_hdi": row.y_hdi,
            "y_pred_hdi": value, "residual": row.y_hdi - value,
            "absolute_error": abs(row.y_hdi - value),
        })

    b_values, c_values = {}, {}
    for target in COMPONENTS:
        b_alpha = selected[(target, b_model)][0]
        b_raw, b_kept = fit_predict(train, test, models[b_model], target, b_alpha)
        b_values[target] = np.clip(b_raw, 0.0, 1.0)
        c_model = c_models[target][0]
        c_alpha = selected[(target, c_model)][0]
        c_raw, c_kept = fit_predict(train, test, models[c_model], target, c_alpha)
        c_values[target] = np.clip(c_raw, 0.0, 1.0)
        for row, raw, clipped in zip(test.itertuples(index=False), b_raw, b_values[target]):
            component_rows.append({
                "adm1_code": row.adm1_code, "adm2_code": row.adm2_code,
                "outer_test_province": outer_code, "prediction_group": "B_common_components",
                "target": target, "model": b_model, "selected_alpha": b_alpha,
                "kept_predictors": b_kept, "y_true": getattr(row, target),
                "y_pred_raw": raw, "y_pred_clipped": clipped,
            })
        for row, raw, clipped in zip(test.itertuples(index=False), c_raw, c_values[target]):
            component_rows.append({
                "adm1_code": row.adm1_code, "adm2_code": row.adm2_code,
                "outer_test_province": outer_code, "prediction_group": "C_dimension_specific",
                "target": target, "model": c_model, "selected_alpha": c_alpha,
                "kept_predictors": c_kept, "y_true": getattr(row, target),
                "y_pred_raw": raw, "y_pred_clipped": clipped,
            })
    b_hdi = np.cbrt(b_values["y_health_index"] * b_values["y_education_index"] * b_values["y_income_index"])
    c_hdi = np.cbrt(c_values["y_health_index"] * c_values["y_education_index"] * c_values["y_income_index"])
    for group, model_label, values in [
        ("B_common_components", b_model, b_hdi),
        ("C_dimension_specific", "+".join(c_models[x][0] for x in COMPONENTS), c_hdi),
    ]:
        for row, value in zip(test.itertuples(index=False), values):
            prediction_rows.append({
                "adm1_code": row.adm1_code, "adm2_code": row.adm2_code,
                "outer_test_province": outer_code, "prediction_group": group,
                "model": model_label, "selected_alpha": np.nan, "kept_predictors": np.nan,
                "y_true_hdi": row.y_hdi, "y_pred_hdi": value,
                "residual": row.y_hdi - value, "absolute_error": abs(row.y_hdi - value),
            })
    return tuning, fold_metrics, pd.DataFrame(selection_rows), pd.DataFrame(prediction_rows), pd.DataFrame(component_rows)


def summarize_predictions(predictions: pd.DataFrame):
    rows = []
    groups = []
    fixed = predictions.loc[predictions.prediction_group.eq("fixed_direct")]
    groups.extend(("fixed_direct", model, data) for model, data in fixed.groupby("model", sort=True))
    for group in ["A_direct", "B_common_components", "C_dimension_specific"]:
        groups.append((group, "selected_by_outer_fold", predictions.loc[predictions.prediction_group.eq(group)]))
    for group, model, data in groups:
        rows.append({
            "prediction_group": group, "model": model, "cities": len(data),
            "provinces": data.adm1_code.nunique(),
            "province_equal_MAE": province_equal_mae(data.y_true_hdi, data.y_pred_hdi, data.adm1_code),
            "city_equal_MAE": mean_absolute_error(data.y_true_hdi, data.y_pred_hdi),
            "city_equal_RMSE": math.sqrt(mean_squared_error(data.y_true_hdi, data.y_pred_hdi)),
        })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    matrix, splits, models, model_order = validate_frozen_inputs()
    all_tuning, all_folds, all_selection, all_predictions, all_components = [], [], [], [], []
    for number, outer in enumerate(splits, 1):
        results = run_outer(matrix, outer, models, model_order)
        for bucket, frame in zip(
            [all_tuning, all_folds, all_selection, all_predictions, all_components], results
        ):
            bucket.append(frame)
        print(f"[{number:02d}/{len(splits)}] outer province {outer['outer_test_province']} complete", flush=True)
    tuning = pd.concat(all_tuning, ignore_index=True).sort_values(
        ["outer_test_province", "target", "model", "alpha"]
    )
    folds = pd.concat(all_folds, ignore_index=True).sort_values(
        ["outer_test_province", "inner_fold", "target", "model", "alpha"]
    )
    selection = pd.concat(all_selection, ignore_index=True).sort_values(
        ["outer_test_province", "selection_group", "selection_target", "selected_model"]
    )
    predictions = pd.concat(all_predictions, ignore_index=True).sort_values(
        ["prediction_group", "model", "adm1_code", "adm2_code"]
    )
    components = pd.concat(all_components, ignore_index=True).sort_values(
        ["prediction_group", "target", "adm1_code", "adm2_code"]
    )
    metrics = summarize_predictions(predictions)

    paths = {
        "inner_tuning_scores": OUT / "inner_tuning_scores.csv",
        "inner_fold_metrics": OUT / "inner_fold_metrics.csv",
        "outer_selection_records": OUT / "outer_selection_records.csv",
        "outer_hdi_predictions": OUT / "outer_hdi_predictions.csv",
        "outer_component_predictions": OUT / "outer_component_predictions.csv",
        "overall_metrics": OUT / "overall_metrics.csv",
    }
    for key, frame in [
        ("inner_tuning_scores", tuning), ("inner_fold_metrics", folds),
        ("outer_selection_records", selection), ("outer_hdi_predictions", predictions),
        ("outer_component_predictions", components), ("overall_metrics", metrics),
    ]:
        write_csv(frame, paths[key])

    # Structural checks; these inspect file completeness, not scientific success.
    expected_fixed = len(matrix) * len(model_order)
    expected_abc = len(matrix) * 3
    fixed = predictions.loc[predictions.prediction_group.eq("fixed_direct")]
    abc = predictions.loc[~predictions.prediction_group.eq("fixed_direct")]
    criteria = {
        "frozen_input_hashes_match": True,
        "all_34_outer_folds_completed": selection.outer_test_province.nunique() == 34,
        "fixed_direct_prediction_rows_2525": len(fixed) == expected_fixed == 2525,
        "abc_prediction_rows_1515": len(abc) == expected_abc == 1515,
        "each_fixed_model_predicts_each_city_once": bool((fixed.groupby(["model", "adm2_code"]).size() == 1).all()),
        "each_abc_group_predicts_each_city_once": bool((abc.groupby(["prediction_group", "adm2_code"]).size() == 1).all()),
        "all_predictions_finite": bool(np.isfinite(predictions[["y_true_hdi", "y_pred_hdi"]].to_numpy(float)).all()),
        "inner_grid_complete": len(tuning) == 34 * 4 * 5 * 5,
        "inner_fold_records_complete": len(folds) == 34 * 5 * 4 * 5 * 5,
        "selection_records_complete": len(selection) == 34 * (5 + 1 + 3 + 3),
    }
    manifest = {
        "run_protocol": RUN_PROTOCOL,
        "frozen_protocol": PROTOCOL,
        "status": "PASS" if all(criteria.values()) else "FAIL",
        "criteria": criteria,
        "sample_cities": len(matrix), "sample_provinces": matrix.adm1_code.nunique(),
        "outer_folds": len(splits), "alphas": ALPHAS,
        "models": {name: len(models[name]) for name in model_order},
        "software": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        },
        "outputs": {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()},
    }
    manifest_path = OUT / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    hash_rows = [
        {"item": MATRIX.name, "path": str(MATRIX), "sha256": sha256(MATRIX)},
        {"item": SPLITS.name, "path": str(SPLITS), "sha256": sha256(SPLITS)},
        {"item": MODELS_FILE.name, "path": str(MODELS_FILE), "sha256": sha256(MODELS_FILE)},
        *[{"item": path.name, "path": str(path), "sha256": sha256(path)} for path in paths.values()],
        {"item": manifest_path.name, "path": str(manifest_path), "sha256": sha256(manifest_path)},
    ]
    write_csv(pd.DataFrame(hash_rows), OUT / "run_hashes_sha256.csv")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
