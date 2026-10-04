"""Nested spatial comparison of three HDI aggregation rules.

P0: fixed geometric mean of strategy-C component predictions.
P1: sigmoid of a learned linear function of the three C predictions.
P2: P1 plus all pairwise component interactions.

The outer predictions are the manuscript's archived C predictions.  For each
outer fold, P1/P2 are trained only on cross-fitted component predictions from
the corresponding outer-training regions.  No outer-test outcome is used to
fit an aggregation rule.  Coefficients are predictive, not welfare weights.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
MANUSCRIPT = WORKSPACE / "JAG_manuscript_US_uncertainty_20260927"
SUPP = MANUSCRIPT / "supplementary_data"
ROOT = Path(r"E:\BeyondSurfaceHDI")
CORE_DIR = ROOT / "src"
sys.path.insert(0, str(CORE_DIR))
import run_indonesia_abc_2020 as core  # noqa: E402
import run_usa_c_validation_suite as usa_suite  # noqa: E402

SEED = 20260928
EPS = 1e-6
ALPHAS = [0.0, 0.01, 0.1, 1.0, 10.0]
BOOTSTRAP_REPS = 20_000
SIGNFLIP_REPS = 100_000
COMPONENTS = ["y_health_index", "y_education_index", "y_income_index"]
SHORT = {"y_health_index": "H", "y_education_index": "E", "y_income_index": "I"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sigmoid(x):
    x = np.asarray(x, float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1 / (1 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1 + ex)
    return out


def logit(y):
    y = np.clip(np.asarray(y, float), EPS, 1 - EPS)
    return np.log(y / (1 - y))


def meta_features(z: np.ndarray, interactions: bool) -> tuple[np.ndarray, list[str]]:
    z = np.asarray(z, float)
    columns = [z[:, 0], z[:, 1], z[:, 2]]
    names = ["H", "E", "I"]
    if interactions:
        columns.extend([z[:, 0] * z[:, 1], z[:, 0] * z[:, 2], z[:, 1] * z[:, 2]])
        names.extend(["H_E", "H_I", "E_I"])
    return np.column_stack(columns), names


def group_equal_mae(y, p, groups) -> float:
    tmp = pd.DataFrame({"y": y, "p": p, "g": groups})
    return float(tmp.assign(a=lambda x: abs(x.y - x.p)).groupby("g").a.mean().mean())


def fit_meta(z: np.ndarray, y: np.ndarray, groups: np.ndarray, interactions: bool,
             tune: bool = True) -> tuple[dict, pd.DataFrame]:
    x, names = meta_features(z, interactions)
    rows = []
    if tune:
        n_splits = min(5, len(np.unique(groups)))
        folds = list(GroupKFold(n_splits=n_splits).split(x, groups=groups))
        for alpha in ALPHAS:
            pred = np.full(len(y), np.nan)
            for tr, va in folds:
                scaler = StandardScaler().fit(x[tr])
                model = Ridge(alpha=alpha, fit_intercept=True, solver="svd").fit(
                    scaler.transform(x[tr]), logit(y[tr])
                )
                pred[va] = sigmoid(model.predict(scaler.transform(x[va])))
            rows.append({"alpha": alpha, "group_equal_MAE": group_equal_mae(y, pred, groups)})
        tuning = pd.DataFrame(rows)
        best_score = tuning.group_equal_MAE.min()
        alpha = float(tuning.loc[tuning.group_equal_MAE <= best_score + 1e-12, "alpha"].min())
    else:
        alpha = 1.0
        tuning = pd.DataFrame([{"alpha": alpha, "group_equal_MAE": np.nan}])
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=alpha, fit_intercept=True, solver="svd").fit(scaler.transform(x), logit(y))
    raw_coef = model.coef_ / scaler.scale_
    raw_intercept = float(model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_))
    return {
        "model": model, "scaler": scaler, "alpha": alpha, "names": names,
        "raw_intercept": raw_intercept, "raw_coef": raw_coef,
    }, tuning


def predict_meta(fitted: dict, z: np.ndarray, interactions: bool) -> np.ndarray:
    x, _ = meta_features(z, interactions)
    return sigmoid(fitted["model"].predict(fitted["scaler"].transform(x)))


def monotonicity_min_derivatives(fitted: dict, interactions: bool) -> dict:
    b = dict(zip(fitted["names"], fitted["raw_coef"]))
    if not interactions:
        return {f"min_deta_d{k}": float(b[k]) for k in ["H", "E", "I"]}
    # Bilinear eta has extrema in each derivative at the other variables' corners.
    return {
        "min_deta_dH": float(b["H"] + min(0, b["H_E"]) + min(0, b["H_I"])),
        "min_deta_dE": float(b["E"] + min(0, b["H_E"]) + min(0, b["E_I"])),
        "min_deta_dI": float(b["I"] + min(0, b["H_I"]) + min(0, b["E_I"])),
    }


def model_columns(prefix=True):
    p = "feature_" if prefix else ""
    pop = [p + x for x in [
        "control_log1p_population", "control_log1p_population_density",
        "control_worldpop_child_share", "control_worldpop_elderly_share",
        "control_urbanization_rate"]]
    eo = [p + f"ae320_A{i:02d}_popmean" for i in range(64)]
    econ = [p + "economic_log1p_retail_yuan_per_resident"]
    med = [p + x for x in ["health_log1p_motorized_mean_minutes",
        "health_log1p_q90_minutes", "health_log1p_sd_minutes",
        "health_population_fraction_ge60min"]]
    return {"B69": pop + eo, "R6": pop + econ, "B_R70": pop + eo + econ,
            "R_S10": pop + econ + med, "B_RS74": pop + eo + econ + med}


def make_group_splits(frame: pd.DataFrame, id_col="adm2_code", group_col="adm1_code"):
    splits = []
    for group in sorted(frame[group_col].astype(str).unique()):
        test = frame.loc[frame[group_col].astype(str).eq(group)].sort_values(id_col)
        train = frame.loc[~frame[group_col].astype(str).eq(group)].sort_values(id_col)
        inner = []
        gkf = GroupKFold(n_splits=5)
        for fold, (ti, vi) in enumerate(gkf.split(train, groups=train[group_col].astype(str)), 1):
            inner.append({"fold": fold,
                "train_ids": train.iloc[ti][id_col].astype(str).tolist(),
                "validation_ids": train.iloc[vi][id_col].astype(str).tolist()})
        splits.append({"outer_group": group,
            "outer_train_ids": train[id_col].astype(str).tolist(),
            "outer_test_ids": test[id_col].astype(str).tolist(), "inner_folds": inner})
    return splits


def load_country(country: str):
    if country == "China":
        matrix_path = SUPP / "D49_China_neural_input_matrix.csv"
        matrix = pd.read_csv(matrix_path, dtype={"city_code": str, "province_code": str})
        matrix = matrix.rename(columns={"city_code": "unit", "province_code": "group",
            "hdi": "y_hdi", "lifeindex": "y_health_index", "eduindex": "y_education_index",
            "incomeindex": "y_income_index"})
        payload = json.loads((SUPP / "D50_China_neural_splits.json").read_text(encoding="utf-8"))
        splits = []
        for outer in payload["splits"]:
            splits.append({"outer_group": str(outer["outer_test_province"]),
                "outer_train_ids": list(map(str, outer["outer_train_city_codes"])),
                "outer_test_ids": list(map(str, outer["outer_test_city_codes"])),
                "inner_folds": [{"fold": x["fold"],
                    "train_ids": list(map(str, x["train_city_codes"])),
                    "validation_ids": list(map(str, x["validation_city_codes"]))}
                    for x in outer["inner_folds"]]})
        choices = pd.read_csv(SUPP / "D10_China_selection_decisions.csv", dtype={"outer_test_province": str})
        choices = choices.loc[(choices.eo_year.eq(2020)) & choices.group.eq("C")]
        target_map = {"health": "y_health_index", "education": "y_education_index", "income": "y_income_index"}
        fixed = {(str(r.outer_test_province), target_map[r.selection_target]):
                 (r.selected_model, None) for r in choices.itertuples()}
        models = model_columns(True)
        pred = pd.read_csv(SUPP / "D08_component_predictions.csv", dtype={"unit": str, "province": str})
        pred = pred.loc[(pred.country.eq("China")) & pred.method.eq("C")]
        pred["target"] = pred.target.map({"health": "y_health_index", "education": "y_education_index", "income": "y_income_index"})
        pred = pred.rename(columns={"unit": "unit", "province": "group", "predicted": "pred"})
        sources = [matrix_path, SUPP / "D50_China_neural_splits.json", SUPP / "D10_China_selection_decisions.csv", SUPP / "D08_component_predictions.csv"]
    elif country == "USA":
        matrix_path = ROOT / "data_work" / "usa_2019" / "model_freeze_pop100_2019" / "usa_model_freeze_pop100_2019.csv"
        matrix = pd.read_csv(matrix_path, dtype={"geoid": str, "state_fips": str})
        matrix["geoid"] = matrix.geoid.str.zfill(5); matrix["state_fips"] = matrix.state_fips.str.zfill(2)
        eligible = pd.read_csv(ROOT / "data_work" / "usa_2019" / "model_freeze_2019" / "usa_model_freeze_2019.csv",
                               dtype={"geoid": str})
        eligible_ids = set(eligible.loc[eligible.model_eligible.astype(bool), "geoid"].str.zfill(5))
        matrix = matrix.loc[matrix.model_eligible.astype(bool) & matrix.geoid.isin(eligible_ids)].copy()
        matrix = matrix.rename(columns={"geoid": "unit", "state_fips": "group"})
        splits = make_group_splits(matrix, "unit", "group")
        choices = pd.read_csv(SUPP / "D20_USA_100m_selection_records.csv", dtype={"outer_test_province": str})
        choices["outer_test_province"] = choices.outer_test_province.str.zfill(2)
        choices = choices.loc[choices.selection_group.eq("C_dimension_specific")]
        fixed = {(r.outer_test_province, r.selection_target): (r.selected_model, float(r.selected_alpha)) for r in choices.itertuples()}
        models = usa_suite.FULL_MODELS
        pred = pd.read_csv(SUPP / "D19_USA_100m_component_predictions.csv", dtype={"adm1_code": str, "adm2_code": str})
        pred = pred.loc[pred.prediction_group.eq("C_dimension_specific")].rename(
            columns={"adm2_code": "unit", "adm1_code": "group", "y_pred_clipped": "pred"})
        pred["group"] = pred.group.str.zfill(2)
        sources = [matrix_path, SUPP / "D20_USA_100m_selection_records.csv", SUPP / "D19_USA_100m_component_predictions.csv"]
    elif country == "Indonesia":
        matrix_path = ROOT / "data_raw" / "indonesia_validation" / "frozen_experiment_2020" / "indonesia_adm2_model_matrix_2020.csv"
        matrix = pd.read_csv(matrix_path, dtype={"adm2_code": str, "adm1_code": str})
        cohort = pd.read_csv(SUPP / "D04_Indonesia_cohort_514.csv", dtype={"adm2_code": str})
        ids = set(cohort.loc[cohort.included.astype(bool), "adm2_code"])
        matrix = matrix.loc[matrix.adm2_code.isin(ids)].copy().rename(columns={"adm2_code": "unit", "adm1_code": "group"})
        # The manuscript's corrected 443-unit labels supersede the older 505-unit
        # freeze labels.  Replace all four targets before any cross-fitting.
        current_components = pd.read_csv(SUPP / "D08_component_predictions.csv", dtype={"unit": str})
        current_components = current_components.loc[(current_components.country.eq("Indonesia")) &
                                                    current_components.method.eq("C")]
        current_components["target"] = current_components.target.map(
            {"health": "y_health_index", "education": "y_education_index", "income": "y_income_index"})
        current_components = current_components.pivot(index="unit", columns="target", values="observed")
        current_hdi = pd.read_csv(SUPP / "D01_ABC_predictions.csv", dtype={"unit": str})
        current_hdi = current_hdi.loc[(current_hdi.country.eq("Indonesia")) & current_hdi.method.eq("C")]
        current_hdi = current_hdi.drop_duplicates("unit").set_index("unit")["observed"]
        matrix = matrix.set_index("unit", drop=False)
        if set(matrix.index) != set(current_components.index) or set(matrix.index) != set(current_hdi.index):
            raise RuntimeError("Corrected Indonesia target IDs do not match the 443-unit matrix")
        for target in COMPONENTS:
            matrix.loc[current_components.index, target] = current_components[target]
        matrix.loc[current_hdi.index, "y_hdi"] = current_hdi
        matrix = matrix.reset_index(drop=True)
        splits = make_group_splits(matrix, "unit", "group")
        choices = pd.read_csv(SUPP / "D11_Indonesia_selection_decisions.csv", dtype={"outer_test_province": str})
        choices = choices.loc[choices.selection_group.eq("C_dimension_specific")]
        fixed = {(r.outer_test_province, r.selection_target): (r.selected_model, float(r.selected_alpha)) for r in choices.itertuples()}
        models_payload = json.loads((ROOT / "data_raw" / "indonesia_validation" / "frozen_experiment_2020" / "model_input_combinations_2020.json").read_text(encoding="utf-8"))
        models = models_payload["models"] if "models" in models_payload else models_payload
        pred = pd.read_csv(SUPP / "D08_component_predictions.csv", dtype={"unit": str, "province": str})
        pred = pred.loc[(pred.country.eq("Indonesia")) & pred.method.eq("C")]
        pred["target"] = pred.target.map({"health": "y_health_index", "education": "y_education_index", "income": "y_income_index"})
        pred = pred.rename(columns={"province": "group", "predicted": "pred"})
        sources = [matrix_path, SUPP / "D04_Indonesia_cohort_514.csv", SUPP / "D11_Indonesia_selection_decisions.csv",
                   SUPP / "D08_component_predictions.csv", SUPP / "D01_ABC_predictions.csv"]
    else:
        raise ValueError(country)
    matrix["unit"] = matrix.unit.astype(str); matrix["group"] = matrix.group.astype(str)
    matrix = matrix.sort_values("unit").reset_index(drop=True)
    pred["unit"] = pred.unit.astype(str); pred["group"] = pred.group.astype(str)
    if country == "China":
        pred = pred.loc[pred.unit.isin(set(sum([x["outer_test_ids"] for x in splits], [])))]
    counts = matrix.shape[0], matrix.group.nunique(), len(splits), pred.unit.nunique()
    return matrix, splits, models, fixed, pred[["unit", "group", "target", "pred"]], sources, counts


def fit_base(train: pd.DataFrame, valid: pd.DataFrame, columns: list[str], target: str, alpha: float):
    xtr = train[columns].to_numpy(float); xva = valid[columns].to_numpy(float)
    keep = np.ptp(xtr, axis=0) > 0
    scaler = StandardScaler().fit(xtr[:, keep])
    model = Ridge(alpha=alpha, fit_intercept=True, solver="svd").fit(scaler.transform(xtr[:, keep]), train[target].to_numpy(float))
    return np.clip(model.predict(scaler.transform(xva[:, keep])), 0, 1)


def crossfit_outer_training(matrix: pd.DataFrame, outer: dict, models: dict, fixed: dict, country: str):
    index = matrix.set_index("unit", drop=False)
    train_all = index.loc[outer["outer_train_ids"]].copy()
    predictions = {target: pd.Series(index=train_all.unit, dtype=float) for target in COMPONENTS}
    selected_rows = []
    for target in COMPONENTS:
        model_name, fixed_alpha = fixed[(outer["outer_group"], target)]
        candidate_alphas = core.ALPHAS if fixed_alpha is None else [fixed_alpha]
        per_alpha = {}
        for alpha in candidate_alphas:
            values = pd.Series(index=train_all.unit, dtype=float)
            for inner in outer["inner_folds"]:
                tr = index.loc[inner["train_ids"]]
                va = index.loc[inner["validation_ids"]]
                values.loc[va.unit] = fit_base(tr, va, models[model_name], target, float(alpha))
            if values.isna().any():
                raise RuntimeError(f"Incomplete crossfit: {country} {outer['outer_group']} {target}")
            score = group_equal_mae(train_all[target].to_numpy(float), values.loc[train_all.unit].to_numpy(float), train_all.group.to_numpy())
            per_alpha[float(alpha)] = (score, values)
        best = min(v[0] for v in per_alpha.values())
        alpha = min(a for a, v in per_alpha.items() if v[0] <= best + 1e-12)
        predictions[target] = per_alpha[alpha][1]
        selected_rows.append({"country": country, "outer_group": outer["outer_group"], "target": target,
                              "model": model_name, "alpha": alpha, "inner_group_equal_MAE": best})
    z = np.column_stack([predictions[t].loc[train_all.unit].to_numpy(float) for t in COMPONENTS])
    return train_all, z, selected_rows


def metric_rows(country: str, predictions: pd.DataFrame):
    rows = []
    for method, d in predictions.groupby("method", sort=False):
        y = d.y_true.to_numpy(float); p = d.y_pred.to_numpy(float)
        rel = np.abs(y - p) / y
        rows.append({"country": country, "method": method, "n": len(d), "groups": d.group.nunique(),
            "MAE": mean_absolute_error(y, p), "RMSE": math.sqrt(mean_squared_error(y, p)),
            "R2": r2_score(y, p), "bias_pred_minus_true": float(np.mean(p-y)),
            "P3": float(np.mean(rel <= .03)), "P5": float(np.mean(rel <= .05)),
            "P8": float(np.mean(rel <= .08)), "P10": float(np.mean(rel <= .10))})
    return rows


def cluster_inference(country: str, d: pd.DataFrame, candidate: str, rng: np.random.Generator):
    base = d.loc[d.method.eq("P0_geometric")].set_index("unit")
    alt = d.loc[d.method.eq(candidate)].set_index("unit").loc[base.index]
    groups = base.group.astype(str)
    unique = sorted(groups.unique())
    arrays = {}
    for loss in ["MAE", "MSE"]:
        lb = np.abs(base.y_true-base.y_pred).to_numpy() if loss == "MAE" else (base.y_true-base.y_pred).to_numpy()**2
        la = np.abs(alt.y_true-alt.y_pred).to_numpy() if loss == "MAE" else (alt.y_true-alt.y_pred).to_numpy()**2
        arrays[loss] = (lb, la)
    counts = np.array([(groups == g).sum() for g in unique], float)
    out = []
    for loss, (lb, la) in arrays.items():
        sb = np.array([lb[groups.eq(g)].sum() for g in unique]); sa = np.array([la[groups.eq(g)].sum() for g in unique])
        observed = float((sb.sum()-sa.sum())/counts.sum())
        idx = rng.integers(0, len(unique), size=(BOOTSTRAP_REPS, len(unique)))
        gain = (sb[idx].sum(1)-sa[idx].sum(1))/counts[idx].sum(1)
        signs = rng.choice([-1.0, 1.0], size=(SIGNFLIP_REPS, len(unique)))
        null = (signs @ (sb-sa))/counts.sum()
        p = float((np.count_nonzero(np.abs(null) >= abs(observed)) + 1)/(SIGNFLIP_REPS+1))
        row = {"country": country, "candidate": candidate, "metric": loss,
               "gain_P0_minus_candidate": observed, "ci_low": float(np.quantile(gain,.025)),
               "ci_high": float(np.quantile(gain,.975)), "group_signflip_p": p}
        if loss == "MSE":
            rmse_obs = math.sqrt(sb.sum()/counts.sum())-math.sqrt(sa.sum()/counts.sum())
            rmse_gain = np.sqrt(sb[idx].sum(1)/counts[idx].sum(1))-np.sqrt(sa[idx].sum(1)/counts[idx].sum(1))
            row.update({"RMSE_gain": rmse_obs, "RMSE_ci_low": float(np.quantile(rmse_gain,.025)),
                        "RMSE_ci_high": float(np.quantile(rmse_gain,.975))})
        out.append(row)
    return out


def holm(pvalues):
    p = np.asarray(pvalues, float); order = np.argsort(p); adjusted = np.empty_like(p); running = 0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m-rank)*p[idx]); adjusted[idx] = min(1.0, running)
    return adjusted


def main():
    all_predictions, all_metrics, all_coefficients, all_tuning, all_base_selection = [], [], [], [], []
    audit = {"status": "RUNNING", "seed": SEED, "protocol": "nested_spatial_C_aggregation_v1",
             "methods": {"P0_geometric": "(Hhat*Ehat*Ihat)^(1/3)",
                         "P1_linear": "sigmoid(b+bH*Hhat+bE*Ehat+bI*Ihat)",
                         "P2_interactions": "P1 logit plus bHE*Hhat*Ehat+bHI*Hhat*Ihat+bEI*Ehat*Ihat"},
             "meta_alpha_grid": ALPHAS, "sources": {}}
    for country in ["China", "USA", "Indonesia"]:
        matrix, splits, models, fixed, outer_components, sources, counts = load_country(country)
        audit["sources"][country] = {str(p): sha256(p) for p in sources}
        audit.setdefault("counts", {})[country] = {"matrix_rows": counts[0], "groups": counts[1], "outer_folds": counts[2], "scored_units": counts[3]}
        outer_wide = outer_components.pivot(index=["unit","group"], columns="target", values="pred").reset_index()
        matrix_index = matrix.set_index("unit")
        country_rows = []
        for number, outer in enumerate(splits, 1):
            train, z_train, selected_rows = crossfit_outer_training(matrix, outer, models, fixed, country)
            all_base_selection.extend(selected_rows)
            p1, tune1 = fit_meta(z_train, train.y_hdi.to_numpy(float), train.group.to_numpy(), False)
            p2, tune2 = fit_meta(z_train, train.y_hdi.to_numpy(float), train.group.to_numpy(), True)
            for label, fitted, tuning, inter in [("P1_linear",p1,tune1,False),("P2_interactions",p2,tune2,True)]:
                for row in tuning.itertuples():
                    all_tuning.append({"country": country, "outer_group": outer["outer_group"], "method": label,
                                       "alpha": row.alpha, "inner_group_equal_MAE": row.group_equal_MAE,
                                       "selected": abs(row.alpha-fitted["alpha"]) < 1e-12})
                coeff = {"country": country, "record_type": "outer_fold", "outer_group": outer["outer_group"],
                         "method": label, "alpha": fitted["alpha"], "intercept": fitted["raw_intercept"],
                         **monotonicity_min_derivatives(fitted, inter)}
                coeff.update({f"coef_{n}": float(v) for n,v in zip(fitted["names"],fitted["raw_coef"])})
                all_coefficients.append(coeff)
            test = outer_wide.loc[outer_wide.group.astype(str).eq(str(outer["outer_group"]))].copy()
            if set(test.unit) != set(outer["outer_test_ids"]):
                raise RuntimeError(f"Outer archive mismatch {country} {outer['outer_group']}")
            test = test.sort_values("unit")
            z_test = test[COMPONENTS].to_numpy(float)
            y_test = matrix_index.loc[test.unit, "y_hdi"].to_numpy(float)
            preds = {"P0_geometric": np.cbrt(np.prod(z_test,axis=1)),
                     "P1_linear": predict_meta(p1,z_test,False),
                     "P2_interactions": predict_meta(p2,z_test,True)}
            for method, values in preds.items():
                for unit, group, y, value, h, e, inc in zip(test.unit,test.group,y_test,values,*z_test.T):
                    country_rows.append({"country":country,"unit":unit,"group":group,"outer_group":outer["outer_group"],
                        "method":method,"y_true":y,"y_pred":float(value),"pred_H":h,"pred_E":e,"pred_I":inc})
            print(f"{country} outer {number:02d}/{len(splits)} {outer['outer_group']} complete", flush=True)
        country_pred = pd.DataFrame(country_rows)
        all_predictions.append(country_pred)
        all_metrics.extend(metric_rows(country, country_pred))
        # A post-evaluation deployment fit on archived country-wide OOF components.
        z = outer_wide[COMPONENTS].to_numpy(float)
        y = matrix_index.loc[outer_wide.unit,"y_hdi"].to_numpy(float)
        groups = outer_wide.group.to_numpy()
        for label, inter in [("P1_linear",False),("P2_interactions",True)]:
            fitted, tuning = fit_meta(z,y,groups,inter)
            coeff = {"country":country,"record_type":"post_evaluation_deployment","outer_group":"ALL",
                     "method":label,"alpha":fitted["alpha"],"intercept":fitted["raw_intercept"],
                     **monotonicity_min_derivatives(fitted,inter)}
            coeff.update({f"coef_{n}":float(v) for n,v in zip(fitted["names"],fitted["raw_coef"])})
            all_coefficients.append(coeff)
    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.DataFrame(all_metrics)
    coefficients = pd.DataFrame(all_coefficients)
    tuning = pd.DataFrame(all_tuning)
    base_selection = pd.DataFrame(all_base_selection)
    rng = np.random.default_rng(SEED)
    inference = []
    for country in ["China","USA","Indonesia"]:
        d = predictions.loc[predictions.country.eq(country)]
        for candidate in ["P1_linear","P2_interactions"]:
            inference.extend(cluster_inference(country,d,candidate,rng))
    inference = pd.DataFrame(inference)
    inference["holm_p_12_tests"] = holm(inference.group_signflip_p)
    for name, frame in [("outer_predictions.csv",predictions),("metrics.csv",metrics),
                        ("paired_inference.csv",inference),("aggregation_coefficients.csv",coefficients),
                        ("meta_tuning.csv",tuning),("base_C_inner_selection.csv",base_selection)]:
        frame.to_csv(HERE/name,index=False,encoding="utf-8-sig",float_format="%.12g")
    # Verification invariants.
    expected = {"China":240,"USA":3104,"Indonesia":443}
    checks = {}
    for country,n in expected.items():
        part = predictions.loc[predictions.country.eq(country)]
        checks[country] = {"rows":len(part),"each_method_n":part.groupby("method").size().to_dict(),
                           "unique_units":part.unit.nunique(),"expected_units":n,
                           "all_finite":bool(np.isfinite(part[["y_true","y_pred","pred_H","pred_E","pred_I"]]).all().all())}
        if part.unit.nunique()!=n or not (part.groupby("method").size()==n).all(): raise RuntimeError(checks[country])
    audit.update({"status":"PASS","bootstrap_replicates":BOOTSTRAP_REPS,"signflip_replicates":SIGNFLIP_REPS,
                  "primary_comparisons":["P1-P0","P2-P0"],"multiplicity":"Holm over 3 countries x 2 methods x 2 losses",
                  "outer_test_independence":"Aggregation learned from outer-training cross-fitted C predictions only",
                  "selection_induced_note":"Base model/alpha selection uses pooled inner-fold outcomes; outer test remains untouched",
                  "coefficient_interpretation":"predictive parameters, not normative weights or causal effects",
                  "deployment_coefficients":"post-evaluation descriptive fit; no independent performance claim",
                  "checks":checks})
    (HERE/"audit.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding="utf-8")
    print("\nMETRICS\n"+metrics.to_string(index=False),flush=True)
    print("\nINFERENCE\n"+inference.to_string(index=False),flush=True)
    print("\nDEPLOYMENT COEFFICIENTS\n"+coefficients.loc[coefficients.record_type.eq("post_evaluation_deployment")].to_string(index=False),flush=True)


if __name__ == "__main__":
    main()
