"""Complete the predefined USA C-focused validation suite.

The primary model formula is unchanged: component-specific Ridge selection for
health, education and income, followed by the fixed geometric mean.  All
exclusions are based on pre-fit label/predictor/coverage availability.
"""
from __future__ import annotations

import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(r"E:\BeyondSurfaceHDI")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

import run_indonesia_abc_2020 as core

FREEZE = ROOT / "data_work" / "usa_2019" / "model_freeze_2019" / "usa_model_freeze_2019.csv"
BASE_RUN = ROOT / "runs" / "usa_abc_2019"
OUT = ROOT / "runs" / "usa_c_validation_suite_2019"
REPORT = ROOT / "reports" / "usa_2019"

POP = [
    "control_log1p_population",
    "control_log1p_population_density",
    "control_worldpop_child_share",
    "control_worldpop_elderly_share",
    "control_urbanization_rate",
]
EO = [f"ae320_A{i:02d}_popmean" for i in range(64)]
ECON = ["economy_log1p_retail_payroll_per_capita"]
MED = [
    "health_log1p_motorized_mean_minutes",
    "health_log1p_motorized_q90_minutes",
    "health_log1p_motorized_sd_minutes",
    "health_motorized_share_ge60min",
]
FULL_MODELS = {
    "B69": POP + EO,
    "R6": POP + ECON,
    "B_R70": POP + EO + ECON,
    "R_S10": POP + ECON + MED,
    "B_RS74": POP + EO + ECON + MED,
}
FULL_ORDER = ["B69", "R6", "B_R70", "R_S10", "B_RS74"]
ABLATIONS = {
    "no_EO": ({"R6": POP + ECON, "R_S10": POP + ECON + MED}, ["R6", "R_S10"]),
    "no_economy": (
        {"B69": POP + EO, "P_S9": POP + MED, "B_S73": POP + EO + MED},
        ["P_S9", "B69", "B_S73"],
    ),
    "no_healthcare": (
        {"B69": POP + EO, "R6": POP + ECON, "B_R70": POP + EO + ECON},
        ["R6", "B69", "B_R70"],
    ),
}

DIVISIONS = {
    "New England": ["09", "23", "25", "33", "44", "50"],
    "Middle Atlantic": ["34", "36", "42"],
    "East North Central": ["17", "18", "26", "39", "55"],
    "West North Central": ["19", "20", "27", "29", "31", "38", "46"],
    "South Atlantic": ["10", "11", "12", "13", "24", "37", "45", "51", "54"],
    "East South Central": ["01", "21", "28", "47"],
    "West South Central": ["05", "22", "40", "48"],
    "Mountain": ["04", "08", "16", "30", "32", "35", "49", "56"],
    "Pacific": ["02", "06", "15", "41", "53"],
}

G_MATRIX = None
G_MODELS = None
G_ORDER = None


def load_freeze() -> pd.DataFrame:
    frame = pd.read_csv(FREEZE, dtype={"state_fips": str, "geoid": str})
    frame["state_fips"] = frame.state_fips.str.zfill(2)
    frame["geoid"] = frame.geoid.str.zfill(5)
    return frame.rename(columns={"state_fips": "adm1_code", "geoid": "adm2_code"})


def state_splits(matrix: pd.DataFrame) -> list[dict]:
    rows = []
    for state in sorted(matrix.adm1_code.unique()):
        test = matrix.loc[matrix.adm1_code.eq(state)].sort_values("adm2_code")
        train = matrix.loc[~matrix.adm1_code.eq(state)].sort_values("adm2_code")
        splitter = GroupKFold(n_splits=5)
        inner = []
        for fold, (ti, vi) in enumerate(splitter.split(train, groups=train.adm1_code), 1):
            inner.append({
                "fold": fold,
                "train_adm2_codes": train.iloc[ti].adm2_code.tolist(),
                "validation_adm2_codes": train.iloc[vi].adm2_code.tolist(),
            })
        rows.append({
            "outer_test_province": state,
            "outer_train_adm2_codes": train.adm2_code.tolist(),
            "outer_test_adm2_codes": test.adm2_code.tolist(),
            "inner_folds": inner,
        })
    return rows


def division_splits(matrix: pd.DataFrame) -> list[dict]:
    rows = []
    all_states = set(matrix.adm1_code.unique())
    mapped = {state for values in DIVISIONS.values() for state in values}
    if all_states != mapped:
        raise RuntimeError(f"Census Division map mismatch: data={sorted(all_states)} map={sorted(mapped)}")
    for division, states in DIVISIONS.items():
        test = matrix.loc[matrix.adm1_code.isin(states)].sort_values("adm2_code")
        train = matrix.loc[~matrix.adm1_code.isin(states)].sort_values("adm2_code")
        splitter = GroupKFold(n_splits=5)
        inner = []
        for fold, (ti, vi) in enumerate(splitter.split(train, groups=train.adm1_code), 1):
            inner.append({
                "fold": fold,
                "train_adm2_codes": train.iloc[ti].adm2_code.tolist(),
                "validation_adm2_codes": train.iloc[vi].adm2_code.tolist(),
            })
        rows.append({
            "outer_test_province": division,
            "outer_train_adm2_codes": train.adm2_code.tolist(),
            "outer_test_adm2_codes": test.adm2_code.tolist(),
            "inner_folds": inner,
        })
    return rows


def init_worker(matrix_path: str, models: dict, order: list[str]):
    global G_MATRIX, G_MODELS, G_ORDER
    G_MATRIX = pd.read_pickle(matrix_path)
    G_MODELS = models
    G_ORDER = order


def run_worker(split: dict):
    return core.run_outer(G_MATRIX, split, G_MODELS, G_ORDER)


def run_nested(tag: str, matrix: pd.DataFrame, splits: list[dict], models: dict, order: list[str]):
    folder = OUT / tag
    folder.mkdir(parents=True, exist_ok=True)
    matrix_path = folder / "matrix.pkl"
    matrix.to_pickle(matrix_path)
    buckets = [[], [], [], [], []]
    workers = min(8, max(1, (os.cpu_count() or 4) - 1))
    print(f"{tag}: {len(splits)} outer folds with {workers} workers", flush=True)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(str(matrix_path), models, order),
    ) as pool:
        for i, result in enumerate(pool.map(run_worker, splits, chunksize=1), 1):
            for bucket, frame in zip(buckets, result):
                bucket.append(frame)
            print(f"{tag}: [{i}/{len(splits)}]", flush=True)
    names = [
        "inner_tuning_scores",
        "inner_fold_metrics",
        "outer_selection_records",
        "outer_hdi_predictions",
        "outer_component_predictions",
    ]
    frames = {name: pd.concat(parts, ignore_index=True) for name, parts in zip(names, buckets)}
    for name, frame in frames.items():
        frame.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    matrix_path.unlink()
    return frames


def metric_row(data: pd.DataFrame, true: str, pred: str, group_col: str = "adm1_code") -> dict:
    y = data[true].to_numpy(float)
    p = data[pred].to_numpy(float)
    rel = np.abs(p - y) / y
    group_mae = data.assign(_ae=np.abs(p - y)).groupby(group_col)._ae.mean()
    return {
        "N": len(data),
        "groups": data[group_col].nunique(),
        "group_equal_MAE": float(group_mae.mean()),
        "unit_equal_MAE": mean_absolute_error(y, p),
        "RMSE": math.sqrt(mean_squared_error(y, p)),
        "MAPE_pct": float(rel.mean() * 100),
        "R2": r2_score(y, p),
        "within_5pct": float((rel <= 0.05).mean()),
        "within_8pct": float((rel <= 0.08).mean()),
    }


def component_metrics() -> pd.DataFrame:
    frame = pd.read_csv(
        BASE_RUN / "outer_component_predictions.csv",
        dtype={"adm1_code": str, "adm2_code": str},
    )
    rows = []
    for (strategy, target), data in frame.groupby(["prediction_group", "target"]):
        rows.append({
            "strategy": strategy,
            "component": target.removeprefix("y_").removesuffix("_index"),
            **metric_row(data, "y_true", "y_pred_clipped"),
        })
    return pd.DataFrame(rows)


def bootstrap_main_predictions(reps: int = 20000) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(BASE_RUN / "outer_hdi_predictions.csv", dtype={"adm1_code": str, "adm2_code": str})
    frame = frame.loc[frame.prediction_group.isin(["A_direct", "B_common_components", "C_dimension_specific"])].copy()
    frame["absolute_error"] = np.abs(frame.y_pred_hdi - frame.y_true_hdi)
    frame["within5"] = (frame.absolute_error / frame.y_true_hdi <= 0.05).astype(float)
    frame["within8"] = (frame.absolute_error / frame.y_true_hdi <= 0.08).astype(float)
    labels = ["A_direct", "B_common_components", "C_dimension_specific"]
    states = sorted(frame.adm1_code.unique())
    state_arrays = {}
    for label in labels:
        subset = frame.loc[frame.prediction_group.eq(label)]
        grouped = subset.groupby("adm1_code", sort=True).agg(
            n=("adm2_code", "size"),
            absolute_error_sum=("absolute_error", "sum"),
            within5_sum=("within5", "sum"),
            within8_sum=("within8", "sum"),
        ).reindex(states)
        state_arrays[label] = grouped[["n", "absolute_error_sum", "within5_sum", "within8_sum"]].to_numpy(float)
    rng = np.random.default_rng(20260926)
    sampled = rng.integers(0, len(states), size=(reps, len(states)))
    draws = {}
    for label in labels:
        values = state_arrays[label][sampled].sum(axis=1)
        denominator = values[:, 0]
        draws[label] = {
            "MAE": values[:, 1] / denominator,
            "P5": values[:, 2] / denominator,
            "P8": values[:, 3] / denominator,
        }
    diffs = {}
    for name, other in [("C_minus_A", "A_direct"), ("C_minus_B", "B_common_components")]:
        diffs[name] = {
            "MAE_gain": draws[other]["MAE"] - draws["C_dimension_specific"]["MAE"],
            "P5_gain": draws["C_dimension_specific"]["P5"] - draws[other]["P5"],
            "P8_gain": draws["C_dimension_specific"]["P8"] - draws[other]["P8"],
        }
    rows = []
    for label in labels:
        for metric, values in draws[label].items():
            lo, hi = np.quantile(values, [0.025, 0.975])
            observed = metric_row(frame.loc[frame.prediction_group.eq(label)], "y_true_hdi", "y_pred_hdi")
            key = {"MAE": "unit_equal_MAE", "P5": "within_5pct", "P8": "within_8pct"}[metric]
            rows.append({"strategy": label, "metric": metric, "estimate": observed[key], "cluster_bootstrap_lo": lo, "cluster_bootstrap_hi": hi})
    comp_rows = []
    for name, metrics in diffs.items():
        for metric, values in metrics.items():
            lo, hi = np.quantile(values, [0.025, 0.975])
            comp_rows.append({"contrast": name, "metric": metric, "bootstrap_mean": float(np.mean(values)), "cluster_bootstrap_lo": lo, "cluster_bootstrap_hi": hi})
    return pd.DataFrame(rows), pd.DataFrame(comp_rows)


def subgroup_metrics() -> pd.DataFrame:
    pred = pd.read_csv(BASE_RUN / "outer_hdi_predictions.csv", dtype={"adm1_code": str, "adm2_code": str})
    pred = pred.loc[pred.prediction_group.eq("C_dimension_specific")].copy()
    source = load_freeze()[[
        "adm2_code", "county_name_acs", "population_2020_ghsl", "control_urbanization_rate",
        "y_hdi", "state_fips" if "state_fips" in load_freeze().columns else "adm1_code",
    ]].copy().rename(columns={"y_hdi": "y_true_hdi"})
    source = source.drop(columns=[c for c in ["state_fips"] if c in source.columns])
    data = pred.merge(source, on=["adm2_code", "adm1_code", "y_true_hdi"], how="left", validate="one_to_one")
    data["HDI quintile"] = pd.qcut(data.y_true_hdi, 5, labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"])
    data["Population quartile"] = pd.qcut(data.population_2020_ghsl, 4, labels=["Q1 smallest", "Q2", "Q3", "Q4 largest"])
    data["Urbanization"] = pd.cut(data.control_urbanization_rate, [-np.inf, 0.25, 0.5, 0.75, np.inf], labels=["<25%", "25-50%", "50-75%", ">=75%"], right=False)
    rows = []
    for variable in ["HDI quintile", "Population quartile", "Urbanization"]:
        for level, part in data.groupby(variable, observed=True):
            rows.append({"stratifier": variable, "level": str(level), **metric_row(part, "y_true_hdi", "y_pred_hdi")})
    return pd.DataFrame(rows)


def c_result(tag: str, frames: dict, group_col: str = "adm1_code") -> dict:
    c = frames["outer_hdi_predictions"].loc[lambda x: x.prediction_group.eq("C_dimension_specific")]
    return {"experiment": tag, **metric_row(c, "y_true_hdi", "y_pred_hdi", group_col=group_col)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    freeze = load_freeze()
    main_matrix = freeze.loc[freeze.model_eligible.astype(bool)].sort_values("adm2_code").copy()
    required = core.TARGETS + sum(FULL_MODELS.values(), [])
    if len(main_matrix) != 3104 or not np.isfinite(main_matrix[required].to_numpy(float)).all():
        raise RuntimeError("Main frozen matrix failed verification")

    # Analysis of already locked predictions.
    comp = component_metrics()
    intervals, comparisons = bootstrap_main_predictions()
    subgroups = subgroup_metrics()
    comp.to_csv(REPORT / "usa_c_component_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    intervals.to_csv(REPORT / "usa_abc_cluster_bootstrap_intervals.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    comparisons.to_csv(REPORT / "usa_c_paired_cluster_bootstrap.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    subgroups.to_csv(REPORT / "usa_c_subgroup_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.12g")

    results = []
    # Same-structure C modality ablations.
    for tag, (models, order) in ABLATIONS.items():
        frames = run_nested(tag, main_matrix, state_splits(main_matrix), models, order)
        results.append(c_result(tag, frames))

    # Relax only the pre-fit population-coverage threshold from 0.99 to 0.95.
    relaxed = freeze.loc[
        freeze.label_complete.astype(bool)
        & freeze[required].notna().all(axis=1)
        & freeze.eo_population_coverage.ge(0.95)
        & freeze.health_population_coverage_after_coastfill.ge(0.95)
    ].sort_values("adm2_code").copy()
    relaxed_frames = run_nested("coverage_ge095", relaxed, state_splits(relaxed), FULL_MODELS, FULL_ORDER)
    results.append(c_result("coverage_ge095", relaxed_frames))

    # Strict regional extrapolation: hold out one of nine Census Divisions.
    regional_frames = run_nested("leave_one_census_division_out", main_matrix, division_splits(main_matrix), FULL_MODELS, FULL_ORDER)
    results.append(c_result("leave_one_census_division_out", regional_frames, group_col="outer_test_province"))

    baseline = pd.read_csv(BASE_RUN / "outer_hdi_predictions.csv", dtype={"adm1_code": str})
    baseline = baseline.loc[baseline.prediction_group.eq("C_dimension_specific")]
    results.insert(0, {"experiment": "main_leave_one_state_out", **metric_row(baseline, "y_true_hdi", "y_pred_hdi")})
    experiment_metrics = pd.DataFrame(results)
    experiment_metrics.to_csv(REPORT / "usa_c_validation_suite_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.12g")

    manifest = {
        "status": "PASS",
        "protocol": "usa_c_focused_validation_suite_2019_v1",
        "primary_formula": "component-specific Ridge selection for H/E/I; fixed equal-weight geometric mean",
        "main_cohort": len(main_matrix),
        "relaxed_coverage_cohort": len(relaxed),
        "bootstrap_repetitions": 20000,
        "ablation_libraries": {name: list(models) for name, (models, _) in ABLATIONS.items()},
        "census_divisions": DIVISIONS,
        "outputs": [
            "usa_c_component_metrics.csv",
            "usa_abc_cluster_bootstrap_intervals.csv",
            "usa_c_paired_cluster_bootstrap.csv",
            "usa_c_subgroup_metrics.csv",
            "usa_c_validation_suite_metrics.csv",
        ],
    }
    (REPORT / "usa_c_validation_suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(experiment_metrics.to_string(index=False))
    print(comp.to_string(index=False))
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
