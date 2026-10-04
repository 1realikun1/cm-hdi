"""Matched neural baselines for the Chinese Mainland HDI experiment.

The script fits a feature-token Transformer and a residual MLP to the same
74 regional predictors used by B_RS74. Both models predict the three official
HDI component indices; held-out HDI is the fixed geometric mean. Architecture
selection is confined to the five province-grouped inner folds of each outer
leave-one-province-out split. The selected architecture is refit on the full
outer training cohort with three deterministic seeds, and seed predictions
are averaged before scoring.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "D49_China_neural_input_matrix.csv"
SPLITS_JSON = HERE / "D50_China_neural_splits.json"
PREDICTIONS_CSV = HERE / "D51_China_neural_predictions.csv"
TUNING_CSV = HERE / "D52_China_neural_tuning.csv"
METRICS_CSV = HERE / "D53_China_neural_metrics.csv"
INFERENCE_CSV = HERE / "D54_China_neural_inference.csv"
AUDIT_JSON = HERE / "D55_China_neural_audit.json"
C_PREDICTIONS = HERE / "D01_ABC_predictions.csv"

BASE_SEED = 20260928
COMPONENTS = ["lifeindex", "eduindex", "incomeindex"]
THRESHOLDS = [3, 5, 8, 10]
BOOTSTRAP_REPLICATES = 20_000


@dataclass(frozen=True)
class Config:
    config_id: str
    family: str
    d_hidden: int
    n_layers: int
    n_heads: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int = 512
    max_epochs: int = 200
    patience: int = 20


CONFIGS = [
    Config("resmlp_64", "Residual MLP", 64, 2, 1, 0.10, 1.0e-3, 1.0e-4),
    Config("resmlp_128_reg", "Residual MLP", 128, 2, 1, 0.20, 5.0e-4, 1.0e-3),
    Config("ftt_16", "FT-Transformer", 16, 2, 4, 0.10, 1.0e-3, 1.0e-4,
           max_epochs=160, patience=15),
    Config("ftt_24_reg", "FT-Transformer", 24, 2, 4, 0.20, 5.0e-4, 1.0e-3,
           max_epochs=160, patience=15),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return BASE_SEED + int.from_bytes(digest[:4], "big") % 1_000_000_000


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResidualMLP(nn.Module):
    def __init__(self, n_features: int, cfg: Config) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(n_features, cfg.d_hidden), nn.GELU())
        self.blocks = nn.Sequential(
            *[ResidualBlock(cfg.d_hidden, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.head = nn.Sequential(nn.LayerNorm(cfg.d_hidden), nn.Linear(cfg.d_hidden, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


class NumericalTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.xavier_uniform_(self.weight)
        nn.init.normal_(self.bias, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTBlock(nn.Module):
    """Feature-token attention with the prediction token as the sole query.

    This is the efficient last-layer form of FT-Transformer attention: all
    numerical feature tokens remain keys and values, while only the prediction
    token is updated. It preserves data-dependent feature attention without the
    quadratic feature-by-feature activation tensor.
    """

    def __init__(self, d_token: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_kv = nn.LayerNorm(d_token)
        self.norm_q = nn.LayerNorm(d_token)
        self.attention = nn.MultiheadAttention(
            d_token, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(d_token)
        self.ff = nn.Sequential(
            nn.Linear(d_token, d_token * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token * 2, d_token),
            nn.Dropout(dropout),
        )

    def forward(self, cls: torch.Tensor, feature_tokens: torch.Tensor) -> torch.Tensor:
        kv = torch.cat([cls, feature_tokens], dim=1)
        update, _ = self.attention(
            self.norm_q(cls), self.norm_kv(kv), self.norm_kv(kv), need_weights=False
        )
        cls = cls + update
        return cls + self.ff(self.norm_ff(cls))


class FTTransformer(nn.Module):
    def __init__(self, n_features: int, cfg: Config) -> None:
        super().__init__()
        self.tokenizer = NumericalTokenizer(n_features, cfg.d_hidden)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_hidden))
        nn.init.normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([
            FTBlock(cfg.d_hidden, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.head = nn.Sequential(nn.LayerNorm(cfg.d_hidden), nn.Linear(cfg.d_hidden, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls = self.cls.expand(len(x), -1, -1)
        for block in self.blocks:
            cls = block(cls, tokens)
        return self.head(cls[:, 0])


def make_model(n_features: int, cfg: Config) -> nn.Module:
    if cfg.family == "Residual MLP":
        return ResidualMLP(n_features, cfg)
    if cfg.family == "FT-Transformer":
        return FTTransformer(n_features, cfg)
    raise ValueError(cfg.family)


def geometric_hdi(components: np.ndarray) -> np.ndarray:
    return np.prod(np.clip(components, 0.0, 1.0), axis=1) ** (1.0 / 3.0)


def province_equal_mae(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> float:
    return float(np.mean([
        np.mean(np.abs(pred[groups == group] - y[groups == group]))
        for group in np.unique(groups)
    ]))


def standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std <= 1e-12] = 1.0
    return tuple(((array - mean) / std).astype(np.float32) for array in (train,) + others)


def predict(model: nn.Module, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            result.append(model(torch.from_numpy(x[start:start + batch_size])).cpu().numpy())
    return np.concatenate(result)


def invert_target_scale(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip(raw * std + mean, 0.0, 1.0)


def fit_with_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid_components: np.ndarray,
    y_valid_hdi: np.ndarray,
    valid_groups: np.ndarray,
    cfg: Config,
    seed: int,
) -> tuple[nn.Module, int, float, np.ndarray]:
    seed_everything(seed)
    model = make_model(x_train.shape[1], cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    y_mean = y_train.mean(axis=0).astype(np.float32)
    y_std = y_train.std(axis=0).astype(np.float32)
    y_std[y_std <= 1e-12] = 1.0
    y_scaled = ((y_train - y_mean) / y_std).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_scaled)),
        batch_size=min(cfg.batch_size, len(x_train)), shuffle=True, generator=generator,
    )
    best_score = math.inf
    best_epoch = 0
    best_state = None
    best_pred = None
    stale = 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        if epoch == 1 or epoch % 5 == 0:
            comp_pred = invert_target_scale(predict(model, x_valid), y_mean, y_std)
            hdi_pred = geometric_hdi(comp_pred)
            score = province_equal_mae(y_valid_hdi, hdi_pred, valid_groups)
            if score < best_score - 1e-8:
                best_score = score
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                best_pred = comp_pred.copy()
                stale = 0
            else:
                stale += 5
            if epoch >= 25 and stale >= cfg.patience:
                break
    assert best_state is not None and best_pred is not None
    model.load_state_dict(best_state)
    return model, best_epoch, best_score, best_pred


def fit_fixed_epochs(
    x: np.ndarray, y: np.ndarray, cfg: Config, epochs: int, seed: int
) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    seed_everything(seed)
    model = make_model(x.shape[1], cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    y_mean = y.mean(axis=0).astype(np.float32)
    y_std = y.std(axis=0).astype(np.float32)
    y_std[y_std <= 1e-12] = 1.0
    y_scaled = ((y - y_mean) / y_std).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y_scaled)),
        batch_size=min(cfg.batch_size, len(x)), shuffle=True, generator=generator,
    )
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model, y_mean, y_std


def holm(pvalues: Iterable[float]) -> list[float]:
    values = np.asarray(list(pvalues), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(values) - rank) * values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def metric_row(model: str, y: np.ndarray, pred: np.ndarray) -> dict:
    error = pred - y
    abs_error = np.abs(error)
    rel = abs_error / y
    row = {
        "model": model,
        "n": len(y),
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(100 * rel.mean()),
        "r2": float(1 - np.sum(error**2) / np.sum((y - y.mean())**2)),
    }
    for threshold in THRESHOLDS:
        passed = rel <= threshold / 100
        row[f"p{threshold}_n"] = int(passed.sum())
        row[f"p{threshold}_pct"] = float(100 * passed.mean())
    return row


def paired_inference(predictions: pd.DataFrame) -> pd.DataFrame:
    wide = predictions.pivot(
        index=["city_code", "province_code"],
        columns="model", values=["observed_hdi", "predicted_hdi"]
    )
    reference_observed = wide[("observed_hdi", "Component-adaptive Ridge (C)")].to_numpy(float)
    for comparator in ("FT-Transformer", "Residual MLP"):
        assert np.allclose(
            reference_observed, wide[("observed_hdi", comparator)].to_numpy(float), atol=1e-12
        )
    records = []
    y = reference_observed
    c = wide[("predicted_hdi", "Component-adaptive Ridge (C)")].to_numpy(float)
    groups, inverse = np.unique(wide.index.get_level_values("province_code").astype(str), return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    for comparator in ("FT-Transformer", "Residual MLP"):
        other = wide[("predicted_hdi", comparator)].to_numpy(float)
        for metric in ("P3", "P5", "P8", "P10", "MAE", "MSE"):
            if metric.startswith("P"):
                tau = int(metric[1:]) / 100
                delta = 100 * (
                    (np.abs(c - y) / y <= tau).astype(float)
                    - (np.abs(other - y) / y <= tau).astype(float)
                )
                direction = "C_minus_neural_percentage_points"
            elif metric == "MAE":
                delta = np.abs(other - y) - np.abs(c - y)
                direction = "neural_error_minus_C_error"
            else:
                delta = (other - y) ** 2 - (c - y) ** 2
                direction = "neural_error_minus_C_error"
            group_sums = np.bincount(inverse, weights=delta)
            estimate = float(delta.mean())
            loo = (delta.sum() - group_sums) / (len(delta) - counts)
            label = f"{comparator}:{metric}"
            rng = np.random.default_rng(stable_seed(label + ":bootstrap"))
            indices = rng.integers(0, len(groups), size=(BOOTSTRAP_REPLICATES, len(groups)))
            boot = group_sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            rng = np.random.default_rng(stable_seed(label + ":signflip"))
            signs = rng.choice(np.array([-1.0, 1.0]), size=(BOOTSTRAP_REPLICATES, len(groups)))
            null = signs @ group_sums / len(delta)
            p = (1 + int(np.count_nonzero(np.abs(null) >= abs(estimate) - 1e-15))) / (
                BOOTSTRAP_REPLICATES + 1
            )
            records.append({
                "comparator": comparator,
                "metric": metric,
                "contrast_direction": direction,
                "estimate": estimate,
                "bootstrap_95_lo": float(lo),
                "bootstrap_95_hi": float(hi),
                "leave_one_province_out_min": float(loo.min()),
                "leave_one_province_out_max": float(loo.max()),
                "signflip_p_two_sided": p,
                "n_units": len(delta),
                "n_provinces": len(groups),
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BASE_SEED,
            })
    adjusted = holm(row["signflip_p_two_sided"] for row in records)
    for row, p_adjusted in zip(records, adjusted):
        row["holm_p_12_tests"] = p_adjusted
    return pd.DataFrame(records)


def main(
    outer_limit: int | None = None,
    outer_start: int = 0,
    shard: str | None = None,
    merge_shards: bool = False,
) -> None:
    torch.set_num_threads(1)
    data = pd.read_csv(INPUT_CSV, dtype={"city_code": str, "province_code": str})
    splits = json.loads(SPLITS_JSON.read_text(encoding="utf-8"))["splits"]
    feature_cols = [column for column in data if column.startswith("feature_")]
    assert len(data) == 280 and len(feature_cols) == 74 and len(splits) == 24
    assert data.city_code.is_unique
    index = data.set_index("city_code", drop=False)
    scored = data.loc[data.role.eq("nationwide_score")]
    assert len(scored) == 240 and scored.province_code.nunique() == 24
    config_by_family = {
        family: [cfg for cfg in CONFIGS if cfg.family == family]
        for family in ("Residual MLP", "FT-Transformer")
    }

    tuning_rows: list[dict] = []
    prediction_rows: list[dict] = []
    selected_records: list[dict] = []
    if merge_shards:
        prediction_files = sorted(HERE.glob("_neural_shard*_predictions.csv"))
        tuning_files = sorted(HERE.glob("_neural_shard*_tuning.csv"))
        selected_files = sorted(HERE.glob("_neural_shard*_selected.json"))
        assert len(prediction_files) == len(tuning_files) == len(selected_files) == 4
        predictions = pd.concat([pd.read_csv(path, dtype={"city_code": str, "province_code": str}) for path in prediction_files], ignore_index=True)
        tuning_rows = pd.concat([pd.read_csv(path) for path in tuning_files], ignore_index=True).to_dict("records")
        for path in selected_files:
            selected_records.extend(json.loads(path.read_text(encoding="utf-8")))
        active_splits = []
    else:
        stop = outer_start + outer_limit if outer_limit else None
        active_splits = splits[outer_start:stop]
    for outer_number, split in enumerate(active_splits, 1):
        outer_province = str(split["outer_test_province"])
        train_codes = list(map(str, split["outer_train_city_codes"]))
        test_codes = list(map(str, split["outer_test_city_codes"]))
        train_frame = index.loc[train_codes]
        test_frame = index.loc[test_codes]
        x_outer_train = train_frame[feature_cols].to_numpy(np.float32)
        x_outer_test = test_frame[feature_cols].to_numpy(np.float32)
        y_outer_train = train_frame[COMPONENTS].to_numpy(np.float32)

        for family, candidates in config_by_family.items():
            candidate_summaries = []
            for cfg in candidates:
                pooled = []
                fold_epochs = []
                for inner in split["inner_folds"]:
                    inner_train_codes = list(map(str, inner["train_city_codes"]))
                    valid_codes = list(map(str, inner["validation_city_codes"]))
                    inner_train = index.loc[inner_train_codes]
                    valid = index.loc[valid_codes]
                    x_train = inner_train[feature_cols].to_numpy(np.float32)
                    x_valid = valid[feature_cols].to_numpy(np.float32)
                    x_train, x_valid = standardize(x_train, x_valid)
                    seed = stable_seed(
                        f"select:{outer_province}:{family}:{cfg.config_id}:{inner['fold']}"
                    )
                    _, best_epoch, score, component_pred = fit_with_validation(
                        x_train,
                        inner_train[COMPONENTS].to_numpy(np.float32),
                        x_valid,
                        valid[COMPONENTS].to_numpy(np.float32),
                        valid.hdi.to_numpy(float),
                        valid.province_code.astype(str).to_numpy(),
                        cfg,
                        seed,
                    )
                    hdi_pred = geometric_hdi(component_pred)
                    fold_epochs.append(best_epoch)
                    pooled.append(pd.DataFrame({
                        "city_code": valid.city_code.to_numpy(),
                        "province_code": valid.province_code.to_numpy(),
                        "observed": valid.hdi.to_numpy(float),
                        "predicted": hdi_pred,
                    }))
                    tuning_rows.append({
                        "outer_test_province": outer_province,
                        "family": family,
                        "config_id": cfg.config_id,
                        "record_type": "inner_fold",
                        "inner_fold": inner["fold"],
                        "validation_n": len(valid),
                        "province_equal_mae": score,
                        "best_epoch": best_epoch,
                        "selected": False,
                    })
                pooled_frame = pd.concat(pooled, ignore_index=True)
                assert len(pooled_frame) == len(train_frame)
                pooled_score = province_equal_mae(
                    pooled_frame.observed.to_numpy(float),
                    pooled_frame.predicted.to_numpy(float),
                    pooled_frame.province_code.astype(str).to_numpy(),
                )
                final_epochs = int(np.median(fold_epochs))
                candidate_summaries.append((pooled_score, cfg.config_id, cfg, final_epochs))
                tuning_rows.append({
                    "outer_test_province": outer_province,
                    "family": family,
                    "config_id": cfg.config_id,
                    "record_type": "pooled_inner",
                    "inner_fold": "all",
                    "validation_n": len(pooled_frame),
                    "province_equal_mae": pooled_score,
                    "best_epoch": final_epochs,
                    "selected": False,
                })
            pooled_score, _, selected_cfg, final_epochs = min(
                candidate_summaries, key=lambda item: (item[0], item[1])
            )
            for row in tuning_rows:
                if (
                    row["outer_test_province"] == outer_province
                    and row["family"] == family
                    and row["config_id"] == selected_cfg.config_id
                    and row["record_type"] == "pooled_inner"
                ):
                    row["selected"] = True
            x_outer_train_std, x_outer_test_std = standardize(x_outer_train, x_outer_test)
            seed_predictions = []
            final_seeds = []
            for repeat in range(3):
                seed = stable_seed(
                    f"final:{outer_province}:{family}:{selected_cfg.config_id}:{repeat}"
                )
                final_seeds.append(seed)
                model, target_mean, target_std = fit_fixed_epochs(
                    x_outer_train_std, y_outer_train, selected_cfg, final_epochs, seed
                )
                seed_predictions.append(invert_target_scale(
                    predict(model, x_outer_test_std), target_mean, target_std
                ))
            component_pred = np.mean(seed_predictions, axis=0)
            hdi_pred = geometric_hdi(component_pred)
            selected_records.append({
                "outer_test_province": outer_province,
                "family": family,
                "selected_config": selected_cfg.config_id,
                "pooled_inner_province_equal_mae": pooled_score,
                "final_epochs": final_epochs,
                "final_seeds": final_seeds,
            })
            for position, (_, row) in enumerate(test_frame.iterrows()):
                prediction_rows.append({
                    "city_code": row.city_code,
                    "province_code": row.province_code,
                    "model": family,
                    "observed_hdi": float(row.hdi),
                    "predicted_hdi": float(hdi_pred[position]),
                    "predicted_health": float(component_pred[position, 0]),
                    "predicted_education": float(component_pred[position, 1]),
                    "predicted_income": float(component_pred[position, 2]),
                    "absolute_error": float(abs(hdi_pred[position] - row.hdi)),
                    "absolute_relative_error_pct": float(100 * abs(hdi_pred[position] - row.hdi) / row.hdi),
                    "selected_config": selected_cfg.config_id,
                    "final_epochs": final_epochs,
                })
        print(f"Completed outer fold {outer_number}/{len(active_splits)}: {outer_province}", flush=True)

    if not merge_shards:
        predictions = pd.DataFrame(prediction_rows)
    if shard:
        predictions.to_csv(HERE / f"_neural_{shard}_predictions.csv", index=False)
        pd.DataFrame(tuning_rows).to_csv(HERE / f"_neural_{shard}_tuning.csv", index=False)
        (HERE / f"_neural_{shard}_selected.json").write_text(
            json.dumps(selected_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Shard {shard} wrote {len(predictions)} predictions", flush=True)
        return
    if outer_limit and not merge_shards:
        preview = HERE / "_neural_preview_predictions.csv"
        predictions.to_csv(preview, index=False)
        print(f"Preview wrote {preview}")
        return

    assert len(predictions) == 480 and predictions.groupby("model").size().eq(240).all()
    c_rows = pd.read_csv(C_PREDICTIONS, dtype={"unit": str, "province": str})
    c_rows = c_rows.loc[(c_rows.country == "China") & (c_rows.method == "C")]
    assert len(c_rows) == 240
    c_predictions = pd.DataFrame({
        "city_code": c_rows.unit,
        "province_code": c_rows.province,
        "model": "Component-adaptive Ridge (C)",
        "observed_hdi": c_rows.observed,
        "predicted_hdi": c_rows.predicted,
        "predicted_health": np.nan,
        "predicted_education": np.nan,
        "predicted_income": np.nan,
        "absolute_error": np.abs(c_rows.predicted - c_rows.observed),
        "absolute_relative_error_pct": 100 * np.abs(c_rows.predicted - c_rows.observed) / c_rows.observed,
        "selected_config": "component-specific nested Ridge",
        "final_epochs": np.nan,
    })
    predictions = pd.concat([c_predictions, predictions], ignore_index=True)
    predictions = predictions.sort_values(["model", "province_code", "city_code"])
    predictions.to_csv(PREDICTIONS_CSV, index=False)
    pd.DataFrame(tuning_rows).to_csv(TUNING_CSV, index=False)

    metric_rows = []
    for model, group in predictions.groupby("model", sort=False):
        metric_rows.append(metric_row(
            model, group.observed_hdi.to_numpy(float), group.predicted_hdi.to_numpy(float)
        ))
    metrics = pd.DataFrame(metric_rows)
    column_order = [
        "model", "n", "p3_n", "p3_pct", "p5_n", "p5_pct", "p8_n", "p8_pct",
        "p10_n", "p10_pct", "mae", "rmse", "mape_pct", "r2",
    ]
    metrics[column_order].to_csv(METRICS_CSV, index=False)
    inference = paired_inference(predictions)
    inference.to_csv(INFERENCE_CSV, index=False)

    audit = {
        "status": "COMPLETED_MATCHED_NEURAL_BASELINES",
        "created_with_seed": BASE_SEED,
        "scope": {
            "country": "Chinese Mainland",
            "cohort": 280,
            "outer_scored_cities": 240,
            "development_train_only_cities": 40,
            "outer_folds": 24,
            "inner_folds_per_outer": 5,
        },
        "estimand": "downstream fusion comparison on identical 74 regional numerical predictors",
        "targets": COMPONENTS,
        "synthesis": "clip component predictions to [0,1], then fixed geometric mean",
        "selection_metric": "province-equal HDI MAE from pooled inner out-of-fold predictions",
        "target_preprocessing": "component-wise mean and standard deviation fitted on each training partition; predictions inverse-transformed and clipped to [0,1]",
        "final_training": "median inner early-stopping epoch; three deterministic seeds averaged",
        "architectures": [asdict(cfg) for cfg in CONFIGS],
        "selected_outer_models": selected_records,
        "inference": {
            "unit": "province cluster",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "signflip_replicates": BOOTSTRAP_REPLICATES,
            "holm_family": "2 neural comparators x 6 metrics = 12 tests",
            "conditioning": "inference conditions on completed out-of-fold predictions",
        },
        "exclusions": {
            "GeoViSTA": "requires image-patch tensors unavailable in the regional-vector benchmark",
            "GNN": "requires a prespecified graph and a graph-safe geographic holdout estimand",
        },
        "software": {
            "python": __import__("sys").version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "sha256_inputs": {
            INPUT_CSV.name: sha256(INPUT_CSV),
            SPLITS_JSON.name: sha256(SPLITS_JSON),
            C_PREDICTIONS.name: sha256(C_PREDICTIONS),
        },
        "sha256_outputs": {
            PREDICTIONS_CSV.name: sha256(PREDICTIONS_CSV),
            TUNING_CSV.name: sha256(TUNING_CSV),
            METRICS_CSV.name: sha256(METRICS_CSV),
            INFERENCE_CSV.name: sha256(INFERENCE_CSV),
        },
    }
    AUDIT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics[column_order].to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(inference.to_string(index=False, float_format=lambda value: f"{value:.6g}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-limit", type=int, default=None)
    parser.add_argument("--outer-start", type=int, default=0)
    parser.add_argument("--shard", type=str, default=None)
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()
    main(args.outer_limit, args.outer_start, args.shard, args.merge_shards)
