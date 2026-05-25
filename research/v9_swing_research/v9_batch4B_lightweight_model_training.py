#!/usr/bin/env python3
"""Batch 4B: train frozen lightweight v9 swing models.

This script strictly follows Batch 4A:
- primary horizon: 5d
- primary label: fwd_ret_5d_open
- primary features: frozen features only
- no parameter search, no horizon switching, no concept/theme fields
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT / "data_tushare" / "clean" / "v9"
REPORT_ROOT = ROOT / "reports" / "tushare" / "v9_swing_research"
REPORT_DIR = REPORT_ROOT / "batch4B_lightweight_model_training"
MODEL_DIR = REPORT_DIR / "models"
LOCAL_LARGE_DIR = REPORT_DIR / "_local_large"

B4A_DIR = REPORT_ROOT / "batch4A_model_training_plan"
B3D_DIR = REPORT_ROOT / "batch3D_simple_rule_baseline"

FEATURE_PATH = CLEAN_DIR / "v9_stock_industry_features.parquet"
LABEL_PATH = CLEAN_DIR / "v9_swing_labels_3_5_10.parquet"

GLOBAL_HANDOFF_PATH = REPORT_ROOT / "v9_current_handoff.md"
ROADMAP_PATH = REPORT_ROOT / "v9_execution_roadmap.md"
STAGE_STATUS_PATH = REPORT_ROOT / "v9_stage_status.csv"

PRIMARY_LABEL = "fwd_ret_5d_open"
PRIMARY_BINARY_LABEL = "label_win_5d"
TOP_N = 20
RANDOM_SEED = 42

TRAIN_START = "20180102"
TRAIN_END = "20221230"
VALIDATION_START = "20230103"
VALIDATION_END = "20241231"
HOLDOUT_START = "20250102"
HOLDOUT_END = "20260522"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_family: str
    objective: str
    features: list[str]
    training_role: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def max_drawdown(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return np.nan
    equity = (1.0 + daily_returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def profit_factor(returns: pd.Series) -> float:
    pos = returns[returns > 0].sum()
    neg = returns[returns < 0].sum()
    if neg == 0:
        return np.inf if pos > 0 else np.nan
    return float(pos / abs(neg))


def summarize_daily_returns(daily: pd.DataFrame, meta: dict[str, object]) -> dict[str, object]:
    if daily.empty:
        return {
            **meta,
            "signal_days": 0,
            "total_trades": 0,
            "avg_selected_count": np.nan,
            "cumulative_return": np.nan,
            "mean_daily_return": np.nan,
            "median_daily_return": np.nan,
            "daily_win_rate": np.nan,
            "trade_win_rate": np.nan,
            "profit_factor": np.nan,
            "trade_profit_factor": np.nan,
            "max_drawdown": np.nan,
            "avg_trade_return": np.nan,
            "max_daily_return": np.nan,
            "min_daily_return": np.nan,
        }
    if "trade_date" in daily.columns:
        daily = daily.sort_values("trade_date").copy()
    r = finite_numeric(daily["daily_return"]).dropna()
    trade_sum = finite_numeric(daily["trade_return_sum"]).fillna(0.0)
    pos_trade_sum = finite_numeric(daily["positive_trade_return_sum"]).fillna(0.0).sum()
    neg_trade_sum = finite_numeric(daily["negative_trade_return_sum"]).fillna(0.0).sum()
    total_trades = int(daily["selected_count"].sum())
    return {
        **meta,
        "signal_days": int(len(daily)),
        "total_trades": total_trades,
        "avg_selected_count": float(daily["selected_count"].mean()),
        "cumulative_return": float((1.0 + r).prod() - 1.0) if len(r) else np.nan,
        "mean_daily_return": float(r.mean()) if len(r) else np.nan,
        "median_daily_return": float(r.median()) if len(r) else np.nan,
        "daily_win_rate": float(r.gt(0).mean()) if len(r) else np.nan,
        "trade_win_rate": float(daily["win_count"].sum() / max(total_trades, 1)),
        "profit_factor": profit_factor(r),
        "trade_profit_factor": float(pos_trade_sum / abs(neg_trade_sum)) if neg_trade_sum < 0 else (np.inf if pos_trade_sum > 0 else np.nan),
        "max_drawdown": max_drawdown(r),
        "avg_trade_return": float(trade_sum.sum() / max(total_trades, 1)),
        "max_daily_return": float(r.max()) if len(r) else np.nan,
        "min_daily_return": float(r.min()) if len(r) else np.nan,
    }


def split_name(trade_date: pd.Series) -> pd.Series:
    d = trade_date.astype(str)
    out = pd.Series("outside", index=trade_date.index, dtype="object")
    out[d.between(TRAIN_START, TRAIN_END)] = "train"
    out[d.between(VALIDATION_START, VALIDATION_END)] = "validation"
    out[d.between(HOLDOUT_START, HOLDOUT_END)] = "research_holdout"
    return out


def read_frozen_features() -> tuple[list[str], list[str]]:
    ff = pd.read_csv(B4A_DIR / "batch4A_feature_freeze.csv")
    primary = ff.loc[ff["training_status"].eq("primary_training_feature"), "factor"].tolist()
    conditional = ff.loc[ff["training_status"].eq("conditional_ablation_only"), "factor"].tolist()
    return primary, conditional


def read_dataset(features_all: list[str]) -> pd.DataFrame:
    feature_cols = [
        "trade_date",
        "ts_code",
        "industry",
        "trend_regime",
        "vol_regime",
        "industry_crowding_regime",
        "stock_concentration_regime",
        "market_regime_id",
        "listed_days",
        "st_flag",
        "suspend_flag",
        "limit_up_close_flag",
        "limit_down_close_flag",
    ] + features_all
    label_cols = ["trade_date", "ts_code", PRIMARY_LABEL, PRIMARY_BINARY_LABEL]
    print({"event": "read_features", "columns": len(feature_cols)}, flush=True)
    features = pd.read_parquet(FEATURE_PATH, columns=feature_cols)
    print({"event": "read_labels", "columns": len(label_cols)}, flush=True)
    labels = pd.read_parquet(LABEL_PATH, columns=label_cols)
    features["trade_date"] = features["trade_date"].astype(str)
    labels["trade_date"] = labels["trade_date"].astype(str)
    print({"event": "merge", "feature_rows": int(len(features)), "label_rows": int(len(labels))}, flush=True)
    df = features.merge(labels, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
    df["industry"] = df["industry"].fillna("UNKNOWN").astype(str)
    for flag in ["st_flag", "suspend_flag", "limit_up_close_flag", "limit_down_close_flag"]:
        df[flag] = df[flag].fillna(False).astype(bool)
    df["listed_days"] = finite_numeric(df["listed_days"])
    for col in features_all + [PRIMARY_LABEL]:
        df[col] = finite_numeric(df[col])
    df[PRIMARY_BINARY_LABEL] = df[PRIMARY_BINARY_LABEL].fillna(df[PRIMARY_LABEL].gt(0)).astype(bool)
    df["screened_universe_flag"] = (
        df["listed_days"].ge(120)
        & ~df["st_flag"]
        & ~df["suspend_flag"]
        & ~df["limit_up_close_flag"]
        & ~df["limit_down_close_flag"]
        & df[PRIMARY_LABEL].notna()
    )
    df["split"] = split_name(df["trade_date"])
    df = df[df["screened_universe_flag"] & df["split"].ne("outside")].copy()
    df["year"] = df["trade_date"].str[:4]
    return df


def add_rank_features(df: pd.DataFrame, features_all: list[str]) -> list[str]:
    rank_cols = []
    for idx, feature in enumerate(features_all, start=1):
        print({"event": "rank_feature", "feature": feature, "done": idx - 1, "total": len(features_all)}, flush=True)
        rank_col = f"rank_{feature}"
        rank = df.groupby("trade_date", sort=False)[feature].rank(pct=True, method="average")
        df[rank_col] = (rank - 0.5).fillna(0.0).astype("float32")
        rank_cols.append(rank_col)
    df["size_bucket"] = make_bucket_from_rank(df["rank_log_total_mv"], "size_q") if "rank_log_total_mv" in df else "missing"
    df["liquidity_bucket"] = make_bucket_from_rank(df["rank_log_amount"], "liq_q") if "rank_log_amount" in df else "missing"
    return rank_cols


def make_bucket_from_rank(centered_rank: pd.Series, prefix: str) -> pd.Series:
    pct = centered_rank.astype(float) + 0.5
    q = np.ceil(pct.clip(0, 1) * 5).clip(1, 5).astype("int8")
    return pd.Series([f"{prefix}{x}" for x in q], index=centered_rank.index, dtype="object")


def build_model_specs(primary_features: list[str], conditional_features: list[str]) -> list[ModelSpec]:
    primary_rank = [f"rank_{f}" for f in primary_features]
    specs = [
        ModelSpec("ridge_primary_alpha1", "ridge", "regression_fwd_ret_5d_open", primary_rank, "primary_candidate"),
        ModelSpec("logistic_primary_sgd_l2", "logistic_sgd", "classification_label_win_5d", primary_rank, "secondary_diagnostic"),
    ]
    if conditional_features:
        specs.append(
            ModelSpec(
                "ridge_conditional_industry_ablation",
                "ridge",
                "regression_fwd_ret_5d_open",
                primary_rank + [f"rank_{f}" for f in conditional_features],
                "conditional_ablation_only",
            )
        )
    return specs


def train_models(df: pd.DataFrame, specs: list[ModelSpec]) -> dict[str, object]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_mask = df["split"].eq("train")
    models: dict[str, object] = {}
    for spec in specs:
        print({"event": "train_model", "model_id": spec.model_id, "features": spec.features}, flush=True)
        x_train = df.loc[train_mask, spec.features].to_numpy(dtype=np.float32)
        if spec.model_family == "ridge":
            y_train = df.loc[train_mask, PRIMARY_LABEL].to_numpy(dtype=np.float32)
            model = Ridge(alpha=1.0, fit_intercept=True, random_state=RANDOM_SEED)
            model.fit(x_train, y_train)
        elif spec.model_family == "logistic_sgd":
            y_train = df.loc[train_mask, PRIMARY_BINARY_LABEL].astype(int).to_numpy()
            model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                max_iter=20,
                tol=1e-3,
                random_state=RANDOM_SEED,
                average=True,
            )
            model.fit(x_train, y_train)
        else:
            raise ValueError(f"unsupported model_family={spec.model_family}")
        models[spec.model_id] = model
        with (MODEL_DIR / f"{spec.model_id}.pkl").open("wb") as f:
            pickle.dump({"spec": spec, "model": model}, f)
    return models


def score_models(df: pd.DataFrame, specs: list[ModelSpec], models: dict[str, object]) -> list[str]:
    score_cols = []
    for spec in specs:
        score_col = f"score_{spec.model_id}"
        print({"event": "score_model", "model_id": spec.model_id}, flush=True)
        x = df[spec.features].to_numpy(dtype=np.float32)
        model = models[spec.model_id]
        if spec.model_family == "logistic_sgd":
            score = model.decision_function(x)
        else:
            score = model.predict(x)
        df[score_col] = score.astype("float32")
        score_cols.append(score_col)
    return score_cols


def selected_topn(df: pd.DataFrame, specs: list[ModelSpec]) -> pd.DataFrame:
    frames = []
    keep_cols = [
        "trade_date",
        "ts_code",
        "industry",
        "split",
        "year",
        "trend_regime",
        "vol_regime",
        "industry_crowding_regime",
        "stock_concentration_regime",
        "market_regime_id",
        "size_bucket",
        "liquidity_bucket",
        PRIMARY_LABEL,
        PRIMARY_BINARY_LABEL,
    ]
    for spec in specs:
        score_col = f"score_{spec.model_id}"
        print({"event": "select_topn", "model_id": spec.model_id, "top_n": TOP_N}, flush=True)
        sub = df[keep_cols + [score_col]].dropna(subset=[score_col, PRIMARY_LABEL]).copy()
        sub = sub.sort_values(["trade_date", score_col], ascending=[True, False], kind="mergesort")
        top = sub.groupby("trade_date", sort=False).head(TOP_N).copy()
        top["model_id"] = spec.model_id
        top["model_family"] = spec.model_family
        top["training_role"] = spec.training_role
        top["rank"] = top.groupby(["model_id", "trade_date"], sort=False)[score_col].rank(ascending=False, method="first").astype("int16")
        top = top.rename(columns={score_col: "score"})
        frames.append(top)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def daily_from_selected(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    g = selected.groupby(["model_id", "split", "trade_date"], sort=False)
    daily = g.agg(
        selected_count=(PRIMARY_LABEL, "size"),
        daily_return=(PRIMARY_LABEL, "mean"),
        trade_return_sum=(PRIMARY_LABEL, "sum"),
        win_count=(PRIMARY_LABEL, lambda x: int((x > 0).sum())),
        loss_count=(PRIMARY_LABEL, lambda x: int((x <= 0).sum())),
        positive_trade_return_sum=(PRIMARY_LABEL, lambda x: float(x[x > 0].sum())),
        negative_trade_return_sum=(PRIMARY_LABEL, lambda x: float(x[x < 0].sum())),
        avg_score=("score", "mean"),
        unique_industries=("industry", "nunique"),
    ).reset_index()
    return daily


def rank_ic_metrics(df: pd.DataFrame, specs: list[ModelSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        score_col = f"score_{spec.model_id}"
        for split, sub in df.groupby("split", sort=False):
            print({"event": "rank_ic", "model_id": spec.model_id, "split": split}, flush=True)
            ic_rows = []
            for trade_date, g in sub.groupby("trade_date", sort=False):
                if len(g) < 30:
                    continue
                corr = g[score_col].rank(pct=True).corr(g[PRIMARY_LABEL].rank(pct=True))
                if pd.notna(corr):
                    ic_rows.append(corr)
            ic = pd.Series(ic_rows, dtype="float64")
            y_true = sub[PRIMARY_BINARY_LABEL].astype(int)
            score = sub[score_col]
            try:
                auc = float(roc_auc_score(y_true, score))
            except ValueError:
                auc = np.nan
            rows.append(
                {
                    "model_id": spec.model_id,
                    "split": split,
                    "auc": auc,
                    "rank_ic_days": int(ic.size),
                    "mean_rank_ic": float(ic.mean()) if ic.size else np.nan,
                    "median_rank_ic": float(ic.median()) if ic.size else np.nan,
                    "rank_ic_std": float(ic.std(ddof=1)) if ic.size > 1 else np.nan,
                    "rank_icir": float(ic.mean() / ic.std(ddof=1) * np.sqrt(252)) if ic.size > 1 and ic.std(ddof=1) != 0 else np.nan,
                    "positive_rank_ic_rate": float(ic.gt(0).mean()) if ic.size else np.nan,
                    "scored_rows": int(len(sub)),
                }
            )
    return pd.DataFrame(rows)


def validation_metrics(df: pd.DataFrame, daily: pd.DataFrame, specs: list[ModelSpec]) -> pd.DataFrame:
    rows = []
    rankic = rank_ic_metrics(df, specs)
    for spec in specs:
        for split, d in daily[daily["model_id"].eq(spec.model_id)].groupby("split", sort=False):
            row = summarize_daily_returns(d, {"model_id": spec.model_id, "split": split, "top_n": TOP_N})
            rows.append(row)
    out = pd.DataFrame(rows)
    out = out.merge(rankic, on=["model_id", "split"], how="left")
    out.to_csv(REPORT_DIR / "batch4B_validation_metrics.csv", index=False)
    return out


def summarize_selected_by_daily_group(selected: pd.DataFrame, group_col: str, out_name: str) -> pd.DataFrame:
    g = selected.groupby(["model_id", "split", group_col, "trade_date"], dropna=False, sort=False)
    daily = g.agg(
        selected_count=(PRIMARY_LABEL, "size"),
        daily_return=(PRIMARY_LABEL, "mean"),
        trade_return_sum=(PRIMARY_LABEL, "sum"),
        win_count=(PRIMARY_LABEL, lambda x: int((x > 0).sum())),
        loss_count=(PRIMARY_LABEL, lambda x: int((x <= 0).sum())),
        positive_trade_return_sum=(PRIMARY_LABEL, lambda x: float(x[x > 0].sum())),
        negative_trade_return_sum=(PRIMARY_LABEL, lambda x: float(x[x < 0].sum())),
    ).reset_index()
    rows = []
    for keys, d in daily.groupby(["model_id", "split", group_col], dropna=False, sort=False):
        model_id, split, value = keys
        rows.append(summarize_daily_returns(d, {"model_id": model_id, "split": split, "group_type": group_col, "group_value": value}))
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / out_name, index=False)
    return out


def summarize_by_regime(selected: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for col in ["trend_regime", "vol_regime", "industry_crowding_regime", "stock_concentration_regime"]:
        frames.append(summarize_selected_by_daily_group(selected, col, f"_tmp_{col}.csv"))
        tmp = REPORT_DIR / f"_tmp_{col}.csv"
        if tmp.exists():
            tmp.unlink()
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(REPORT_DIR / "batch4B_by_regime_metrics.csv", index=False)
    return out


def summarize_trade_group(selected: pd.DataFrame, group_col: str, out_name: str) -> pd.DataFrame:
    g = selected.groupby(["model_id", "split", group_col], dropna=False, sort=False)
    out = g.agg(
        trade_count=(PRIMARY_LABEL, "size"),
        avg_trade_return=(PRIMARY_LABEL, "mean"),
        median_trade_return=(PRIMARY_LABEL, "median"),
        trade_win_rate=(PRIMARY_LABEL, lambda x: float((x > 0).mean())),
        positive_trade_return_sum=(PRIMARY_LABEL, lambda x: float(x[x > 0].sum())),
        negative_trade_return_sum=(PRIMARY_LABEL, lambda x: float(x[x < 0].sum())),
        avg_score=("score", "mean"),
    ).reset_index()
    out["trade_profit_factor"] = out.apply(
        lambda r: r["positive_trade_return_sum"] / abs(r["negative_trade_return_sum"]) if r["negative_trade_return_sum"] < 0 else np.nan,
        axis=1,
    )
    out.to_csv(REPORT_DIR / out_name, index=False)
    return out


def cost_sensitivity(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cost_bps in [0, 20, 40, 60, 100]:
        adjusted = daily.copy()
        adjusted["daily_return"] = adjusted["daily_return"] - cost_bps / 10000.0
        for keys, d in adjusted.groupby(["model_id", "split"], sort=False):
            model_id, split = keys
            rows.append(summarize_daily_returns(d, {"model_id": model_id, "split": split, "roundtrip_cost_bps": cost_bps}))
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "batch4B_cost_sensitivity.csv", index=False)
    return out


def remove_winner_concentration(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, d in daily.groupby(["model_id", "split"], sort=False):
        model_id, split = keys
        for remove_n in [0, 1, 3, 5, 10]:
            dd = d.sort_values("daily_return", ascending=False).iloc[remove_n:].copy()
            rows.append(summarize_daily_returns(dd, {"model_id": model_id, "split": split, "remove_top_winning_days": remove_n}))
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "batch4B_profit_concentration.csv", index=False)
    return out


def feature_importance(specs: list[ModelSpec], models: dict[str, object]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        model = models[spec.model_id]
        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_).reshape(-1)
        else:
            coef = np.full(len(spec.features), np.nan)
        intercept = float(np.asarray(getattr(model, "intercept_", [np.nan])).reshape(-1)[0])
        for feature, value in zip(spec.features, coef):
            rows.append(
                {
                    "model_id": spec.model_id,
                    "model_family": spec.model_family,
                    "training_role": spec.training_role,
                    "feature": feature.replace("rank_", ""),
                    "rank_feature": feature,
                    "coefficient": float(value),
                    "abs_coefficient": float(abs(value)),
                    "intercept": intercept,
                }
            )
    out = pd.DataFrame(rows).sort_values(["model_id", "abs_coefficient"], ascending=[True, False])
    out.to_csv(REPORT_DIR / "batch4B_feature_importance.csv", index=False)
    return out


def model_registry(specs: list[ModelSpec], metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in specs:
        val = metrics[(metrics["model_id"].eq(spec.model_id)) & metrics["split"].eq("validation")]
        hold = metrics[(metrics["model_id"].eq(spec.model_id)) & metrics["split"].eq("research_holdout")]
        rows.append(
            {
                "model_id": spec.model_id,
                "model_family": spec.model_family,
                "objective": spec.objective,
                "training_role": spec.training_role,
                "features": ", ".join(f.replace("rank_", "") for f in spec.features),
                "feature_count": len(spec.features),
                "model_file": str(MODEL_DIR / f"{spec.model_id}.pkl"),
                "validation_mean_daily_return": float(val["mean_daily_return"].iloc[0]) if not val.empty else np.nan,
                "validation_profit_factor": float(val["profit_factor"].iloc[0]) if not val.empty else np.nan,
                "validation_max_drawdown": float(val["max_drawdown"].iloc[0]) if not val.empty else np.nan,
                "validation_auc": float(val["auc"].iloc[0]) if not val.empty else np.nan,
                "validation_rank_ic": float(val["mean_rank_ic"].iloc[0]) if not val.empty else np.nan,
                "holdout_mean_daily_return": float(hold["mean_daily_return"].iloc[0]) if not hold.empty else np.nan,
                "holdout_profit_factor": float(hold["profit_factor"].iloc[0]) if not hold.empty else np.nan,
                "holdout_max_drawdown": float(hold["max_drawdown"].iloc[0]) if not hold.empty else np.nan,
                "holdout_auc": float(hold["auc"].iloc[0]) if not hold.empty else np.nan,
                "holdout_rank_ic": float(hold["mean_rank_ic"].iloc[0]) if not hold.empty else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "batch4B_model_registry.csv", index=False)
    return out


def baseline_metrics_by_split() -> pd.DataFrame:
    daily_path = B3D_DIR / "batch3D_rule_daily_returns.csv"
    ref = pd.read_csv(B4A_DIR / "batch4A_baseline_reference.csv")
    min_rules = set(ref.loc[ref["baseline_role"].eq("minimum_to_beat"), "rule_id"].astype(str))
    daily = pd.read_csv(daily_path, usecols=["rule_id", "horizon", "trade_date", "selected_count", "daily_return", "trade_return_sum", "win_count", "loss_count", "positive_trade_return_sum", "negative_trade_return_sum"])
    daily = daily[daily["horizon"].eq("5d") & daily["rule_id"].isin(min_rules)].copy()
    daily["trade_date"] = daily["trade_date"].astype(str)
    daily["split"] = split_name(daily["trade_date"])
    daily = daily[daily["split"].ne("outside")]
    rows = []
    for keys, d in daily.groupby(["rule_id", "split"], sort=False):
        rule_id, split = keys
        rows.append(summarize_daily_returns(d, {"baseline_id": rule_id, "split": split, "baseline_type": "batch3D_simple_rule_5d"}))
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "batch4B_simple_baseline_by_split.csv", index=False)
    return out


def compare_vs_baseline(metrics: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    best = (
        baseline.groupby("split", as_index=False)
        .agg(
            best_baseline_pf=("profit_factor", "max"),
            best_baseline_mean_daily_return=("mean_daily_return", "max"),
            least_bad_baseline_drawdown=("max_drawdown", "max"),
        )
    )
    comp = metrics.merge(best, on="split", how="left")
    comp["beats_best_baseline_pf"] = comp["profit_factor"] > comp["best_baseline_pf"]
    comp["beats_best_baseline_mean_daily_return"] = comp["mean_daily_return"] > comp["best_baseline_mean_daily_return"]
    comp["drawdown_not_worse_than_least_bad_baseline"] = comp["max_drawdown"] >= comp["least_bad_baseline_drawdown"]
    comp.to_csv(REPORT_DIR / "batch4B_vs_simple_baseline.csv", index=False)
    return comp


def write_train_config(specs: list[ModelSpec], primary_features: list[str], conditional_features: list[str]) -> None:
    lines = [
        "# Batch 4B train config",
        "",
        "primary_horizon: 5d",
        f"primary_label: {PRIMARY_LABEL}",
        f"binary_label: {PRIMARY_BINARY_LABEL}",
        f"top_n: {TOP_N}",
        f"random_seed: {RANDOM_SEED}",
        "feature_transform: daily_cross_sectional_rank_pct_then_center",
        "missing_value_policy: centered_rank_zero",
        "splits:",
        f"  train: [{TRAIN_START}, {TRAIN_END}]",
        f"  validation: [{VALIDATION_START}, {VALIDATION_END}]",
        f"  research_holdout: [{HOLDOUT_START}, {HOLDOUT_END}]",
        "primary_features:",
    ]
    lines.extend([f"  - {f}" for f in primary_features])
    lines.append("conditional_ablation_features:")
    lines.extend([f"  - {f}" for f in conditional_features] or ["  []"])
    lines.append("models:")
    for spec in specs:
        lines.append(f"  - model_id: {spec.model_id}")
        lines.append(f"    family: {spec.model_family}")
        lines.append(f"    objective: {spec.objective}")
        lines.append(f"    role: {spec.training_role}")
        lines.append(f"    features: [{', '.join(f.replace('rank_', '') for f in spec.features)}]")
    lines.append("blocked:")
    lines.append("  - no hyperparameter search")
    lines.append("  - no horizon switching")
    lines.append("  - no non-frozen features")
    lines.append("  - no concept/theme fields")
    (REPORT_DIR / "batch4B_train_config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_predictions(df: pd.DataFrame, specs: list[ModelSpec]) -> Path:
    LOCAL_LARGE_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["trade_date", "ts_code", "split", PRIMARY_LABEL, PRIMARY_BINARY_LABEL] + [f"score_{s.model_id}" for s in specs]
    out = LOCAL_LARGE_DIR / "batch4B_prediction_scores.parquet"
    print({"event": "write_prediction_scores", "file": str(out), "rows": int(len(df))}, flush=True)
    df[cols].to_parquet(out, index=False, compression="zstd")
    return out


def write_conclusion(registry: pd.DataFrame, comp: pd.DataFrame) -> str:
    val = registry[registry["training_role"].eq("primary_candidate")].head(1)
    val_pass = False
    if not val.empty:
        comp_val = comp[(comp["model_id"].eq(val.iloc[0]["model_id"])) & comp["split"].eq("validation")]
        if not comp_val.empty:
            val_pass = bool(
                comp_val.iloc[0]["mean_daily_return"] > 0
                and comp_val.iloc[0]["profit_factor"] > 1.05
                and comp_val.iloc[0]["auc"] > 0.5
                and comp_val.iloc[0]["beats_best_baseline_pf"]
                and comp_val.iloc[0]["beats_best_baseline_mean_daily_return"]
                and comp_val.iloc[0]["drawdown_not_worse_than_least_bad_baseline"]
            )
    gate = "pass_to_batch5A_robustness_after_review" if val_pass else "review_required_before_batch5A"
    top = registry.sort_values(["validation_profit_factor", "validation_mean_daily_return"], ascending=[False, False]).head(5)
    lines = [
        "# Batch 4B Conclusion",
        "",
        "## What Changed",
        "",
        "- Trained frozen lightweight models from Batch 4A.",
        "- Generated model registry, coefficients, prediction scores, Top20 daily returns and split/regime/year/industry diagnostics.",
        "",
        "## What Did Not Change",
        "",
        "- Did not change primary horizon, label or frozen feature list.",
        "- Did not run hyperparameter search.",
        "- Did not enable concept/theme fields.",
        "- Did not move or modify `v7_locked`.",
        "",
        "## Top Models",
        "",
        top.to_markdown(index=False),
        "",
        "## Gate",
        "",
        f"- Gate result: `{gate}`.",
        "- Batch 4B is not considered passed unless the primary model also beats the frozen Batch3D 5d simple-rule baseline on validation.",
        "- If accepted in a later review, next step is Batch 5A robustness and exposure audit, not walk-forward or paper tracking.",
        "- If rejected, do not tune parameters on the same validation result; revise the plan explicitly in a new Batch 4A amendment.",
        "",
        "## Key Limitations",
        "",
        "- Research holdout is not a pristine final test because Batch3 diagnostics used full history.",
        "- Returns remain gross cohort diagnostics; execution ledger and capacity are only stress diagnostics here.",
        "- Low-size and low-liquidity exposure remains a central risk to audit in Batch 5A.",
    ]
    (REPORT_DIR / "batch4B_conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return gate


def write_handoff(gate: str, registry: pd.DataFrame) -> None:
    title = "Handoff: Batch 4B to Batch 5A" if gate.startswith("pass") else "Handoff: Batch 4B Review Required"
    lines = [
        f"# {title}",
        "",
        "## Completed Batch",
        "",
        "- Completed: `Batch 4B lightweight model training`.",
        "- Output directory: `reports/tushare/v9_swing_research/batch4B_lightweight_model_training/`.",
        "- Models trained only with Batch 4A frozen features and 5d label.",
        "",
        "## Gate",
        "",
        f"- Gate result: `{gate}`.",
        "",
        "## Required Inputs For Next Step",
        "",
        "- `batch4B_model_registry.csv`",
        "- `batch4B_validation_metrics.csv`",
        "- `batch4B_vs_simple_baseline.csv`",
        "- `batch4B_by_regime_metrics.csv`",
        "- `batch4B_by_year_metrics.csv`",
        "- `batch4B_by_industry_metrics.csv`",
        "- `batch4B_cost_sensitivity.csv`",
        "- `batch4B_profit_concentration.csv`",
        "",
        "## Blocked Actions",
        "",
        "- Do not start Batch 5A unless the Batch4B review explicitly accepts the model-vs-baseline gap.",
        "- Do not start walk-forward before Batch 5A robustness audit.",
        "- Do not add features or switch horizon without a new plan freeze.",
        "- Do not treat this as live readiness.",
    ]
    text = "\n".join(lines) + "\n"
    (REPORT_DIR / "batch4B_handoff_to_batch5A.md").write_text(text, encoding="utf-8")
    GLOBAL_HANDOFF_PATH.write_text(text, encoding="utf-8")


def update_stage_status(gate: str) -> None:
    if not STAGE_STATUS_PATH.exists():
        return
    df = pd.read_csv(STAGE_STATUS_PATH)
    mask4b = df["batch"].eq("batch4B_lightweight_model_training")
    df.loc[mask4b, "status"] = "completed"
    df.loc[mask4b, "output_dir"] = "reports/tushare/v9_swing_research/batch4B_lightweight_model_training"
    df.loc[mask4b, "gate_result"] = "pass" if gate.startswith("pass") else "review"
    df.loc[mask4b, "next_action"] = (
        "Proceed to Batch 5A robustness audit only after reviewing model-vs-baseline results; do not start walk-forward."
        if gate.startswith("pass")
        else "Review required: primary model did not beat frozen Batch3D simple-rule baseline; do not proceed to Batch5A without explicit acceptance."
    )
    mask5a = df["batch"].eq("batch5A_robustness_exposure_audit")
    if gate.startswith("pass"):
        df.loc[mask5a, "status"] = "pending"
        df.loc[mask5a, "gate_result"] = "pending"
        df.loc[mask5a, "next_action"] = "Audit frozen Batch4B model by regime, year, industry, stock concentration, liquidity, costs and profit concentration."
    else:
        df.loc[mask5a, "status"] = "blocked"
        df.loc[mask5a, "gate_result"] = "blocked"
        df.loc[mask5a, "next_action"] = "Blocked until Batch4B model-vs-baseline gap is reviewed; do not tune on validation results."
    df.to_csv(STAGE_STATUS_PATH, index=False)


def update_roadmap(gate: str) -> None:
    if not ROADMAP_PATH.exists():
        return
    text = ROADMAP_PATH.read_text(encoding="utf-8")
    addition = """### Stage 7B 轻量模型训练

输出目录：`reports/tushare/v9_swing_research/batch4B_lightweight_model_training/`

执行边界：

- 严格使用 Batch4A 冻结的 `5d` 标签和冻结特征。
- 未做超参搜索，未切换 horizon，未启用概念/题材字段。
- 输出模型注册、权重、Top20 逐日收益、分市场状态/年度/行业诊断和简单规则对比。

下一步：

- 若 Batch4B gate 通过，只能进入 Batch5A 鲁棒性和暴露审计。
- 若 Batch4B gate 为 review，则不得进入 Batch5A；需要先人工审查模型未打败简单规则的问题。
- 不得直接进入 walk-forward、forward paper tracking 或实盘相关流程。

"""
    marker = "### Step 5A: 鲁棒性和暴露审计"
    if addition.strip() not in text:
        text = text.replace(marker, addition + marker)
    text = text.replace("- 内容来自 `Batch 4A -> Batch 4B`", "- 内容来自 `Batch 4B review`")
    if gate.startswith("pass"):
        recommendation = "> Batch 5A: robustness and exposure audit, only if Batch 4B is accepted"
    else:
        recommendation = "> Review Batch 4B model-vs-baseline gap; do not enter Batch 5A yet"
    text = text.replace(
        "> Batch 4B: lightweight model training, only if the Batch 4A frozen plan is accepted",
        recommendation,
    )
    text = text.replace(
        "> Batch 5A: robustness and exposure audit, only if Batch 4B is accepted",
        recommendation,
    )
    ROADMAP_PATH.write_text(text, encoding="utf-8")


def write_hashes(extra_large_path: Path | None = None) -> None:
    rows = []
    for path in sorted(REPORT_DIR.glob("batch4B_*")):
        if path.name == "batch4B_file_sha256.csv" or not path.is_file():
            continue
        rows.append({"file": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    for path in sorted(MODEL_DIR.glob("*.pkl")):
        rows.append({"file": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    if extra_large_path and extra_large_path.exists():
        rows.append({"file": str(extra_large_path), "sha256": sha256_file(extra_large_path), "size_bytes": extra_large_path.stat().st_size})
    for path in [GLOBAL_HANDOFF_PATH, ROADMAP_PATH, STAGE_STATUS_PATH]:
        if path.exists():
            rows.append({"file": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    with (REPORT_DIR / "batch4B_file_sha256.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "sha256", "size_bytes"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    primary_features, conditional_features = read_frozen_features()
    all_features = primary_features + conditional_features
    specs = build_model_specs(primary_features, conditional_features)
    write_train_config(specs, primary_features, conditional_features)
    print({"event": "start", "primary_features": primary_features, "conditional_features": conditional_features}, flush=True)
    df = read_dataset(all_features)
    print({"event": "dataset_ready", "rows": int(len(df)), "splits": df["split"].value_counts().to_dict()}, flush=True)
    add_rank_features(df, all_features)
    models = train_models(df, specs)
    score_cols = score_models(df, specs, models)
    pred_path = write_predictions(df, specs)
    selected = selected_topn(df, specs)
    selected.to_parquet(REPORT_DIR / "batch4B_selected_trades.parquet", index=False, compression="zstd")
    daily = daily_from_selected(selected)
    daily.to_csv(REPORT_DIR / "batch4B_top20_daily_returns.csv", index=False)
    metrics = validation_metrics(df, daily, specs)
    summarize_by_regime(selected)
    summarize_selected_by_daily_group(selected, "year", "batch4B_by_year_metrics.csv")
    summarize_trade_group(selected, "industry", "batch4B_by_industry_metrics.csv")
    summarize_trade_group(selected, "size_bucket", "batch4B_by_size_metrics.csv")
    summarize_trade_group(selected, "liquidity_bucket", "batch4B_by_liquidity_metrics.csv")
    cost_sensitivity(daily)
    remove_winner_concentration(daily)
    feature_importance(specs, models)
    registry = model_registry(specs, metrics)
    baseline = baseline_metrics_by_split()
    comp = compare_vs_baseline(metrics, baseline)
    gate = write_conclusion(registry, comp)
    write_handoff(gate, registry)
    update_stage_status(gate)
    update_roadmap(gate)
    write_hashes(pred_path)
    print(
        {
            "out_dir": str(REPORT_DIR),
            "models": [s.model_id for s in specs],
            "prediction_scores": str(pred_path),
            "gate_result": gate,
            "next_step": "Batch 5A robustness audit after review" if gate.startswith("pass") else "review Batch 4B before robustness audit",
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
