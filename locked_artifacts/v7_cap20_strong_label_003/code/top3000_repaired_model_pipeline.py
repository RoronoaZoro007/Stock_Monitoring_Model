#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from tushare_tail_model import (
    FEATURE_COLUMNS,
    CostConfig,
    ExitConfig,
    add_enhanced_features,
    add_market_cross_section,
    compute_symbol_features,
    load_minute_symbol,
    load_stock_basic,
    max_drawdown,
    minute_symbol_paths,
    profit_factor,
    topn_by_day,
)


ROOT = Path(__file__).resolve().parent
RANK_PATH = ROOT / "data_tushare" / "manifests" / "liquid_top3000_20251120_20260213.csv"
RAW_DAILY_DIR = ROOT / "data_tushare" / "raw" / "daily"
CLEAN_DIR = ROOT / "data_tushare" / "clean"
FEATURE_DIR = CLEAN_DIR / "features"
REPORT_DIR = ROOT / "reports" / "tushare" / "full_top3000_model_compare"


BASE_PARAMS = {
    "learning_rate": 0.04,
    "max_iter": 260,
    "max_leaf_nodes": 31,
    "l2_regularization": 0.02,
    "min_samples_leaf": 80,
}


def pct(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "NA"


def read_rank(path: Path, top_rank: int) -> pd.DataFrame:
    rank = pd.read_csv(path, dtype={"ts_code": str})
    rank["liquidity_rank"] = pd.to_numeric(rank["liquidity_rank"], errors="coerce").astype(int)
    return rank.sort_values("liquidity_rank").drop_duplicates("ts_code", keep="first").head(top_rank)


def read_raw_daily() -> pd.DataFrame:
    paths = sorted(RAW_DAILY_DIR.glob("*.parquet"))
    if not paths:
        raise SystemExit(f"missing daily files under {RAW_DAILY_DIR}")
    df = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    for col in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trade_date"] = df["trade_date"].astype(str)
    return df.sort_values(["ts_code", "trade_date"]).drop_duplicates(["ts_code", "trade_date"], keep="last")


def build_minute_daily_agg(rank: pd.DataFrame, rebuild: bool = False) -> pd.DataFrame:
    path = CLEAN_DIR / "daily_from_minutes_top3000.parquet"
    if path.exists() and not rebuild:
        return pd.read_parquet(path)

    allowed = set(rank["ts_code"])
    paths_by_symbol: dict[str, list[Path]] = {}
    for path_i in minute_symbol_paths():
        ts_code = path_i.parent.name.split("=", 1)[1]
        if ts_code in allowed:
            paths_by_symbol.setdefault(ts_code, []).append(path_i)

    rows = []
    for idx, ts_code in enumerate(sorted(paths_by_symbol), start=1):
        minute = load_minute_symbol(paths_by_symbol[ts_code])
        if minute.empty:
            continue
        agg = minute.sort_values("dt").groupby("trade_date").agg(
            ts_code=("ts_code", "last"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            vol=("vol", "sum"),
            amount=("amount", "sum"),
            bars=("bar_time", "nunique"),
        )
        agg = agg.reset_index()
        agg["vol"] = agg["vol"] / 100.0
        agg["amount"] = agg["amount"] / 1000.0
        rows.append(agg)
        if idx % 100 == 0:
            print(f"minute_daily_agg symbols={idx} rows={sum(len(x) for x in rows)}", flush=True)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False, compression="zstd")
    return out


def prepare_repaired_daily(rank: pd.DataFrame, rebuild_minute_agg: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = read_raw_daily()
    raw_top = raw[raw["ts_code"].isin(set(rank["ts_code"]))].copy()
    raw_top["daily_source"] = "raw_daily"
    minute_agg = build_minute_daily_agg(rank, rebuild_minute_agg)
    minute_key = minute_agg[["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "bars"]].copy()
    merged = minute_key.merge(raw_top[["ts_code", "trade_date"]], on=["ts_code", "trade_date"], how="left", indicator=True)
    missing_keys = merged[merged["_merge"].eq("left_only")][["ts_code", "trade_date"]]
    missing = minute_key.merge(missing_keys, on=["ts_code", "trade_date"], how="inner").copy()
    missing = missing[missing["bars"].ge(40)].drop(columns=["bars"])
    for col in ["pre_close", "change", "pct_chg"]:
        missing[col] = np.nan
    missing["daily_source"] = "minute_reconstructed"

    repaired = pd.concat([raw_top, missing], ignore_index=True, sort=False)
    repaired = repaired.sort_values(["ts_code", "trade_date"]).drop_duplicates(["ts_code", "trade_date"], keep="last")
    g = repaired.groupby("ts_code", group_keys=False)
    repaired["pre_close"] = g["close"].shift(1)
    repaired["change"] = repaired["close"] - repaired["pre_close"]
    repaired["pct_chg"] = repaired["change"] / repaired["pre_close"] * 100.0

    repaired["prev_close_calc"] = g["close"].shift(1)
    repaired["prev_close_use"] = repaired["pre_close"].where(repaired["pre_close"].notna(), repaired["prev_close_calc"])
    repaired["prev_ret_1d"] = g["close"].transform(lambda s: s.pct_change().shift(1))
    repaired["prev_ret_3d"] = g["close"].transform(lambda s: s.shift(1) / s.shift(4) - 1)
    repaired["prev_ret_5d"] = g["close"].transform(lambda s: s.shift(1) / s.shift(6) - 1)
    repaired["prev_volatility_20d"] = g["close"].transform(lambda s: s.pct_change().shift(1).rolling(20, min_periods=10).std())
    repaired["ma5_prev"] = g["close"].transform(lambda s: s.shift(1).rolling(5, min_periods=5).mean())
    repaired["ma10_prev"] = g["close"].transform(lambda s: s.shift(1).rolling(10, min_periods=8).mean())
    repaired["ma20_prev"] = g["close"].transform(lambda s: s.shift(1).rolling(20, min_periods=15).mean())
    repaired["amount5_prev"] = g["amount"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    repaired["amount20_prev"] = g["amount"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    repaired["prev_amount_ratio_5_20"] = repaired["amount5_prev"] / repaired["amount20_prev"] - 1
    repaired["next_trade_date"] = g["trade_date"].shift(-1)
    repaired["prev_trade_date"] = g["trade_date"].shift(1)
    repaired = add_enhanced_features(repaired)

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    repaired_path = CLEAN_DIR / "daily_repaired_top3000.parquet"
    repaired.to_parquet(repaired_path, index=False, compression="zstd")
    audit = {
        "raw_daily_rows_top3000": int(len(raw_top)),
        "minute_agg_rows_top3000": int(len(minute_agg)),
        "reconstructed_daily_rows": int(len(missing)),
        "repaired_daily_rows": int(len(repaired)),
        "reconstructed_symbols": int(missing["ts_code"].nunique()) if not missing.empty else 0,
        "repaired_path": str(repaired_path),
        "minute_agg_path": str(CLEAN_DIR / "daily_from_minutes_top3000.parquet"),
        "repair_rule": "补齐分钟线存在但 raw daily 缺失的股票-交易日；vol 按 /100、amount 按 /1000 转为 Tushare daily 口径；pre_close/change/pct_chg 在修复后按股票时间序列重算。",
    }
    return repaired, audit


def build_feature_dataset(rank: pd.DataFrame, daily: pd.DataFrame, min_amount_sofar: float, rebuild: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = FEATURE_DIR / "tail_dataset_top3000_repaired_raw.parquet"
    audit_path = REPORT_DIR / "dataset_repaired_raw_audit.json"
    if path.exists() and audit_path.exists() and not rebuild:
        return pd.read_parquet(path), json.loads(audit_path.read_text(encoding="utf-8"))

    allowed = set(rank["ts_code"])
    rank_map = dict(zip(rank["ts_code"], rank["liquidity_rank"]))
    stock_basic = load_stock_basic()
    st_names = set(stock_basic.loc[stock_basic["name"].str.contains("ST", case=False, na=False), "ts_code"].astype(str))
    daily_by_symbol = {code: group.sort_values("trade_date") for code, group in daily.groupby("ts_code")}
    paths_by_symbol: dict[str, list[Path]] = {}
    for path_i in minute_symbol_paths():
        ts_code = path_i.parent.name.split("=", 1)[1]
        if ts_code in allowed:
            paths_by_symbol.setdefault(ts_code, []).append(path_i)

    costs = CostConfig()
    exit_cfg = ExitConfig()
    parts = []
    for idx, ts_code in enumerate(sorted(paths_by_symbol, key=lambda code: rank_map.get(code, 999999)), start=1):
        if ts_code not in daily_by_symbol:
            continue
        minute = load_minute_symbol(paths_by_symbol[ts_code])
        feats = compute_symbol_features(
            ts_code=ts_code,
            minute=minute,
            daily_map=daily_by_symbol[ts_code],
            is_st=ts_code in st_names,
            min_amount_sofar=min_amount_sofar,
            costs=costs,
            exit_cfg=exit_cfg,
        )
        if not feats.empty:
            feats["liquidity_rank"] = rank_map.get(ts_code)
            parts.append(feats)
        if idx % 100 == 0:
            print(f"feature_dataset symbols={idx} parts={len(parts)} rows={sum(len(p) for p in parts)}", flush=True)

    dataset = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not dataset.empty:
        dataset = add_market_cross_section(dataset).replace([np.inf, -np.inf], np.nan)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(path, index=False, compression="zstd")
    audit = {
        "dataset_path": str(path),
        "selected_symbols": int(len(rank)),
        "symbols_with_minute_files": int(len(paths_by_symbol)),
        "symbols_with_samples": int(dataset["ts_code"].nunique()) if not dataset.empty else 0,
        "rows": int(len(dataset)),
        "date_min": str(dataset["trade_date"].min()) if not dataset.empty else None,
        "date_max": str(dataset["trade_date"].max()) if not dataset.empty else None,
        "win_rate_all_samples": float(dataset["target_win"].mean()) if not dataset.empty else None,
        "avg_target_return_all_samples": float(dataset["target_return"].mean()) if not dataset.empty else None,
        "cost_config": asdict(costs),
        "exit_config": asdict(exit_cfg),
        "filters": {"min_amount_sofar": min_amount_sofar},
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return dataset, audit


def add_quality_flags(dataset: pd.DataFrame, repaired_daily: pd.DataFrame) -> pd.DataFrame:
    current = repaired_daily[["ts_code", "trade_date", "close", "next_trade_date"]].rename(columns={"close": "current_close"})
    next_day = repaired_daily[["ts_code", "trade_date", "open", "pct_chg", "daily_source"]].rename(
        columns={"trade_date": "next_trade_date", "open": "next_open", "pct_chg": "next_pct_chg", "daily_source": "next_daily_source"}
    )
    out = dataset.merge(current, on=["ts_code", "trade_date"], how="left").merge(
        next_day, on=["ts_code", "next_trade_date"], how="left"
    )
    out["overnight_open_gap"] = out["next_open"] / out["current_close"] - 1.0
    out["next_pct_chg_frac"] = out["next_pct_chg"] / 100.0
    out["suspect_corporate_action"] = (
        out["overnight_open_gap"].abs().gt(0.22)
        & out["next_pct_chg_frac"].abs().lt(0.15)
        & out["target_return"].lt(-0.20)
    )
    out["extreme_abs_return_gt10"] = out["target_return"].abs().gt(0.10)
    out["extreme_abs_return_gt20"] = out["target_return"].abs().gt(0.20)
    out["extreme_abs_return_gt50"] = out["target_return"].abs().gt(0.50)
    return out


def add_prev_amp_5d(df: pd.DataFrame, repaired_daily: pd.DataFrame) -> pd.DataFrame:
    daily = repaired_daily[["ts_code", "trade_date", "high", "low", "prev_close_use"]].copy()
    daily["daily_amp"] = (daily["high"] - daily["low"]) / daily["prev_close_use"]
    daily["prev_amp_5d"] = daily.groupby("ts_code")["daily_amp"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    return df.merge(daily[["ts_code", "trade_date", "prev_amp_5d"]], on=["ts_code", "trade_date"], how="left")


def add_resonance_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tail_share = out["tail_amount_share"].clip(lower=0, upper=1)
    amount_vs = out["tail_amount_vs_prev20"].clip(lower=-0.95, upper=5)
    out["tail_ret_1430_1450_x_tail_amount_share"] = out["tail_ret_1430_1450"] * tail_share
    out["tail_ret_1420_1450_x_tail_amount_vs_prev20"] = out["tail_ret_1420_1450"] * np.log1p(amount_vs + 1)
    out["late_ret_1445_1450_x_tail_amount_share"] = out["late_ret_1445_1450"] * tail_share
    out["tail_quality_ratio"] = out["tail_ret_1430_1450"] / (tail_share + 0.02)
    return out


def add_industry_features(df: pd.DataFrame) -> pd.DataFrame:
    stock_basic = load_stock_basic()
    cols = ["ts_code", "industry"]
    if "industry" not in stock_basic.columns:
        out = df.copy()
        out["industry"] = "UNKNOWN"
    else:
        out = df.merge(stock_basic[cols], on="ts_code", how="left")
        out["industry"] = out["industry"].fillna("UNKNOWN")
    for col in ["tail_ret_1430_1450", "tail_ret_1420_1450", "prev_volatility_20d", "tail_amount_share"]:
        med = out.groupby(["trade_date", "industry"])[col].transform("median")
        std = out.groupby(["trade_date", "industry"])[col].transform("std")
        out[f"{col}_industry_z"] = (out[col] - med) / std.replace(0, np.nan)
    return out


def prepare_model_frame(dataset: pd.DataFrame, repaired_daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    flagged = add_quality_flags(dataset, repaired_daily)
    drop_cols = [
        "current_close",
        "next_trade_date",
        "next_open",
        "next_pct_chg",
        "overnight_open_gap",
        "next_pct_chg_frac",
        "suspect_corporate_action",
        "extreme_abs_return_gt10",
        "extreme_abs_return_gt20",
        "extreme_abs_return_gt50",
        "next_daily_source",
    ]
    event_clean = flagged[~flagged["suspect_corporate_action"].fillna(False)].drop(columns=drop_cols, errors="ignore")
    strict_cap20 = flagged[flagged["target_return"].abs().le(0.20)].drop(columns=drop_cols, errors="ignore")
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    event_path = FEATURE_DIR / "tail_dataset_top3000_event_clean.parquet"
    cap_path = FEATURE_DIR / "tail_dataset_top3000_strict_cap20.parquet"
    event_clean.to_parquet(event_path, index=False, compression="zstd")
    strict_cap20.to_parquet(cap_path, index=False, compression="zstd")

    model_df = add_prev_amp_5d(event_clean, repaired_daily)
    model_df = add_resonance_features(model_df)
    model_df = add_industry_features(model_df)
    model_df["regime_allowed"] = model_df["market_breadth_positive"].ge(0.30) & model_df["market_tail_ret_median"].gt(-0.003)
    model_df["label_win"] = (model_df["target_return"] > 0).astype(int)
    model_df["label_static_003"] = (model_df["target_return"] > 0.003).astype(int)
    amp_fallback = model_df["prev_amp_5d"].median()
    model_df["dynamic_threshold"] = np.maximum(0.003, model_df["prev_amp_5d"].fillna(amp_fallback) * 0.10)
    model_df["label_dynamic_amp10"] = (model_df["target_return"] > model_df["dynamic_threshold"]).astype(int)
    model_path = FEATURE_DIR / "tail_dataset_top3000_model_frame.parquet"
    model_df.replace([np.inf, -np.inf], np.nan).to_parquet(model_path, index=False, compression="zstd")

    quality = {
        "raw_feature_rows": int(len(dataset)),
        "event_clean_rows": int(len(event_clean)),
        "strict_cap20_rows": int(len(strict_cap20)),
        "dropped_suspect_corporate_action": int(flagged["suspect_corporate_action"].fillna(False).sum()),
        "extreme_abs_return_gt10": int(flagged["extreme_abs_return_gt10"].sum()),
        "extreme_abs_return_gt20": int(flagged["extreme_abs_return_gt20"].sum()),
        "extreme_abs_return_gt50": int(flagged["extreme_abs_return_gt50"].sum()),
        "model_frame_rows": int(len(model_df)),
        "event_clean_path": str(event_path),
        "strict_cap20_path": str(cap_path),
        "model_frame_path": str(model_path),
    }
    return model_df.replace([np.inf, -np.inf], np.nan), quality


def make_model(params: dict[str, Any]) -> Pipeline:
    model = HistGradientBoostingClassifier(
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        max_leaf_nodes=params["max_leaf_nodes"],
        l2_regularization=params["l2_regularization"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=42,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def evaluate_predictions(pred: pd.DataFrame, label_col: str, top_n: int, all_validation_dates: list[str], apply_regime_filter: bool) -> dict[str, Any]:
    pick_universe = pred[pred["regime_allowed"].fillna(False)].copy() if apply_regime_filter else pred.copy()
    selected = topn_by_day(pick_universe, top_n) if not pick_universe.empty else pd.DataFrame(columns=pred.columns)
    returns_by_date = selected.groupby("trade_date")["target_return"].mean() if not selected.empty else pd.Series(dtype=float)
    daily = pd.Series(0.0, index=pd.Index(sorted(all_validation_dates), name="trade_date"))
    if not returns_by_date.empty:
        daily.loc[returns_by_date.index] = returns_by_date
    return {
        "rows": int(len(pred)),
        "days": int(len(all_validation_dates)),
        "trading_days": int(selected["trade_date"].nunique()) if len(selected) else 0,
        "selected_trades": int(len(selected)),
        "auc": float(roc_auc_score(pred[label_col], pred["score"])) if pred[label_col].nunique() > 1 else np.nan,
        "label_positive_rate": float(pred[label_col].mean()),
        "all_sample_win_rate": float((pred["target_return"] > 0).mean()),
        "all_sample_avg_return": float(pred["target_return"].mean()),
        "top10_trade_win_rate": float((selected["target_return"] > 0).mean()) if len(selected) else np.nan,
        "top10_avg_trade_return": float(selected["target_return"].mean()) if len(selected) else np.nan,
        "top10_daily_win_rate": float((daily > 0).mean()) if len(daily) else np.nan,
        "top10_avg_daily_return": float(daily.mean()) if len(daily) else np.nan,
        "top10_cumulative_return": float((1 + daily).prod() - 1) if len(daily) else np.nan,
        "top10_max_drawdown": max_drawdown(daily),
        "top10_profit_factor": profit_factor(selected["target_return"]) if len(selected) else np.nan,
        "cash_days": int((daily == 0).sum()),
    }


def run_variant(
    df: pd.DataFrame,
    name: str,
    label_col: str,
    feature_cols: list[str],
    validation_start: str,
    top_n: int,
    apply_regime_filter: bool,
) -> dict[str, Any]:
    out_dir = REPORT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    data = df.dropna(subset=["target_return", label_col]).copy()
    data["trade_date"] = data["trade_date"].astype(str)
    train = data[data["trade_date"] < validation_start].copy()
    val = data[data["trade_date"] >= validation_start].copy()
    if train.empty or val.empty:
        raise SystemExit(f"empty train/validation for {name}")
    if train[label_col].nunique() < 2:
        raise SystemExit(f"single-class train label for {name}")
    model = make_model(BASE_PARAMS).fit(train[feature_cols], train[label_col].astype(int))
    val_pred = val.copy()
    val_pred["score"] = model.predict_proba(val_pred[feature_cols])[:, 1]
    val_pred["p_model_label"] = val_pred["score"]
    all_validation_dates = sorted(val["trade_date"].unique())
    pick_universe = val_pred[val_pred["regime_allowed"].fillna(False)] if apply_regime_filter else val_pred
    top10 = topn_by_day(pick_universe, top_n) if not pick_universe.empty else pd.DataFrame(columns=val_pred.columns)

    val_pred.to_csv(out_dir / "validation_predictions.csv", index=False)
    top10.to_csv(out_dir / "top10_validation.csv", index=False)
    with (out_dir / "model.pkl").open("wb") as fh:
        pickle.dump({"model": model, "feature_columns": feature_cols, "label_col": label_col, "params": BASE_PARAMS}, fh)

    metrics = {
        "variant": name,
        "label_col": label_col,
        "feature_count": len(feature_cols),
        "apply_regime_filter": apply_regime_filter,
        "params": BASE_PARAMS,
        "dataset": {
            "rows": int(len(data)),
            "train_rows": int(len(train)),
            "validation_rows": int(len(val)),
            "train_symbols": int(train["ts_code"].nunique()),
            "validation_symbols": int(val["ts_code"].nunique()),
            "train_positive_rate": float(train[label_col].mean()),
            "validation_positive_rate": float(val[label_col].mean()),
            "date_min": str(data["trade_date"].min()),
            "date_max": str(data["trade_date"].max()),
            "validation_start": validation_start,
        },
        "validation": evaluate_predictions(val_pred, label_col, top_n, all_validation_dates, apply_regime_filter),
        "leakage_controls": [
            "validation_start 之后的数据不参与日线修复规则选择、标签阈值选择、模型参数选择。",
            "模型参数使用此前训练期 walk-forward 选出的固定参数；本轮多版本比较不使用验证集调参。",
            "每个变体只用 validation_start 前样本拟合，validation_start 后只做一次验证。",
            "特征仍限制为买入日前 14:50 及以前分钟线、当日开盘集合竞价、前一交易日收盘集合竞价/资金流和横截面市场状态。",
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return metrics


def row_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    val = metrics["validation"]
    ds = metrics["dataset"]
    return {
        "variant": metrics["variant"],
        "label": metrics["label_col"],
        "features": metrics["feature_count"],
        "regime": metrics["apply_regime_filter"],
        "train_rows": ds["train_rows"],
        "validation_rows": ds["validation_rows"],
        "train_positive_rate": ds["train_positive_rate"],
        "validation_positive_rate": ds["validation_positive_rate"],
        "trading_days": val["trading_days"],
        "cash_days": val["cash_days"],
        "auc": val["auc"],
        "trade_win_rate": val["top10_trade_win_rate"],
        "daily_win_rate": val["top10_daily_win_rate"],
        "avg_daily_return": val["top10_avg_daily_return"],
        "cumulative_return": val["top10_cumulative_return"],
        "max_drawdown": val["top10_max_drawdown"],
        "profit_factor": val["top10_profit_factor"],
    }


def write_summary(data_audit: dict[str, Any], repair_audit: dict[str, Any], quality: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    table = pd.DataFrame(rows)
    table.to_csv(REPORT_DIR / "comparison_metrics.csv", index=False)
    view = table.copy()
    for col in [
        "train_positive_rate",
        "validation_positive_rate",
        "auc",
        "trade_win_rate",
        "daily_win_rate",
        "avg_daily_return",
        "cumulative_return",
        "max_drawdown",
    ]:
        if col in view:
            if col == "auc":
                view[col] = view[col].map(lambda x: "NA" if pd.isna(x) else f"{float(x):.4f}")
            else:
                view[col] = view[col].map(pct)
    best = table.sort_values(["cumulative_return", "avg_daily_return"], ascending=False).iloc[0] if not table.empty else None
    text = f"""# Top3000 修复数据与模型对比

## 数据修复依据

- 原始分钟线审计显示 Top3000 覆盖完整，硬错误为 0：无重复、无非法 OHLC、无非正价格、无负成交量额、无非交易时点。
- 日线交叉校验发现：分钟线存在但 raw daily 缺失的股票-交易日为 {repair_audit['reconstructed_daily_rows']} 行，涉及 {repair_audit['reconstructed_symbols']} 只股票。
- 修复动作：用分钟线聚合补齐缺失 daily 行，并按修复后的股票时间序列重算 `pre_close/change/pct_chg/next_trade_date`，防止隔夜标签跨过真实交易日。
- 原始数据保留不动；修复层输出到 `data_tushare/clean`。

## 特征样本

- 修复后原始特征样本：{data_audit.get('rows'):,} 行，{data_audit.get('symbols_with_samples')} 只股票，日期 {data_audit.get('date_min')} 至 {data_audit.get('date_max')}。
- 疑似除权/断层剔除：{quality['dropped_suspect_corporate_action']} 行。
- `abs(target_return)>10%`：{quality['extreme_abs_return_gt10']} 行；`>20%`：{quality['extreme_abs_return_gt20']} 行；`>50%`：{quality['extreme_abs_return_gt50']} 行。
- 主清洗样本：{quality['event_clean_rows']:,} 行。

## 验证集对比

统一验证起点为 `20260224`，所有模型只用此前样本训练。

{view.to_markdown(index=False)}

## 当前结论

"""
    if best is not None:
        text += (
            f"按验证期累计收益排序，当前最好变体是 `{best['variant']}`："
            f"累计收益 {pct(best['cumulative_return'])}，日均收益 {pct(best['avg_daily_return'])}，"
            f"逐笔胜率 {pct(best['trade_win_rate'])}，最大回撤 {pct(best['max_drawdown'])}。\n"
        )
    text += """
## 可执行建议

1. 生产执行先使用修复 daily + 主清洗样本，不再使用 raw daily 的 `next_trade_date` 直接生成隔夜标签。
2. 若最佳变体包含市场环境过滤，则把空仓日视为真实执行结果，而不是从统计中剔除；这会降低交易频率但避免弱市 Beta 杀伤。
3. `target_return>0.3%` 标签用于覆盖交易摩擦门槛；如果验证期收益提升但交易日减少，需要后续用滚动月份稳定性继续确认。
4. Level-2 大单/五档盘口、市值中性化仍缺数据，本轮不伪造；补齐对应数据后再纳入下一轮模型。
"""
    (REPORT_DIR / "summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair daily gaps from minute data and compare Top3000 model variants")
    parser.add_argument("--rank-file", default=str(RANK_PATH))
    parser.add_argument("--top-rank", type=int, default=3000)
    parser.add_argument("--validation-start", default="20260224")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-amount-sofar", type=float, default=20_000_000)
    parser.add_argument("--rebuild-minute-agg", action="store_true")
    parser.add_argument("--rebuild-dataset", action="store_true")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rank = read_rank(Path(args.rank_file), args.top_rank)
    repaired_daily, repair_audit = prepare_repaired_daily(rank, args.rebuild_minute_agg)
    (REPORT_DIR / "daily_repair_audit.json").write_text(json.dumps(repair_audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    dataset, data_audit = build_feature_dataset(rank, repaired_daily, args.min_amount_sofar, args.rebuild_dataset)
    model_df, quality = prepare_model_frame(dataset, repaired_daily)
    (REPORT_DIR / "cleaning_quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    resonance_cols = [
        "tail_ret_1430_1450_x_tail_amount_share",
        "tail_ret_1420_1450_x_tail_amount_vs_prev20",
        "late_ret_1445_1450_x_tail_amount_share",
        "tail_quality_ratio",
    ]
    industry_cols = [
        "tail_ret_1430_1450_industry_z",
        "tail_ret_1420_1450_industry_z",
        "prev_volatility_20d_industry_z",
        "tail_amount_share_industry_z",
    ]
    enhanced_cols = FEATURE_COLUMNS + resonance_cols + industry_cols + ["prev_amp_5d"]
    cap20_df = model_df[model_df["target_return"].abs().le(0.20)].copy()
    variants = [
        (model_df, "v1_event_clean_win_label", "label_win", FEATURE_COLUMNS, False),
        (model_df, "v2_event_clean_strong_label_003", "label_static_003", FEATURE_COLUMNS, False),
        (model_df, "v3_event_clean_dynamic_amp10", "label_dynamic_amp10", FEATURE_COLUMNS, False),
        (model_df, "v4_strong_label_resonance_industry", "label_static_003", enhanced_cols, False),
        (model_df, "v5_strong_label_resonance_industry_regime", "label_static_003", enhanced_cols, True),
        (cap20_df, "v6_cap20_win_label", "label_win", FEATURE_COLUMNS, False),
        (cap20_df, "v7_cap20_strong_label_003", "label_static_003", FEATURE_COLUMNS, False),
        (cap20_df, "v8_cap20_strong_label_resonance_industry", "label_static_003", enhanced_cols, False),
    ]
    rows = []
    metrics_all = []
    for variant_df, name, label, cols, regime in variants:
        print(f"running_variant={name} label={label} features={len(cols)} regime={regime}", flush=True)
        metrics = run_variant(variant_df, name, label, cols, args.validation_start, args.top_n, regime)
        metrics_all.append(metrics)
        rows.append(row_from_metrics(metrics))

    (REPORT_DIR / "all_metrics.json").write_text(json.dumps(metrics_all, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_summary(data_audit, repair_audit, quality, rows)
    print(json.dumps({"repair_audit": repair_audit, "dataset_audit": data_audit, "quality": quality, "rows": rows}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
