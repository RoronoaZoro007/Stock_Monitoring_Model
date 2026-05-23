#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tushare_tail_model import (
    FEATURE_COLUMNS,
    CostConfig,
    ExitConfig,
    add_enhanced_features,
    add_market_cross_section,
    compute_symbol_features,
    evaluate,
    load_daily,
    load_minute_symbol,
    load_stock_basic,
    minute_symbol_paths,
    model_from_params,
    random_baseline,
    topn_by_day,
    tune_on_train,
)


ROOT = Path(__file__).resolve().parent
STAGE_ROOT = ROOT / "reports" / "tushare" / "stages"
DATASET_ROOT = ROOT / "data_tushare" / "features" / "stages"
BEIJING_TZ = timezone(timedelta(hours=8))


def read_rank(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ts_code": str})
    df["liquidity_rank"] = pd.to_numeric(df["liquidity_rank"], errors="coerce").astype(int)
    return df.sort_values("liquidity_rank").drop_duplicates("ts_code")


def pct(value: Any) -> str:
    try:
        if value is None or math.isnan(float(value)):
            return "NA"
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "NA"


def build_stage_dataset(rank_file: Path, top_rank: int, out_dir: Path, min_amount_sofar: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    ranked = read_rank(rank_file)
    selected = ranked[ranked["liquidity_rank"].le(top_rank)].copy()
    allowed = set(selected["ts_code"])
    rank_map = dict(zip(selected["ts_code"], selected["liquidity_rank"]))

    costs = CostConfig()
    exit_cfg = ExitConfig()
    stock_basic = load_stock_basic()
    st_names = set(stock_basic.loc[stock_basic["name"].str.contains("ST", case=False, na=False), "ts_code"].astype(str))
    daily = add_enhanced_features(load_daily())
    daily_by_symbol = {code: group.sort_values("trade_date") for code, group in daily.groupby("ts_code")}

    paths_by_symbol: dict[str, list[Path]] = {}
    for path in minute_symbol_paths():
        ts_code = path.parent.name.split("=", 1)[1]
        if ts_code in allowed:
            paths_by_symbol.setdefault(ts_code, []).append(path)

    parts = []
    processed = 0
    for ts_code in sorted(paths_by_symbol, key=lambda code: rank_map.get(code, 999999)):
        processed += 1
        if ts_code not in daily_by_symbol:
            continue
        minute = load_minute_symbol(paths_by_symbol[ts_code])
        feats = compute_symbol_features(
            ts_code,
            minute,
            daily_by_symbol[ts_code],
            ts_code in st_names,
            min_amount_sofar,
            costs,
            exit_cfg,
        )
        if not feats.empty:
            feats["liquidity_rank"] = rank_map.get(ts_code)
            parts.append(feats)
        if processed % 100 == 0:
            print(f"stage_dataset processed_symbols={processed} feature_parts={len(parts)} rows={sum(len(p) for p in parts)}", flush=True)

    if parts:
        dataset = pd.concat(parts, ignore_index=True)
        dataset = add_market_cross_section(dataset).replace([np.inf, -np.inf], np.nan)
    else:
        dataset = pd.DataFrame()

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_path = DATASET_ROOT / f"tail_dataset_top{top_rank}.parquet"
    if not dataset.empty:
        dataset.to_parquet(dataset_path, index=False, compression="zstd")

    audit = {
        "dataset_path": str(dataset_path),
        "top_rank": top_rank,
        "rank_file": str(rank_file.resolve()),
        "selected_symbols": int(len(selected)),
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
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return dataset, audit


def daily_return_series(selected: pd.DataFrame, all_dates: pd.Series, top_n: int) -> pd.DataFrame:
    rows = []
    by_date = {date: group for date, group in selected.groupby("trade_date")}
    for date in sorted(all_dates.astype(str).unique()):
        picks = by_date.get(date)
        daily_return = 0.0 if picks is None or picks.empty else float(picks["target_return"].sum() / top_n)
        rows.append({"trade_date": date, "daily_return": daily_return})
    return pd.DataFrame(rows)


def monthly_stability(pred: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    selected = topn_by_day(pred, top_n)
    daily = daily_return_series(selected, pred["trade_date"], top_n)
    if daily.empty:
        return pd.DataFrame()
    daily["month"] = daily["trade_date"].str[:6]
    selected = selected.copy()
    selected["month"] = selected["trade_date"].astype(str).str[:6]
    rows = []
    for month, group in daily.groupby("month", sort=True):
        trades = selected[selected["month"].eq(month)]
        rows.append(
            {
                "month": month,
                "days": int(len(group)),
                "selected_trades": int(len(trades)),
                "daily_win_rate": float((group["daily_return"] > 0).mean()) if len(group) else np.nan,
                "avg_daily_return": float(group["daily_return"].mean()) if len(group) else np.nan,
                "cumulative_return": float((1 + group["daily_return"]).prod() - 1) if len(group) else np.nan,
                "trade_win_rate": float((trades["target_return"] > 0).mean()) if len(trades) else np.nan,
                "avg_trade_return": float(trades["target_return"].mean()) if len(trades) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def train_and_evaluate(dataset: pd.DataFrame, out_dir: Path, stage: str, top_n: int, validation_start: str, random_seeds: int) -> dict[str, Any]:
    if dataset.empty:
        return {"stage": stage, "error": "empty dataset"}
    df = dataset.replace([np.inf, -np.inf], np.nan).dropna(subset=["target_return", "target_win"]).copy()
    df["trade_date"] = df["trade_date"].astype(str)
    train_df = df[df["trade_date"] < validation_start].copy()
    val_df = df[df["trade_date"] >= validation_start].copy()
    if train_df.empty or val_df.empty:
        return {
            "stage": stage,
            "error": "empty train or validation split",
            "validation_start": validation_start,
            "rows": int(len(df)),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
        }
    if train_df["target_win"].nunique() < 2:
        return {
            "stage": stage,
            "error": "training labels only have one class",
            "rows": int(len(df)),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
        }

    best_params, tuning, wf_pred = tune_on_train(train_df, top_n)
    tuning.to_csv(out_dir / "classifier_tuning.csv", index=False)
    wf_pred.to_csv(out_dir / "walk_forward_predictions.csv", index=False)

    final_model = model_from_params(best_params).fit(train_df[FEATURE_COLUMNS], train_df["target_win"].astype(int))
    val_pred = val_df.copy()
    val_pred["p_win"] = final_model.predict_proba(val_pred[FEATURE_COLUMNS])[:, 1]
    val_pred["score"] = val_pred["p_win"]
    val_pred.to_csv(out_dir / "validation_predictions.csv", index=False)
    top10 = topn_by_day(val_pred, top_n)
    top10.to_csv(out_dir / "top10_validation.csv", index=False)

    wf_monthly = monthly_stability(wf_pred, top_n)
    val_monthly = monthly_stability(val_pred, top_n)
    wf_monthly.to_csv(out_dir / "walk_forward_monthly_stability.csv", index=False)
    val_monthly.to_csv(out_dir / "validation_monthly_stability.csv", index=False)

    metrics = {
        "stage": stage,
        "dataset": {
            "rows": int(len(df)),
            "validation_start": validation_start,
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "train_symbols": int(train_df["ts_code"].nunique()),
            "validation_symbols": int(val_df["ts_code"].nunique()),
            "date_min": str(df["trade_date"].min()),
            "date_max": str(df["trade_date"].max()),
        },
        "best_params_selected_on_train_only": best_params,
        "walk_forward": evaluate(wf_pred, "train_walk_forward", top_n),
        "validation": evaluate(val_pred, "validation", top_n),
        "validation_random_baseline": random_baseline(val_pred, top_n, random_seeds),
        "leakage_controls": [
            "训练集日期严格早于 validation_start。",
            "超参数搜索只使用 validation_start 之前的训练期 walk-forward。",
            "验证集只使用训练期选出的参数做一次评估。",
        ],
    }

    with (out_dir / "classifier_model.pkl").open("wb") as fh:
        pickle.dump({"model": final_model, "feature_columns": FEATURE_COLUMNS, "metrics": metrics}, fh)
    return metrics


def write_markdown(out_dir: Path, stage: str, mode: str, audit: dict[str, Any], metrics: dict[str, Any]) -> None:
    validation = metrics.get("validation", {}) if isinstance(metrics, dict) else {}
    wf = metrics.get("walk_forward", {}) if isinstance(metrics, dict) else {}
    baseline = metrics.get("validation_random_baseline", {}) if isinstance(metrics, dict) else {}
    text = f"""# {stage} 阶段产出

模式：{mode}

## 数据审计

- Top rank：{audit.get('top_rank')}
- 入选股票：{audit.get('selected_symbols')}
- 有分钟文件股票：{audit.get('symbols_with_minute_files')}
- 生成样本股票：{audit.get('symbols_with_samples')}
- 样本行数：{audit.get('rows')}
- 日期范围：{audit.get('date_min')} 至 {audit.get('date_max')}
- 全样本胜率：{pct(audit.get('win_rate_all_samples'))}
- 全样本平均收益：{pct(audit.get('avg_target_return_all_samples'))}

## 训练期 Walk-Forward

- Top10 日均收益：{pct(wf.get('top10_avg_daily_return'))}
- Top10 累计收益：{pct(wf.get('top10_cumulative_return'))}
- Top10 逐笔胜率：{pct(wf.get('top10_trade_win_rate'))}
- Top10 日胜率：{pct(wf.get('top10_daily_win_rate'))}
- 最大回撤：{pct(wf.get('top10_max_drawdown'))}

## 最近三个月验证

- Top10 日均收益：{pct(validation.get('top10_avg_daily_return'))}
- Top10 累计收益：{pct(validation.get('top10_cumulative_return'))}
- Top10 逐笔胜率：{pct(validation.get('top10_trade_win_rate'))}
- Top10 日胜率：{pct(validation.get('top10_daily_win_rate'))}
- 最大回撤：{pct(validation.get('top10_max_drawdown'))}
- Profit Factor：{validation.get('top10_profit_factor')}
- AUC：{validation.get('auc')}

## 随机基准

- 随机 Top10 日均收益均值：{pct(baseline.get('random_top10_avg_daily_return_mean'))}
- 随机 Top10 日均收益 95 分位：{pct(baseline.get('random_top10_avg_daily_return_p95'))}
- 随机 Top10 累计收益均值：{pct(baseline.get('random_top10_cumulative_return_mean'))}
- 随机 Top10 累计收益 95 分位：{pct(baseline.get('random_top10_cumulative_return_p95'))}
"""
    (out_dir / "summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="build stage-specific model outputs for Tushare tail strategy")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--top-rank", type=int, required=True)
    parser.add_argument("--rank-file", required=True)
    parser.add_argument("--validation-start", default="20260224")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--random-seeds", type=int, default=100)
    parser.add_argument("--min-amount-sofar", type=float, default=20_000_000)
    args = parser.parse_args()

    out_dir = STAGE_ROOT / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_path = out_dir / "STAGE_OUTPUT_FAILED.json"
    running_path = out_dir / "STAGE_OUTPUT_RUNNING.json"
    failed_path.unlink(missing_ok=True)
    try:
        dataset, audit = build_stage_dataset(Path(args.rank_file), args.top_rank, out_dir, args.min_amount_sofar)
        metrics = train_and_evaluate(dataset, out_dir, args.stage, args.top_n, args.validation_start, args.random_seeds)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        write_markdown(out_dir, args.stage, args.mode, audit, metrics)
        done = {"stage": args.stage, "mode": args.mode, "top_rank": args.top_rank, "outputs": str(out_dir), "metrics_error": metrics.get("error")}
        (out_dir / "STAGE_OUTPUT_DONE.json").write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
        running_path.unlink(missing_ok=True)
        failed_path.unlink(missing_ok=True)
        print(json.dumps(done, ensure_ascii=False, indent=2))
    except Exception as exc:
        failed = {
            "stage": args.stage,
            "mode": args.mode,
            "top_rank": args.top_rank,
            "failed_at_beijing": datetime.now(BEIJING_TZ).isoformat(timespec="seconds"),
            "error": repr(exc),
        }
        failed_path.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
        running_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
