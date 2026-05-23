#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage_model_outputs import build_stage_dataset, monthly_stability
from tushare_tail_model import (
    FEATURE_COLUMNS,
    add_predictions,
    evaluate,
    load_daily,
    model_from_params,
    random_baseline,
    topn_by_day,
)


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "data_tushare" / "features" / "stages"
REPORT_ROOT = ROOT / "reports" / "tushare" / "stages"


def pct(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "NA"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def add_daily_quality_flags(dataset: pd.DataFrame) -> pd.DataFrame:
    daily = load_daily()
    current = daily[["ts_code", "trade_date", "close", "next_trade_date"]].rename(columns={"close": "current_close"})
    next_day = daily[["ts_code", "trade_date", "open", "pct_chg"]].rename(
        columns={"trade_date": "next_trade_date", "open": "next_open", "pct_chg": "next_pct_chg"}
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
    out["extreme_abs_return_gt20"] = out["target_return"].abs().gt(0.20)
    out["extreme_abs_return_gt50"] = out["target_return"].abs().gt(0.50)
    return out


def quality_summary(dataset: pd.DataFrame, name: str) -> dict[str, Any]:
    df = add_daily_quality_flags(dataset)
    ret = pd.to_numeric(df["target_return"], errors="coerce")
    return {
        "variant": name,
        "rows": int(len(df)),
        "symbols": int(df["ts_code"].nunique()),
        "date_min": str(df["trade_date"].min()),
        "date_max": str(df["trade_date"].max()),
        "win_rate": float((ret > 0).mean()),
        "avg_return": float(ret.mean()),
        "std_return": float(ret.std()),
        "min_return": float(ret.min()),
        "max_return": float(ret.max()),
        "abs_gt_10pct": int(ret.abs().gt(0.10).sum()),
        "abs_gt_20pct": int(ret.abs().gt(0.20).sum()),
        "abs_gt_50pct": int(ret.abs().gt(0.50).sum()),
        "abs_gt_100pct": int(ret.abs().gt(1.00).sum()),
        "suspect_corporate_action": int(df["suspect_corporate_action"].sum()),
    }


def evaluate_fixed_params(
    dataset: pd.DataFrame,
    out_dir: Path,
    stage: str,
    params: dict[str, Any],
    validation_start: str,
    top_n: int,
    random_seeds: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = dataset.replace([np.inf, -np.inf], np.nan).dropna(subset=["target_return", "target_win"]).copy()
    df["trade_date"] = df["trade_date"].astype(str)
    train_df = df[df["trade_date"] < validation_start].copy()
    val_df = df[df["trade_date"] >= validation_start].copy()
    model = model_from_params(params).fit(train_df[FEATURE_COLUMNS], train_df["target_win"].astype(int))
    val_pred = add_predictions(model, val_df)
    val_pred.to_csv(out_dir / "validation_predictions.csv", index=False)
    top10 = topn_by_day(val_pred, top_n)
    top10.to_csv(out_dir / "top10_validation.csv", index=False)
    monthly_stability(val_pred, top_n).to_csv(out_dir / "validation_monthly_stability.csv", index=False)
    metrics = {
        "stage": stage,
        "procedure": "same_hgb_params_no_retune",
        "params": params,
        "dataset": {
            "rows": int(len(df)),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "symbols": int(df["ts_code"].nunique()),
            "date_min": str(df["trade_date"].min()),
            "date_max": str(df["trade_date"].max()),
            "validation_start": validation_start,
        },
        "quality": quality_summary(df, stage),
        "validation": evaluate(val_pred, "validation", top_n),
        "validation_random_baseline": random_baseline(val_pred, top_n, random_seeds),
        "leakage_controls": [
            "训练集日期严格早于 validation_start。",
            "本对比固定同一组 HGB 参数，不使用验证集调参。",
            "验证集只在训练后用于一次性评估。",
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return metrics


def metric_row(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    val = metrics.get("validation", {})
    rnd = metrics.get("validation_random_baseline", {})
    ds = metrics.get("dataset", {})
    quality = metrics.get("quality", {})
    return {
        "variant": name,
        "procedure": metrics.get("procedure", "train_only_tuned"),
        "rows": ds.get("rows"),
        "train_rows": ds.get("train_rows"),
        "validation_rows": ds.get("validation_rows"),
        "quality_avg_return": quality.get("avg_return"),
        "quality_abs_gt20": quality.get("abs_gt_20pct"),
        "quality_abs_gt50": quality.get("abs_gt_50pct"),
        "suspect_corporate_action": quality.get("suspect_corporate_action"),
        "auc": val.get("auc"),
        "top10_trade_win_rate": val.get("top10_trade_win_rate"),
        "top10_daily_win_rate": val.get("top10_daily_win_rate"),
        "top10_avg_daily_return": val.get("top10_avg_daily_return"),
        "top10_cumulative_return": val.get("top10_cumulative_return"),
        "top10_max_drawdown": val.get("top10_max_drawdown"),
        "top10_profit_factor": val.get("top10_profit_factor"),
        "random_avg_daily_return": rnd.get("random_top10_avg_daily_return_mean"),
        "random_cumulative_return": rnd.get("random_top10_cumulative_return_mean"),
    }


def write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    table = pd.DataFrame(rows)
    view = table.copy()
    for col in [
        "quality_avg_return",
        "auc",
        "top10_trade_win_rate",
        "top10_daily_win_rate",
        "top10_avg_daily_return",
        "top10_cumulative_return",
        "top10_max_drawdown",
        "random_avg_daily_return",
        "random_cumulative_return",
    ]:
        if col in view:
            if col == "auc":
                view[col] = view[col].map(lambda x: "NA" if pd.isna(x) else f"{float(x):.4f}")
            else:
                view[col] = view[col].map(pct)
    text = "# Top500 数据质量模型对比\n\n"
    text += "本对比包含旧口径已产出结果，以及固定同一组 HGB 参数后的数据清洗口径对比。固定参数对比用于隔离数据修复影响，不使用验证集调参。\n\n"
    text += view.to_markdown(index=False)
    text += "\n\n## 口径说明\n\n"
    text += "- `old_vwap_existing_tuned`：原 Top500 阶段产出，使用旧 VWAP 口径并在训练集 walk-forward 调参。\n"
    text += "- `old_vwap_same_params`：旧数据集，固定同一组 HGB 参数重训。\n"
    text += "- `vwap_fixed_same_params`：重建后的 VWAP 修复数据集，固定同一组 HGB 参数重训。\n"
    text += "- `vwap_fixed_event_clean_same_params`：在 VWAP 修复基础上剔除疑似除权/拆分断层样本。\n"
    text += "- `vwap_fixed_strict_cap20_same_params`：在 VWAP 修复基础上剔除所有 `abs(target_return)>20%` 样本，用于压力对照。\n"
    (out_dir / "comparison_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="compare Top500 model results across data-quality variants")
    parser.add_argument("--rank-file", required=True)
    parser.add_argument("--top-rank", type=int, default=500)
    parser.add_argument("--validation-start", default="20260224")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--random-seeds", type=int, default=100)
    parser.add_argument("--min-amount-sofar", type=float, default=20_000_000)
    args = parser.parse_args()

    out_root = REPORT_ROOT / "top500_data_quality_comparison"
    out_root.mkdir(parents=True, exist_ok=True)
    old_dataset_path = DATASET_ROOT / "tail_dataset_top500.parquet"
    old_backup_path = DATASET_ROOT / "tail_dataset_top500_old_vwap.parquet"
    if old_dataset_path.exists() and not old_backup_path.exists():
        shutil.copy2(old_dataset_path, old_backup_path)
    if not old_backup_path.exists():
        raise SystemExit(f"missing old Top500 dataset: {old_backup_path}")

    old_dataset = pd.read_parquet(old_backup_path)
    old_metrics = load_json(REPORT_ROOT / "top500" / "metrics.json")
    params = old_metrics.get("best_params_selected_on_train_only") or {
        "learning_rate": 0.04,
        "max_iter": 260,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.10,
        "min_samples_leaf": 80,
    }

    print("building_vwap_fixed_dataset", flush=True)
    fixed_stage_dir = REPORT_ROOT / "top500_vwap_fixed_rebuild"
    fixed_dataset, fixed_audit = build_stage_dataset(Path(args.rank_file), args.top_rank, fixed_stage_dir, args.min_amount_sofar)
    (fixed_stage_dir / "dataset_audit.json").write_text(
        json.dumps(fixed_audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    fixed_dataset = fixed_dataset.copy()
    fixed_flagged = add_daily_quality_flags(fixed_dataset)
    event_clean = fixed_flagged[~fixed_flagged["suspect_corporate_action"]].drop(
        columns=[
            "current_close",
            "next_trade_date",
            "next_open",
            "next_pct_chg",
            "overnight_open_gap",
            "next_pct_chg_frac",
            "suspect_corporate_action",
            "extreme_abs_return_gt20",
            "extreme_abs_return_gt50",
        ],
        errors="ignore",
    )
    strict_cap20 = fixed_flagged[fixed_flagged["target_return"].abs().le(0.20)].drop(
        columns=[
            "current_close",
            "next_trade_date",
            "next_open",
            "next_pct_chg",
            "overnight_open_gap",
            "next_pct_chg_frac",
            "suspect_corporate_action",
            "extreme_abs_return_gt20",
            "extreme_abs_return_gt50",
        ],
        errors="ignore",
    )

    variants = [
        ("old_vwap_same_params", old_dataset),
        ("vwap_fixed_same_params", fixed_dataset),
        ("vwap_fixed_event_clean_same_params", event_clean),
        ("vwap_fixed_strict_cap20_same_params", strict_cap20),
    ]

    rows = []
    if old_metrics:
        old_existing = dict(old_metrics)
        old_existing["procedure"] = "existing_train_only_tuned_old_vwap"
        old_existing["quality"] = quality_summary(old_dataset, "old_vwap_existing_tuned")
        rows.append(metric_row("old_vwap_existing_tuned", old_existing))

    for name, dataset in variants:
        print(f"evaluating {name} rows={len(dataset)}", flush=True)
        metrics = evaluate_fixed_params(
            dataset,
            out_root / name,
            name,
            params,
            args.validation_start,
            args.top_n,
            args.random_seeds,
        )
        rows.append(metric_row(name, metrics))

    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_root / "comparison_metrics.csv", index=False)
    write_summary(out_root, rows)
    print(json.dumps({"output_dir": str(out_root), "rows": rows}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

