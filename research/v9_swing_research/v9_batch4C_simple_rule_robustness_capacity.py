#!/usr/bin/env python3
"""v9 Batch 4C: simple-rule robustness and capacity audit.

This batch audits the Batch3D minimum-to-beat simple rules after Batch4B models
failed to beat them. It does not train models, tune parameters, add rules,
change labels, change horizons, or modify v7_locked.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT / "data_tushare" / "clean" / "v9"
REPORT_ROOT = ROOT / "reports" / "tushare" / "v9_swing_research"
B4A_DIR = REPORT_ROOT / "batch4A_model_training_plan"
B4B_REVIEW_DIR = REPORT_ROOT / "batch4B_review"
REPORT_DIR = REPORT_ROOT / "batch4C_simple_rule_robustness_capacity"

FEATURE_PATH = CLEAN_DIR / "v9_stock_industry_features.parquet"
LABEL_PATH = CLEAN_DIR / "v9_swing_labels_3_5_10.parquet"
REVIEW_RULE_SELECTED_PATH = B4B_REVIEW_DIR / "batch4B_review_rule_selected_trades.parquet"

GLOBAL_HANDOFF_PATH = REPORT_ROOT / "v9_current_handoff.md"
ROADMAP_PATH = REPORT_ROOT / "v9_execution_roadmap.md"
STAGE_STATUS_PATH = REPORT_ROOT / "v9_stage_status.csv"

PRIMARY_LABEL = "fwd_ret_5d_open"
PRIMARY_TOP_N = 20

TRAIN_START = "20180102"
TRAIN_END = "20221230"
VALIDATION_START = "20230103"
VALIDATION_END = "20241231"
HOLDOUT_START = "20250102"
HOLDOUT_END = "20260522"

ROUNDTRIP_COST_BPS = [0, 20, 40, 60, 100, 150, 200]
CAPITAL_YUAN = [100_000, 300_000, 500_000, 1_000_000, 5_000_000]
PARTICIPATION_CAPS = [0.01, 0.03, 0.05, 0.10]
CAPACITY_REFERENCE_COLUMNS = ["signal_amount", "entry_amount"]
PRIMARY_CAPITAL_YUAN = 1_000_000
PRIMARY_PARTICIPATION_CAP = 0.05
PRIMARY_COST_BPS = 40

REGIME_COLUMNS = [
    "trend_regime",
    "vol_regime",
    "industry_crowding_regime",
    "stock_concentration_regime",
    "market_regime_id",
    "extreme_selloff_flag",
]


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    factors: tuple[str, ...]
    directions: tuple[str, ...]


RULE_SPECS = {
    "low_log_amount": RuleSpec("low_log_amount", ("log_amount",), ("low",)),
    "low_stock_amount_share_in_industry": RuleSpec(
        "low_stock_amount_share_in_industry",
        ("stock_amount_share_in_industry",),
        ("low",),
    ),
    "low_log_total_mv": RuleSpec("low_log_total_mv", ("log_total_mv",), ("low",)),
    "low_stock_ret_60d": RuleSpec("low_stock_ret_60d", ("stock_ret_60d",), ("low",)),
    "combo_low_liquidity_weak_momentum": RuleSpec(
        "combo_low_liquidity_weak_momentum",
        ("log_amount", "turnover_rate", "stock_ret_20d", "stock_ret_60d"),
        ("low", "low", "low", "low"),
    ),
    "combo_low_size_low_liquidity": RuleSpec(
        "combo_low_size_low_liquidity",
        ("log_total_mv", "log_amount", "turnover_rate"),
        ("low", "low", "low"),
    ),
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def split_name(trade_date: pd.Series) -> pd.Series:
    d = trade_date.astype(str)
    out = pd.Series("outside", index=trade_date.index, dtype="object")
    out[d.between(TRAIN_START, TRAIN_END)] = "train"
    out[d.between(VALIDATION_START, VALIDATION_END)] = "validation"
    out[d.between(HOLDOUT_START, HOLDOUT_END)] = "research_holdout"
    return out


def max_drawdown(returns: pd.Series) -> float:
    r = finite_numeric(returns).dropna()
    if r.empty:
        return np.nan
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def profit_factor(returns: pd.Series) -> float:
    r = finite_numeric(returns).dropna()
    gains = r.loc[r.gt(0)].sum()
    losses = r.loc[r.lt(0)].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / abs(losses))


def summarize_daily(daily: pd.DataFrame, meta: dict[str, object], return_col: str = "daily_return") -> dict[str, object]:
    row = dict(meta)
    if daily.empty:
        row.update(
            {
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
        )
        return row
    d = daily.sort_values("trade_date").copy()
    r = finite_numeric(d[return_col]).fillna(0.0)
    total_trades = int(d["selected_count"].sum())
    pos_sum = float(d.get("positive_trade_return_sum", pd.Series(dtype=float)).sum()) if total_trades else np.nan
    neg_sum = float(d.get("negative_trade_return_sum", pd.Series(dtype=float)).sum()) if total_trades else np.nan
    trade_sum = float(d.get("trade_return_sum", pd.Series(dtype=float)).sum()) if total_trades else np.nan
    win_count = int(d.get("win_count", pd.Series(dtype=float)).sum()) if total_trades else 0
    row.update(
        {
            "signal_days": int(d["trade_date"].nunique()),
            "total_trades": total_trades,
            "avg_selected_count": float(d["selected_count"].mean()),
            "cumulative_return": float((1.0 + r).prod() - 1.0),
            "mean_daily_return": float(r.mean()),
            "median_daily_return": float(r.median()),
            "daily_win_rate": float(r.gt(0).mean()),
            "trade_win_rate": float(win_count / total_trades) if total_trades else np.nan,
            "profit_factor": profit_factor(r),
            "trade_profit_factor": float(pos_sum / abs(neg_sum)) if pd.notna(neg_sum) and neg_sum < 0 else np.nan,
            "max_drawdown": max_drawdown(r),
            "avg_trade_return": float(trade_sum / total_trades) if total_trades else np.nan,
            "max_daily_return": float(r.max()),
            "min_daily_return": float(r.min()),
        }
    )
    return row


def daily_from_trades(trades: pd.DataFrame, return_col: str, group_cols: list[str]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby(group_cols + ["trade_date"], dropna=False, sort=False)
    daily = g.agg(
        selected_count=(return_col, "size"),
        daily_return=(return_col, "mean"),
        trade_return_sum=(return_col, "sum"),
        win_count=(return_col, lambda x: int((x > 0).sum())),
        loss_count=(return_col, lambda x: int((x <= 0).sum())),
        positive_trade_return_sum=(return_col, lambda x: float(x[x > 0].sum())),
        negative_trade_return_sum=(return_col, lambda x: float(x[x < 0].sum())),
    ).reset_index()
    return daily


def make_rule_score(sub: pd.DataFrame, rule: RuleSpec) -> pd.Series:
    if len(rule.factors) == 1:
        raw = finite_numeric(sub[rule.factors[0]])
        return -raw if rule.directions[0] == "low" else raw
    parts = []
    for factor, direction in zip(rule.factors, rule.directions):
        rank_pct = sub.groupby("trade_date", sort=False)[factor].rank(pct=True, method="average")
        preferred = 1.0 - rank_pct if direction == "low" else rank_pct
        parts.append(preferred)
    return pd.concat(parts, axis=1).mean(axis=1)


def read_minimum_rules() -> list[str]:
    ref = pd.read_csv(B4A_DIR / "batch4A_baseline_reference.csv")
    rules = ref.loc[ref["baseline_role"].eq("minimum_to_beat") & ref["horizon"].eq("5d"), "rule_id"].astype(str).tolist()
    return [rule_id for rule_id in rules if rule_id in RULE_SPECS]


def read_selection_dataset(minimum_rules: list[str]) -> pd.DataFrame:
    factor_cols = sorted({f for rid in minimum_rules for f in RULE_SPECS[rid].factors})
    feature_cols = [
        "trade_date",
        "ts_code",
        "industry",
        "amount",
        "total_mv",
        "turnover_rate",
        "listed_days",
        "st_flag",
        "suspend_flag",
        "limit_up_close_flag",
        "limit_down_close_flag",
    ] + REGIME_COLUMNS + factor_cols
    feature_cols = list(dict.fromkeys(feature_cols))
    label_cols = ["trade_date", "ts_code", PRIMARY_LABEL, "label_win_5d"]
    print({"event": "read_features", "columns": len(feature_cols)}, flush=True)
    features = pd.read_parquet(FEATURE_PATH, columns=feature_cols)
    print({"event": "read_labels", "columns": len(label_cols)}, flush=True)
    labels = pd.read_parquet(LABEL_PATH, columns=label_cols)
    features["trade_date"] = features["trade_date"].astype(str)
    labels["trade_date"] = labels["trade_date"].astype(str)
    df = features.merge(labels, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
    df["industry"] = df["industry"].fillna("UNKNOWN").astype(str)
    for flag in ["st_flag", "suspend_flag", "limit_up_close_flag", "limit_down_close_flag", "extreme_selloff_flag"]:
        df[flag] = df[flag].fillna(False).astype(bool)
    numeric_cols = sorted(set(factor_cols + ["amount", "total_mv", "turnover_rate", "listed_days", PRIMARY_LABEL]))
    for col in numeric_cols:
        df[col] = finite_numeric(df[col])
    df["split"] = split_name(df["trade_date"])
    df["screened_universe_flag"] = (
        df["split"].ne("outside")
        & df["listed_days"].ge(120)
        & ~df["st_flag"]
        & ~df["suspend_flag"]
        & ~df["limit_up_close_flag"]
        & ~df["limit_down_close_flag"]
        & df[PRIMARY_LABEL].notna()
    )
    return df[df["split"].ne("outside")].copy()


def reconstruct_rule_trades(df: pd.DataFrame, minimum_rules: list[str]) -> pd.DataFrame:
    frames = []
    for idx, rule_id in enumerate(minimum_rules, start=1):
        rule = RULE_SPECS[rule_id]
        print({"event": "reconstruct_rule", "done": idx - 1, "total": len(minimum_rules), "rule_id": rule_id}, flush=True)
        needed = [
            "trade_date",
            "ts_code",
            "industry",
            "split",
            "amount",
            "total_mv",
            "turnover_rate",
            PRIMARY_LABEL,
            "label_win_5d",
        ] + REGIME_COLUMNS + list(rule.factors)
        needed = list(dict.fromkeys(needed))
        sub = df.loc[df["screened_universe_flag"], needed].dropna(subset=[PRIMARY_LABEL] + list(rule.factors)).copy()
        sub["score"] = make_rule_score(sub, rule)
        sub = sub.dropna(subset=["score"])
        sub = sub.sort_values(["trade_date", "score"], ascending=[True, False], kind="mergesort")
        top = sub.groupby("trade_date", sort=False).head(PRIMARY_TOP_N).copy()
        top["rule_id"] = rule_id
        top["rank"] = top.groupby(["rule_id", "trade_date"], sort=False)["score"].rank(ascending=False, method="first").astype("int16")
        frames.append(top)
    return pd.concat(frames, ignore_index=True)


def load_or_reconstruct_trades(minimum_rules: list[str]) -> pd.DataFrame:
    if REVIEW_RULE_SELECTED_PATH.exists():
        print({"event": "read_existing_rule_selected", "path": str(REVIEW_RULE_SELECTED_PATH)}, flush=True)
        selected = pd.read_parquet(REVIEW_RULE_SELECTED_PATH)
        selected = selected.rename(columns={"baseline_id": "rule_id"})
        selected["trade_date"] = selected["trade_date"].astype(str)
        return selected
    df = read_selection_dataset(minimum_rules)
    return reconstruct_rule_trades(df, minimum_rules)


def add_execution_context(selected: pd.DataFrame) -> pd.DataFrame:
    missing_regime_cols = [col for col in REGIME_COLUMNS if col not in selected.columns]
    if missing_regime_cols:
        regime_patch = pd.read_parquet(FEATURE_PATH, columns=["trade_date", "ts_code"] + missing_regime_cols)
        regime_patch["trade_date"] = regime_patch["trade_date"].astype(str)
        selected = selected.merge(regime_patch, on=["trade_date", "ts_code"], how="left", validate="many_to_one")
    label_cols = [
        "trade_date",
        "ts_code",
        "amount",
        "entry_date",
        "exit_date_5d",
        "entry_open",
        "entry_st_flag",
        "entry_suspend_flag",
        "entry_limit_up_open_flag",
        "exit_st_flag_5d",
        "exit_suspend_flag_5d",
        "exit_limit_down_close_flag_5d",
        "exit_limit_up_close_flag_5d",
    ]
    labels = pd.read_parquet(LABEL_PATH, columns=label_cols)
    labels["trade_date"] = labels["trade_date"].astype(str)
    labels["entry_date"] = labels["entry_date"].astype(str)
    labels["exit_date_5d"] = labels["exit_date_5d"].astype(str)
    context = labels[
        [
            "trade_date",
            "ts_code",
            "entry_date",
            "exit_date_5d",
            "entry_open",
            "entry_st_flag",
            "entry_suspend_flag",
            "entry_limit_up_open_flag",
            "exit_st_flag_5d",
            "exit_suspend_flag_5d",
            "exit_limit_down_close_flag_5d",
            "exit_limit_up_close_flag_5d",
        ]
    ]
    out = selected.merge(context, on=["trade_date", "ts_code"], how="left", validate="many_to_one")
    amount_lookup = labels[["trade_date", "ts_code", "amount"]].copy()
    entry_amount = amount_lookup.rename(columns={"trade_date": "entry_date", "amount": "entry_amount"})
    exit_amount = amount_lookup.rename(columns={"trade_date": "exit_date_5d", "amount": "exit_amount_5d"})
    out = out.merge(entry_amount, on=["entry_date", "ts_code"], how="left", validate="many_to_one")
    out = out.merge(exit_amount, on=["exit_date_5d", "ts_code"], how="left", validate="many_to_one")
    out = out.rename(columns={"amount": "signal_amount"})
    for col in ["signal_amount", "entry_amount", "exit_amount_5d"]:
        out[col] = finite_numeric(out[col])
        out[f"{col}_yuan"] = out[col] * 1000.0
    for flag in [
        "entry_st_flag",
        "entry_suspend_flag",
        "entry_limit_up_open_flag",
        "exit_st_flag_5d",
        "exit_suspend_flag_5d",
        "exit_limit_down_close_flag_5d",
        "exit_limit_up_close_flag_5d",
    ]:
        out[flag] = out[flag].fillna(False).astype(bool)
    if "extreme_selloff_flag" in out.columns:
        out["extreme_selloff_flag"] = out["extreme_selloff_flag"].fillna(False).astype(bool)
    out[PRIMARY_LABEL] = finite_numeric(out[PRIMARY_LABEL])
    out["year"] = out["trade_date"].str[:4]
    local_path = REPORT_DIR / "_local_batch4C_rule_trades_with_execution.parquet"
    out.to_parquet(local_path, index=False, compression="zstd")
    return out


def read_cached_csv(name: str) -> pd.DataFrame | None:
    path = REPORT_DIR / name
    if path.exists() and path.stat().st_size > 0:
        print({"event": "cache_hit", "file": name}, flush=True)
        return pd.read_csv(path)
    return None


def build_base_summary(trades: pd.DataFrame) -> pd.DataFrame:
    daily = daily_from_trades(trades, PRIMARY_LABEL, ["rule_id", "split"])
    rows = []
    for keys, g in daily.groupby(["rule_id", "split"], sort=False):
        rule_id, split = keys
        rows.append(summarize_daily(g, {"rule_id": rule_id, "split": split, "cost_bps": 0}))
    out = pd.DataFrame(rows).sort_values(["split", "mean_daily_return"], ascending=[True, False])
    out.to_csv(REPORT_DIR / "batch4C_rule_robustness_summary.csv", index=False)
    return out


def build_cost_sensitivity(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cost_bps in ROUNDTRIP_COST_BPS:
        sub = trades.copy()
        sub["return_after_cost"] = sub[PRIMARY_LABEL] - cost_bps / 10000.0
        daily = daily_from_trades(sub, "return_after_cost", ["rule_id", "split"])
        for keys, g in daily.groupby(["rule_id", "split"], sort=False):
            rule_id, split = keys
            rows.append(summarize_daily(g, {"rule_id": rule_id, "split": split, "roundtrip_cost_bps": cost_bps}))
    out = pd.DataFrame(rows).sort_values(["split", "rule_id", "roundtrip_cost_bps"])
    out.to_csv(REPORT_DIR / "batch4C_cost_sensitivity.csv", index=False)
    return out


def build_capacity_matrix(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_cols = ["rule_id", "split", "trade_date", PRIMARY_LABEL]
    for reference_col in CAPACITY_REFERENCE_COLUMNS:
        ref_yuan = finite_numeric(trades[f"{reference_col}_yuan"])
        for capital_yuan in CAPITAL_YUAN:
            order_yuan = capital_yuan / PRIMARY_TOP_N
            participation = order_yuan / ref_yuan.replace(0, np.nan)
            for cap_idx, cap in enumerate(PARTICIPATION_CAPS, start=1):
                print(
                    {
                        "event": "capacity_progress",
                        "reference_amount": reference_col,
                        "capital_yuan": capital_yuan,
                        "participation_cap": cap,
                        "cap_done": cap_idx - 1,
                        "caps_total": len(PARTICIPATION_CAPS),
                    },
                    flush=True,
                )
                base_fill = (cap / participation).clip(upper=1.0)
                base_fill = base_fill.where(participation.notna() & participation.gt(0), 0.0).fillna(0.0)
                cost_bps = PRIMARY_COST_BPS
                sub = trades[base_cols].copy()
                sub["participation_rate"] = participation
                sub["fill_ratio"] = base_fill
                sub["return_after_cost"] = sub[PRIMARY_LABEL] - cost_bps / 10000.0
                # Cash drag: unfilled notional earns zero return and is not redistributed.
                sub["capacity_adjusted_return"] = sub["return_after_cost"] * sub["fill_ratio"]
                daily = daily_from_trades(sub, "capacity_adjusted_return", ["rule_id", "split"])
                fill_daily = (
                    sub.groupby(["rule_id", "split", "trade_date"], sort=False)
                    .agg(
                        avg_fill_ratio=("fill_ratio", "mean"),
                        partial_fill_count=("fill_ratio", lambda x: int((x.lt(0.999999) & x.gt(0)).sum())),
                        zero_fill_count=("fill_ratio", lambda x: int(x.le(0).sum())),
                        participation_p50=("participation_rate", "median"),
                        participation_p90=("participation_rate", lambda x: float(finite_numeric(x).quantile(0.90))),
                        participation_p99=("participation_rate", lambda x: float(finite_numeric(x).quantile(0.99))),
                    )
                    .reset_index()
                )
                daily = daily.merge(fill_daily, on=["rule_id", "split", "trade_date"], how="left", validate="one_to_one")
                for keys, g in daily.groupby(["rule_id", "split"], sort=False):
                    rule_id, split = keys
                    row = summarize_daily(
                        g,
                        {
                            "rule_id": rule_id,
                            "split": split,
                            "reference_amount": reference_col,
                            "capital_yuan": capital_yuan,
                            "per_stock_order_yuan": order_yuan,
                            "participation_cap": cap,
                            "roundtrip_cost_bps": cost_bps,
                            "fill_policy": "partial_fill_cash_drag_no_refill",
                        },
                    )
                    row.update(
                        {
                            "avg_fill_ratio": float(g["avg_fill_ratio"].mean()),
                            "partial_fill_days": int(g["partial_fill_count"].gt(0).sum()),
                            "zero_fill_days": int(g["zero_fill_count"].gt(0).sum()),
                            "partial_fill_trades": int(g["partial_fill_count"].sum()),
                            "zero_fill_trades": int(g["zero_fill_count"].sum()),
                            "participation_p50": float(g["participation_p50"].median()),
                            "participation_p90": float(g["participation_p90"].median()),
                            "participation_p99": float(g["participation_p99"].median()),
                        }
                    )
                    rows.append(row)
    out = pd.DataFrame(rows).sort_values(["split", "rule_id", "reference_amount", "capital_yuan", "participation_cap", "roundtrip_cost_bps"])
    out.to_csv(REPORT_DIR / "batch4C_capacity_matrix.csv", index=False)
    return out


def build_regime_robustness(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cost_bps in [0, PRIMARY_COST_BPS]:
        sub = trades.copy()
        sub["return_after_cost"] = sub[PRIMARY_LABEL] - cost_bps / 10000.0
        for regime_col in REGIME_COLUMNS:
            daily = daily_from_trades(sub, "return_after_cost", ["rule_id", "split", regime_col])
            for keys, g in daily.groupby(["rule_id", "split", regime_col], dropna=False, sort=False):
                rule_id, split, regime_value = keys
                rows.append(
                    summarize_daily(
                        g,
                        {
                            "rule_id": rule_id,
                            "split": split,
                            "roundtrip_cost_bps": cost_bps,
                            "regime_type": regime_col,
                            "regime_value": regime_value,
                        },
                    )
                )
    out = pd.DataFrame(rows).sort_values(["split", "rule_id", "roundtrip_cost_bps", "regime_type", "regime_value"])
    out.to_csv(REPORT_DIR / "batch4C_regime_robustness.csv", index=False)
    return out


def build_yearly_robustness(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cost_bps in [0, PRIMARY_COST_BPS]:
        sub = trades.copy()
        sub["return_after_cost"] = sub[PRIMARY_LABEL] - cost_bps / 10000.0
        daily = daily_from_trades(sub, "return_after_cost", ["rule_id", "split", "year"])
        for keys, g in daily.groupby(["rule_id", "split", "year"], sort=False):
            rule_id, split, year = keys
            rows.append(summarize_daily(g, {"rule_id": rule_id, "split": split, "year": year, "roundtrip_cost_bps": cost_bps}))
    out = pd.DataFrame(rows).sort_values(["rule_id", "year", "roundtrip_cost_bps"])
    out.to_csv(REPORT_DIR / "batch4C_yearly_robustness.csv", index=False)
    return out


def build_concentration(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in trades.groupby(["rule_id", "split"], sort=False):
        rule_id, split = keys
        stock_counts = g["ts_code"].value_counts()
        industry_counts = g["industry"].fillna("UNKNOWN").value_counts()
        stock_returns = g.groupby("ts_code")[PRIMARY_LABEL].sum().sort_values(ascending=False)
        industry_returns = g.groupby("industry")[PRIMARY_LABEL].sum().sort_values(ascending=False)
        total_positive_stock = stock_returns[stock_returns.gt(0)].sum()
        rows.append(
            {
                "rule_id": rule_id,
                "split": split,
                "trades": int(len(g)),
                "unique_stocks": int(g["ts_code"].nunique()),
                "unique_industries": int(g["industry"].nunique()),
                "top1_stock_trade_share": float(stock_counts.iloc[0] / len(g)) if len(g) else np.nan,
                "top10_stock_trade_share": float(stock_counts.head(10).sum() / len(g)) if len(g) else np.nan,
                "top1_industry_trade_share": float(industry_counts.iloc[0] / len(g)) if len(g) else np.nan,
                "top5_industry_trade_share": float(industry_counts.head(5).sum() / len(g)) if len(g) else np.nan,
                "top10_stock_return_sum": float(stock_returns.head(10).sum()) if not stock_returns.empty else np.nan,
                "bottom10_stock_return_sum": float(stock_returns.tail(10).sum()) if not stock_returns.empty else np.nan,
                "top10_stock_positive_contribution_ratio": float(stock_returns.head(10).sum() / total_positive_stock)
                if total_positive_stock and total_positive_stock > 0
                else np.nan,
                "top5_industry_return_sum": float(industry_returns.head(5).sum()) if not industry_returns.empty else np.nan,
                "bottom5_industry_return_sum": float(industry_returns.tail(5).sum()) if not industry_returns.empty else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values(["split", "rule_id"])
    out.to_csv(REPORT_DIR / "batch4C_concentration_summary.csv", index=False)
    return out


def build_profit_concentration(trades: pd.DataFrame) -> pd.DataFrame:
    daily = daily_from_trades(trades, PRIMARY_LABEL, ["rule_id", "split"])
    rows = []
    for keys, g in daily.groupby(["rule_id", "split"], sort=False):
        rule_id, split = keys
        ordered = g.sort_values("daily_return", ascending=False)
        for remove_n in [0, 1, 3, 5, 10, 20]:
            rows.append(summarize_daily(ordered.iloc[remove_n:].copy(), {"rule_id": rule_id, "split": split, "remove_top_winning_days": remove_n}))
    out = pd.DataFrame(rows).sort_values(["split", "rule_id", "remove_top_winning_days"])
    out.to_csv(REPORT_DIR / "batch4C_profit_concentration.csv", index=False)
    return out


def build_execution_risk(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    flags = [
        "entry_st_flag",
        "entry_suspend_flag",
        "entry_limit_up_open_flag",
        "exit_st_flag_5d",
        "exit_suspend_flag_5d",
        "exit_limit_down_close_flag_5d",
        "exit_limit_up_close_flag_5d",
    ]
    for keys, g in trades.groupby(["rule_id", "split"], sort=False):
        rule_id, split = keys
        row = {"rule_id": rule_id, "split": split, "trades": int(len(g))}
        for flag in flags:
            row[f"{flag}_count"] = int(g[flag].sum())
            row[f"{flag}_ratio"] = float(g[flag].mean())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["split", "rule_id"])
    out.to_csv(REPORT_DIR / "batch4C_execution_risk_summary.csv", index=False)
    return out


def build_rule_ranking(
    base: pd.DataFrame,
    cost: pd.DataFrame,
    capacity: pd.DataFrame,
    regime: pd.DataFrame,
    concentration: pd.DataFrame,
    profit_concentration: pd.DataFrame,
) -> pd.DataFrame:
    val = base[base["split"].eq("validation")].set_index("rule_id")
    hold = base[base["split"].eq("research_holdout")].set_index("rule_id")
    cost40 = cost[cost["roundtrip_cost_bps"].eq(PRIMARY_COST_BPS)]
    val_cost = cost40[cost40["split"].eq("validation")].set_index("rule_id")
    hold_cost = cost40[cost40["split"].eq("research_holdout")].set_index("rule_id")
    cap_primary = capacity[
        capacity["reference_amount"].eq("signal_amount")
        & capacity["capital_yuan"].eq(PRIMARY_CAPITAL_YUAN)
        & capacity["participation_cap"].eq(PRIMARY_PARTICIPATION_CAP)
        & capacity["roundtrip_cost_bps"].eq(PRIMARY_COST_BPS)
    ]
    cap_hold = cap_primary[cap_primary["split"].eq("research_holdout")].set_index("rule_id")
    weak = regime[
        regime["roundtrip_cost_bps"].eq(PRIMARY_COST_BPS)
        & regime["regime_type"].eq("trend_regime")
        & regime["regime_value"].eq("weak")
        & regime["split"].eq("research_holdout")
    ].set_index("rule_id")
    pc = profit_concentration[
        profit_concentration["split"].eq("research_holdout") & profit_concentration["remove_top_winning_days"].eq(5)
    ].set_index("rule_id")
    conc = concentration[concentration["split"].eq("research_holdout")].set_index("rule_id")
    rows = []
    for rule_id in sorted(base["rule_id"].unique()):
        row = {"rule_id": rule_id}
        for prefix, table in [
            ("validation_gross", val),
            ("holdout_gross", hold),
            ("validation_cost40", val_cost),
            ("holdout_cost40", hold_cost),
            ("holdout_capacity_1m_5pct_signal_cost40", cap_hold),
            ("holdout_weak_cost40", weak),
            ("holdout_remove5_gross", pc),
        ]:
            if rule_id in table.index:
                r = table.loc[rule_id]
                row[f"{prefix}_mean_daily_return"] = float(r["mean_daily_return"])
                row[f"{prefix}_profit_factor"] = float(r["profit_factor"])
                row[f"{prefix}_max_drawdown"] = float(r["max_drawdown"])
                row[f"{prefix}_cumulative_return"] = float(r["cumulative_return"])
        if rule_id in cap_hold.index:
            row["holdout_capacity_avg_fill_ratio"] = float(cap_hold.loc[rule_id]["avg_fill_ratio"])
            row["holdout_capacity_zero_fill_trades"] = int(cap_hold.loc[rule_id]["zero_fill_trades"])
            row["holdout_capacity_partial_fill_trades"] = int(cap_hold.loc[rule_id]["partial_fill_trades"])
        if rule_id in conc.index:
            row["holdout_top10_stock_trade_share"] = float(conc.loc[rule_id]["top10_stock_trade_share"])
            row["holdout_top5_industry_trade_share"] = float(conc.loc[rule_id]["top5_industry_trade_share"])
        row["passes_basic_robustness_screen"] = (
            row.get("validation_cost40_mean_daily_return", -np.inf) > 0
            and row.get("holdout_cost40_mean_daily_return", -np.inf) > 0
            and row.get("holdout_capacity_avg_fill_ratio", 0) >= 0.80
            and row.get("holdout_weak_cost40_mean_daily_return", -np.inf) > 0
            and row.get("holdout_remove5_gross_mean_daily_return", -np.inf) > 0
            and row.get("holdout_top10_stock_trade_share", 1) <= 0.30
        )
        row["audit_status"] = "candidate_for_deeper_rule_audit" if row["passes_basic_robustness_screen"] else "not_ready"
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(
        ["passes_basic_robustness_screen", "holdout_cost40_profit_factor", "validation_cost40_profit_factor"],
        ascending=[False, False, False],
    )
    out.to_csv(REPORT_DIR / "batch4C_rule_ranking.csv", index=False)
    return out


def write_limitations() -> None:
    lines = [
        "# Batch 4C Limitations",
        "",
        "- This is a simple-rule audit, not a trading strategy and not a model training batch.",
        "- No rules, TopN, labels, horizons, features or model files are changed.",
        "- Returns are still gross or stress-adjusted daily cohort returns, not a full overlapping-capital portfolio ledger.",
        "- `amount` is Tushare daily amount in thousand yuan; capacity calculations multiply it by 1000 to estimate yuan notional.",
        "- `signal_amount` is available at T close and is used as an ex-ante capacity proxy.",
        "- `entry_amount` is the T+1 full-day amount and is used only as post-trade execution-quality audit, not for signal selection.",
        "- Capacity stress uses partial-fill cash drag: unfilled notional earns zero and is not redistributed to other names.",
        "- Cost bps are stress assumptions only; this batch does not encode an official historical tax/commission schedule.",
        "- Entry/exit flags are daily bar proxies and do not prove actual open/close executability.",
        "- Industry classification remains current-snapshot and is not point-in-time clean.",
    ]
    (REPORT_DIR / "batch4C_limitations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_conclusion(
    base: pd.DataFrame,
    capacity: pd.DataFrame,
    regime: pd.DataFrame,
    ranking: pd.DataFrame,
) -> str:
    hold = base[base["split"].eq("research_holdout")].sort_values("mean_daily_return", ascending=False)
    best_hold_rule = str(hold["rule_id"].iloc[0]) if not hold.empty else ""
    best_hold_ret = float(hold["mean_daily_return"].iloc[0]) if not hold.empty else np.nan
    cap_primary = capacity[
        capacity["reference_amount"].eq("signal_amount")
        & capacity["capital_yuan"].eq(PRIMARY_CAPITAL_YUAN)
        & capacity["participation_cap"].eq(PRIMARY_PARTICIPATION_CAP)
        & capacity["roundtrip_cost_bps"].eq(PRIMARY_COST_BPS)
        & capacity["split"].eq("research_holdout")
    ].sort_values("mean_daily_return", ascending=False)
    cap_stress = capacity[
        capacity["reference_amount"].eq("signal_amount")
        & capacity["capital_yuan"].eq(5_000_000)
        & capacity["participation_cap"].eq(0.01)
        & capacity["roundtrip_cost_bps"].eq(PRIMARY_COST_BPS)
        & capacity["split"].eq("research_holdout")
    ].sort_values("mean_daily_return", ascending=False)
    best_cap_rule = str(cap_primary["rule_id"].iloc[0]) if not cap_primary.empty else ""
    best_cap_ret = float(cap_primary["mean_daily_return"].iloc[0]) if not cap_primary.empty else np.nan
    best_cap_fill = float(cap_primary["avg_fill_ratio"].iloc[0]) if not cap_primary.empty else np.nan
    best_stress_rule = str(cap_stress["rule_id"].iloc[0]) if not cap_stress.empty else ""
    best_stress_ret = float(cap_stress["mean_daily_return"].iloc[0]) if not cap_stress.empty else np.nan
    best_stress_fill = float(cap_stress["avg_fill_ratio"].iloc[0]) if not cap_stress.empty else np.nan
    weak = regime[
        regime["split"].eq("research_holdout")
        & regime["roundtrip_cost_bps"].eq(PRIMARY_COST_BPS)
        & regime["regime_type"].eq("trend_regime")
        & regime["regime_value"].eq("weak")
    ].sort_values("mean_daily_return", ascending=False)
    best_weak_rule = str(weak["rule_id"].iloc[0]) if not weak.empty else ""
    best_weak_ret = float(weak["mean_daily_return"].iloc[0]) if not weak.empty else np.nan
    validation_weak = regime[
        regime["split"].eq("validation")
        & regime["roundtrip_cost_bps"].eq(PRIMARY_COST_BPS)
        & regime["regime_type"].eq("trend_regime")
        & regime["regime_value"].eq("weak")
    ].sort_values("mean_daily_return", ascending=False)
    validation_weak_text = (
        f"`{validation_weak['rule_id'].iloc[0]}`, mean daily return `{float(validation_weak['mean_daily_return'].iloc[0]):.6f}`"
        if not validation_weak.empty
        else "not available"
    )
    passed = ranking[ranking["passes_basic_robustness_screen"].astype(bool)]
    gate = "review_required_no_forward_tracking"
    if len(passed) > 0:
        gate = "pass_to_deeper_simple_rule_audit_not_forward_tracking"
    lines = [
        "# Batch 4C Simple Rule Robustness and Capacity Audit",
        "",
        "## Scope",
        "",
        "- Audited Batch3D minimum-to-beat simple rules after Batch4B models failed to beat them.",
        "- No model retraining, rule search, parameter tuning, feature change, label change, horizon change or v7_locked modification was performed.",
        "",
        "## Core Findings",
        "",
        f"- Best research-holdout gross rule: `{best_hold_rule}`, mean daily return `{best_hold_ret:.6f}`.",
        f"- Best research-holdout capacity-stressed rule under signal_amount, 1,000,000 yuan capital, 5% participation cap, 40bps roundtrip cost: `{best_cap_rule}`, mean daily return `{best_cap_ret:.6f}`, avg fill ratio `{best_cap_fill:.4f}`.",
        f"- Under a stricter signal_amount, 5,000,000 yuan capital, 1% participation cap, 40bps roundtrip cost stress, best research-holdout rule: `{best_stress_rule}`, mean daily return `{best_stress_ret:.6f}`, avg fill ratio `{best_stress_fill:.4f}`.",
        f"- Research-holdout has no `weak` trend-regime days, so weak-market robustness cannot be validated in the most recent holdout. Validation weak-regime best rule: {validation_weak_text}.",
        f"- Basic robustness screen pass count: `{len(passed)}` of `{ranking['rule_id'].nunique()}` rules.",
        "",
        "## Interpretation",
        "",
        "- Simple rules are stronger than Batch4B models under gross cohort comparison, but their edge is still highly exposed to small-cap / low-liquidity implementation risk.",
        "- Capacity is acceptable for small capital under 5% daily-amount participation, but stricter 1% participation at 5,000,000 yuan materially reduces fill ratio for the lower-liquidity rules.",
        "- Weak-market and profit-concentration diagnostics must be treated as gating evidence before any forward paper tracking.",
        "- Passing this audit would only justify a deeper simple-rule audit; it is not live readiness and not permission to resume complex model training.",
        "",
        "## Gate",
        "",
        f"- Gate result: `{gate}`.",
        "- If accepted, the next step should be a deeper simple-rule audit with a full overlapping-capital ledger, execution assumptions and point-in-time industry caveat review.",
        "- Do not enter model Batch5A for the current Batch4B models.",
    ]
    (REPORT_DIR / "batch4C_conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return gate


def update_handoff(gate: str) -> None:
    text = "\n".join(
        [
            "# Handoff: Batch 4C Simple Rule Audit Completed",
            "",
            "## Completed Batch",
            "",
            "- Completed: `Batch 4C simple-rule robustness and capacity audit`.",
            "- Output directory: `reports/tushare/v9_swing_research/batch4C_simple_rule_robustness_capacity/`.",
            "- Scope: costs, capacity, market regime, yearly stability, concentration and execution-risk proxies for Batch3D minimum-to-beat simple rules.",
            "",
            "## Gate",
            "",
            f"- Gate result: `{gate}`.",
            "",
            "## Required Inputs For Next Step",
            "",
            "- `batch4C_rule_ranking.csv`",
            "- `batch4C_capacity_matrix.csv`",
            "- `batch4C_regime_robustness.csv`",
            "- `batch4C_profit_concentration.csv`",
            "- `batch4C_concentration_summary.csv`",
            "- `batch4C_conclusion.md`",
            "",
            "## Next Step",
            "",
            "- Only consider a deeper simple-rule audit if the user accepts the robustness/capacity evidence.",
            "- Do not resume complex model training or Batch5A for current Batch4B models.",
            "- Do not start forward tracking until a full overlapping-capital ledger and execution assumptions are frozen.",
        ]
    )
    GLOBAL_HANDOFF_PATH.write_text(text + "\n", encoding="utf-8")


def update_stage_status(gate: str) -> None:
    if not STAGE_STATUS_PATH.exists():
        return
    df = pd.read_csv(STAGE_STATUS_PATH)
    row = {
        "stage": "Stage 7 Simple Rule Robustness Capacity",
        "batch": "batch4C_simple_rule_robustness_capacity",
        "status": "completed",
        "output_dir": "reports/tushare/v9_swing_research/batch4C_simple_rule_robustness_capacity",
        "gate_result": "review" if gate.startswith("pass") else "blocked",
        "next_action": "Review simple-rule cost/capacity/regime evidence; do not start forward tracking without a full overlapping-capital ledger.",
    }
    df = df[~df["batch"].eq("batch4C_simple_rule_robustness_capacity")]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(STAGE_STATUS_PATH, index=False)


def update_roadmap(gate: str) -> None:
    if not ROADMAP_PATH.exists():
        return
    text = ROADMAP_PATH.read_text(encoding="utf-8")
    addition = """### Stage 7D Batch 4C Simple Rule Robustness and Capacity

输出目录：`reports/tushare/v9_swing_research/batch4C_simple_rule_robustness_capacity/`

执行边界：

- 不训练模型、不调参、不新增规则、不修改特征/标签/horizon/TopN。
- 只审计 Batch3D minimum-to-beat 简单规则。
- 输出成本敏感性、容量矩阵、市场状态、年度稳定性、收益集中度、行业/个股集中度和执行风险代理。

结论：

- 简单规则虽然强于 Batch4B 轻量模型，但需要接受成本、容量、弱市和集中度审计后，才可能进入更深的纸面流程设计。
- 当前仍不建议恢复复杂模型训练，也不建议直接 forward tracking。

"""
    marker = "### Step 5A: 鲁棒性和暴露审计"
    if addition.strip() not in text:
        text = text.replace(marker, addition + marker)
    text = text.replace(
        "> Batch 4B Review completed; do not enter Batch 5A with current lightweight models",
        "> Batch 4C completed; review simple-rule robustness/capacity before any deeper audit",
    )
    ROADMAP_PATH.write_text(text, encoding="utf-8")


def write_hashes() -> None:
    rows = []
    for path in sorted(REPORT_DIR.glob("batch4C_*")):
        if path.name == "batch4C_file_sha256.csv" or not path.is_file():
            continue
        rows.append({"file": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    for path in [GLOBAL_HANDOFF_PATH, ROADMAP_PATH, STAGE_STATUS_PATH]:
        if path.exists():
            rows.append({"file": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    with (REPORT_DIR / "batch4C_file_sha256.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "sha256", "size_bytes"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    minimum_rules = read_minimum_rules()
    print({"event": "start", "out_dir": str(REPORT_DIR), "rules": minimum_rules}, flush=True)
    selected = load_or_reconstruct_trades(minimum_rules)
    selected = add_execution_context(selected)
    print({"event": "selected_ready", "rows": int(len(selected)), "rules": int(selected["rule_id"].nunique())}, flush=True)

    base = read_cached_csv("batch4C_rule_robustness_summary.csv")
    if base is None:
        base = build_base_summary(selected)
    cost = read_cached_csv("batch4C_cost_sensitivity.csv")
    if cost is None:
        cost = build_cost_sensitivity(selected)
    capacity = read_cached_csv("batch4C_capacity_matrix.csv")
    if capacity is None:
        capacity = build_capacity_matrix(selected)
    regime = read_cached_csv("batch4C_regime_robustness.csv")
    if regime is None:
        regime = build_regime_robustness(selected)
    yearly = read_cached_csv("batch4C_yearly_robustness.csv")
    if yearly is None:
        yearly = build_yearly_robustness(selected)
    concentration = read_cached_csv("batch4C_concentration_summary.csv")
    if concentration is None:
        concentration = build_concentration(selected)
    profit_concentration = read_cached_csv("batch4C_profit_concentration.csv")
    if profit_concentration is None:
        profit_concentration = build_profit_concentration(selected)
    execution_risk = read_cached_csv("batch4C_execution_risk_summary.csv")
    if execution_risk is None:
        build_execution_risk(selected)
    ranking = build_rule_ranking(base, cost, capacity, regime, concentration, profit_concentration)
    write_limitations()
    gate = write_conclusion(base, capacity, regime, ranking)
    update_handoff(gate)
    update_stage_status(gate)
    update_roadmap(gate)
    write_hashes()
    print({"event": "done", "out_dir": str(REPORT_DIR), "gate_result": gate}, flush=True)


if __name__ == "__main__":
    main()
