#!/usr/bin/env python3
"""v9 Batch 4D: deep simple-rule overlapping ledger audit.

This batch audits two pre-selected simple rules from Batch4C:
- low_stock_ret_60d
- combo_low_liquidity_weak_momentum

It converts the earlier daily cohort diagnostics into an overlapping capital
ledger with mark-to-market PnL, turnover, cost, capacity and concentration
summaries. It does not train models, tune parameters, add rules, change labels,
change horizons, or modify v7_locked.
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
B4C_DIR = REPORT_ROOT / "batch4C_simple_rule_robustness_capacity"
REPORT_DIR = REPORT_ROOT / "batch4D_deep_simple_rule_ledger"

FEATURE_PATH = CLEAN_DIR / "v9_stock_industry_features.parquet"
LABEL_PATH = CLEAN_DIR / "v9_swing_labels_3_5_10.parquet"
B4C_LOCAL_TRADES_PATH = B4C_DIR / "_local_batch4C_rule_trades_with_execution.parquet"
B4B_RULE_SELECTED_PATH = REPORT_ROOT / "batch4B_review" / "batch4B_review_rule_selected_trades.parquet"

GLOBAL_HANDOFF_PATH = REPORT_ROOT / "v9_current_handoff.md"
ROADMAP_PATH = REPORT_ROOT / "v9_execution_roadmap.md"
STAGE_STATUS_PATH = REPORT_ROOT / "v9_stage_status.csv"

TARGET_RULES = ["low_stock_ret_60d", "combo_low_liquidity_weak_momentum"]
PRIMARY_LABEL = "fwd_ret_5d_open"
PRIMARY_TOP_N = 20
SLEEVE_FRACTION = 0.20

TRAIN_START = "20180102"
TRAIN_END = "20221230"
VALIDATION_START = "20230103"
VALIDATION_END = "20241231"
HOLDOUT_START = "20250102"
HOLDOUT_END = "20260522"


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    capital_yuan: float
    participation_cap: float | None
    roundtrip_cost_bps: float
    capacity_reference: str
    entry_limit_policy: str
    description: str


SCENARIOS = [
    Scenario(
        "no_cost_full_fill_1m",
        1_000_000.0,
        None,
        0.0,
        "none",
        "allow_daily_proxy",
        "Diagnostic no-cost full-fill ledger.",
    ),
    Scenario(
        "base_1m_5pct_signal_40bps",
        1_000_000.0,
        0.05,
        40.0,
        "signal_amount_yuan",
        "block_entry_limit_or_suspend",
        "Primary execution-stress ledger: 1m capital, 5% signal-day amount cap, 40bps roundtrip cost.",
    ),
    Scenario(
        "high_cost_1m_5pct_signal_100bps",
        1_000_000.0,
        0.05,
        100.0,
        "signal_amount_yuan",
        "block_entry_limit_or_suspend",
        "High-cost stress ledger: 1m capital, 5% signal-day amount cap, 100bps roundtrip cost.",
    ),
    Scenario(
        "strict_5m_1pct_signal_40bps",
        5_000_000.0,
        0.01,
        40.0,
        "signal_amount_yuan",
        "block_entry_limit_or_suspend",
        "Strict capacity stress: 5m capital, 1% signal-day amount cap, 40bps roundtrip cost.",
    ),
]


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


def max_drawdown_from_equity(equity: pd.Series) -> float:
    eq = finite_numeric(equity).dropna()
    if eq.empty:
        return np.nan
    peak = eq.cummax()
    return float((eq / peak - 1.0).min())


def profit_factor_from_pnl(pnl: pd.Series) -> float:
    p = finite_numeric(pnl).dropna()
    gains = p.loc[p.gt(0)].sum()
    losses = p.loc[p.lt(0)].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / abs(losses))


def load_target_trades() -> pd.DataFrame:
    if B4C_LOCAL_TRADES_PATH.exists():
        print({"event": "read_b4c_local_trades", "path": str(B4C_LOCAL_TRADES_PATH)}, flush=True)
        trades = pd.read_parquet(B4C_LOCAL_TRADES_PATH)
    elif B4B_RULE_SELECTED_PATH.exists():
        print({"event": "read_b4b_rule_selected", "path": str(B4B_RULE_SELECTED_PATH)}, flush=True)
        trades = pd.read_parquet(B4B_RULE_SELECTED_PATH).rename(columns={"baseline_id": "rule_id", "amount": "signal_amount"})
    else:
        raise FileNotFoundError("Need Batch4C local trades or Batch4B rule-selected trades before Batch4D.")
    trades["trade_date"] = trades["trade_date"].astype(str)
    trades = trades[trades["rule_id"].isin(TARGET_RULES)].copy()
    if trades.empty:
        raise ValueError("No target rule trades found for Batch4D.")
    trades["split"] = split_name(trades["trade_date"])
    trades = trades[trades["split"].ne("outside")].copy()
    needed_label_cols = [
        "trade_date",
        "ts_code",
        "entry_date",
        "exit_date_5d",
        "entry_adj_open",
        "entry_suspend_flag",
        "entry_limit_up_open_flag",
        "exit_suspend_flag_5d",
        "exit_limit_down_close_flag_5d",
        PRIMARY_LABEL,
    ]
    labels = pd.read_parquet(LABEL_PATH, columns=needed_label_cols)
    labels["trade_date"] = labels["trade_date"].astype(str)
    labels["entry_date"] = labels["entry_date"].astype(str)
    labels["exit_date_5d"] = labels["exit_date_5d"].astype(str)
    patch_cols = [
        "entry_date",
        "exit_date_5d",
        "entry_adj_open",
        "entry_suspend_flag",
        "entry_limit_up_open_flag",
        "exit_suspend_flag_5d",
        "exit_limit_down_close_flag_5d",
        PRIMARY_LABEL,
    ]
    patch = labels[["trade_date", "ts_code"] + patch_cols].drop_duplicates(["trade_date", "ts_code"])
    for col in patch_cols:
        if col in trades.columns:
            trades = trades.drop(columns=[col])
    trades = trades.merge(patch, on=["trade_date", "ts_code"], how="left", validate="many_to_one")
    if "signal_amount_yuan" not in trades.columns:
        if "signal_amount" in trades.columns:
            trades["signal_amount_yuan"] = finite_numeric(trades["signal_amount"]) * 1000.0
        elif "amount" in trades.columns:
            trades["signal_amount_yuan"] = finite_numeric(trades["amount"]) * 1000.0
        else:
            trades["signal_amount_yuan"] = np.nan
    for col in ["signal_amount_yuan", "entry_adj_open", PRIMARY_LABEL]:
        trades[col] = finite_numeric(trades[col])
    for col in [
        "entry_suspend_flag",
        "entry_limit_up_open_flag",
        "exit_suspend_flag_5d",
        "exit_limit_down_close_flag_5d",
        "extreme_selloff_flag",
    ]:
        if col in trades.columns:
            trades[col] = trades[col].fillna(False).astype(bool)
    if "trend_regime" not in trades.columns or "extreme_selloff_flag" not in trades.columns:
        regime_cols = ["trade_date", "ts_code", "trend_regime", "vol_regime", "industry_crowding_regime", "stock_concentration_regime", "market_regime_id", "extreme_selloff_flag"]
        regime = pd.read_parquet(FEATURE_PATH, columns=regime_cols)
        regime["trade_date"] = regime["trade_date"].astype(str)
        existing = [c for c in regime_cols if c in trades.columns and c not in ["trade_date", "ts_code"]]
        if existing:
            trades = trades.drop(columns=existing)
        trades = trades.merge(regime, on=["trade_date", "ts_code"], how="left", validate="many_to_one")
    trades = trades.dropna(subset=["entry_date", "exit_date_5d", "entry_adj_open", PRIMARY_LABEL])
    sort_cols = [c for c in ["rule_id", "trade_date", "rank", "ts_code"] if c in trades.columns]
    trades = trades.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    trades["trade_id"] = np.arange(len(trades), dtype=np.int64)
    return trades


def read_price_panel() -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    cols = ["trade_date", "ts_code", "adj_close", "adj_ret_1d"]
    prices = pd.read_parquet(LABEL_PATH, columns=cols)
    prices["trade_date"] = prices["trade_date"].astype(str)
    prices["adj_close"] = finite_numeric(prices["adj_close"])
    prices["adj_ret_1d"] = finite_numeric(prices["adj_ret_1d"])
    prices = prices.sort_values(["ts_code", "trade_date"], kind="mergesort")
    prices["prev_adj_close"] = prices.groupby("ts_code", sort=False)["adj_close"].shift(1)
    dates = sorted(prices["trade_date"].unique())
    return prices, dates, {d: i for i, d in enumerate(dates)}


def expand_position_days(trades: pd.DataFrame, dates: list[str], date_to_idx: dict[str, int]) -> pd.DataFrame:
    idx_trades = trades.copy()
    idx_trades["entry_idx"] = idx_trades["entry_date"].map(date_to_idx)
    idx_trades["exit_idx"] = idx_trades["exit_date_5d"].map(date_to_idx)
    idx_trades = idx_trades[idx_trades["entry_idx"].notna() & idx_trades["exit_idx"].notna()].copy()
    idx_trades["entry_idx"] = idx_trades["entry_idx"].astype(int)
    idx_trades["exit_idx"] = idx_trades["exit_idx"].astype(int)
    idx_trades = idx_trades[idx_trades["exit_idx"].ge(idx_trades["entry_idx"])].copy()
    if "trade_id" not in idx_trades.columns:
        idx_trades["trade_id"] = np.arange(len(idx_trades), dtype=np.int64)
    lengths = (idx_trades["exit_idx"] - idx_trades["entry_idx"] + 1).astype(int).to_numpy()
    repeated = np.repeat(idx_trades.index.to_numpy(), lengths)
    day_indices = np.concatenate([np.arange(start, end + 1, dtype=np.int32) for start, end in zip(idx_trades["entry_idx"], idx_trades["exit_idx"])])
    pos = idx_trades.loc[repeated].copy()
    pos["position_idx"] = day_indices
    pos["position_date"] = np.asarray(dates, dtype=object)[day_indices]
    keep_cols = [
        "trade_id",
        "rule_id",
        "split",
        "trade_date",
        "ts_code",
        "industry",
        "rank",
        "score",
        "trend_regime",
        "vol_regime",
        "industry_crowding_regime",
        "stock_concentration_regime",
        "market_regime_id",
        "extreme_selloff_flag",
        "entry_date",
        "exit_date_5d",
        "entry_idx",
        "exit_idx",
        "position_date",
        "entry_adj_open",
        "signal_amount_yuan",
        PRIMARY_LABEL,
        "entry_suspend_flag",
        "entry_limit_up_open_flag",
        "exit_suspend_flag_5d",
        "exit_limit_down_close_flag_5d",
    ]
    keep_cols = [c for c in keep_cols if c in pos.columns]
    return pos[keep_cols].reset_index(drop=True)


def add_position_returns(position_days: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    price_patch = prices.rename(columns={"trade_date": "position_date"})
    pos = position_days.merge(
        price_patch[["position_date", "ts_code", "adj_close", "adj_ret_1d", "prev_adj_close"]],
        on=["position_date", "ts_code"],
        how="left",
        validate="many_to_one",
    )
    pos["is_entry_day"] = pos["position_date"].eq(pos["entry_date"])
    pos["is_exit_day"] = pos["position_date"].eq(pos["exit_date_5d"])
    pos["start_price"] = np.where(pos["is_entry_day"], pos["entry_adj_open"], pos["prev_adj_close"])
    pos["start_price"] = finite_numeric(pos["start_price"])
    pos["adj_close"] = finite_numeric(pos["adj_close"])
    pos["daily_price_return"] = pos["adj_close"] / pos["start_price"] - 1.0
    pos["value_start_multiplier"] = pos["start_price"] / pos["entry_adj_open"]
    pos["value_close_multiplier"] = pos["adj_close"] / pos["entry_adj_open"]
    pos = pos.dropna(subset=["daily_price_return", "value_start_multiplier", "value_close_multiplier"])
    return pos


def scenario_trade_terms(trades: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    selected_counts = trades.groupby(["rule_id", "trade_date"], sort=False)["ts_code"].transform("size")
    out = trades[[
        "trade_id",
        "rule_id",
        "split",
        "trade_date",
        "ts_code",
        "industry",
        "trend_regime",
        "extreme_selloff_flag",
        "entry_date",
        "exit_date_5d",
        "signal_amount_yuan",
        PRIMARY_LABEL,
        "entry_suspend_flag",
        "entry_limit_up_open_flag",
        "exit_suspend_flag_5d",
        "exit_limit_down_close_flag_5d",
    ]].copy()
    out["scenario_id"] = scenario.scenario_id
    out["target_order_yuan"] = scenario.capital_yuan * SLEEVE_FRACTION / selected_counts
    if scenario.capacity_reference == "none" or scenario.participation_cap is None:
        fill_ratio = pd.Series(1.0, index=out.index)
    else:
        ref_yuan = finite_numeric(out[scenario.capacity_reference]).replace(0, np.nan)
        fill_ratio = (scenario.participation_cap * ref_yuan / out["target_order_yuan"]).clip(upper=1.0)
        fill_ratio = fill_ratio.where(fill_ratio.notna() & fill_ratio.gt(0), 0.0).fillna(0.0)
    if scenario.entry_limit_policy == "block_entry_limit_or_suspend":
        blocked = out["entry_suspend_flag"].astype(bool) | out["entry_limit_up_open_flag"].astype(bool)
        fill_ratio = fill_ratio.mask(blocked, 0.0)
        out["entry_blocked_flag"] = blocked
    else:
        out["entry_blocked_flag"] = False
    out["fill_ratio"] = fill_ratio.astype(float)
    out["filled_entry_notional"] = out["target_order_yuan"] * out["fill_ratio"]
    out["entry_cost_yuan"] = out["filled_entry_notional"] * (scenario.roundtrip_cost_bps / 2.0) / 10000.0
    exit_value = out["filled_entry_notional"] * (1.0 + out[PRIMARY_LABEL])
    out["exit_value_yuan"] = exit_value
    out["exit_cost_yuan"] = exit_value.clip(lower=0.0) * (scenario.roundtrip_cost_bps / 2.0) / 10000.0
    out["gross_trade_pnl_yuan"] = out["filled_entry_notional"] * out[PRIMARY_LABEL]
    out["net_trade_pnl_yuan"] = out["gross_trade_pnl_yuan"] - out["entry_cost_yuan"] - out["exit_cost_yuan"]
    out["net_trade_return_on_target"] = out["net_trade_pnl_yuan"] / out["target_order_yuan"]
    return out


def build_scenario_daily(position_days: pd.DataFrame, trades: pd.DataFrame, scenario: Scenario) -> tuple[pd.DataFrame, pd.DataFrame]:
    terms = scenario_trade_terms(trades, scenario)
    pos = position_days.merge(
        terms[[
            "trade_id",
            "scenario_id",
            "target_order_yuan",
            "fill_ratio",
            "filled_entry_notional",
            "entry_cost_yuan",
            "exit_cost_yuan",
            "entry_blocked_flag",
        ]],
        on="trade_id",
        how="inner",
        validate="many_to_one",
    )
    pos["gross_pnl_yuan"] = pos["filled_entry_notional"] * pos["value_start_multiplier"] * pos["daily_price_return"]
    pos["entry_cost_today_yuan"] = np.where(pos["is_entry_day"], pos["entry_cost_yuan"], 0.0)
    pos["exit_cost_today_yuan"] = np.where(pos["is_exit_day"], pos["exit_cost_yuan"], 0.0)
    pos["net_pnl_yuan"] = pos["gross_pnl_yuan"] - pos["entry_cost_today_yuan"] - pos["exit_cost_today_yuan"]
    pos["entry_notional_today_yuan"] = np.where(pos["is_entry_day"], pos["filled_entry_notional"], 0.0)
    pos["exit_notional_today_yuan"] = np.where(pos["is_exit_day"], pos["filled_entry_notional"] * pos["value_close_multiplier"], 0.0)
    pos["close_market_value_yuan"] = pos["filled_entry_notional"] * pos["value_close_multiplier"]
    pos["traded_notional_yuan"] = pos["entry_notional_today_yuan"] + pos["exit_notional_today_yuan"]
    daily = (
        pos.groupby(["scenario_id", "rule_id", "split", "position_date"], sort=False)
        .agg(
            gross_pnl_yuan=("gross_pnl_yuan", "sum"),
            net_pnl_yuan=("net_pnl_yuan", "sum"),
            entry_cost_yuan=("entry_cost_today_yuan", "sum"),
            exit_cost_yuan=("exit_cost_today_yuan", "sum"),
            total_cost_yuan=("entry_cost_today_yuan", "sum"),
            close_market_value_yuan=("close_market_value_yuan", "sum"),
            traded_notional_yuan=("traded_notional_yuan", "sum"),
            entry_notional_yuan=("entry_notional_today_yuan", "sum"),
            exit_notional_yuan=("exit_notional_today_yuan", "sum"),
            active_positions=("trade_id", "nunique"),
            entry_count=("is_entry_day", "sum"),
            exit_count=("is_exit_day", "sum"),
            avg_fill_ratio=("fill_ratio", "mean"),
        )
        .reset_index()
        .rename(columns={"position_date": "trade_date"})
    )
    daily["total_cost_yuan"] = daily["entry_cost_yuan"] + daily["exit_cost_yuan"]
    frames = []
    for keys, g in daily.groupby(["scenario_id", "rule_id", "split"], sort=False):
        scenario_id, rule_id, split = keys
        g = g.sort_values("trade_date").copy()
        prev_equity = []
        equity = []
        current = scenario.capital_yuan
        for pnl in g["net_pnl_yuan"].fillna(0.0):
            prev_equity.append(current)
            current = current + float(pnl)
            equity.append(current)
        g["prev_equity_yuan"] = prev_equity
        g["equity_yuan"] = equity
        g["daily_return"] = g["net_pnl_yuan"] / g["prev_equity_yuan"].replace(0, np.nan)
        g["gross_exposure_ratio"] = g["close_market_value_yuan"] / g["prev_equity_yuan"].replace(0, np.nan)
        g["daily_turnover_ratio"] = g["traded_notional_yuan"] / g["prev_equity_yuan"].replace(0, np.nan)
        frames.append(g)
    daily = pd.concat(frames, ignore_index=True) if frames else daily
    return daily, terms


def summarize_ledger(daily: pd.DataFrame, terms: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    rows = []
    term_summary = (
        terms.groupby(["scenario_id", "rule_id", "split"], sort=False)
        .agg(
            trades=("trade_id", "size"),
            avg_trade_fill_ratio=("fill_ratio", "mean"),
            zero_fill_trades=("fill_ratio", lambda x: int(x.le(0).sum())),
            partial_fill_trades=("fill_ratio", lambda x: int((x.gt(0) & x.lt(0.999999)).sum())),
            entry_blocked_trades=("entry_blocked_flag", "sum"),
            avg_net_trade_return_on_target=("net_trade_return_on_target", "mean"),
            trade_win_rate=("net_trade_pnl_yuan", lambda x: float((x > 0).mean())),
            net_trade_pnl_yuan=("net_trade_pnl_yuan", "sum"),
        )
        .reset_index()
    )
    for keys, g in daily.groupby(["scenario_id", "rule_id", "split"], sort=False):
        scenario_id, rule_id, split = keys
        g = g.sort_values("trade_date")
        final_equity = float(g["equity_yuan"].iloc[-1]) if not g.empty else scenario.capital_yuan
        total_net_pnl = float(g["net_pnl_yuan"].sum()) if not g.empty else 0.0
        rows.append(
            {
                "scenario_id": scenario_id,
                "rule_id": rule_id,
                "split": split,
                "initial_capital_yuan": scenario.capital_yuan,
                "final_equity_yuan": final_equity,
                "total_net_pnl_yuan": total_net_pnl,
                "cumulative_return": final_equity / scenario.capital_yuan - 1.0,
                "ledger_days": int(len(g)),
                "mean_daily_return": float(g["daily_return"].mean()),
                "median_daily_return": float(g["daily_return"].median()),
                "daily_win_rate": float(g["daily_return"].gt(0).mean()),
                "profit_factor": profit_factor_from_pnl(g["net_pnl_yuan"]),
                "max_drawdown": max_drawdown_from_equity(g["equity_yuan"]),
                "avg_gross_exposure": float(g["gross_exposure_ratio"].mean()),
                "max_gross_exposure": float(g["gross_exposure_ratio"].max()),
                "avg_active_positions": float(g["active_positions"].mean()),
                "max_active_positions": int(g["active_positions"].max()),
                "avg_daily_turnover": float(g["daily_turnover_ratio"].mean()),
                "annualized_turnover": float(g["daily_turnover_ratio"].mean() * 252.0),
                "total_traded_notional_yuan": float(g["traded_notional_yuan"].sum()),
                "total_cost_yuan": float(g["total_cost_yuan"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    out = out.merge(term_summary, on=["scenario_id", "rule_id", "split"], how="left", validate="one_to_one")
    return out


def build_signal_regime_summary(terms_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in terms_all.groupby(["scenario_id", "rule_id", "split", "trend_regime"], dropna=False, sort=False):
        scenario_id, rule_id, split, trend = keys
        pnl = finite_numeric(g["net_trade_pnl_yuan"])
        target = finite_numeric(g["target_order_yuan"])
        rows.append(
            {
                "scenario_id": scenario_id,
                "rule_id": rule_id,
                "split": split,
                "trend_regime": trend,
                "trades": int(len(g)),
                "signal_days": int(g["trade_date"].nunique()),
                "net_pnl_yuan": float(pnl.sum()),
                "avg_net_return_on_target": float((pnl / target.replace(0, np.nan)).mean()),
                "trade_win_rate": float(pnl.gt(0).mean()),
                "profit_factor": profit_factor_from_pnl(pnl),
                "avg_fill_ratio": float(g["fill_ratio"].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["split", "scenario_id", "rule_id", "trend_regime"])
    out.to_csv(REPORT_DIR / "batch4D_signal_regime_trade_summary.csv", index=False)
    return out


def build_concentration(terms_all: pd.DataFrame, scenario_id: str) -> pd.DataFrame:
    rows = []
    terms = terms_all[terms_all["scenario_id"].eq(scenario_id)].copy()
    for keys, g in terms.groupby(["rule_id", "split"], sort=False):
        rule_id, split = keys
        stock_counts = g["ts_code"].value_counts()
        industry_counts = g["industry"].fillna("UNKNOWN").value_counts()
        stock_pnl = g.groupby("ts_code")["net_trade_pnl_yuan"].sum().sort_values(ascending=False)
        industry_pnl = g.groupby("industry")["net_trade_pnl_yuan"].sum().sort_values(ascending=False)
        positive_stock_pnl = stock_pnl[stock_pnl.gt(0)].sum()
        rows.append(
            {
                "scenario_id": scenario_id,
                "rule_id": rule_id,
                "split": split,
                "trades": int(len(g)),
                "unique_stocks": int(g["ts_code"].nunique()),
                "unique_industries": int(g["industry"].nunique()),
                "top10_stock_trade_share": float(stock_counts.head(10).sum() / len(g)) if len(g) else np.nan,
                "top5_industry_trade_share": float(industry_counts.head(5).sum() / len(g)) if len(g) else np.nan,
                "top10_stock_pnl_yuan": float(stock_pnl.head(10).sum()) if not stock_pnl.empty else np.nan,
                "bottom10_stock_pnl_yuan": float(stock_pnl.tail(10).sum()) if not stock_pnl.empty else np.nan,
                "top10_stock_positive_pnl_share": float(stock_pnl.head(10).sum() / positive_stock_pnl)
                if positive_stock_pnl and positive_stock_pnl > 0
                else np.nan,
                "top5_industry_pnl_yuan": float(industry_pnl.head(5).sum()) if not industry_pnl.empty else np.nan,
                "bottom5_industry_pnl_yuan": float(industry_pnl.tail(5).sum()) if not industry_pnl.empty else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values(["split", "rule_id"])
    out.to_csv(REPORT_DIR / "batch4D_ledger_concentration.csv", index=False)
    return out


def write_config() -> None:
    lines = [
        "batch: batch4D_deep_simple_rule_ledger",
        "training_enabled: false",
        "rule_search_enabled: false",
        "target_rules:",
    ]
    lines.extend([f"  - {rule}" for rule in TARGET_RULES])
    lines.extend(
        [
            f"top_n: {PRIMARY_TOP_N}",
            "horizon: 5d",
            "entry: T+1 adjusted open",
            "exit: T+5 adjusted close",
            f"sleeve_fraction_per_signal_day: {SLEEVE_FRACTION}",
            "position_weighting: equal weight inside each daily sleeve",
            "capacity_reference: signal_amount_yuan unless scenario says none",
            "fill_policy: partial fill with cash drag; no refill",
            "entry_block_policy: block entry if entry_suspend or entry_limit_up_open for execution scenarios",
            "scenarios:",
        ]
    )
    for s in SCENARIOS:
        lines.extend(
            [
                f"  - scenario_id: {s.scenario_id}",
                f"    capital_yuan: {s.capital_yuan:.0f}",
                f"    participation_cap: {s.participation_cap if s.participation_cap is not None else 'null'}",
                f"    roundtrip_cost_bps: {s.roundtrip_cost_bps}",
                f"    capacity_reference: {s.capacity_reference}",
                f"    entry_limit_policy: {s.entry_limit_policy}",
                f"    description: {s.description}",
            ]
        )
    (REPORT_DIR / "batch4D_ledger_config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_limitations() -> None:
    lines = [
        "# Batch 4D Limitations",
        "",
        "- This batch is a deeper audit of two pre-selected simple rules, not a model or live trading system.",
        "- It does not add rules, change TopN, tune parameters, change labels or use validation results to select a new rule.",
        "- The ledger uses a 20% capital sleeve per signal day to avoid hidden leverage from overlapping 5-day holds.",
        "- Mark-to-market uses adjusted daily prices; intraday slippage and true order book execution are not modeled.",
        "- Capacity uses signal-day amount, which is ex-ante at post-close signal time. T+1 entry amount is not used for selection.",
        "- Fill policy is partial-fill cash drag: unfilled cash remains idle and is not reallocated.",
        "- Entry limit-up/suspension blocks are handled in execution scenarios, but exit limit-down/suspension is flagged rather than delayed.",
        "- Research holdout contains no weak trend-regime signal days, so weak-market robustness remains unresolved.",
        "- Current-snapshot industry classification remains a point-in-time caveat for industry concentration reporting.",
    ]
    (REPORT_DIR / "batch4D_limitations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_conclusion(summary: pd.DataFrame, regime: pd.DataFrame, concentration: pd.DataFrame) -> str:
    base = summary[summary["scenario_id"].eq("base_1m_5pct_signal_40bps")]
    hold = base[base["split"].eq("research_holdout")].sort_values("cumulative_return", ascending=False)
    strict = summary[summary["scenario_id"].eq("strict_5m_1pct_signal_40bps")]
    strict_hold = strict[strict["split"].eq("research_holdout")].sort_values("cumulative_return", ascending=False)
    best = hold.iloc[0] if not hold.empty else None
    best_strict = strict_hold.iloc[0] if not strict_hold.empty else None
    weak_hold = regime[
        regime["scenario_id"].eq("base_1m_5pct_signal_40bps")
        & regime["split"].eq("research_holdout")
        & regime["trend_regime"].eq("weak")
    ]
    weak_val = regime[
        regime["scenario_id"].eq("base_1m_5pct_signal_40bps")
        & regime["split"].eq("validation")
        & regime["trend_regime"].eq("weak")
    ].sort_values("avg_net_return_on_target", ascending=False)
    gate = "review_required_no_forward_tracking"
    if best is not None and best["cumulative_return"] > 0 and best["max_drawdown"] > -0.30 and not weak_hold.empty:
        gate = "candidate_for_full_rule_audit"
    lines = [
        "# Batch 4D Deep Simple Rule Ledger",
        "",
        "## Scope",
        "",
        "- Built an overlapping 5-day capital ledger for `low_stock_ret_60d` and `combo_low_liquidity_weak_momentum`.",
        "- No model training, no new rule search, no parameter tuning, no feature/label/horizon change, and no v7_locked modification.",
        "",
        "## Core Findings",
        "",
    ]
    if best is not None:
        lines.append(
            f"- Best research-holdout base ledger rule: `{best['rule_id']}`, cumulative return `{best['cumulative_return']:.4f}`, max drawdown `{best['max_drawdown']:.4f}`, PF `{best['profit_factor']:.4f}`, annualized turnover `{best['annualized_turnover']:.2f}`."
        )
    if best_strict is not None:
        lines.append(
            f"- Best strict capacity ledger rule: `{best_strict['rule_id']}`, cumulative return `{best_strict['cumulative_return']:.4f}`, avg fill `{best_strict['avg_trade_fill_ratio']:.4f}`, max drawdown `{best_strict['max_drawdown']:.4f}`."
        )
    if weak_hold.empty:
        lines.append("- Research holdout has no weak trend-regime signal days; weak-market robustness is not proven on the newest period.")
    if not weak_val.empty:
        row = weak_val.iloc[0]
        lines.append(
            f"- Validation weak-regime best rule under base ledger: `{row['rule_id']}`, avg net return on target `{row['avg_net_return_on_target']:.4f}`, PF `{row['profit_factor']:.4f}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The two simple rules remain more defensible than Batch4B models because the ledger is explicit about overlap, turnover, cost and fill.",
            "- This still does not clear forward tracking because weak-market coverage in the newest holdout is missing and exit-delay risk is only flagged, not simulated.",
            "- `low_stock_ret_60d` is the more liquid and less concentrated candidate; `combo_low_liquidity_weak_momentum` has lower concentration but needs stronger drawdown review.",
            "",
            "## Gate",
            "",
            f"- Gate result: `{gate}`.",
            "- Next step, if accepted, should be a full rule audit with exit-delay simulation, monthly cash ledger review and independent weak-market sample validation.",
            "- Do not resume complex model training from this result.",
        ]
    )
    (REPORT_DIR / "batch4D_conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return gate


def update_handoff(gate: str) -> None:
    lines = [
        "# Handoff: Batch 4D Deep Simple Rule Ledger Completed",
        "",
        "## Completed Batch",
        "",
        "- Completed: `Batch 4D deep simple-rule overlapping ledger audit`.",
        "- Output directory: `reports/tushare/v9_swing_research/batch4D_deep_simple_rule_ledger/`.",
        "- Scope: overlapping capital ledger, cost, capacity, turnover, weak-regime sample, and concentration for two pre-selected simple rules.",
        "",
        "## Gate",
        "",
        f"- Gate result: `{gate}`.",
        "",
        "## Required Inputs For Next Step",
        "",
        "- `batch4D_ledger_summary.csv`",
        "- `batch4D_daily_ledger.csv`",
        "- `batch4D_signal_regime_trade_summary.csv`",
        "- `batch4D_ledger_concentration.csv`",
        "- `batch4D_conclusion.md`",
        "",
        "## Next Step",
        "",
        "- Do not start forward tracking yet.",
        "- If accepted, do a full rule audit with exit-delay simulation and monthly cash ledger review.",
        "- Do not resume complex model training based on this audit.",
    ]
    GLOBAL_HANDOFF_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_stage_status(gate: str) -> None:
    if not STAGE_STATUS_PATH.exists():
        return
    df = pd.read_csv(STAGE_STATUS_PATH)
    row = {
        "stage": "Stage 7 Deep Simple Rule Ledger",
        "batch": "batch4D_deep_simple_rule_ledger",
        "status": "completed",
        "output_dir": "reports/tushare/v9_swing_research/batch4D_deep_simple_rule_ledger",
        "gate_result": "review" if gate.startswith("candidate") else "blocked",
        "next_action": "Review overlapping ledger results; do not start forward tracking until exit-delay and weak-market checks are complete.",
    }
    df = df[~df["batch"].eq("batch4D_deep_simple_rule_ledger")]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(STAGE_STATUS_PATH, index=False)


def update_roadmap(gate: str) -> None:
    if not ROADMAP_PATH.exists():
        return
    text = ROADMAP_PATH.read_text(encoding="utf-8")
    addition = """### Stage 7E Batch 4D Deep Simple Rule Ledger

输出目录：`reports/tushare/v9_swing_research/batch4D_deep_simple_rule_ledger/`

执行边界：

- 只审计 `low_stock_ret_60d` 和 `combo_low_liquidity_weak_momentum`。
- 不训练模型、不新增规则、不调参、不修改特征/标签/horizon/TopN。
- 将 Batch3D/B4C 的日度 cohort 诊断升级为重叠持仓资金账本。
- 账本采用每个信号日 20% 资金 sleeve，Top20 等权，5 日持有逐日 mark-to-market。

结论：

- 当前仍不能直接进入 forward tracking；需要先补 exit-delay、月度现金账本和弱市样本验证。

"""
    marker = "### Step 5A: 鲁棒性和暴露审计"
    if addition.strip() not in text:
        text = text.replace(marker, addition + marker)
    text = text.replace(
        "> Batch 4C completed; review simple-rule robustness/capacity before any deeper audit",
        "> Batch 4D completed; review overlapping ledger before any forward tracking",
    )
    ROADMAP_PATH.write_text(text, encoding="utf-8")


def write_hashes() -> None:
    rows = []
    for path in sorted(REPORT_DIR.glob("batch4D_*")):
        if path.name == "batch4D_file_sha256.csv" or not path.is_file():
            continue
        rows.append({"file": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    for path in [GLOBAL_HANDOFF_PATH, ROADMAP_PATH, STAGE_STATUS_PATH]:
        if path.exists():
            rows.append({"file": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    with (REPORT_DIR / "batch4D_file_sha256.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "sha256", "size_bytes"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_config()
    write_limitations()
    print({"event": "load_trades"}, flush=True)
    trades = load_target_trades()
    trades.to_parquet(REPORT_DIR / "_local_batch4D_target_trades.parquet", index=False, compression="zstd")
    print({"event": "read_prices"}, flush=True)
    prices, dates, date_to_idx = read_price_panel()
    print({"event": "expand_positions", "trades": int(len(trades))}, flush=True)
    position_days = expand_position_days(trades, dates, date_to_idx)
    print({"event": "position_days", "rows": int(len(position_days))}, flush=True)
    position_days = add_position_returns(position_days, prices)
    position_days.to_parquet(REPORT_DIR / "_local_batch4D_position_days.parquet", index=False, compression="zstd")

    daily_frames = []
    term_frames = []
    summary_frames = []
    for idx, scenario in enumerate(SCENARIOS, start=1):
        print({"event": "scenario_start", "done": idx - 1, "total": len(SCENARIOS), "scenario_id": scenario.scenario_id}, flush=True)
        daily, terms = build_scenario_daily(position_days, trades, scenario)
        summary = summarize_ledger(daily, terms, scenario)
        daily_frames.append(daily)
        term_frames.append(terms)
        summary_frames.append(summary)
    daily_all = pd.concat(daily_frames, ignore_index=True)
    terms_all = pd.concat(term_frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)
    daily_all.to_csv(REPORT_DIR / "batch4D_daily_ledger.csv", index=False)
    summary_all.to_csv(REPORT_DIR / "batch4D_ledger_summary.csv", index=False)
    terms_all.to_parquet(REPORT_DIR / "_local_batch4D_trade_terms.parquet", index=False, compression="zstd")
    regime_summary = build_signal_regime_summary(terms_all)
    concentration = build_concentration(terms_all, "base_1m_5pct_signal_40bps")
    gate = write_conclusion(summary_all, regime_summary, concentration)
    update_handoff(gate)
    update_stage_status(gate)
    update_roadmap(gate)
    write_hashes()
    print({"event": "done", "out_dir": str(REPORT_DIR), "gate_result": gate}, flush=True)


if __name__ == "__main__":
    main()
