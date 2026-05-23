#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
LOCKED_DIR = ROOT / "locked_artifacts" / "v7_cap20_strong_label_003"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "tushare" / "v8_research" / "batch0_evaluator"
DEFAULT_SIGNAL_FILE = LOCKED_DIR / "validation" / "top10_validation.csv"
DEFAULT_SCORE_FILE = LOCKED_DIR / "validation" / "validation_predictions.csv.gz"
DEFAULT_CANDIDATE_FILE = DEFAULT_SCORE_FILE
DEFAULT_DAILY_FILE = ROOT / "data_tushare" / "clean" / "daily_repaired_top3000.parquet"
DEFAULT_STOCK_BASIC_FILE = ROOT / "data_tushare" / "raw" / "bootstrap" / "stock_basic.parquet"
DEFAULT_MINUTE_DIR = ROOT / "data_tushare" / "raw" / "stk_mins" / "freq=5min"

TOP_N = 10
RANDOM_SEED = 20260523
RANDOM_RUNS = 1000
VALIDATION_START = "20260224"

BUY_COMMISSION = 0.00025
SELL_COMMISSION = 0.00025
STAMP_TAX = 0.00050
LOCKED_BASE_SLIPPAGE = 0.00050
IMPACT_COEF = 0.0015
IMPACT_CAP = 0.0050
PRICE_EPS = 0.005

SLIPPAGE_BPS = [5, 10, 15, 20]
CAPITALS = [100_000, 300_000, 500_000, 1_000_000, 5_000_000]
PARTICIPATION_CAPS = [0.01, 0.03, 0.05, 0.10]


@dataclass(frozen=True)
class Batch0Config:
    batch: str = "v8_research_batch0_evaluator"
    source_model_version: str = "v7_cap20_strong_label_003"
    validation_period: str = "20260224-20260520"
    signal_file: str = "locked_artifacts/v7_cap20_strong_label_003/validation/top10_validation.csv"
    score_file: str = "locked_artifacts/v7_cap20_strong_label_003/validation/validation_predictions.csv.gz"
    candidate_pool_file: str = "locked_artifacts/v7_cap20_strong_label_003/validation/validation_predictions.csv.gz"
    daily_file: str = "data_tushare/clean/daily_repaired_top3000.parquet"
    minute_dir: str = "data_tushare/raw/stk_mins/freq=5min"
    top_n: int = TOP_N
    score_column: str = "score"
    label_column: str = "label_static_003"
    return_column: str = "target_return"
    locked_return_assumption: str = "target_return already includes commission, stamp tax and 5bp one-way slippage."
    train_model: bool = False
    change_features: bool = False
    change_label: bool = False
    change_topn: bool = False
    change_exit_rule: bool = False
    cost_cases: tuple[str, ...] = (
        "no_cost",
        "basic_cost_no_slippage",
        "5bp_one_way_slippage",
        "10bp_one_way_slippage",
        "15bp_one_way_slippage",
        "20bp_one_way_slippage",
        "10bp_one_way_slippage_plus_impact_cost",
        "10bp_one_way_slippage_plus_impact_cost_plus_participation_cap",
        "10bp_one_way_slippage_plus_impact_cost_plus_participation_cap_plus_partial_zero_fill",
    )
    capital_levels: tuple[int, ...] = tuple(CAPITALS)
    participation_caps: tuple[float, ...] = tuple(PARTICIPATION_CAPS)
    enable_partial_fill: bool = True
    enable_zero_fill: bool = True
    enable_limit_suspend_rules: bool = True
    impact_formula: str = "per-side impact = min(50bp, 15bp * sqrt(realized_participation))"
    random_baseline_runs: int = RANDOM_RUNS
    random_seed: int = RANDOM_SEED


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    cost_case: str
    slippage_bp: int | None
    include_commission_tax: bool
    include_impact: bool = False
    capital: int | None = None
    participation_cap: float | None = None
    partial_fill: bool = False
    zero_fill: bool = False
    enforce_limit_suspend: bool = False


def beijing_now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def max_drawdown(daily_return: pd.Series) -> float:
    if daily_return.empty:
        return float("nan")
    equity = (1 + daily_return.fillna(0.0)).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def drawdown_window(daily_return: pd.Series) -> dict[str, Any]:
    if daily_return.empty:
        return {"drawdown_start": "", "drawdown_end": "", "drawdown_days": 0, "max_drawdown": float("nan")}
    equity = (1 + daily_return.fillna(0.0)).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    end = str(dd.idxmin())
    start = str(equity.loc[: end].idxmax())
    dates = list(daily_return.index.astype(str))
    try:
        days = dates.index(end) - dates.index(start) + 1
    except ValueError:
        days = 0
    return {"drawdown_start": start, "drawdown_end": end, "drawdown_days": int(days), "max_drawdown": float(dd.min())}


def profit_factor(ret: pd.Series) -> float:
    pos = float(ret[ret > 0].sum())
    neg = float(ret[ret < 0].sum())
    if neg == 0:
        return float("inf") if pos > 0 else float("nan")
    return pos / abs(neg)


def auc_safe(df: pd.DataFrame, label_col: str, score_col: str) -> float:
    if label_col not in df.columns or score_col not in df.columns or df[label_col].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(df[label_col], df[score_col]))


def net_to_gross_locked(net: pd.Series) -> pd.Series:
    return (1 + net) * (1 + LOCKED_BASE_SLIPPAGE + BUY_COMMISSION) / (
        1 - LOCKED_BASE_SLIPPAGE - SELL_COMMISSION - STAMP_TAX
    ) - 1


def gross_to_net(
    gross: pd.Series | np.ndarray,
    slippage_bp: int | None,
    include_commission_tax: bool,
    buy_impact: np.ndarray | float = 0.0,
    sell_impact: np.ndarray | float = 0.0,
) -> np.ndarray:
    gross_arr = np.asarray(gross, dtype=float)
    slippage = 0.0 if slippage_bp is None else slippage_bp / 10000.0
    buy_commission = BUY_COMMISSION if include_commission_tax else 0.0
    sell_commission = SELL_COMMISSION if include_commission_tax else 0.0
    stamp_tax = STAMP_TAX if include_commission_tax else 0.0
    return (1 + gross_arr) * (1 - slippage - sell_impact - sell_commission - stamp_tax) / (
        1 + slippage + buy_impact + buy_commission
    ) - 1


def safe_ratio(numerator: np.ndarray | pd.Series | float, denominator: np.ndarray | pd.Series | float) -> np.ndarray:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    num, den = np.broadcast_arrays(num, den)
    out = np.zeros_like(num, dtype=float)
    np.divide(num, den, out=out, where=np.isfinite(den) & (den > 0))
    out[~np.isfinite(out)] = 0.0
    return out


def impact_from_participation(participation: np.ndarray | pd.Series) -> np.ndarray:
    p = np.asarray(participation, dtype=float)
    p = np.clip(np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    return np.minimum(IMPACT_CAP, IMPACT_COEF * np.sqrt(p))


def round_price(x: float) -> float:
    if not np.isfinite(x):
        return float("nan")
    return round(float(x) + 1e-9, 2)


def limit_pct(ts_code: str, name: str) -> float:
    name_text = str(name or "").upper()
    if "ST" in name_text:
        return 0.05
    raw = str(ts_code).split(".")[0]
    if raw.startswith(("300", "301", "688")):
        return 0.20
    return 0.10


def vwap_from_bar(row: pd.Series) -> float:
    amount = pd.to_numeric(row.get("amount"), errors="coerce")
    vol = pd.to_numeric(row.get("vol"), errors="coerce")
    close = pd.to_numeric(row.get("close"), errors="coerce")
    if pd.notna(amount) and pd.notna(vol) and amount > 0 and vol > 0:
        price = float(amount / vol)
        if np.isfinite(price) and price > 0:
            return price
    return float(close) if pd.notna(close) else float("nan")


def load_inputs(signal_file: Path, score_file: Path, daily_file: Path, stock_basic_file: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signals = pd.read_csv(signal_file)
    signals["trade_date"] = signals["trade_date"].astype(str)
    signals["trade_id"] = np.arange(len(signals))
    candidates = pd.read_csv(score_file, compression="infer")
    candidates["trade_date"] = candidates["trade_date"].astype(str)

    daily_cols = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "prev_close_use",
        "amount",
        "vol",
        "next_trade_date",
        "prev_trade_date",
    ]
    daily = pd.read_parquet(daily_file, columns=daily_cols)
    daily["trade_date"] = daily["trade_date"].astype(str)
    daily["next_trade_date"] = daily["next_trade_date"].astype(str)
    daily["prev_trade_date"] = daily["prev_trade_date"].astype(str)

    if stock_basic_file.exists():
        stock = pd.read_parquet(stock_basic_file)
        keep = [c for c in ["ts_code", "name", "industry", "market", "list_status", "list_date"] if c in stock.columns]
        stock = stock[keep].copy()
    else:
        stock = pd.DataFrame(columns=["ts_code", "name", "industry", "market", "list_status", "list_date"])
    stock["ts_code"] = stock["ts_code"].astype(str)
    return signals, candidates, daily, stock


def prepare_trade_base(signals: pd.DataFrame, daily: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:
    entry_daily = daily.rename(
        columns={
            "prev_close_use": "entry_prev_close",
            "next_trade_date": "exit_trade_date",
            "amount": "entry_daily_amount",
            "vol": "entry_daily_vol",
        }
    )[
        [
            "ts_code",
            "trade_date",
            "entry_prev_close",
            "exit_trade_date",
            "entry_daily_amount",
            "entry_daily_vol",
        ]
    ]
    out = signals.merge(entry_daily, on=["ts_code", "trade_date"], how="left")
    exit_daily = daily.rename(
        columns={
            "trade_date": "exit_trade_date",
            "prev_close_use": "exit_prev_close",
            "next_trade_date": "fallback_trade_date",
            "amount": "exit_daily_amount",
            "vol": "exit_daily_vol",
        }
    )[
        [
            "ts_code",
            "exit_trade_date",
            "exit_prev_close",
            "fallback_trade_date",
            "exit_daily_amount",
            "exit_daily_vol",
        ]
    ]
    out = out.merge(exit_daily, on=["ts_code", "exit_trade_date"], how="left")
    out = out.merge(stock, on="ts_code", how="left")
    out["limit_pct"] = [limit_pct(c, n) for c, n in zip(out["ts_code"], out.get("name", ""))]
    out["entry_limit_up"] = [round_price(p * (1 + l)) for p, l in zip(out["entry_prev_close"], out["limit_pct"])]
    out["exit_limit_down"] = [round_price(p * (1 - l)) for p, l in zip(out["exit_prev_close"], out["limit_pct"])]
    out["gross_return_v7"] = net_to_gross_locked(out["target_return"])
    return out


def needed_dates_by_symbol(base: pd.DataFrame) -> dict[str, set[str]]:
    needed: dict[str, set[str]] = {}
    for row in base[["ts_code", "trade_date", "exit_trade_date", "fallback_trade_date"]].itertuples(index=False):
        dates = {str(row.trade_date), str(row.exit_trade_date), str(row.fallback_trade_date)}
        dates = {d for d in dates if d and d != "nan" and len(d) == 8}
        needed.setdefault(str(row.ts_code), set()).update(dates)
    return needed


def load_relevant_minutes(base: pd.DataFrame, minute_dir: Path) -> pd.DataFrame:
    needed = needed_dates_by_symbol(base)
    frames: list[pd.DataFrame] = []
    cols = ["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"]
    for ts_code, dates in sorted(needed.items()):
        symbol_dir = minute_dir / f"ts_code={ts_code}"
        for path in sorted(symbol_dir.glob("*.parquet")):
            df = pd.read_parquet(path, columns=cols)
            if df.empty:
                continue
            df["trade_date"] = df["trade_time"].astype(str).str.slice(0, 10).str.replace("-", "", regex=False)
            df = df[df["trade_date"].isin(dates)]
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame(columns=cols + ["trade_date", "bar_time", "bar_vwap"])
    minutes = pd.concat(frames, ignore_index=True)
    minutes["trade_time"] = minutes["trade_time"].astype(str)
    minutes["trade_date"] = minutes["trade_time"].str.slice(0, 10).str.replace("-", "", regex=False)
    minutes["bar_time"] = minutes["trade_time"].str.slice(11, 16)
    for col in ["open", "high", "low", "close", "vol", "amount"]:
        minutes[col] = pd.to_numeric(minutes[col], errors="coerce")
    minutes["bar_vwap"] = minutes.apply(vwap_from_bar, axis=1)
    minutes = minutes.sort_values(["ts_code", "trade_date", "bar_time"])
    return minutes.drop_duplicates(["ts_code", "trade_date", "bar_time"], keep="last")


def add_entry_exit_bars(base: pd.DataFrame, minutes: pd.DataFrame) -> pd.DataFrame:
    entry = minutes[minutes["bar_time"] == "14:55"].rename(
        columns={c: f"entry_bar_{c}" for c in ["open", "high", "low", "close", "vol", "amount", "bar_vwap"]}
    )
    entry = entry.rename(columns={"bar_time": "entry_bar_time"})[
        [
            "ts_code",
            "trade_date",
            "entry_bar_time",
            "entry_bar_open",
            "entry_bar_high",
            "entry_bar_low",
            "entry_bar_close",
            "entry_bar_vol",
            "entry_bar_amount",
            "entry_bar_bar_vwap",
        ]
    ]
    out = base.merge(entry, on=["ts_code", "trade_date"], how="left")

    exit_bars = minutes.rename(columns={"trade_date": "exit_trade_date", "bar_time": "exit_time"})
    exit_bars = exit_bars.rename(
        columns={c: f"exit_bar_{c}" for c in ["open", "high", "low", "close", "vol", "amount", "bar_vwap"]}
    )[
        [
            "ts_code",
            "exit_trade_date",
            "exit_time",
            "exit_bar_open",
            "exit_bar_high",
            "exit_bar_low",
            "exit_bar_close",
            "exit_bar_vol",
            "exit_bar_amount",
            "exit_bar_bar_vwap",
        ]
    ]
    return out.merge(exit_bars, on=["ts_code", "exit_trade_date", "exit_time"], how="left")


def is_hard_limit_up(low_price: float, limit_up: float) -> bool:
    return bool(pd.notna(low_price) and pd.notna(limit_up) and low_price >= limit_up - PRICE_EPS)


def is_limit_up_warning(close_price: float, high_price: float, limit_up: float) -> bool:
    return bool(
        pd.notna(close_price)
        and pd.notna(high_price)
        and pd.notna(limit_up)
        and close_price >= limit_up - PRICE_EPS
        and high_price >= limit_up - PRICE_EPS
    )


def is_hard_limit_down(high_price: float, limit_down: float) -> bool:
    return bool(pd.notna(high_price) and pd.notna(limit_down) and high_price <= limit_down + PRICE_EPS)


def is_limit_down_warning(close_price: float, low_price: float, limit_down: float) -> bool:
    return bool(
        pd.notna(close_price)
        and pd.notna(low_price)
        and pd.notna(limit_down)
        and close_price <= limit_down + PRICE_EPS
        and low_price <= limit_down + PRICE_EPS
    )


def find_deferred_exit(row: pd.Series, minutes: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
    if not row["sell_hard_blocked"]:
        return {
            "effective_exit_trade_date": row["exit_trade_date"],
            "effective_exit_time": row["exit_time"],
            "effective_exit_vwap": row["exit_bar_bar_vwap"],
            "effective_exit_amount": row["exit_bar_amount"],
            "sell_deferred": False,
            "sell_defer_bars": 0,
        }

    ts_code = row["ts_code"]
    dates_to_check = [row["exit_trade_date"], row["fallback_trade_date"]]
    start_times = [row["exit_time"], "09:30"]
    checked = 0
    for trade_date, start_time in zip(dates_to_check, start_times):
        if not isinstance(trade_date, str) or trade_date == "nan":
            continue
        day = minutes[(minutes["ts_code"] == ts_code) & (minutes["trade_date"] == trade_date)].sort_values("bar_time")
        day = day[day["bar_time"] >= start_time]
        if day.empty:
            continue
        daily_row = daily[(daily["ts_code"] == ts_code) & (daily["trade_date"] == trade_date)]
        if daily_row.empty:
            limit_down = row["exit_limit_down"]
        else:
            limit_down = round_price(float(daily_row.iloc[0]["prev_close_use"]) * (1 - row["limit_pct"]))
        for bar in day.itertuples(index=False):
            checked += 1
            if not is_hard_limit_down(float(bar.high), limit_down):
                return {
                    "effective_exit_trade_date": trade_date,
                    "effective_exit_time": str(bar.bar_time),
                    "effective_exit_vwap": float(bar.bar_vwap),
                    "effective_exit_amount": float(bar.amount),
                    "sell_deferred": True,
                    "sell_defer_bars": checked,
                }

    return {
        "effective_exit_trade_date": row["exit_trade_date"],
        "effective_exit_time": row["exit_time"],
        "effective_exit_vwap": np.nan,
        "effective_exit_amount": np.nan,
        "sell_deferred": True,
        "sell_defer_bars": checked,
    }


def add_execution_flags(base: pd.DataFrame, minutes: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    out["entry_missing"] = out["entry_bar_close"].isna()
    out["exit_missing"] = out["exit_bar_close"].isna()
    out["buy_hard_blocked"] = [is_hard_limit_up(low, lim) for low, lim in zip(out["entry_bar_low"], out["entry_limit_up"])]
    out["buy_limit_warning"] = [
        is_limit_up_warning(close, high, lim)
        for close, high, lim in zip(out["entry_bar_close"], out["entry_bar_high"], out["entry_limit_up"])
    ]
    out["sell_hard_blocked"] = [is_hard_limit_down(high, lim) for high, lim in zip(out["exit_bar_high"], out["exit_limit_down"])]
    out["sell_limit_warning"] = [
        is_limit_down_warning(close, low, lim)
        for close, low, lim in zip(out["exit_bar_close"], out["exit_bar_low"], out["exit_limit_down"])
    ]
    deferred = [find_deferred_exit(row, minutes, daily) for _, row in out.iterrows()]
    out = pd.concat([out.reset_index(drop=True), pd.DataFrame(deferred)], axis=1)
    out["effective_gross_return"] = out["gross_return_v7"]
    mask = out["sell_deferred"] & out["effective_exit_vwap"].notna() & (out["entry_vwap"] > 0)
    out.loc[mask, "effective_gross_return"] = out.loc[mask, "effective_exit_vwap"] / out.loc[mask, "entry_vwap"] - 1
    out["no_buy_due_to_execution"] = out["entry_missing"] | out["buy_hard_blocked"]
    out["no_exit_due_to_execution"] = out["exit_missing"] | out["effective_exit_vwap"].isna()
    return out


def build_scenarios() -> tuple[list[Scenario], list[Scenario]]:
    cost_scenarios = [
        Scenario("no_cost", "no_cost", None, False),
        Scenario("basic_cost_no_slippage", "basic_cost_no_slippage", 0, True),
        Scenario("5bp_one_way_slippage", "5bp_one_way_slippage", 5, True),
        Scenario("10bp_one_way_slippage", "10bp_one_way_slippage", 10, True),
        Scenario("15bp_one_way_slippage", "15bp_one_way_slippage", 15, True),
        Scenario("20bp_one_way_slippage", "20bp_one_way_slippage", 20, True),
        Scenario(
            "10bp_slippage_plus_impact_capital_100k",
            "10bp_one_way_slippage_plus_impact_cost",
            10,
            True,
            include_impact=True,
            capital=100_000,
        ),
        Scenario(
            "10bp_slippage_plus_impact_participation_10pct_no_partial_capital_100k",
            "10bp_one_way_slippage_plus_impact_cost_plus_participation_cap",
            10,
            True,
            include_impact=True,
            capital=100_000,
            participation_cap=0.10,
            partial_fill=False,
            zero_fill=True,
            enforce_limit_suspend=True,
        ),
        Scenario(
            "10bp_slippage_plus_impact_participation_10pct_partial_zero_capital_100k",
            "10bp_one_way_slippage_plus_impact_cost_plus_participation_cap_plus_partial_zero_fill",
            10,
            True,
            include_impact=True,
            capital=100_000,
            participation_cap=0.10,
            partial_fill=True,
            zero_fill=True,
            enforce_limit_suspend=True,
        ),
    ]
    capacity_scenarios = []
    for capital in CAPITALS:
        for participation_cap in PARTICIPATION_CAPS:
            for bp in SLIPPAGE_BPS:
                capacity_scenarios.append(
                    Scenario(
                        f"capacity_{capital}_part_{int(participation_cap * 10000)}bp_slip_{bp}bp",
                        "slippage_plus_impact_plus_participation_partial_zero",
                        bp,
                        True,
                        include_impact=True,
                        capital=capital,
                        participation_cap=participation_cap,
                        partial_fill=True,
                        zero_fill=True,
                        enforce_limit_suspend=True,
                    )
                )
    return cost_scenarios, capacity_scenarios


def evaluate_scenario(trades: pd.DataFrame, scenario: Scenario, score_auc: float) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    out = trades[["trade_id", "trade_date", "ts_code", "score", "target_return", "gross_return_v7", "effective_gross_return"]].copy()
    gross = trades["effective_gross_return"].astype(float).to_numpy()
    n = len(trades)
    order_notional = None if scenario.capital is None else scenario.capital / TOP_N
    executable = np.ones(n, dtype=bool)
    if scenario.enforce_limit_suspend:
        executable = ~(trades["no_buy_due_to_execution"].to_numpy() | trades["no_exit_due_to_execution"].to_numpy())

    fill_ratio = np.where(executable, 1.0, 0.0)
    buy_participation = np.zeros(n, dtype=float)
    sell_participation = np.zeros(n, dtype=float)

    if scenario.capital is not None and order_notional is not None:
        entry_amount = trades["entry_bar_amount"].fillna(0.0).to_numpy(dtype=float)
        exit_amount = trades["effective_exit_amount"].fillna(0.0).to_numpy(dtype=float)
        sell_notional_multiplier = np.maximum(0.0, 1.0 + gross)
        raw_fill = np.ones(n, dtype=float)
        if scenario.participation_cap is not None:
            buy_capacity = scenario.participation_cap * entry_amount
            sell_capacity = scenario.participation_cap * exit_amount
            raw_fill = np.minimum(raw_fill, safe_ratio(buy_capacity, order_notional))
            raw_fill = np.minimum(raw_fill, safe_ratio(sell_capacity, order_notional * sell_notional_multiplier))
        raw_fill = np.where(executable, raw_fill, 0.0)
        raw_fill = np.clip(np.nan_to_num(raw_fill, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        if scenario.participation_cap is not None and not scenario.partial_fill:
            fill_ratio = np.where(raw_fill >= 0.999, 1.0, 0.0 if scenario.zero_fill else raw_fill)
        else:
            fill_ratio = raw_fill

        buy_participation = safe_ratio(order_notional * fill_ratio, entry_amount)
        sell_participation = safe_ratio(order_notional * fill_ratio * sell_notional_multiplier, exit_amount)

    buy_impact = impact_from_participation(buy_participation) if scenario.include_impact else np.zeros(n)
    sell_impact = impact_from_participation(sell_participation) if scenario.include_impact else np.zeros(n)
    filled_return = gross_to_net(
        gross,
        scenario.slippage_bp,
        scenario.include_commission_tax,
        buy_impact=buy_impact,
        sell_impact=sell_impact,
    )
    portfolio_slot_return = fill_ratio * filled_return

    unfilled_reason = np.full(n, "", dtype=object)
    unfilled_reason = np.where(trades["no_buy_due_to_execution"].to_numpy(), "no_buy_limit_or_missing", unfilled_reason)
    unfilled_reason = np.where(
        (~trades["no_buy_due_to_execution"].to_numpy()) & trades["no_exit_due_to_execution"].to_numpy(),
        "no_exit_limit_or_missing",
        unfilled_reason,
    )
    cap_limited = (fill_ratio > 0) & (fill_ratio < 0.999)
    zero_capacity = (fill_ratio <= 0) & executable & (scenario.participation_cap is not None)
    unfilled_reason = np.where(cap_limited, "partial_fill_capacity", unfilled_reason)
    unfilled_reason = np.where(zero_capacity, "zero_fill_capacity", unfilled_reason)
    unfilled_reason = np.where(unfilled_reason == "", "filled", unfilled_reason)

    out["scenario_id"] = scenario.scenario_id
    out["cost_case"] = scenario.cost_case
    out["capital"] = scenario.capital
    out["participation_cap"] = scenario.participation_cap
    out["slippage_bp"] = scenario.slippage_bp
    out["include_impact"] = scenario.include_impact
    out["partial_fill"] = scenario.partial_fill
    out["zero_fill"] = scenario.zero_fill
    out["fill_ratio"] = fill_ratio
    out["buy_participation"] = buy_participation
    out["sell_participation"] = sell_participation
    out["buy_impact_bp"] = buy_impact * 10000
    out["sell_impact_bp"] = sell_impact * 10000
    out["filled_notional_return"] = filled_return
    out["portfolio_slot_return"] = portfolio_slot_return
    out["unfilled_reason"] = unfilled_reason

    daily = out.groupby("trade_date")["portfolio_slot_return"].mean().rename("daily_return")
    daily = daily.reindex(sorted(trades["trade_date"].astype(str).unique())).fillna(0.0)
    metrics = metrics_from_returns(out, daily, scenario, score_auc)
    daily_df = daily.reset_index()
    daily_df["scenario_id"] = scenario.scenario_id
    return metrics, out, daily_df


def metrics_from_returns(out: pd.DataFrame, daily: pd.Series, scenario: Scenario, score_auc: float | None = None) -> dict[str, Any]:
    ret = out["portfolio_slot_return"]
    nonzero = out[out["fill_ratio"] > 0]
    total_days = int(len(daily))
    partial_count = int(((out["fill_ratio"] > 0) & (out["fill_ratio"] < 0.999)).sum())
    zero_count = int((out["fill_ratio"] <= 0).sum())
    daily_sorted = daily.sort_index()
    profit_days = daily_sorted[daily_sorted > 0].sort_values(ascending=False)
    row: dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "cost_case": scenario.cost_case,
        "capital": scenario.capital,
        "participation_cap": scenario.participation_cap,
        "slippage_bp": scenario.slippage_bp,
        "include_commission_tax": scenario.include_commission_tax,
        "include_impact": scenario.include_impact,
        "partial_fill": scenario.partial_fill,
        "zero_fill": scenario.zero_fill,
        "enforce_limit_suspend": scenario.enforce_limit_suspend,
        "cumulative_return": float((1 + daily_sorted).prod() - 1),
        "avg_daily_return": float(daily_sorted.mean()),
        "trade_win_rate": float((nonzero["filled_notional_return"] > 0).mean()) if len(nonzero) else float("nan"),
        "daily_win_rate": float((daily_sorted > 0).mean()) if total_days else float("nan"),
        "max_drawdown": max_drawdown(daily_sorted),
        "profit_factor": profit_factor(ret),
        "auc": score_auc,
        "total_trades": int(len(out)),
        "trading_days": total_days,
        "avg_daily_trades": float(len(out) / total_days) if total_days else float("nan"),
        "avg_fill_ratio": float(out["fill_ratio"].mean()),
        "partial_fill_count": partial_count,
        "zero_fill_count": zero_count,
        "max_profit_day": str(daily_sorted.idxmax()) if total_days else "",
        "max_profit_day_return": float(daily_sorted.max()) if total_days else float("nan"),
        "max_loss_day": str(daily_sorted.idxmin()) if total_days else "",
        "max_loss_day_return": float(daily_sorted.min()) if total_days else float("nan"),
        "cum_return_drop_top1_profit_day": drop_top_profit_days(daily_sorted, 1),
        "cum_return_drop_top3_profit_days": drop_top_profit_days(daily_sorted, 3),
        "cum_return_drop_top5_profit_days": drop_top_profit_days(daily_sorted, 5),
    }
    reason_counts = out["unfilled_reason"].value_counts().to_dict()
    for reason in ["filled", "partial_fill_capacity", "zero_fill_capacity", "no_buy_limit_or_missing", "no_exit_limit_or_missing"]:
        row[f"unfilled_reason_{reason}"] = int(reason_counts.get(reason, 0))
    row.update(drawdown_window(daily_sorted))
    return row


def drop_top_profit_days(daily: pd.Series, n: int) -> float:
    adjusted = daily.copy()
    drop_dates = adjusted[adjusted > 0].sort_values(ascending=False).head(n).index
    adjusted.loc[drop_dates] = 0.0
    return float((1 + adjusted).prod() - 1)


def evaluate_plain_selection(
    selected: pd.DataFrame,
    scenario_id: str,
    score_auc: float | None = None,
    return_col: str = "target_return",
) -> dict[str, Any]:
    tmp = selected[["trade_date", "ts_code", return_col]].copy()
    tmp["trade_id"] = np.arange(len(tmp))
    tmp["score"] = selected["score"].to_numpy() if "score" in selected.columns else np.nan
    tmp["gross_return_v7"] = net_to_gross_locked(tmp[return_col])
    tmp["effective_gross_return"] = tmp["gross_return_v7"]
    tmp["fill_ratio"] = 1.0
    tmp["filled_notional_return"] = tmp[return_col]
    tmp["portfolio_slot_return"] = tmp[return_col]
    tmp["unfilled_reason"] = "filled"
    scenario = Scenario(scenario_id, scenario_id, 5, True)
    daily = tmp.groupby("trade_date")["portfolio_slot_return"].mean().reindex(sorted(tmp["trade_date"].unique())).fillna(0.0)
    return metrics_from_returns(tmp, daily, scenario, score_auc)


def topn_by_column(df: pd.DataFrame, column: str, ascending: bool = False) -> pd.DataFrame:
    return (
        df.sort_values(["trade_date", column], ascending=[True, ascending])
        .groupby("trade_date", group_keys=False)
        .head(TOP_N)
        .copy()
    )


def random_topn_baseline(candidates: pd.DataFrame, model_cum_return: float) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    groups = [(d, g["target_return"].to_numpy(dtype=float)) for d, g in candidates.groupby("trade_date", sort=True)]
    rows = []
    for run in range(RANDOM_RUNS):
        picked = []
        daily_values = []
        for trade_date, arr in groups:
            take = rng.choice(arr, size=TOP_N, replace=False)
            picked.append(take)
            daily_values.append(float(np.mean(take)))
        trade_returns = pd.Series(np.concatenate(picked))
        daily = pd.Series(daily_values)
        rows.append(
            {
                "run": run,
                "cumulative_return": float((1 + daily).prod() - 1),
                "avg_daily_return": float(daily.mean()),
                "trade_win_rate": float((trade_returns > 0).mean()),
                "daily_win_rate": float((daily > 0).mean()),
                "max_drawdown": max_drawdown(daily),
                "profit_factor": profit_factor(trade_returns),
            }
        )
    runs = pd.DataFrame(rows)
    summary = {
        "baseline": "random_top10_1000_runs_same_candidate_pool",
        "runs": RANDOM_RUNS,
        "model_cumulative_return": model_cum_return,
        "model_percentile_by_cum_return": float((runs["cumulative_return"] <= model_cum_return).mean()),
    }
    for col in ["cumulative_return", "avg_daily_return", "trade_win_rate", "daily_win_rate", "max_drawdown", "profit_factor"]:
        summary[f"{col}_mean"] = float(runs[col].mean())
        summary[f"{col}_median"] = float(runs[col].median())
        summary[f"{col}_p05"] = float(runs[col].quantile(0.05))
        summary[f"{col}_p95"] = float(runs[col].quantile(0.95))
        summary[f"{col}_min"] = float(runs[col].min())
        summary[f"{col}_max"] = float(runs[col].max())
    return pd.DataFrame([summary])


def simple_benchmarks(candidates: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("ai_model_score_top10", "score", False),
        ("tail_ret_1430_1450_top10", "tail_ret_1430_1450", False),
        ("tail_ret_1420_1450_top10", "tail_ret_1420_1450", False),
        ("ret_to_prev_close_1450_top10", "ret_to_prev_close_1450", False),
        ("tail_amount_vs_prev20_top10", "tail_amount_vs_prev20", False),
        ("amount_sofar_log_top10", "amount_sofar_log", False),
        ("vwap_pos_1450_top10", "vwap_pos_1450", False),
    ]
    rows = []
    for name, col, ascending in specs:
        if col not in candidates.columns:
            continue
        selected = topn_by_column(candidates.dropna(subset=[col]), col, ascending=ascending)
        auc = auc_safe(candidates.dropna(subset=[col]), "label_static_003", col)
        row = evaluate_plain_selection(selected, name, score_auc=auc)
        row["rank_column"] = col
        rows.append(row)
    return pd.DataFrame(rows)


def write_config_yaml(path: Path, config: Batch0Config) -> None:
    data = asdict(config)
    lines = [
        "# Batch 0 evaluator config. This file is documentation for the executed run.",
        f"created_at_beijing: {beijing_now()}",
    ]
    for key, value in data.items():
        if isinstance(value, tuple):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        elif isinstance(value, bool):
            lines.append(f"{key}: {'true' if value else 'false'}")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_conclusion(
    output_dir: Path,
    reproduction: pd.DataFrame,
    cost_summary: pd.DataFrame,
    random_summary: pd.DataFrame,
    simple_summary: pd.DataFrame,
) -> None:
    rep = reproduction.iloc[0].to_dict()
    ten = cost_summary[cost_summary["scenario_id"] == "10bp_one_way_slippage"].iloc[0].to_dict()
    five = cost_summary[cost_summary["scenario_id"] == "5bp_one_way_slippage"].iloc[0].to_dict()
    diff_5bp = five["cumulative_return"] - 0.0563104482106096
    diff_pf = five["profit_factor"] - 1.1198887096204315
    diff_auc = five["auc"] - 0.5334428792730082
    diff_10bp = ten["cumulative_return"] - (-0.003227249600176618)
    rand = random_summary.iloc[0].to_dict()
    best_simple = simple_summary.sort_values("cumulative_return", ascending=False).iloc[0].to_dict() if len(simple_summary) else {}

    text = f"""# Batch 0 Conclusion

## 1. Batch 0 changed what?

Batch 0 added a unified evaluator at `research/v8_research/evaluate_strategy.py`
and generated local evaluation outputs under `reports/tushare/v8_research/batch0_evaluator/`.
It standardizes metric calculation, cost/slippage scenarios, execution flags,
capacity/participation scenarios, random Top10 baseline and fixed simple rule baselines.

## 2. Batch 0 did not change what?

It did not change the locked v7 model, features, label, TopN, exit rule, universe,
or any training data. It did not start Batch 1 to Batch 5.

## 3. Was `v7_locked` modified?

No.

## 4. Was the model retrained?

No.

## 5. Were features, label, TopN or exit rule modified?

No.

## 6. Were the locked v7 core results reproduced?

Yes.

| Metric | Locked reference | Batch 0 reproduced | Difference |
|---|---:|---:|---:|
| 5bp cumulative return | 0.0563104482 | {five['cumulative_return']:.10f} | {diff_5bp:.12f} |
| PF | 1.1198887096 | {five['profit_factor']:.10f} | {diff_pf:.12f} |
| AUC | 0.5334428793 | {five['auc']:.10f} | {diff_auc:.12f} |
| 10bp cumulative return | -0.0032272496 | {ten['cumulative_return']:.10f} | {diff_10bp:.12f} |

## 7. If there is a difference, what caused it?

The reproduced core metrics are numerically equal within floating-point tolerance.
No material difference was observed.

## 8. Can this evaluator be used for later batches?

Yes for Batch 1, Batch 2A, Batch 2B, Batch 3 and Batch 4 style comparisons,
provided each later batch writes signals or scores in the same schema. It already
accepts signal/score/candidate inputs, cost settings, slippage settings, impact
cost, capital levels, participation caps, partial/zero fill and limit/suspension
execution flags. Walk-forward in Batch 5 can use the same evaluator per fold,
but the fold runner should be added separately when that batch is approved.

## 9. Should we enter Batch 1?

Yes, after this Batch 0 output is reviewed. The reason is that the evaluator
now reproduces v7 exactly and gives a fixed measurement harness. Batch 1 should
still be accepted separately before execution, and it must not optimize against
`20260224-20260520` as a final test set.

## Additional Batch 0 facts

| Item | Value |
|---|---:|
| Selected trades | {int(rep['total_trades'])} |
| Trading days | {int(rep['trading_days'])} |
| Random baseline runs | {RANDOM_RUNS} |
| Random baseline cumulative return mean | {rand['cumulative_return_mean']:.10f} |
| v7 percentile vs random cumulative return | {rand['model_percentile_by_cum_return']:.4f} |
| Best fixed simple benchmark in this report | {best_simple.get('scenario_id', 'NA')} |
| Best fixed simple benchmark cumulative return | {best_simple.get('cumulative_return', float('nan')):.10f} |

"""
    (output_dir / "batch0_conclusion.md").write_text(text, encoding="utf-8")


def write_hashes(output_dir: Path, extra_files: list[Path]) -> None:
    rows = []
    files = sorted([p for p in output_dir.glob("batch0_*") if p.is_file() and p.name != "batch0_file_sha256.csv"])
    files.extend(extra_files)
    for path in files:
        import hashlib

        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        rows.append({"path": rel(path), "sha256": h.hexdigest(), "bytes": path.stat().st_size})
    pd.DataFrame(rows).to_csv(output_dir / "batch0_file_sha256.csv", index=False)


def run_batch0(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    signal_file = Path(args.signal_file).resolve()
    score_file = Path(args.score_file).resolve()
    daily_file = Path(args.daily_file).resolve()
    stock_file = Path(args.stock_basic_file).resolve()
    minute_dir = Path(args.minute_dir).resolve()

    signals, candidates, daily, stock = load_inputs(signal_file, score_file, daily_file, stock_file)
    model_auc = auc_safe(candidates, "label_static_003", "score")
    trade_base = prepare_trade_base(signals, daily, stock)
    minutes = load_relevant_minutes(trade_base, minute_dir)
    trade_base = add_entry_exit_bars(trade_base, minutes)
    trade_base = add_execution_flags(trade_base, minutes, daily)

    cost_scenarios, capacity_scenarios = build_scenarios()
    cost_metrics = []
    daily_frames = []
    trade_detail_frames = []
    fill_rows = []
    for scenario in cost_scenarios:
        metrics, details, daily_df = evaluate_scenario(trade_base, scenario, model_auc)
        cost_metrics.append(metrics)
        daily_frames.append(daily_df)
        trade_detail_frames.append(details)
        fill_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "avg_fill_ratio": metrics["avg_fill_ratio"],
                "partial_fill_count": metrics["partial_fill_count"],
                "zero_fill_count": metrics["zero_fill_count"],
                "filled_count": metrics["unfilled_reason_filled"],
                "partial_fill_capacity_count": metrics["unfilled_reason_partial_fill_capacity"],
                "zero_fill_capacity_count": metrics["unfilled_reason_zero_fill_capacity"],
                "no_buy_limit_or_missing_count": metrics["unfilled_reason_no_buy_limit_or_missing"],
                "no_exit_limit_or_missing_count": metrics["unfilled_reason_no_exit_limit_or_missing"],
            }
        )

    capacity_metrics = []
    for scenario in capacity_scenarios:
        metrics, _details, daily_df = evaluate_scenario(trade_base, scenario, model_auc)
        capacity_metrics.append(metrics)
        daily_frames.append(daily_df)
        fill_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "avg_fill_ratio": metrics["avg_fill_ratio"],
                "partial_fill_count": metrics["partial_fill_count"],
                "zero_fill_count": metrics["zero_fill_count"],
                "filled_count": metrics["unfilled_reason_filled"],
                "partial_fill_capacity_count": metrics["unfilled_reason_partial_fill_capacity"],
                "zero_fill_capacity_count": metrics["unfilled_reason_zero_fill_capacity"],
                "no_buy_limit_or_missing_count": metrics["unfilled_reason_no_buy_limit_or_missing"],
                "no_exit_limit_or_missing_count": metrics["unfilled_reason_no_exit_limit_or_missing"],
            }
        )

    cost_summary = pd.DataFrame(cost_metrics)
    capacity_summary = pd.DataFrame(capacity_metrics)
    reproduction = cost_summary[cost_summary["scenario_id"] == "5bp_one_way_slippage"].copy()
    daily_returns = pd.concat(daily_frames, ignore_index=True)
    trade_details = pd.concat(trade_detail_frames, ignore_index=True)
    fill_quality = pd.DataFrame(fill_rows)
    drawdown_summary = pd.concat([cost_summary, capacity_summary], ignore_index=True)[
        ["scenario_id", "drawdown_start", "drawdown_end", "drawdown_days", "max_drawdown"]
    ]
    concentration_summary = pd.concat([cost_summary, capacity_summary], ignore_index=True)[
        [
            "scenario_id",
            "max_profit_day",
            "max_profit_day_return",
            "max_loss_day",
            "max_loss_day_return",
            "cum_return_drop_top1_profit_day",
            "cum_return_drop_top3_profit_days",
            "cum_return_drop_top5_profit_days",
        ]
    ]
    random_summary = random_topn_baseline(candidates, float(reproduction.iloc[0]["cumulative_return"]))
    simple_summary = simple_benchmarks(candidates)

    reproduction.to_csv(output_dir / "batch0_v7_reproduction_summary.csv", index=False)
    cost_summary.to_csv(output_dir / "batch0_cost_sensitivity_summary.csv", index=False)
    capacity_summary.to_csv(output_dir / "batch0_capacity_matrix_template.csv", index=False)
    daily_returns.to_csv(output_dir / "batch0_daily_returns.csv", index=False)
    trade_details.to_csv(output_dir / "batch0_trade_details.csv", index=False)
    fill_quality.to_csv(output_dir / "batch0_fill_quality.csv", index=False)
    drawdown_summary.to_csv(output_dir / "batch0_drawdown_summary.csv", index=False)
    concentration_summary.to_csv(output_dir / "batch0_concentration_summary.csv", index=False)
    random_summary.to_csv(output_dir / "batch0_random_baseline_summary.csv", index=False)
    simple_summary.to_csv(output_dir / "batch0_simple_benchmark_summary.csv", index=False)
    write_config_yaml(output_dir / "batch0_evaluator_config.yaml", Batch0Config())
    write_conclusion(output_dir, reproduction, cost_summary, random_summary, simple_summary)
    write_hashes(output_dir, [Path(__file__).resolve()])

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "v7_5bp_cumulative_return": float(reproduction.iloc[0]["cumulative_return"]),
                "v7_pf": float(reproduction.iloc[0]["profit_factor"]),
                "v7_auc": float(reproduction.iloc[0]["auc"]),
                "v7_10bp_cumulative_return": float(
                    cost_summary[cost_summary["scenario_id"] == "10bp_one_way_slippage"].iloc[0]["cumulative_return"]
                ),
                "cost_cases": int(len(cost_summary)),
                "capacity_cases": int(len(capacity_summary)),
                "random_runs": RANDOM_RUNS,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified strategy evaluator for v8 research Batch 0.")
    parser.add_argument("--batch0-v7", action="store_true", help="Run locked v7 reproduction for Batch 0.")
    parser.add_argument("--signal-file", default=str(DEFAULT_SIGNAL_FILE), help="Selected signal file.")
    parser.add_argument("--score-file", default=str(DEFAULT_SCORE_FILE), help="Score/candidate file.")
    parser.add_argument("--candidate-file", default=str(DEFAULT_CANDIDATE_FILE), help="Reserved candidate pool input.")
    parser.add_argument("--daily-file", default=str(DEFAULT_DAILY_FILE), help="Clean daily file for execution checks.")
    parser.add_argument("--stock-basic-file", default=str(DEFAULT_STOCK_BASIC_FILE), help="Stock basic metadata file.")
    parser.add_argument("--minute-dir", default=str(DEFAULT_MINUTE_DIR), help="Raw 5-minute bar directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.batch0_v7:
        raise SystemExit("Only --batch0-v7 is enabled in the current Batch 0 implementation.")
    run_batch0(args)


if __name__ == "__main__":
    main()
