#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
STORE_DIR = ROOT / "data_tushare"
RAW_DIR = STORE_DIR / "raw"
REPORT_DIR = ROOT / "reports" / "tushare"
DATASET_PATH = STORE_DIR / "features" / "tail_dataset.parquet"


@dataclass(frozen=True)
class CostConfig:
    buy_commission: float = 0.00025
    sell_commission: float = 0.00025
    stamp_tax: float = 0.00050
    buy_slippage: float = 0.00050
    sell_slippage: float = 0.00050


@dataclass(frozen=True)
class ExitConfig:
    take_profit_0935: float = 0.015
    take_profit_0945: float = 0.012
    stop_loss: float = -0.018
    weak_loss: float = -0.004
    final_exit_time: str = "10:30"


FEATURE_COLUMNS = [
    "ret_to_prev_close_1450",
    "intraday_ret_1450",
    "tail_ret_1420_1450",
    "tail_ret_1430_1450",
    "late_ret_1445_1450",
    "afternoon_ret_1305_1450",
    "vwap_pos_1450",
    "close_to_high_sofar",
    "close_to_low_sofar",
    "range_sofar",
    "tail_amount_share",
    "tail_amount_vs_prev20",
    "amount_sofar_log",
    "open_gap",
    "prev_ret_1d",
    "prev_ret_3d",
    "prev_ret_5d",
    "prev_volatility_20d",
    "price_vs_ma5_prev",
    "price_vs_ma10_prev",
    "price_vs_ma20_prev",
    "prev_amount_ratio_5_20",
    "open_auction_ret",
    "open_auction_amount_log",
    "prev_close_auction_ret",
    "prev_close_auction_amount_log",
    "prev_net_mf_amount_ratio",
    "prev_elg_net_amount_ratio",
    "prev_ths_net_amount_ratio",
    "market_ret_median_1450",
    "market_breadth_positive",
    "market_tail_ret_median",
    "market_tail_breadth_positive",
    "market_amount_log",
]


def ensure_dirs() -> None:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def safe_div(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return float(a / b - 1.0)


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def vwap_from_amount_vol(
    amount_value: Any,
    vol_value: Any,
    reference_price: Any,
    low_price: Any = np.nan,
    high_price: Any = np.nan,
) -> float:
    amount = _float_or_nan(amount_value)
    vol = _float_or_nan(vol_value)
    reference = _float_or_nan(reference_price)
    fallback = reference if np.isfinite(reference) and reference > 0 else np.nan
    if not (np.isfinite(amount) and amount > 0 and np.isfinite(vol) and vol > 0):
        return fallback

    base = amount / vol
    candidates = [base, base * 100.0, base / 100.0]
    candidates = [c for c in candidates if np.isfinite(c) and c > 0]
    if not candidates:
        return fallback

    low = _float_or_nan(low_price)
    high = _float_or_nan(high_price)
    if np.isfinite(low) and np.isfinite(high) and low > 0 and high >= low:
        lower = low * 0.98
        upper = high * 1.02
        bounded = [c for c in candidates if lower <= c <= upper]
        if bounded:
            return float(min(bounded, key=lambda c: abs(c / reference - 1.0) if fallback else 0.0))

    if fallback:
        closest = min(candidates, key=lambda c: abs(c / reference - 1.0))
        if abs(closest / reference - 1.0) <= 0.10:
            return float(closest)
    return fallback


def vwap(row: pd.Series) -> float:
    return vwap_from_amount_vol(row.get("amount"), row.get("vol"), row.get("close"), row.get("low"), row.get("high"))


def code_limit_threshold(ts_code: str, is_st: bool) -> float:
    if is_st:
        return 0.05
    raw = ts_code.split(".")[0]
    if raw.startswith(("300", "301", "688")):
        return 0.20
    return 0.10


def read_parquet_dir(api_name: str) -> pd.DataFrame:
    base = RAW_DIR / api_name
    paths = sorted(base.glob("*.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)


def load_daily() -> pd.DataFrame:
    df = read_parquet_dir("daily")
    if df.empty:
        raise SystemExit("缺少 daily 数据")
    for col in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).drop_duplicates(["ts_code", "trade_date"], keep="last")
    g = df.groupby("ts_code", group_keys=False)
    df["prev_close_calc"] = g["close"].shift(1)
    df["prev_close_use"] = df["pre_close"].where(df["pre_close"].notna(), df["prev_close_calc"])
    df["prev_ret_1d"] = g["close"].transform(lambda s: s.pct_change().shift(1))
    df["prev_ret_3d"] = g["close"].transform(lambda s: s.shift(1) / s.shift(4) - 1)
    df["prev_ret_5d"] = g["close"].transform(lambda s: s.shift(1) / s.shift(6) - 1)
    df["prev_volatility_20d"] = g["close"].transform(lambda s: s.pct_change().shift(1).rolling(20, min_periods=10).std())
    df["ma5_prev"] = g["close"].transform(lambda s: s.shift(1).rolling(5, min_periods=5).mean())
    df["ma10_prev"] = g["close"].transform(lambda s: s.shift(1).rolling(10, min_periods=8).mean())
    df["ma20_prev"] = g["close"].transform(lambda s: s.shift(1).rolling(20, min_periods=15).mean())
    df["amount5_prev"] = g["amount"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    df["amount20_prev"] = g["amount"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    df["prev_amount_ratio_5_20"] = df["amount5_prev"] / df["amount20_prev"] - 1
    df["next_trade_date"] = g["trade_date"].shift(-1)
    df["prev_trade_date"] = g["trade_date"].shift(1)
    return df


def load_stock_basic() -> pd.DataFrame:
    path = RAW_DIR / "bootstrap" / "stock_basic.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["ts_code", "name"])
    df = pd.read_parquet(path)
    df["name"] = df["name"].astype(str)
    return df


def add_enhanced_features(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    open_auc = read_parquet_dir("stk_auction_o")
    close_auc = read_parquet_dir("stk_auction_c")
    money = read_parquet_dir("moneyflow")
    money_ths = read_parquet_dir("moneyflow_ths")

    if not open_auc.empty:
        for col in ["close", "amount"]:
            open_auc[col] = pd.to_numeric(open_auc[col], errors="coerce")
        open_auc = open_auc[["ts_code", "trade_date", "close", "amount"]].rename(
            columns={"close": "open_auction_close", "amount": "open_auction_amount"}
        )
        out = out.merge(open_auc, on=["ts_code", "trade_date"], how="left")
        out["open_auction_ret"] = out["open_auction_close"] / out["prev_close_use"] - 1
        out["open_auction_amount_log"] = np.log1p(out["open_auction_amount"])
    else:
        out["open_auction_ret"] = np.nan
        out["open_auction_amount_log"] = np.nan

    if not close_auc.empty:
        for col in ["close", "amount"]:
            close_auc[col] = pd.to_numeric(close_auc[col], errors="coerce")
        close_auc = close_auc[["ts_code", "trade_date", "close", "amount"]].rename(
            columns={
                "trade_date": "prev_trade_date",
                "close": "prev_close_auction_close",
                "amount": "prev_close_auction_amount",
            }
        )
        out = out.merge(close_auc, on=["ts_code", "prev_trade_date"], how="left")
        out["prev_close_auction_ret"] = out["prev_close_auction_close"] / out["prev_close_use"] - 1
        out["prev_close_auction_amount_log"] = np.log1p(out["prev_close_auction_amount"])
    else:
        out["prev_close_auction_ret"] = np.nan
        out["prev_close_auction_amount_log"] = np.nan

    if not money.empty:
        for col in ["net_mf_amount", "buy_elg_amount", "sell_elg_amount"]:
            money[col] = pd.to_numeric(money[col], errors="coerce")
        money["prev_elg_net_amount"] = money["buy_elg_amount"] - money["sell_elg_amount"]
        money = money[["ts_code", "trade_date", "net_mf_amount", "prev_elg_net_amount"]].rename(
            columns={"trade_date": "prev_trade_date", "net_mf_amount": "prev_net_mf_amount"}
        )
        out = out.merge(money, on=["ts_code", "prev_trade_date"], how="left")
        out["prev_net_mf_amount_ratio"] = out["prev_net_mf_amount"] / out["amount20_prev"]
        out["prev_elg_net_amount_ratio"] = out["prev_elg_net_amount"] / out["amount20_prev"]
    else:
        out["prev_net_mf_amount_ratio"] = np.nan
        out["prev_elg_net_amount_ratio"] = np.nan

    if not money_ths.empty:
        money_ths["net_amount"] = pd.to_numeric(money_ths["net_amount"], errors="coerce")
        money_ths = money_ths[["ts_code", "trade_date", "net_amount"]].rename(
            columns={"trade_date": "prev_trade_date", "net_amount": "prev_ths_net_amount"}
        )
        out = out.merge(money_ths, on=["ts_code", "prev_trade_date"], how="left")
        out["prev_ths_net_amount_ratio"] = out["prev_ths_net_amount"] / out["amount20_prev"]
    else:
        out["prev_ths_net_amount_ratio"] = np.nan

    return out


def minute_symbol_paths() -> list[Path]:
    base = RAW_DIR / "stk_mins" / "freq=5min"
    return sorted(base.glob("ts_code=*/*.parquet"))


def load_minute_symbol(paths: list[Path]) -> pd.DataFrame:
    df = pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)
    if df.empty:
        return df
    for col in ["open", "high", "low", "close", "vol", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["dt"] = pd.to_datetime(df["trade_time"], errors="coerce")
    df = df.dropna(subset=["dt", "open", "high", "low", "close"])
    df["trade_date"] = df["dt"].dt.strftime("%Y%m%d")
    df["bar_time"] = df["dt"].dt.strftime("%H:%M")
    return df.sort_values("dt").drop_duplicates(["trade_date", "bar_time"], keep="last")


def price_after_costs(buy_vwap: float, sell_vwap: float, costs: CostConfig) -> float:
    buy_cash = buy_vwap * (1 + costs.buy_slippage + costs.buy_commission)
    sell_cash = sell_vwap * (1 - costs.sell_slippage - costs.sell_commission - costs.stamp_tax)
    return sell_cash / buy_cash - 1


def simulate_exit(next_bars: pd.DataFrame, entry_vwap: float, costs: CostConfig, exit_cfg: ExitConfig) -> tuple[float, str, str]:
    by_time = next_bars.drop_duplicates("bar_time", keep="last").set_index("bar_time")

    def bar_price(t: str) -> float | None:
        if t not in by_time.index:
            return None
        return vwap(by_time.loc[t])

    checks = [("09:35", "09:40"), ("09:45", "09:50"), ("10:00", "10:05")]
    for check_time, exec_time in checks:
        if check_time not in by_time.index:
            continue
        check = by_time.loc[check_time]
        ret_now = safe_div(float(check["close"]), entry_vwap)
        should_sell = False
        reason = ""
        if ret_now <= exit_cfg.stop_loss:
            should_sell = True
            reason = "stop_loss"
        elif check_time == "09:35" and ret_now >= exit_cfg.take_profit_0935:
            should_sell = True
            reason = "take_profit_0935"
        elif check_time == "09:45" and ret_now >= exit_cfg.take_profit_0945:
            prev_close = by_time.loc["09:40", "close"] if "09:40" in by_time.index else np.nan
            should_sell = bool(pd.isna(prev_close) or check["close"] <= prev_close)
            reason = "take_profit_0945"
        elif check_time == "09:45" and ret_now <= exit_cfg.weak_loss:
            open_0935 = by_time.loc["09:35", "open"] if "09:35" in by_time.index else np.nan
            should_sell = bool(pd.notna(open_0935) and check["close"] < open_0935)
            reason = "weak_open"
        elif check_time == "10:00" and ret_now < 0:
            should_sell = True
            reason = "not_recovered_1000"
        if should_sell:
            sell_vwap = bar_price(exec_time)
            if sell_vwap is None:
                sell_vwap = float(check["close"])
                exec_time = check_time
            return price_after_costs(entry_vwap, sell_vwap, costs), exec_time, reason

    final_price = bar_price(exit_cfg.final_exit_time)
    if final_price is None:
        available = by_time[by_time.index <= exit_cfg.final_exit_time]
        if available.empty:
            return np.nan, "", "missing_exit"
        final_time = str(available.index[-1])
        final_price = vwap(available.iloc[-1])
    else:
        final_time = exit_cfg.final_exit_time
    return price_after_costs(entry_vwap, final_price, costs), final_time, "time_exit"


def compute_symbol_features(
    ts_code: str,
    minute: pd.DataFrame,
    daily_map: pd.DataFrame,
    is_st: bool,
    min_amount_sofar: float,
    costs: CostConfig,
    exit_cfg: ExitConfig,
) -> pd.DataFrame:
    if minute.empty or daily_map.empty:
        return pd.DataFrame()
    daily_by_date = daily_map.set_index("trade_date", drop=False)
    minute_groups = {date: group.sort_values("dt") for date, group in minute.groupby("trade_date")}
    rows: list[dict[str, Any]] = []
    for date, bars in minute_groups.items():
        if date not in daily_by_date.index:
            continue
        drow = daily_by_date.loc[date]
        if pd.isna(drow.get("prev_close_use")) or pd.isna(drow.get("ma20_prev")):
            continue
        by_time = bars.drop_duplicates("bar_time", keep="last").set_index("bar_time")
        needed = {"09:35", "13:05", "14:20", "14:30", "14:45", "14:50", "14:55"}
        if not needed.issubset(set(by_time.index)):
            continue
        past = bars[bars["bar_time"] <= "14:50"]
        if past.empty:
            continue
        amount_sofar = float(past["amount"].sum())
        if amount_sofar < min_amount_sofar:
            continue
        close_1450 = float(by_time.loc["14:50", "close"])
        high_sofar = float(past["high"].max())
        low_sofar = float(past["low"].min())
        prev_close = float(drow["prev_close_use"])
        ret_to_prev = safe_div(close_1450, prev_close)
        limit = code_limit_threshold(ts_code, is_st)
        if pd.isna(ret_to_prev) or ret_to_prev > limit - 0.008 or ret_to_prev < -limit + 0.008:
            continue
        next_date = drow.get("next_trade_date")
        if not isinstance(next_date, str) or next_date not in minute_groups:
            continue
        entry_vwap = vwap(by_time.loc["14:55"])
        target_return, exit_time, exit_reason = simulate_exit(minute_groups[next_date], entry_vwap, costs, exit_cfg)
        if pd.isna(target_return):
            continue

        amount20_prev = float(drow.get("amount20_prev", np.nan))
        tail = bars[(bars["bar_time"] >= "14:30") & (bars["bar_time"] <= "14:50")]
        tail_amount = float(tail["amount"].sum())
        sofar_vwap = vwap_from_amount_vol(past["amount"].sum(), past["vol"].sum(), close_1450, low_sofar, high_sofar)
        open_0935 = float(by_time.loc["09:35", "open"])
        rows.append(
            {
                "trade_date": date,
                "date": f"{date[:4]}-{date[4:6]}-{date[6:8]}",
                "ts_code": ts_code,
                "entry_vwap": entry_vwap,
                "exit_time": exit_time,
                "exit_reason": exit_reason,
                "target_return": target_return,
                "target_win": int(target_return > 0),
                "ret_to_prev_close_1450": ret_to_prev,
                "intraday_ret_1450": safe_div(close_1450, open_0935),
                "tail_ret_1420_1450": safe_div(close_1450, float(by_time.loc["14:20", "close"])),
                "tail_ret_1430_1450": safe_div(close_1450, float(by_time.loc["14:30", "close"])),
                "late_ret_1445_1450": safe_div(close_1450, float(by_time.loc["14:45", "close"])),
                "afternoon_ret_1305_1450": safe_div(close_1450, float(by_time.loc["13:05", "close"])),
                "vwap_pos_1450": safe_div(close_1450, sofar_vwap),
                "close_to_high_sofar": safe_div(close_1450, high_sofar),
                "close_to_low_sofar": safe_div(close_1450, low_sofar),
                "range_sofar": safe_div(high_sofar, low_sofar),
                "tail_amount_share": tail_amount / amount_sofar if amount_sofar > 0 else np.nan,
                "tail_amount_vs_prev20": tail_amount / (amount20_prev * (5 / 48)) - 1 if amount20_prev > 0 else np.nan,
                "amount_sofar_log": math.log1p(amount_sofar),
                "open_gap": safe_div(open_0935, prev_close),
                **{col: drow.get(col) for col in FEATURE_COLUMNS if col.startswith("prev_") or col.startswith("open_auction")},
                "price_vs_ma5_prev": safe_div(close_1450, float(drow["ma5_prev"])),
                "price_vs_ma10_prev": safe_div(close_1450, float(drow["ma10_prev"])),
                "price_vs_ma20_prev": safe_div(close_1450, float(drow["ma20_prev"])),
            }
        )
    return pd.DataFrame(rows)


def add_market_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("trade_date")
    market = grouped.agg(
        market_ret_median_1450=("ret_to_prev_close_1450", "median"),
        market_breadth_positive=("ret_to_prev_close_1450", lambda s: float((s > 0).mean())),
        market_tail_ret_median=("tail_ret_1420_1450", "median"),
        market_tail_breadth_positive=("tail_ret_1420_1450", lambda s: float((s > 0).mean())),
        market_amount=("amount_sofar_log", lambda s: float(np.expm1(s).sum())),
    )
    market["market_amount_log"] = np.log1p(market["market_amount"])
    market = market.drop(columns=["market_amount"])
    return df.merge(market.reset_index(), on="trade_date", how="left")


def command_build_dataset(args: argparse.Namespace) -> None:
    ensure_dirs()
    costs = CostConfig()
    exit_cfg = ExitConfig()
    stock_basic = load_stock_basic()
    st_names = set(stock_basic.loc[stock_basic["name"].str.contains("ST", case=False, na=False), "ts_code"].astype(str))
    daily = add_enhanced_features(load_daily())
    daily_by_symbol = {code: group.sort_values("trade_date") for code, group in daily.groupby("ts_code")}
    paths_by_symbol: dict[str, list[Path]] = {}
    for path in minute_symbol_paths():
        ts_code = path.parent.name.split("=", 1)[1]
        paths_by_symbol.setdefault(ts_code, []).append(path)
    if args.max_symbols:
        paths_by_symbol = dict(list(sorted(paths_by_symbol.items()))[: args.max_symbols])

    parts = []
    processed = 0
    for ts_code, paths in sorted(paths_by_symbol.items()):
        processed += 1
        if ts_code not in daily_by_symbol:
            continue
        minute = load_minute_symbol(paths)
        feats = compute_symbol_features(
            ts_code,
            minute,
            daily_by_symbol[ts_code],
            ts_code in st_names,
            args.min_amount_sofar,
            costs,
            exit_cfg,
        )
        if not feats.empty:
            parts.append(feats)
        if processed % 100 == 0:
            print(f"processed_symbols={processed} feature_parts={len(parts)} rows={sum(len(p) for p in parts)}", flush=True)

    if not parts:
        raise SystemExit("没有生成特征样本；需要先下载至少相邻两个交易日的 stk_mins")
    df = pd.concat(parts, ignore_index=True)
    df = add_market_cross_section(df).replace([np.inf, -np.inf], np.nan)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_PATH, index=False, compression="zstd")
    audit = {
        "dataset_path": str(DATASET_PATH),
        "rows": int(len(df)),
        "symbols": int(df["ts_code"].nunique()),
        "date_min": str(df["trade_date"].min()),
        "date_max": str(df["trade_date"].max()),
        "win_rate_all_samples": float(df["target_win"].mean()),
        "avg_target_return_all_samples": float(df["target_return"].mean()),
        "cost_config": asdict(costs),
        "exit_config": asdict(exit_cfg),
        "filters": {"min_amount_sofar": args.min_amount_sofar},
    }
    (REPORT_DIR / "dataset_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def model_from_params(params: dict[str, Any]) -> Pipeline:
    model = HistGradientBoostingClassifier(
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        max_leaf_nodes=params["max_leaf_nodes"],
        l2_regularization=params["l2_regularization"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=42,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def add_predictions(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["p_win"] = model.predict_proba(out[FEATURE_COLUMNS])[:, 1]
    out["score"] = out["p_win"]
    return out


def topn_by_day(pred: pd.DataFrame, n: int) -> pd.DataFrame:
    return pred.sort_values(["trade_date", "score"], ascending=[True, False]).groupby("trade_date", group_keys=False).head(n)


def max_drawdown(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return np.nan
    equity = (1 + daily_returns.fillna(0)).cumprod()
    return float((equity / equity.cummax() - 1).min())


def profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else np.nan
    return float(gains / losses)


def evaluate(pred: pd.DataFrame, name: str, n: int) -> dict[str, Any]:
    if pred.empty:
        return {"name": name, "rows": 0}
    selected = topn_by_day(pred, n)
    daily = selected.groupby("trade_date")["target_return"].mean()
    return {
        "name": name,
        "rows": int(len(pred)),
        "days": int(pred["trade_date"].nunique()),
        "selected_trades": int(len(selected)),
        "selected_days": int(selected["trade_date"].nunique()),
        "auc": float(roc_auc_score(pred["target_win"], pred["p_win"])) if pred["target_win"].nunique() > 1 else np.nan,
        "all_sample_win_rate": float(pred["target_win"].mean()),
        "all_sample_avg_return": float(pred["target_return"].mean()),
        "top10_trade_win_rate": float((selected["target_return"] > 0).mean()) if len(selected) else np.nan,
        "top10_avg_trade_return": float(selected["target_return"].mean()) if len(selected) else np.nan,
        "top10_daily_win_rate": float((daily > 0).mean()) if len(daily) else np.nan,
        "top10_avg_daily_return": float(daily.mean()) if len(daily) else np.nan,
        "top10_cumulative_return": float((1 + daily).prod() - 1) if len(daily) else np.nan,
        "top10_max_drawdown": max_drawdown(daily),
        "top10_profit_factor": profit_factor(selected["target_return"]) if len(selected) else np.nan,
    }


def fold_starts(train_dates: list[str], min_train_days: int = 20, test_days: int = 10) -> list[tuple[str, str]]:
    folds = []
    i = min_train_days
    while i < len(train_dates):
        start = train_dates[i]
        end = train_dates[min(i + test_days, len(train_dates) - 1)]
        folds.append((start, end))
        i += test_days
    return folds


def tune_on_train(train_df: pd.DataFrame, top_n: int) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    grid = [
        {"learning_rate": lr, "max_iter": it, "max_leaf_nodes": leaf, "l2_regularization": l2, "min_samples_leaf": 80}
        for lr in [0.025, 0.04]
        for it in [160, 260]
        for leaf in [15, 31]
        for l2 in [0.02, 0.10]
    ]
    dates = sorted(train_df["trade_date"].unique())
    folds = fold_starts(dates)
    rows = []
    best_score = -1e9
    best_params = grid[0]
    best_pred_parts = []
    for params in grid:
        pred_parts = []
        for start, end in folds:
            fit = train_df[train_df["trade_date"] < start]
            test = train_df[(train_df["trade_date"] >= start) & (train_df["trade_date"] < end)]
            if len(fit) < 500 or test.empty or fit["target_win"].nunique() < 2:
                continue
            model = model_from_params(params).fit(fit[FEATURE_COLUMNS], fit["target_win"].astype(int))
            pred = add_predictions(model, test)
            pred["fold_start"] = start
            pred_parts.append(pred)
        wf_pred = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
        metrics = evaluate(wf_pred, "train_walk_forward", top_n)
        score = float(metrics.get("top10_avg_daily_return", np.nan))
        if np.isnan(score):
            score = -1e9
        rows.append({**params, **metrics, "selection_score": score})
        if score > best_score:
            best_score = score
            best_params = params
            best_pred_parts = pred_parts
    tuning = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
    best_wf = pd.concat(best_pred_parts, ignore_index=True) if best_pred_parts else pd.DataFrame()
    return best_params, tuning, best_wf


def random_baseline(pred: pd.DataFrame, n: int, seeds: int) -> dict[str, float]:
    groups = {d: g["target_return"].to_numpy() for d, g in pred.groupby("trade_date")}
    rng = np.random.default_rng(20260521)
    avg_daily = []
    cum = []
    daily_win = []
    for _ in range(seeds):
        returns = []
        for arr in groups.values():
            if len(arr) <= n:
                pick = arr
            else:
                pick = arr[rng.choice(len(arr), size=n, replace=False)]
            returns.append(float(np.mean(pick)))
        s = pd.Series(returns)
        avg_daily.append(float(s.mean()))
        cum.append(float((1 + s).prod() - 1))
        daily_win.append(float((s > 0).mean()))
    return {
        "random_top10_avg_daily_return_mean": float(np.mean(avg_daily)),
        "random_top10_avg_daily_return_p95": float(np.quantile(avg_daily, 0.95)),
        "random_top10_cumulative_return_mean": float(np.mean(cum)),
        "random_top10_cumulative_return_p95": float(np.quantile(cum, 0.95)),
        "random_top10_daily_win_rate_mean": float(np.mean(daily_win)),
    }


def command_train_evaluate(args: argparse.Namespace) -> None:
    ensure_dirs()
    if not DATASET_PATH.exists():
        raise SystemExit(f"缺少数据集: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["target_return", "target_win"]).copy()
    df["trade_date"] = df["trade_date"].astype(str)
    train_df = df[df["trade_date"] < args.validation_start].copy()
    val_df = df[df["trade_date"] >= args.validation_start].copy()
    if train_df.empty or val_df.empty:
        raise SystemExit(f"训练集或验证集为空: validation_start={args.validation_start}")
    if train_df["target_win"].nunique() < 2:
        raise SystemExit("训练集 target_win 只有一个类别，无法训练分类模型")

    best_params, tuning, wf_pred = tune_on_train(train_df, args.top_n)
    tuning.to_csv(REPORT_DIR / "tuning_results.csv", index=False)
    if not wf_pred.empty:
        wf_pred.to_csv(REPORT_DIR / "walk_forward_predictions.csv", index=False)
    final_model = model_from_params(best_params).fit(train_df[FEATURE_COLUMNS], train_df["target_win"].astype(int))
    val_pred = add_predictions(final_model, val_df)
    val_pred.to_csv(REPORT_DIR / "validation_predictions.csv", index=False)
    top10 = topn_by_day(val_pred, args.top_n)
    top10.to_csv(REPORT_DIR / "top10_validation.csv", index=False)
    metrics = {
        "dataset": {
            "rows": int(len(df)),
            "symbols": int(df["ts_code"].nunique()),
            "date_min": str(df["trade_date"].min()),
            "date_max": str(df["trade_date"].max()),
            "validation_start": args.validation_start,
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "train_symbols": int(train_df["ts_code"].nunique()),
            "validation_symbols": int(val_df["ts_code"].nunique()),
        },
        "best_params_selected_on_train_only": best_params,
        "walk_forward": evaluate(wf_pred, "train_walk_forward", args.top_n),
        "validation": evaluate(val_pred, "validation", args.top_n),
        "validation_random_baseline": random_baseline(val_pred, args.top_n, args.random_seeds),
        "feature_columns": FEATURE_COLUMNS,
        "leakage_controls": [
            "流动性下载排序使用 validation_start 之前的训练期数据。",
            "超参数搜索只使用 validation_start 之前的训练期 walk-forward。",
            "最终模型仅用训练集拟合；验证集只做一次最终评估。",
            "模型特征不使用当日收盘集合竞价、当日全日资金流或次日数据。",
        ],
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_report(metrics, top10)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))


def pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.2f}%"


def write_report(metrics: dict[str, Any], top10: pd.DataFrame) -> None:
    dataset = metrics["dataset"]
    wf = metrics.get("walk_forward", {})
    validation = metrics.get("validation", {})
    baseline = metrics.get("validation_random_baseline", {})
    latest_date = top10["trade_date"].max() if not top10.empty else ""
    latest = top10[top10["trade_date"].eq(latest_date)].copy() if latest_date else pd.DataFrame()
    latest_md = latest[["date", "ts_code", "p_win", "target_return", "exit_time", "exit_reason"]].to_markdown(index=False) if not latest.empty else "无"
    text = f"""# Tushare 尾盘隔夜模型回测报告

## 数据切分

- 样本区间：{dataset['date_min']} 至 {dataset['date_max']}
- 验证集起点：{dataset['validation_start']}
- 总样本：{dataset['rows']}，股票数：{dataset['symbols']}
- 训练样本：{dataset['train_rows']}，验证样本：{dataset['validation_rows']}

## 训练期 Walk-forward

- Top10 日均收益：{pct(wf.get('top10_avg_daily_return'))}
- Top10 累计收益：{pct(wf.get('top10_cumulative_return'))}
- Top10 逐笔胜率：{pct(wf.get('top10_trade_win_rate'))}
- Top10 日胜率：{pct(wf.get('top10_daily_win_rate'))}
- 最大回撤：{pct(wf.get('top10_max_drawdown'))}
- AUC：{wf.get('auc')}

## 最近三个月验证

- Top10 日均收益：{pct(validation.get('top10_avg_daily_return'))}
- Top10 累计收益：{pct(validation.get('top10_cumulative_return'))}
- Top10 逐笔胜率：{pct(validation.get('top10_trade_win_rate'))}
- Top10 日胜率：{pct(validation.get('top10_daily_win_rate'))}
- 最大回撤：{pct(validation.get('top10_max_drawdown'))}
- Profit Factor：{validation.get('top10_profit_factor')}
- AUC：{validation.get('auc')}

## 随机 Top10 基准

- 随机 Top10 日均收益均值：{pct(baseline.get('random_top10_avg_daily_return_mean'))}
- 随机 Top10 日均收益 95 分位：{pct(baseline.get('random_top10_avg_daily_return_p95'))}
- 随机 Top10 累计收益均值：{pct(baseline.get('random_top10_cumulative_return_mean'))}
- 随机 Top10 累计收益 95 分位：{pct(baseline.get('random_top10_cumulative_return_p95'))}
- 随机 Top10 日胜率均值：{pct(baseline.get('random_top10_daily_win_rate_mean'))}

## 最新验证日 Top10

{latest_md}

## 防泄漏说明

验证集不参与下载优先级的流动性排序、不参与调参、不参与模型训练。当前日特征只使用 14:50 及以前分钟线、当日开盘集合竞价、前一交易日集合竞价和前一交易日资金流。
"""
    (REPORT_DIR / "report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tushare local tail-session overnight model")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-dataset")
    build.add_argument("--min-amount-sofar", type=float, default=20_000_000)
    build.add_argument("--max-symbols", type=int, default=None)
    build.set_defaults(func=command_build_dataset)

    train = sub.add_parser("train-evaluate")
    train.add_argument("--validation-start", default="20260224")
    train.add_argument("--top-n", type=int, default=10)
    train.add_argument("--random-seeds", type=int, default=100)
    train.set_defaults(func=command_train_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
