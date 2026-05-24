#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RANK_PATH = ROOT / "data_tushare" / "manifests" / "liquid_top3000_20251120_20260213.csv"
DAILY_FILE = ROOT / "data_tushare" / "clean" / "daily_repaired_top3000.parquet"
MINUTE_DIR = ROOT / "data_tushare" / "raw" / "stk_mins" / "freq=5min"
AUCTION_RAW_DIR = ROOT / "data_tushare" / "raw"
STOCK_BASIC_FILE = ROOT / "data_tushare" / "raw" / "bootstrap" / "stock_basic.parquet"
OUTPUT_ROOT = ROOT / "reports" / "tushare" / "v8_forward_shadow"
REQUIRED_BARS = {"09:35", "13:05", "14:20", "14:30", "14:45", "14:50"}
FORBIDDEN_COLUMNS = {
    "target_return",
    "target_win",
    "label_win",
    "label_static_003",
    "label_dynamic_amp10",
    "entry_vwap",
    "exit_time",
    "exit_reason",
}


def beijing_now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def safe_div(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return float(a / b - 1.0)


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def vwap_from_amount_vol(amount_value: Any, vol_value: Any, reference_price: Any, low_price: Any = np.nan, high_price: Any = np.nan) -> float:
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
            return float(min(bounded, key=lambda c: abs(c / reference - 1.0) if np.isfinite(fallback) else 0.0))
    if np.isfinite(fallback):
        closest = min(candidates, key=lambda c: abs(c / reference - 1.0))
        if abs(closest / reference - 1.0) <= 0.10:
            return float(closest)
    return fallback


def code_limit_threshold(ts_code: str, is_st: bool) -> float:
    if is_st:
        return 0.05
    raw = ts_code.split(".")[0]
    if raw.startswith(("300", "301", "688")):
        return 0.20
    return 0.10


def overlaps(path: Path, trade_date: str) -> bool:
    if "_" not in path.stem:
        return False
    start, end = path.stem.split("_", 1)
    return start <= trade_date <= end


def load_minute_for_date(ts_code: str, trade_date: str, minute_dir: Path = MINUTE_DIR) -> pd.DataFrame:
    symbol_dir = minute_dir / f"ts_code={ts_code}"
    frames = []
    cols = ["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"]
    for path in sorted(symbol_dir.glob("*.parquet")):
        if not overlaps(path, trade_date):
            continue
        df = pd.read_parquet(path, columns=cols)
        if df.empty:
            continue
        df["trade_date"] = df["trade_time"].astype(str).str.slice(0, 10).str.replace("-", "", regex=False)
        df = df[df["trade_date"].eq(trade_date)].copy()
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in ["open", "high", "low", "close", "vol", "amount"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["dt"] = pd.to_datetime(out["trade_time"], errors="coerce")
    out = out.dropna(subset=["dt", "open", "high", "low", "close"])
    out["bar_time"] = out["dt"].dt.strftime("%H:%M")
    return out.sort_values("dt").drop_duplicates(["trade_date", "bar_time"], keep="last")


def load_rank(top_rank: int, rank_file: Path = RANK_PATH) -> pd.DataFrame:
    rank = pd.read_csv(rank_file, dtype={"ts_code": str})
    rank["liquidity_rank"] = pd.to_numeric(rank["liquidity_rank"], errors="coerce")
    return rank.sort_values("liquidity_rank").drop_duplicates("ts_code", keep="first").head(top_rank)


def load_stock_basic() -> pd.DataFrame:
    if not STOCK_BASIC_FILE.exists():
        return pd.DataFrame(columns=["ts_code", "name", "industry", "market"])
    df = pd.read_parquet(STOCK_BASIC_FILE)
    keep = [c for c in ["ts_code", "name", "industry", "market", "list_status", "list_date", "delist_date"] if c in df.columns]
    out = df[keep].copy()
    out["ts_code"] = out["ts_code"].astype(str)
    if "name" in out.columns:
        out["name"] = out["name"].astype(str)
    return out


def raw_auction_path(api_name: str, trade_date: str, auction_raw_dir: Path = AUCTION_RAW_DIR) -> Path:
    return auction_raw_dir / api_name / f"trade_date={trade_date}.parquet"


def read_raw_auction(api_name: str, trade_date: str, auction_raw_dir: Path = AUCTION_RAW_DIR) -> pd.DataFrame:
    path = raw_auction_path(api_name, trade_date, auction_raw_dir)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty or "ts_code" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["ts_code"] = out["ts_code"].astype(str)
    for col in ["close", "amount"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def overlay_auction_sidecar(daily_day: pd.DataFrame, trade_date: str, auction_raw_dir: Path = AUCTION_RAW_DIR) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = daily_day.copy()
    meta: dict[str, Any] = {
        "open_auction_file": str(raw_auction_path("stk_auction_o", trade_date, auction_raw_dir)),
        "open_auction_rows": 0,
        "open_auction_overlay_rows": 0,
        "prev_close_auction_files": [],
        "prev_close_auction_rows": 0,
        "prev_close_auction_overlay_rows": 0,
    }
    open_auc = read_raw_auction("stk_auction_o", trade_date, auction_raw_dir)
    meta["open_auction_rows"] = int(len(open_auc))
    if not open_auc.empty:
        open_auc = open_auc[["ts_code", "close", "amount"]].rename(
            columns={"close": "open_auction_close_sidecar", "amount": "open_auction_amount_sidecar"}
        )
        out = out.merge(open_auc, on="ts_code", how="left")
        mask = out["open_auction_close_sidecar"].notna()
        out.loc[mask, "open_auction_close"] = out.loc[mask, "open_auction_close_sidecar"]
        out.loc[mask, "open_auction_amount"] = out.loc[mask, "open_auction_amount_sidecar"]
        out.loc[mask, "open_auction_ret"] = out.loc[mask, "open_auction_close"] / out.loc[mask, "prev_close_use"] - 1
        out.loc[mask, "open_auction_amount_log"] = np.log1p(out.loc[mask, "open_auction_amount"])
        meta["open_auction_overlay_rows"] = int(mask.sum())
        out = out.drop(columns=["open_auction_close_sidecar", "open_auction_amount_sidecar"])

    close_frames = []
    if "prev_trade_date" in out.columns:
        prev_dates = sorted(set(out["prev_trade_date"].dropna().astype(str)))
        for prev_date in prev_dates:
            close_auc = read_raw_auction("stk_auction_c", prev_date, auction_raw_dir)
            path = raw_auction_path("stk_auction_c", prev_date, auction_raw_dir)
            meta["prev_close_auction_files"].append({"trade_date": prev_date, "path": str(path), "rows": int(len(close_auc))})
            if not close_auc.empty:
                close_auc = close_auc[["ts_code", "close", "amount"]].rename(
                    columns={"close": "prev_close_auction_close_sidecar", "amount": "prev_close_auction_amount_sidecar"}
                )
                close_auc["prev_trade_date"] = prev_date
                close_frames.append(close_auc)
    if close_frames:
        close_all = pd.concat(close_frames, ignore_index=True)
        meta["prev_close_auction_rows"] = int(len(close_all))
        out = out.merge(close_all, on=["ts_code", "prev_trade_date"], how="left")
        mask = out["prev_close_auction_close_sidecar"].notna()
        out.loc[mask, "prev_close_auction_close"] = out.loc[mask, "prev_close_auction_close_sidecar"]
        out.loc[mask, "prev_close_auction_amount"] = out.loc[mask, "prev_close_auction_amount_sidecar"]
        out.loc[mask, "prev_close_auction_ret"] = out.loc[mask, "prev_close_auction_close"] / out.loc[mask, "prev_close_use"] - 1
        out.loc[mask, "prev_close_auction_amount_log"] = np.log1p(out.loc[mask, "prev_close_auction_amount"])
        meta["prev_close_auction_overlay_rows"] = int(mask.sum())
        out = out.drop(columns=["prev_close_auction_close_sidecar", "prev_close_auction_amount_sidecar"])
    return out, meta


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


def build_forward_features(
    trade_date: str,
    top_rank: int,
    min_amount_sofar: float,
    output_root: Path,
    rank_file: Path = RANK_PATH,
    daily_file: Path = DAILY_FILE,
    minute_dir: Path = MINUTE_DIR,
    auction_raw_dir: Path = AUCTION_RAW_DIR,
) -> dict[str, Any]:
    rank = load_rank(top_rank, rank_file)
    rank_map = dict(zip(rank["ts_code"], rank["liquidity_rank"]))
    daily = pd.read_parquet(daily_file)
    daily["trade_date"] = daily["trade_date"].astype(str)
    daily["ts_code"] = daily["ts_code"].astype(str)
    daily_day = daily[daily["trade_date"].eq(trade_date)].copy()
    daily_day, auction_overlay = overlay_auction_sidecar(daily_day, trade_date, auction_raw_dir)
    daily_by_code = daily_day.set_index("ts_code", drop=False)
    stock = load_stock_basic()
    stock_map = stock.set_index("ts_code", drop=False)
    st_names = set(stock.loc[stock.get("name", pd.Series(dtype=str)).astype(str).str.contains("ST", case=False, na=False), "ts_code"]) if not stock.empty else set()

    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for ts_code in rank["ts_code"].astype(str).tolist():
        if ts_code not in daily_by_code.index:
            skipped["missing_daily"] = skipped.get("missing_daily", 0) + 1
            continue
        drow = daily_by_code.loc[ts_code]
        if pd.isna(drow.get("prev_close_use")) or pd.isna(drow.get("ma20_prev")):
            skipped["missing_daily_history"] = skipped.get("missing_daily_history", 0) + 1
            continue
        minute = load_minute_for_date(ts_code, trade_date, minute_dir)
        if minute.empty:
            skipped["missing_minute"] = skipped.get("missing_minute", 0) + 1
            continue
        by_time = minute.drop_duplicates("bar_time", keep="last").set_index("bar_time")
        if not REQUIRED_BARS.issubset(set(by_time.index)):
            skipped["missing_required_bars"] = skipped.get("missing_required_bars", 0) + 1
            continue
        past = minute[minute["bar_time"] <= "14:50"]
        if past.empty:
            skipped["empty_past"] = skipped.get("empty_past", 0) + 1
            continue
        amount_sofar = float(past["amount"].sum())
        if amount_sofar < min_amount_sofar:
            skipped["low_amount_sofar"] = skipped.get("low_amount_sofar", 0) + 1
            continue
        close_1450 = float(by_time.loc["14:50", "close"])
        high_sofar = float(past["high"].max())
        low_sofar = float(past["low"].min())
        prev_close = float(drow["prev_close_use"])
        ret_to_prev = safe_div(close_1450, prev_close)
        is_st = ts_code in st_names
        limit = code_limit_threshold(ts_code, is_st)
        if pd.isna(ret_to_prev) or ret_to_prev > limit - 0.008 or ret_to_prev < -limit + 0.008:
            skipped["near_limit"] = skipped.get("near_limit", 0) + 1
            continue
        amount20_prev = float(drow.get("amount20_prev", np.nan))
        tail = minute[(minute["bar_time"] >= "14:30") & (minute["bar_time"] <= "14:50")]
        tail_amount = float(tail["amount"].sum())
        sofar_vwap = vwap_from_amount_vol(past["amount"].sum(), past["vol"].sum(), close_1450, low_sofar, high_sofar)
        open_0935 = float(by_time.loc["09:35", "open"])
        basic = stock_map.loc[ts_code] if ts_code in stock_map.index else {}
        row = {
            "trade_date": trade_date,
            "date": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}",
            "ts_code": ts_code,
            "name": basic.get("name", "") if hasattr(basic, "get") else "",
            "industry": basic.get("industry", "") if hasattr(basic, "get") else "",
            "market": basic.get("market", "") if hasattr(basic, "get") else "",
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
            "prev_ret_1d": drow.get("prev_ret_1d"),
            "prev_ret_3d": drow.get("prev_ret_3d"),
            "prev_ret_5d": drow.get("prev_ret_5d"),
            "prev_volatility_20d": drow.get("prev_volatility_20d"),
            "price_vs_ma5_prev": safe_div(close_1450, float(drow["ma5_prev"])),
            "price_vs_ma10_prev": safe_div(close_1450, float(drow["ma10_prev"])),
            "price_vs_ma20_prev": safe_div(close_1450, float(drow["ma20_prev"])),
            "prev_amount_ratio_5_20": drow.get("prev_amount_ratio_5_20"),
            "open_auction_ret": drow.get("open_auction_ret"),
            "open_auction_amount_log": drow.get("open_auction_amount_log"),
            "prev_close_auction_ret": drow.get("prev_close_auction_ret"),
            "prev_close_auction_amount_log": drow.get("prev_close_auction_amount_log"),
            "prev_net_mf_amount_ratio": drow.get("prev_net_mf_amount_ratio"),
            "prev_elg_net_amount_ratio": drow.get("prev_elg_net_amount_ratio"),
            "prev_ths_net_amount_ratio": drow.get("prev_ths_net_amount_ratio"),
            "liquidity_rank": rank_map.get(ts_code),
        }
        rows.append(row)

    features = pd.DataFrame(rows)
    if not features.empty:
        features = add_market_cross_section(features).replace([np.inf, -np.inf], np.nan)
    leaked = sorted(FORBIDDEN_COLUMNS & set(features.columns))
    if leaked:
        raise AssertionError(f"forward feature output leaked forbidden columns: {leaked}")

    out_dir = output_root / "test_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{trade_date}_forward_features.csv"
    meta_path = out_dir / f"{trade_date}_forward_features_meta.json"
    features.to_csv(out_path, index=False)
    meta = {
        "trade_date": trade_date,
        "generated_time_beijing": beijing_now(),
        "data_max_timestamp": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} 14:50:00",
        "rows": int(len(features)),
        "top_rank": int(top_rank),
        "min_amount_sofar": float(min_amount_sofar),
        "skipped": skipped,
        "uses_15_or_next_day_for_features": False,
        "paper_replay_feature_file": True,
        "rank_file": str(rank_file),
        "daily_file": str(daily_file),
        "minute_dir": str(minute_dir),
        "auction_raw_dir": str(auction_raw_dir),
        "auction_overlay": auction_overlay,
        "output_path": str(out_path),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"feature_path": str(out_path), "meta_path": str(meta_path), **meta}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build forward-only <=14:50 features for paper replay/testing.")
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--top-rank", type=int, default=3000)
    parser.add_argument("--min-amount-sofar", type=float, default=20_000_000)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--rank-file", default=str(RANK_PATH))
    parser.add_argument("--daily-file", default=str(DAILY_FILE))
    parser.add_argument("--minute-dir", default=str(MINUTE_DIR))
    parser.add_argument("--auction-raw-dir", default=str(AUCTION_RAW_DIR))
    args = parser.parse_args()
    result = build_forward_features(
        trade_date=str(args.trade_date),
        top_rank=int(args.top_rank),
        min_amount_sofar=float(args.min_amount_sofar),
        output_root=Path(args.output_root),
        rank_file=Path(args.rank_file),
        daily_file=Path(args.daily_file),
        minute_dir=Path(args.minute_dir),
        auction_raw_dir=Path(args.auction_raw_dir),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
