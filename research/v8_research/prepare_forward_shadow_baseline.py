#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tushare_data_pipeline import TushareError, TushareProxyClient, TushareRateLimit, get_proxy_url, get_token, write_parquet_atomic  # noqa: E402

LOCKED_RANK_FILE = ROOT / "locked_artifacts" / "v7_cap20_strong_label_003" / "config" / "liquid_top3000_20251120_20260213.csv"
DEFAULT_RANK_FILE = ROOT / "data_tushare" / "manifests" / "liquid_top3000_20251120_20260213.csv"
DEFAULT_STOCK_BASIC = ROOT / "data_tushare" / "raw" / "bootstrap" / "stock_basic.parquet"
DEFAULT_TRADE_CAL = ROOT / "data_tushare" / "raw" / "bootstrap" / "trade_cal.parquet"
DEFAULT_DAILY_FILE = ROOT / "data_tushare" / "clean" / "daily_repaired_top3000.parquet"
RAW_DIR = ROOT / "data_tushare" / "raw"
REPORT_DIR = ROOT / "reports" / "tushare" / "v8_forward_shadow" / "data_preparation"
BEIJING_TZ = timezone(timedelta(hours=8))


class RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.interval = 60.0 / max(1, int(requests_per_minute))
        self.last = 0.0

    def wait(self) -> None:
        delay = self.interval - (time.monotonic() - self.last)
        if delay > 0:
            time.sleep(delay)
        self.last = time.monotonic()


def bj_now() -> str:
    return datetime.now(BEIJING_TZ).isoformat(timespec="seconds")


def ymd(value: datetime) -> str:
    return value.strftime("%Y%m%d")


def ymd_to_dt(value: str) -> datetime:
    return datetime.strptime(str(value), "%Y%m%d")


def ensure_rank_file(rank_file: Path) -> None:
    if rank_file.exists():
        return
    rank_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LOCKED_RANK_FILE, rank_file)


def call_with_retry(
    client: TushareProxyClient,
    limiter: RateLimiter,
    api_name: str,
    params: dict[str, Any],
    fields: str | None,
    max_retries: int,
) -> pd.DataFrame:
    attempt = 0
    while True:
        attempt += 1
        limiter.wait()
        try:
            return client.call(api_name, params, fields=fields)
        except TushareRateLimit:
            if attempt > max_retries:
                raise
            time.sleep(min(90, 8 * attempt))
        except TushareError:
            if attempt > max_retries:
                raise
            time.sleep(min(45, 5 * attempt))


def fetch_to_parquet(
    client: TushareProxyClient,
    limiter: RateLimiter,
    api_name: str,
    params: dict[str, Any],
    fields: str | None,
    path: Path,
    refresh: bool,
    max_retries: int,
) -> dict[str, Any]:
    if path.exists() and path.stat().st_size > 0 and not refresh:
        try:
            rows = len(pd.read_parquet(path))
        except Exception:
            rows = -1
        return {"api_name": api_name, "path": str(path), "status": "cached", "rows": rows}
    df = call_with_retry(client, limiter, api_name, params, fields, max_retries)
    checksum = write_parquet_atomic(df, str(path))
    return {"api_name": api_name, "path": str(path), "status": "fetched", "rows": int(len(df)), "sha1": checksum}


def load_rank(rank_file: Path, top_rank: int) -> pd.DataFrame:
    rank = pd.read_csv(rank_file, dtype={"ts_code": str})
    rank["liquidity_rank"] = pd.to_numeric(rank["liquidity_rank"], errors="coerce")
    return rank.sort_values("liquidity_rank").drop_duplicates("ts_code", keep="first").head(top_rank)


def load_or_fetch_trade_cal(
    client: TushareProxyClient,
    limiter: RateLimiter,
    start_date: str,
    end_date: str,
    refresh: bool,
    max_retries: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    DEFAULT_TRADE_CAL.parent.mkdir(parents=True, exist_ok=True)
    cached = pd.DataFrame()
    cache_covers_request = False
    if DEFAULT_TRADE_CAL.exists() and DEFAULT_TRADE_CAL.stat().st_size > 0:
        cached = pd.read_parquet(DEFAULT_TRADE_CAL)
        if not cached.empty and "cal_date" in cached.columns:
            cached["cal_date"] = cached["cal_date"].astype(str)
            cache_covers_request = bool(cached["cal_date"].min() <= start_date and cached["cal_date"].max() >= end_date)
    if refresh or not DEFAULT_TRADE_CAL.exists() or not cache_covers_request:
        fetched = call_with_retry(
            client,
            limiter,
            "trade_cal",
            {"exchange": "SSE", "start_date": start_date, "end_date": end_date},
            "exchange,cal_date,is_open,pretrade_date",
            max_retries,
        )
        frames = [df for df in [cached, fetched] if not df.empty]
        merged = pd.concat(frames, ignore_index=True, sort=False) if frames else fetched
        if not merged.empty:
            merged["cal_date"] = merged["cal_date"].astype(str)
            keys = ["exchange", "cal_date"] if "exchange" in merged.columns else ["cal_date"]
            merged = merged.drop_duplicates(keys, keep="last").sort_values("cal_date", ascending=False)
        checksum = write_parquet_atomic(merged, str(DEFAULT_TRADE_CAL))
        result = {
            "api_name": "trade_cal",
            "path": str(DEFAULT_TRADE_CAL),
            "status": "fetched_merged" if cache_covers_request else "fetched_missing_range",
            "rows": int(len(merged)),
            "requested_start_date": start_date,
            "requested_end_date": end_date,
            "cache_covers_request_before_fetch": cache_covers_request,
            "sha1": checksum,
        }
    else:
        result = {
            "api_name": "trade_cal",
            "path": str(DEFAULT_TRADE_CAL),
            "status": "cached",
            "rows": int(len(cached)),
            "requested_start_date": start_date,
            "requested_end_date": end_date,
            "cache_covers_request_before_fetch": True,
        }
    cal = pd.read_parquet(DEFAULT_TRADE_CAL)
    cal["cal_date"] = cal["cal_date"].astype(str)
    cal["is_open"] = pd.to_numeric(cal["is_open"], errors="coerce").fillna(0).astype(int)
    cal = cal[(cal["cal_date"] >= start_date) & (cal["cal_date"] <= end_date)].copy()
    return cal, result


def trade_dates_for(cal: pd.DataFrame, trade_date: str, lookback_trading_days: int) -> tuple[list[str], str, str]:
    open_days = sorted(cal[cal["is_open"].eq(1)]["cal_date"].astype(str).tolist())
    if trade_date not in open_days:
        raise SystemExit(f"{trade_date} is not an open trading day in trade_cal; aborting live paper runner")
    prior_days = [d for d in open_days if d < trade_date]
    if not prior_days:
        raise SystemExit(f"no prior open trade date found before {trade_date}")
    prior = prior_days[-1]
    prior2 = prior_days[-2] if len(prior_days) >= 2 else prior_days[-1]
    history = [d for d in open_days if d <= prior]
    history = history[-int(lookback_trading_days) :]
    return history, prior, prior2


def fetch_stock_basic(client: TushareProxyClient, limiter: RateLimiter, refresh: bool, max_retries: int) -> dict[str, Any]:
    DEFAULT_STOCK_BASIC.parent.mkdir(parents=True, exist_ok=True)
    return fetch_to_parquet(
        client,
        limiter,
        "stock_basic",
        {"list_status": "L"},
        "ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date",
        DEFAULT_STOCK_BASIC,
        refresh=refresh or not DEFAULT_STOCK_BASIC.exists(),
        max_retries=max_retries,
    )


def fetch_daily_sidecars(
    client: TushareProxyClient,
    limiter: RateLimiter,
    dates: list[str],
    refresh: bool,
    max_retries: int,
    fetch_moneyflow: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for trade_date in dates:
        results.append(
            fetch_to_parquet(
                client,
                limiter,
                "daily",
                {"trade_date": trade_date},
                None,
                RAW_DIR / "daily" / f"trade_date={trade_date}.parquet",
                refresh=refresh,
                max_retries=max_retries,
            )
        )
        if fetch_moneyflow:
            try:
                results.append(
                    fetch_to_parquet(
                        client,
                        limiter,
                        "moneyflow",
                        {"trade_date": trade_date},
                        None,
                        RAW_DIR / "moneyflow" / f"trade_date={trade_date}.parquet",
                        refresh=refresh,
                        max_retries=max_retries,
                    )
                )
            except TushareError as exc:
                results.append({"api_name": "moneyflow", "trade_date": trade_date, "status": "failed_optional", "error": repr(exc)[:500]})
            try:
                results.append(
                    fetch_to_parquet(
                        client,
                        limiter,
                        "moneyflow_ths",
                        {"trade_date": trade_date},
                        None,
                        RAW_DIR / "moneyflow_ths" / f"trade_date={trade_date}.parquet",
                        refresh=refresh,
                        max_retries=max_retries,
                    )
                )
            except TushareError as exc:
                results.append({"api_name": "moneyflow_ths", "trade_date": trade_date, "status": "failed_optional", "error": repr(exc)[:500]})
    return results


def read_raw_daily(dates: list[str], codes: set[str]) -> pd.DataFrame:
    frames = []
    for trade_date in dates:
        path = RAW_DIR / "daily" / f"trade_date={trade_date}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df["ts_code"] = df["ts_code"].astype(str)
        df["trade_date"] = df["trade_date"].astype(str)
        frames.append(df[df["ts_code"].isin(codes)].copy())
    if not frames:
        raise SystemExit("no daily rows available after provider fetch")
    out = pd.concat(frames, ignore_index=True, sort=False)
    for col in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def read_moneyflow(api_name: str, dates: list[str], codes: set[str]) -> pd.DataFrame:
    frames = []
    for trade_date in dates:
        path = RAW_DIR / api_name / f"trade_date={trade_date}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df.empty or "ts_code" not in df.columns:
            continue
        df["ts_code"] = df["ts_code"].astype(str)
        df["trade_date"] = df["trade_date"].astype(str)
        frames.append(df[df["ts_code"].isin(codes)].copy())
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def add_synthetic_trade_date(daily: pd.DataFrame, trade_date: str, codes: set[str]) -> pd.DataFrame:
    existing = set(daily[daily["trade_date"].eq(trade_date)]["ts_code"].astype(str))
    missing = sorted(codes - existing)
    if not missing:
        return daily
    last = daily[daily["trade_date"] < trade_date].sort_values(["ts_code", "trade_date"]).drop_duplicates("ts_code", keep="last")
    last = last[last["ts_code"].isin(missing)].copy()
    if last.empty:
        return daily
    syn = last.copy()
    syn["trade_date"] = trade_date
    syn["daily_source"] = "synthetic_forward_row_from_t_minus_1"
    for col in ["open", "high", "low", "close", "change", "pct_chg", "vol", "amount"]:
        if col in syn.columns:
            syn[col] = np.nan
    if "pre_close" in syn.columns:
        syn["pre_close"] = last["close"].values
    return pd.concat([daily, syn], ignore_index=True, sort=False)


def compute_daily_features(daily: pd.DataFrame, trade_date: str, rank: pd.DataFrame) -> pd.DataFrame:
    codes = set(rank["ts_code"].astype(str))
    daily = add_synthetic_trade_date(daily, trade_date, codes)
    daily = daily[daily["ts_code"].isin(codes)].copy()
    if "daily_source" not in daily.columns:
        daily["daily_source"] = "provider_daily"
    daily["trade_date"] = daily["trade_date"].astype(str)
    daily["ts_code"] = daily["ts_code"].astype(str)
    daily = daily.sort_values(["ts_code", "trade_date"]).drop_duplicates(["ts_code", "trade_date"], keep="last")

    g = daily.groupby("ts_code", group_keys=False)
    ret = pd.to_numeric(daily["pct_chg"], errors="coerce") / 100.0
    fallback_ret = daily["close"] / daily["pre_close"] - 1.0
    ret = ret.where(ret.notna(), fallback_ret)
    daily["prev_close_calc"] = g["close"].shift(1)
    daily["prev_close_use"] = daily["pre_close"].where(daily["pre_close"].notna(), daily["prev_close_calc"])
    daily["prev_ret_1d"] = ret.groupby(daily["ts_code"]).shift(1)
    daily["prev_ret_3d"] = g["close"].shift(1) / g["close"].shift(4) - 1.0
    daily["prev_ret_5d"] = g["close"].shift(1) / g["close"].shift(6) - 1.0
    daily["prev_volatility_20d"] = ret.groupby(daily["ts_code"]).transform(lambda s: s.shift(1).rolling(20, min_periods=10).std())
    daily["ma5_prev"] = g["close"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    daily["ma10_prev"] = g["close"].transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean())
    daily["ma20_prev"] = g["close"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    daily["amount5_prev"] = g["amount"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    daily["amount20_prev"] = g["amount"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    daily["prev_amount_ratio_5_20"] = daily["amount5_prev"] / daily["amount20_prev"] - 1.0
    daily["prev_trade_date"] = g["trade_date"].shift(1)
    daily["next_trade_date"] = g["trade_date"].shift(-1)
    return daily


def add_moneyflow_features(daily: pd.DataFrame, dates: list[str], codes: set[str]) -> pd.DataFrame:
    out = daily.copy()
    mf = read_moneyflow("moneyflow", dates, codes)
    if not mf.empty:
        for col in ["net_mf_amount", "buy_elg_amount", "sell_elg_amount"]:
            if col in mf.columns:
                mf[col] = pd.to_numeric(mf[col], errors="coerce")
        mf["prev_net_mf_amount"] = mf.get("net_mf_amount", np.nan)
        mf["prev_elg_net_amount"] = mf.get("buy_elg_amount", 0) - mf.get("sell_elg_amount", 0)
        mf_keep = mf[["ts_code", "trade_date", "prev_net_mf_amount", "prev_elg_net_amount"]].copy()
        mf_keep["trade_date"] = mf_keep["trade_date"].astype(str)
        mf_keep = mf_keep.rename(columns={"trade_date": "prev_trade_date"})
        out = out.merge(mf_keep, on=["ts_code", "prev_trade_date"], how="left")
    else:
        out["prev_net_mf_amount"] = np.nan
        out["prev_elg_net_amount"] = np.nan
    ths = read_moneyflow("moneyflow_ths", dates, codes)
    if not ths.empty and "net_amount" in ths.columns:
        ths["prev_ths_net_amount"] = pd.to_numeric(ths["net_amount"], errors="coerce")
        ths_keep = ths[["ts_code", "trade_date", "prev_ths_net_amount"]].copy()
        ths_keep = ths_keep.rename(columns={"trade_date": "prev_trade_date"})
        out = out.merge(ths_keep, on=["ts_code", "prev_trade_date"], how="left")
    else:
        out["prev_ths_net_amount"] = np.nan
    amount_den = pd.to_numeric(out["amount20_prev"], errors="coerce")
    out["prev_net_mf_amount_ratio"] = out["prev_net_mf_amount"] / amount_den
    out["prev_elg_net_amount_ratio"] = out["prev_elg_net_amount"] / amount_den
    out["prev_ths_net_amount_ratio"] = out["prev_ths_net_amount"] / amount_den
    return out


def add_empty_auction_columns(daily: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "open_auction_close",
        "open_auction_amount",
        "open_auction_ret",
        "open_auction_amount_log",
        "prev_close_auction_close",
        "prev_close_auction_amount",
        "prev_close_auction_ret",
        "prev_close_auction_amount_log",
    ]:
        if col not in daily.columns:
            daily[col] = np.nan
    return daily


def write_clean_daily(daily: pd.DataFrame, path: Path) -> None:
    columns = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
        "daily_source",
        "prev_close_calc",
        "prev_close_use",
        "prev_ret_1d",
        "prev_ret_3d",
        "prev_ret_5d",
        "prev_volatility_20d",
        "ma5_prev",
        "ma10_prev",
        "ma20_prev",
        "amount5_prev",
        "amount20_prev",
        "prev_amount_ratio_5_20",
        "next_trade_date",
        "prev_trade_date",
        "open_auction_close",
        "open_auction_amount",
        "open_auction_ret",
        "open_auction_amount_log",
        "prev_close_auction_close",
        "prev_close_auction_amount",
        "prev_close_auction_ret",
        "prev_close_auction_amount_log",
        "prev_net_mf_amount",
        "prev_elg_net_amount",
        "prev_net_mf_amount_ratio",
        "prev_elg_net_amount_ratio",
        "prev_ths_net_amount",
        "prev_ths_net_amount_ratio",
    ]
    for col in columns:
        if col not in daily.columns:
            daily[col] = np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    daily[columns].sort_values(["ts_code", "trade_date"]).to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare portable forward-shadow baseline data from Tushare provider.")
    parser.add_argument("--trade-date", required=True, help="YYYYMMDD live trade date")
    parser.add_argument("--top-rank", type=int, default=3000)
    parser.add_argument("--lookback-trading-days", type=int, default=90)
    parser.add_argument("--rank-file", default=str(DEFAULT_RANK_FILE))
    parser.add_argument("--daily-file", default=str(DEFAULT_DAILY_FILE))
    parser.add_argument("--requests-per-minute", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--skip-moneyflow", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    trade_date = str(args.trade_date)
    start_date = ymd(ymd_to_dt(trade_date) - timedelta(days=max(160, int(args.lookback_trading_days * 2.2))))
    end_date = trade_date
    ensure_rank_file(Path(args.rank_file))
    rank = load_rank(Path(args.rank_file), int(args.top_rank))
    codes = set(rank["ts_code"].astype(str))
    client = TushareProxyClient(get_token(), get_proxy_url(), timeout=int(args.timeout))
    limiter = RateLimiter(int(args.requests_per_minute))

    cal, cal_result = load_or_fetch_trade_cal(client, limiter, start_date, end_date, args.refresh, int(args.max_retries))
    history_dates, prior_trade_date, prior2_trade_date = trade_dates_for(cal, trade_date, int(args.lookback_trading_days))
    stock_result = fetch_stock_basic(client, limiter, args.refresh, int(args.max_retries))
    fetch_results = fetch_daily_sidecars(
        client,
        limiter,
        history_dates,
        args.refresh,
        int(args.max_retries),
        fetch_moneyflow=not bool(args.skip_moneyflow),
    )

    raw_daily = read_raw_daily(history_dates, codes)
    clean = compute_daily_features(raw_daily, trade_date, rank)
    clean = add_moneyflow_features(clean, history_dates, codes)
    clean = add_empty_auction_columns(clean)
    write_clean_daily(clean, Path(args.daily_file))

    day_rows = clean[clean["trade_date"].eq(trade_date)]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_time_beijing": bj_now(),
        "trade_date": trade_date,
        "prior_trade_date": prior_trade_date,
        "prior2_trade_date": prior2_trade_date,
        "lookback_trading_days": int(args.lookback_trading_days),
        "history_start": history_dates[0],
        "history_end": history_dates[-1],
        "top_rank": int(args.top_rank),
        "rank_file": str(Path(args.rank_file)),
        "daily_file": str(Path(args.daily_file)),
        "clean_daily_rows": int(len(clean)),
        "trade_date_rows": int(len(day_rows)),
        "trade_date_rows_with_prev_close": int(day_rows["prev_close_use"].notna().sum()) if not day_rows.empty else 0,
        "trade_date_rows_with_ma20_prev": int(day_rows["ma20_prev"].notna().sum()) if not day_rows.empty else 0,
        "cal_fetch": cal_result,
        "stock_basic_fetch": stock_result,
        "provider_fetch_counts": pd.DataFrame(fetch_results)["status"].value_counts(dropna=False).to_dict() if fetch_results else {},
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    summary_path = REPORT_DIR / f"{trade_date}_baseline_prepare_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({**summary, "summary_path": str(summary_path)}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
