#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tushare_data_pipeline import (  # noqa: E402
    RAW_DIR,
    TushareError,
    TushareProxyClient,
    TushareRateLimit,
    get_proxy_url,
    get_token,
    output_path_for,
    redact_token,
    sha1_file,
    write_parquet_atomic,
)


DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "tushare" / "v8_forward_shadow"
DEFAULT_RANK_FILE = ROOT / "data_tushare" / "manifests" / "liquid_top3000_20251120_20260213.csv"
DEFAULT_MINUTE_DIR = RAW_DIR / "stk_mins" / "freq=5min"
STOCK_BASIC_FILE = RAW_DIR / "bootstrap" / "stock_basic.parquet"
MINUTE_COLS = ["ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount"]


class RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.interval = 60.0 / max(1, int(requests_per_minute))
        self.last = 0.0

    def wait(self) -> None:
        wait_s = self.interval - (time.monotonic() - self.last)
        if wait_s > 0:
            time.sleep(wait_s)
        self.last = time.monotonic()


def bj_now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def ymd_to_iso(value: str) -> str:
    s = value.replace("-", "")[:8]
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def ensure_report_dir(output_root: Path) -> Path:
    path = output_root / "data_guards"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def read_top_codes(rank_file: Path, top_rank: int, trade_date: str) -> list[str]:
    rank = pd.read_csv(rank_file, dtype={"ts_code": str})
    rank["liquidity_rank"] = pd.to_numeric(rank["liquidity_rank"], errors="coerce")
    rank = rank.sort_values("liquidity_rank").drop_duplicates("ts_code", keep="first").head(top_rank)
    if STOCK_BASIC_FILE.exists():
        basic = pd.read_parquet(STOCK_BASIC_FILE)
        basic["ts_code"] = basic["ts_code"].astype(str)
        basic["list_date"] = basic.get("list_date", "").fillna("").astype(str)
        basic["delist_date"] = basic.get("delist_date", "").fillna("").astype(str)
        active = basic[
            basic["list_date"].le(trade_date)
            & (basic["delist_date"].eq("") | basic["delist_date"].ge(trade_date))
        ]["ts_code"]
        rank = rank[rank["ts_code"].isin(set(active))]
    return rank["ts_code"].astype(str).tolist()


def read_prior_entry_codes(output_root: Path, signal_date: str) -> list[str]:
    path = output_root / "daily_entry_prices" / f"{signal_date}_entry_prices.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing prior entry file: {path}")
    df = pd.read_csv(path, dtype={"code": str})
    if "paper_entry_status" in df.columns:
        df = df[df["paper_entry_status"].astype(str).eq("entry_recorded")]
    return sorted(set(df["code"].dropna().astype(str).tolist()))


def minute_output_path(ts_code: str, trade_date: str, minute_dir: Path = DEFAULT_MINUTE_DIR) -> Path:
    return minute_dir / f"ts_code={ts_code}" / f"{trade_date}_{trade_date}.parquet"


def minute_candidate_paths(ts_code: str, trade_date: str, minute_dir: Path) -> list[Path]:
    """Return exact-day path first, then historical range files that may contain the date."""
    code_dir = minute_dir / f"ts_code={ts_code}"
    exact = minute_output_path(ts_code, trade_date, minute_dir)
    paths = [exact]
    if not code_dir.exists():
        return paths
    for path in sorted(code_dir.glob("*.parquet")):
        if path == exact:
            continue
        stem = path.stem
        if "_" not in stem:
            continue
        start, end = stem.split("_", 1)
        if start <= trade_date <= end:
            paths.append(path)
    return paths


def normalize_minute(df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=MINUTE_COLS + ["bar_time"])
    out = df.copy()
    if "trade_time" not in out.columns and "datetime" in out.columns:
        out["trade_time"] = out["datetime"]
    if "trade_time" not in out.columns:
        return pd.DataFrame(columns=MINUTE_COLS + ["bar_time"])
    out["trade_time"] = out["trade_time"].astype(str)
    out["dt"] = pd.to_datetime(out["trade_time"], errors="coerce")
    out = out[out["dt"].notna()]
    out = out[out["dt"].dt.strftime("%Y%m%d").eq(trade_date)]
    out["bar_time"] = out["dt"].dt.strftime("%H:%M")
    for col in ["open", "high", "low", "close", "vol", "amount"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    keep = [c for c in MINUTE_COLS if c in out.columns] + ["bar_time"]
    return out[keep].drop_duplicates(["ts_code", "trade_time"], keep="last").sort_values("trade_time", ascending=False)


def load_minute_for_date(ts_code: str, trade_date: str, minute_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in minute_candidate_paths(ts_code, trade_date, minute_dir):
        if not path.exists():
            continue
        try:
            df = normalize_minute(pd.read_parquet(path), trade_date)
        except Exception:
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=MINUTE_COLS + ["bar_time"])
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out.drop_duplicates(["ts_code", "trade_time"], keep="last").sort_values("trade_time", ascending=False)


def expected_bar_times_until(end_time: str) -> list[str]:
    def build(start_h: int, start_m: int, end_h: int, end_m: int) -> list[str]:
        cur = datetime(2000, 1, 1, start_h, start_m)
        end = datetime(2000, 1, 1, end_h, end_m)
        out = []
        while cur <= end:
            out.append(cur.strftime("%H:%M"))
            cur += timedelta(minutes=5)
        return out

    bars = build(9, 35, 11, 30) + build(13, 5, 15, 0)
    return [t for t in bars if t <= end_time]


def existing_required_codes(codes: list[str], trade_date: str, bar_time: str, mode: str, minute_dir: Path) -> set[str]:
    required = {bar_time} if mode == "bar" else set(expected_bar_times_until(bar_time))
    present: set[str] = set()
    for code in codes:
        df = load_minute_for_date(code, trade_date, minute_dir)
        available = set(df["bar_time"].astype(str)) if not df.empty else set()
        if required.issubset(available):
            present.add(code)
    return present


def write_minute_parts(df: pd.DataFrame, codes: list[str], trade_date: str, minute_dir: Path) -> dict[str, Any]:
    if df.empty:
        df = pd.DataFrame(columns=MINUTE_COLS)
    df = df.copy()
    if "ts_code" not in df.columns:
        df["ts_code"] = ""
    df["ts_code"] = df["ts_code"].astype(str)
    returned_codes = set(df["ts_code"].dropna().astype(str).unique())
    rows_written = 0
    new_rows = 0
    for code in codes:
        new = normalize_minute(df[df["ts_code"].eq(code)].copy(), trade_date)
        path = minute_output_path(code, trade_date, minute_dir)
        if path.exists():
            old = normalize_minute(pd.read_parquet(path), trade_date)
            combined = pd.concat([old, new], ignore_index=True, sort=False)
            combined = combined.drop_duplicates(["ts_code", "trade_time"], keep="last")
        else:
            combined = new
        keep = [c for c in MINUTE_COLS if c in combined.columns]
        combined = combined[keep].sort_values("trade_time", ascending=False) if keep else pd.DataFrame(columns=MINUTE_COLS)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        combined.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(path)
        rows_written += int(len(combined))
        new_rows += int(len(new))
    return {
        "rows_written_total": rows_written,
        "new_rows": new_rows,
        "missing_return_codes": len(set(codes) - returned_codes),
    }


def vwap_from_amount_vol(amount_value: Any, vol_value: Any, reference_price: Any, low_price: Any = None, high_price: Any = None) -> float:
    def to_float(value: Any) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return out if pd.notna(out) else float("nan")

    amount = to_float(amount_value)
    vol = to_float(vol_value)
    reference = to_float(reference_price)
    fallback = reference if pd.notna(reference) and reference > 0 else float("nan")
    if not (pd.notna(amount) and amount > 0 and pd.notna(vol) and vol > 0):
        return fallback
    base = amount / vol
    candidates = [base, base * 100.0, base / 100.0]
    candidates = [c for c in candidates if pd.notna(c) and c > 0]
    if not candidates:
        return fallback
    low = to_float(low_price)
    high = to_float(high_price)
    if pd.notna(low) and pd.notna(high) and low > 0 and high >= low:
        bounded = [c for c in candidates if low * 0.98 <= c <= high * 1.02]
        if bounded:
            return float(min(bounded, key=lambda c: abs(c / reference - 1.0) if pd.notna(fallback) else 0.0))
    if pd.notna(fallback) and fallback > 0:
        closest = min(candidates, key=lambda c: abs(c / reference - 1.0))
        if abs(closest / reference - 1.0) <= 0.10:
            return float(closest)
    return fallback


def record_entry_prices(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    minute_dir = Path(args.minute_dir)
    signals_path = output_root / "daily_signals" / f"{args.trade_date}_signals.csv"
    if not signals_path.exists():
        raise FileNotFoundError(f"missing signals file: {signals_path}")
    signals = pd.read_csv(signals_path)
    if "final_selected_flag" in signals.columns:
        signals = signals[signals["final_selected_flag"].astype(str).str.lower().isin(["true", "1"])]
    signals = signals[signals.get("code", pd.Series(dtype=str)).notna()].copy()
    rows: list[dict[str, Any]] = []
    for row in signals.itertuples(index=False):
        code = str(row.code)
        status = "missing_1455_bar"
        entry_vwap = float("nan")
        entry_amount = float("nan")
        df = load_minute_for_date(code, args.trade_date, minute_dir)
        bar = df[df["bar_time"].eq("14:55")]
        if not bar.empty:
            b = bar.iloc[-1]
            entry_vwap = vwap_from_amount_vol(b.get("amount"), b.get("vol"), b.get("close"), b.get("low"), b.get("high"))
            entry_amount = float(b.get("amount")) if pd.notna(b.get("amount")) else float("nan")
            status = "entry_recorded" if pd.notna(entry_vwap) else "entry_price_nan"
        rows.append(
            {
                "trade_date": args.trade_date,
                "strategy_id": getattr(row, "strategy_id", ""),
                "candidate_id": getattr(row, "candidate_id", ""),
                "code": code,
                "name": getattr(row, "name", ""),
                "original_v7_rank": getattr(row, "original_v7_rank", None),
                "score": getattr(row, "score", None),
                "paper_entry_status": status,
                "expected_entry_time": "14:55",
                "entry_vwap": entry_vwap,
                "entry_amount": entry_amount,
                "entry_price_source": "raw_stk_mins_5min_bar_14:55_vwap",
                "weight": getattr(row, "position_weight", None),
                "paper_tracking_only": True,
                "generated_time_beijing": bj_now(),
            }
        )
    out = pd.DataFrame(rows)
    out_dir = output_root / "daily_entry_prices"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.trade_date}_entry_prices.csv"
    out.to_csv(out_path, index=False)
    status = "success" if not out.empty and out["paper_entry_status"].eq("entry_recorded").any() else "failed"
    summary = {
        "task": "record_entry_1455_vwap",
        "status": status,
        "generated_time_beijing": bj_now(),
        "trade_date": args.trade_date,
        "signals_path": str(signals_path),
        "entry_path": str(out_path),
        "rows": int(len(out)),
        "entry_recorded": int(out["paper_entry_status"].eq("entry_recorded").sum()) if not out.empty else 0,
        "missing_1455_bar": int(out["paper_entry_status"].eq("missing_1455_bar").sum()) if not out.empty else 0,
    }
    report_path = ensure_report_dir(output_root) / f"{args.trade_date}_record_entry_1455_vwap.json"
    write_json(report_path, summary)
    summary["report_path"] = str(report_path)
    if status != "success":
        raise SystemExit(json.dumps(summary, ensure_ascii=False, default=str))
    return summary


def fetch_minute_once(
    client: TushareProxyClient,
    limiter: RateLimiter,
    codes: list[str],
    trade_date: str,
    bar_time: str,
    mode: str,
    batch_size: int,
    max_retries: int,
    minute_dir: Path,
) -> dict[str, Any]:
    if mode == "until":
        start_time = "09:30"
        end_time = bar_time
    elif mode == "bar":
        start_time = bar_time
        end_time = bar_time
    else:
        raise ValueError(f"unsupported mode: {mode}")
    iso = ymd_to_iso(trade_date)
    out = {
        "request_count": 0,
        "rows_returned": 0,
        "rows_written_total": 0,
        "new_rows": 0,
        "missing_return_codes": 0,
        "rate_limit_retries": 0,
        "api_errors": 0,
        "request_seconds": 0.0,
    }
    for batch_codes in chunked(codes, batch_size):
        params = {
            "ts_code": ",".join(batch_codes),
            "start_date": f"{iso} {start_time}:00",
            "end_date": f"{iso} {end_time}:00",
            "freq": "5min",
        }
        attempt = 0
        while True:
            attempt += 1
            limiter.wait()
            started = time.monotonic()
            out["request_count"] += 1
            try:
                df = client.call("stk_mins", params)
                elapsed = time.monotonic() - started
                out["request_seconds"] += elapsed
                out["rows_returned"] += int(len(df))
                write = write_minute_parts(df, batch_codes, trade_date, minute_dir)
                out["rows_written_total"] += int(write["rows_written_total"])
                out["new_rows"] += int(write["new_rows"])
                out["missing_return_codes"] += int(write["missing_return_codes"])
                break
            except TushareRateLimit:
                out["request_seconds"] += time.monotonic() - started
                out["rate_limit_retries"] += 1
                if attempt > max_retries:
                    raise
                time.sleep(min(90, 8 * attempt))
            except TushareError:
                out["request_seconds"] += time.monotonic() - started
                out["api_errors"] += 1
                if attempt > max_retries:
                    raise
                time.sleep(min(45, 5 * attempt))
    out["request_seconds"] = round(float(out["request_seconds"]), 3)
    out["provider_start_time"] = start_time
    out["provider_end_time"] = end_time
    return out


def run_minute_fetch(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    minute_dir = Path(args.minute_dir)
    if args.codes_source == "prior_entries":
        if not args.signal_date:
            raise SystemExit("--signal-date is required for --codes-source prior_entries")
        codes = read_prior_entry_codes(output_root, args.signal_date)
    else:
        codes = read_top_codes(Path(args.rank_file), int(args.top_rank), args.trade_date)
    report_dir = ensure_report_dir(output_root)
    started = time.monotonic()
    present = existing_required_codes(codes, args.trade_date, args.bar_time, args.mode, minute_dir)
    missing = sorted(set(codes) - present)
    fetch_rounds: list[dict[str, Any]] = []
    deadline = time.monotonic() + max(0, float(args.retry_until_complete_seconds))
    client: TushareProxyClient | None = None
    limiter: RateLimiter | None = None
    while missing:
        if client is None:
            client = TushareProxyClient(get_token(), get_proxy_url(), timeout=int(args.timeout))
            limiter = RateLimiter(int(args.requests_per_minute))
        assert limiter is not None
        result = fetch_minute_once(
            client,
            limiter,
            missing,
            args.trade_date,
            args.bar_time,
            args.mode,
            int(args.batch_size),
            int(args.max_retries),
            minute_dir,
        )
        fetch_rounds.append(result)
        present = existing_required_codes(codes, args.trade_date, args.bar_time, args.mode, minute_dir)
        missing = sorted(set(codes) - present)
        if not missing or time.monotonic() >= deadline:
            break
        time.sleep(float(args.retry_interval_seconds))
    status = "success" if not missing else "partial"
    summary = {
        "task": "minute_fetch",
        "status": status,
        "generated_time_beijing": bj_now(),
        "trade_date": args.trade_date,
        "signal_date": args.signal_date,
        "codes_source": args.codes_source,
        "mode": args.mode,
        "bar_time": args.bar_time,
        "required_bars_per_symbol": expected_bar_times_until(args.bar_time) if args.mode == "until" else [args.bar_time],
        "symbols_expected": len(codes),
        "symbols_with_required_bar": len(present),
        "symbols_missing_required_bar": len(missing),
        "missing_codes_sample": missing[:30],
        "fetch_rounds": fetch_rounds,
        "duration_seconds": round(time.monotonic() - started, 3),
        "minute_dir": str(minute_dir),
    }
    path = report_dir / f"{args.trade_date}_{args.codes_source}_{args.mode}_{args.bar_time.replace(':', '')}_minute_fetch.json"
    write_json(path, summary)
    summary["report_path"] = str(path)
    return summary


def auction_output(api_name: str, trade_date: str, auction_raw_dir: Path = RAW_DIR) -> Path:
    return auction_raw_dir / api_name / f"trade_date={trade_date}.parquet"


def read_auction_file(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    if "ts_code" in df.columns:
        df["ts_code"] = df["ts_code"].astype(str)
    return df


def fetch_auction_api(
    client: TushareProxyClient,
    limiter: RateLimiter,
    api_name: str,
    trade_date: str,
    max_retries: int,
    auction_raw_dir: Path = RAW_DIR,
) -> dict[str, Any]:
    params = {"trade_date": trade_date}
    path = auction_output(api_name, trade_date, auction_raw_dir)
    attempt = 0
    while True:
        attempt += 1
        limiter.wait()
        started = time.monotonic()
        try:
            df = client.call(api_name, params)
            checksum = write_parquet_atomic(df, str(path))
            return {
                "api_name": api_name,
                "trade_date": trade_date,
                "status": "done",
                "rows": int(len(df)),
                "seconds": round(time.monotonic() - started, 3),
                "path": str(path),
                "sha1": checksum,
            }
        except TushareRateLimit as exc:
            if attempt > max_retries:
                return {"api_name": api_name, "trade_date": trade_date, "status": "failed", "error": redact_token(repr(exc))}
            time.sleep(min(90, 8 * attempt))
        except TushareError as exc:
            if attempt > max_retries:
                return {"api_name": api_name, "trade_date": trade_date, "status": "failed", "error": redact_token(repr(exc))}
            time.sleep(min(45, 5 * attempt))


def auction_coverage(api_name: str, trade_date: str, codes: list[str], auction_raw_dir: Path = RAW_DIR) -> dict[str, Any]:
    path = auction_output(api_name, trade_date, auction_raw_dir)
    df = read_auction_file(path)
    returned = set(df["ts_code"].dropna().astype(str).tolist()) if "ts_code" in df.columns else set()
    missing = sorted(set(codes) - returned)
    return {
        "api_name": api_name,
        "trade_date": trade_date,
        "path": str(path),
        "exists": bool(path.exists()),
        "rows": int(len(df)),
        "top_symbols_expected": len(codes),
        "top_symbols_present": len(set(codes) & returned),
        "top_symbols_missing": len(missing),
        "missing_codes_sample": missing[:30],
        "sha1": sha1_file(path) if path.exists() and path.stat().st_size > 0 else "",
    }


def run_auction_guard(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    auction_raw_dir = Path(args.auction_raw_dir)
    report_dir = ensure_report_dir(output_root)
    codes = read_top_codes(Path(args.rank_file), int(args.top_rank), args.trade_date)
    targets = [
        ("stk_auction_c", args.prior_trade_date, "prev_close_auction"),
        ("stk_auction_o", args.trade_date, "open_auction"),
    ]
    started = time.monotonic()
    before = [auction_coverage(api, date, codes, auction_raw_dir) | {"role": role} for api, date, role in targets]
    needs_fetch = [
        (api, date, role)
        for api, date, role in targets
        if args.refresh or not auction_output(api, date, auction_raw_dir).exists() or auction_output(api, date, auction_raw_dir).stat().st_size == 0
    ]
    fetch_results: list[dict[str, Any]] = []
    if needs_fetch:
        client = TushareProxyClient(get_token(), get_proxy_url(), timeout=int(args.timeout))
        limiter = RateLimiter(int(args.requests_per_minute))
        for api, date, _role in needs_fetch:
            fetch_results.append(fetch_auction_api(client, limiter, api, date, int(args.max_retries), auction_raw_dir))
    after = [auction_coverage(api, date, codes, auction_raw_dir) | {"role": role} for api, date, role in targets]
    missing_files = [x for x in after if not x["exists"] or x["rows"] <= 0]
    status = "success" if not missing_files else "failed"
    summary = {
        "task": "auction_guard_0925",
        "status": status,
        "generated_time_beijing": bj_now(),
        "trade_date": args.trade_date,
        "prior_trade_date": args.prior_trade_date,
        "top_rank": int(args.top_rank),
        "auction_raw_dir": str(auction_raw_dir),
        "before": before,
        "fetch_results": fetch_results,
        "after": after,
        "duration_seconds": round(time.monotonic() - started, 3),
        "notes": [
            "stk_auction_o is T-day open auction and is allowed before 14:50 signal generation.",
            "stk_auction_c is used only for T-1 close auction in T-day features.",
            "T-day close auction remains archive-only and is not pulled by this guard.",
        ],
    }
    path = report_dir / f"{args.trade_date}_auction_guard_0925.json"
    write_json(path, summary)
    summary["report_path"] = str(path)
    if status != "success":
        raise SystemExit(json.dumps(summary, ensure_ascii=False, default=str))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-only forward shadow data readiness and provider fetch guard.")
    sub = parser.add_subparsers(dest="command", required=True)

    auc = sub.add_parser("auction-guard")
    auc.add_argument("--trade-date", required=True)
    auc.add_argument("--prior-trade-date", required=True)
    auc.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    auc.add_argument("--auction-raw-dir", default=str(RAW_DIR))
    auc.add_argument("--rank-file", default=str(DEFAULT_RANK_FILE))
    auc.add_argument("--top-rank", type=int, default=3000)
    auc.add_argument("--requests-per-minute", type=int, default=120)
    auc.add_argument("--timeout", type=int, default=90)
    auc.add_argument("--max-retries", type=int, default=3)
    auc.add_argument("--refresh", action="store_true")
    auc.set_defaults(func=run_auction_guard)

    mins = sub.add_parser("minute-fetch")
    mins.add_argument("--trade-date", required=True)
    mins.add_argument("--signal-date")
    mins.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    mins.add_argument("--minute-dir", default=str(DEFAULT_MINUTE_DIR))
    mins.add_argument("--rank-file", default=str(DEFAULT_RANK_FILE))
    mins.add_argument("--top-rank", type=int, default=3000)
    mins.add_argument("--codes-source", choices=["prior_entries", "top3000"], required=True)
    mins.add_argument("--bar-time", required=True, help="HH:MM completed 5-minute bar.")
    mins.add_argument("--mode", choices=["bar", "until"], default="bar")
    mins.add_argument("--batch-size", type=int, default=160)
    mins.add_argument("--requests-per-minute", type=int, default=120)
    mins.add_argument("--timeout", type=int, default=90)
    mins.add_argument("--max-retries", type=int, default=3)
    mins.add_argument("--retry-until-complete-seconds", type=float, default=0)
    mins.add_argument("--retry-interval-seconds", type=float, default=5)
    mins.set_defaults(func=run_minute_fetch)

    entry = sub.add_parser("record-entry")
    entry.add_argument("--trade-date", required=True)
    entry.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    entry.add_argument("--minute-dir", default=str(DEFAULT_MINUTE_DIR))
    entry.set_defaults(func=record_entry_prices)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = args.func(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
