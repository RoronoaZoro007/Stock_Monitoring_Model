#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data_tushare" / "raw"
REPORT_DIR = ROOT / "reports" / "tushare" / "v9_swing_research" / "batch1_history_daily"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tushare_data_pipeline import TushareError, TushareProxyClient, TushareRateLimit, get_proxy_url, redact_token


@dataclass(frozen=True)
class EndpointSpec:
    api_name: str
    fields: str | None
    purpose: str


ENDPOINTS = [
    EndpointSpec(
        "daily",
        "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
        "daily OHLCV and amount",
    ),
    EndpointSpec("adj_factor", "ts_code,trade_date,adj_factor", "adjustment factor"),
    EndpointSpec(
        "daily_basic",
        "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,total_share,float_share,free_share,total_mv,circ_mv",
        "market cap, float shares, turnover and valuation fields",
    ),
    EndpointSpec("suspend_d", "ts_code,trade_date,suspend_timing,suspend_type", "suspension status"),
    EndpointSpec("stk_limit", "trade_date,ts_code,up_limit,down_limit", "daily limit up/down prices"),
]

BOOTSTRAP_ENDPOINTS = [
    EndpointSpec(
        "stock_basic",
        "ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date",
        "active/delisted stock metadata",
    ),
    EndpointSpec("trade_cal", "exchange,cal_date,is_open,pretrade_date", "exchange trading calendar"),
    EndpointSpec("namechange", "ts_code,name,start_date,end_date,ann_date,change_reason", "point-in-time ST/name-change proxy"),
]


class RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.interval = 60.0 / max(1, requests_per_minute)
        self.last = 0.0
        self.lock = threading.Lock()
        self.cooldown_until = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            if now < self.cooldown_until:
                time.sleep(self.cooldown_until - now)
            wait_s = self.interval - (time.monotonic() - self.last)
            if wait_s > 0:
                time.sleep(wait_s)
            self.last = time.monotonic()

    def cooldown(self, seconds: float) -> None:
        with self.lock:
            self.cooldown_until = max(self.cooldown_until, time.monotonic() + seconds)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_token() -> str:
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise SystemExit("missing TUSHARE_TOKEN; load it from the shell environment before running")
    return token


def ymd(value: str) -> str:
    return value.replace("-", "")[:8]


def daily_path(api_name: str, trade_date: str) -> Path:
    return RAW_DIR / api_name / f"trade_date={trade_date}.parquet"


def bootstrap_path(name: str, start: str, end: str) -> Path:
    if name == "trade_cal":
        return RAW_DIR / "bootstrap" / f"trade_cal_{start}_{end}.parquet"
    if name == "namechange":
        return RAW_DIR / "bootstrap" / f"namechange_{start}_{end}.parquet"
    return RAW_DIR / "bootstrap" / f"{name}_all_status.parquet"


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)
    return sha256_file(path)


def read_existing_rows(path: Path) -> int:
    try:
        return int(pd.read_parquet(path).shape[0])
    except Exception:
        return -1


def call_with_retry(
    client: TushareProxyClient,
    limiter: RateLimiter,
    api_name: str,
    params: dict[str, Any],
    fields: str | None,
    retries: int,
) -> tuple[pd.DataFrame, float]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        limiter.wait()
        started = time.time()
        try:
            df = client.call(api_name, params, fields)
            return df, time.time() - started
        except TushareRateLimit as exc:
            last_error = exc
            wait_s = min(300, 15 * (attempt + 1))
            limiter.cooldown(wait_s)
            time.sleep(wait_s)
        except TushareError as exc:
            if "429" in str(exc) or "Too Many Requests" in str(exc):
                last_error = exc
                wait_s = min(300, 20 * (attempt + 1))
                limiter.cooldown(wait_s)
                time.sleep(wait_s)
                continue
            last_error = exc
            break
    assert last_error is not None
    raise last_error


def fetch_bootstrap(client: TushareProxyClient, limiter: RateLimiter, start: str, end: str, refresh: bool, retries: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in BOOTSTRAP_ENDPOINTS:
        path = bootstrap_path(spec.api_name, start, end)
        if path.exists() and not refresh:
            rows.append(
                {
                    "api_name": spec.api_name,
                    "status": "exists",
                    "rows": read_existing_rows(path),
                    "path": str(path),
                    "seconds": 0.0,
                    "error": "",
                }
            )
            continue

        if spec.api_name == "stock_basic":
            parts = []
            seconds = 0.0
            for status in ["L", "D", "P"]:
                df, elapsed = call_with_retry(
                    client,
                    limiter,
                    spec.api_name,
                    {"exchange": "", "list_status": status},
                    spec.fields,
                    retries,
                )
                if not df.empty:
                    df["source_list_status"] = status
                    parts.append(df)
                seconds += elapsed
            out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        elif spec.api_name == "trade_cal":
            out, seconds = call_with_retry(
                client,
                limiter,
                spec.api_name,
                {"exchange": "SSE", "start_date": start, "end_date": end},
                spec.fields,
                retries,
            )
        elif spec.api_name == "namechange":
            out, seconds = call_with_retry(
                client,
                limiter,
                spec.api_name,
                {"start_date": start, "end_date": end},
                spec.fields,
                retries,
            )
        else:
            continue
        checksum = write_parquet_atomic(out, path)
        rows.append(
            {
                "api_name": spec.api_name,
                "status": "downloaded",
                "rows": int(len(out)),
                "path": str(path),
                "seconds": round(seconds, 3),
                "sha256": checksum,
                "error": "",
            }
        )
    return rows


def download_daily_task(
    token: str,
    base_url: str,
    limiter: RateLimiter,
    spec: EndpointSpec,
    trade_date: str,
    retries: int,
) -> dict[str, Any]:
    path = daily_path(spec.api_name, trade_date)
    started = time.time()
    try:
        client = TushareProxyClient(token, base_url, timeout=60)
        df, elapsed = call_with_retry(
            client,
            limiter,
            spec.api_name,
            {"trade_date": trade_date},
            spec.fields,
            retries,
        )
        checksum = write_parquet_atomic(df, path)
        return {
            "api_name": spec.api_name,
            "trade_date": trade_date,
            "status": "downloaded",
            "rows": int(len(df)),
            "seconds": round(elapsed, 3),
            "wall_seconds": round(time.time() - started, 3),
            "path": str(path),
            "sha256": checksum,
            "error": "",
        }
    except Exception as exc:
        return {
            "api_name": spec.api_name,
            "trade_date": trade_date,
            "status": "failed",
            "rows": 0,
            "seconds": round(time.time() - started, 3),
            "wall_seconds": round(time.time() - started, 3),
            "path": str(path),
            "error": redact_token(str(exc))[:1000],
        }


def open_dates_from_trade_cal(path: Path) -> list[str]:
    cal = pd.read_parquet(path)
    cal["is_open"] = pd.to_numeric(cal["is_open"], errors="coerce").fillna(0).astype(int)
    dates = cal[cal["is_open"].eq(1)]["cal_date"].astype(str).tolist()
    return sorted(set(dates))


def smoke_test(client: TushareProxyClient, limiter: RateLimiter, sample_date: str, start: str, end: str, retries: int) -> pd.DataFrame:
    tests: list[tuple[str, dict[str, Any], str | None]] = [
        ("trade_cal", {"exchange": "SSE", "start_date": sample_date, "end_date": sample_date}, "exchange,cal_date,is_open,pretrade_date"),
        ("stock_basic", {"exchange": "", "list_status": "L"}, BOOTSTRAP_ENDPOINTS[0].fields),
        ("namechange", {"start_date": start, "end_date": end}, BOOTSTRAP_ENDPOINTS[2].fields),
    ]
    tests.extend((spec.api_name, {"trade_date": sample_date}, spec.fields) for spec in ENDPOINTS)
    rows = []
    for api_name, params, fields in tests:
        started = time.time()
        try:
            df, elapsed = call_with_retry(client, limiter, api_name, params, fields, retries)
            rows.append(
                {
                    "api_name": api_name,
                    "params_mode": "trade_date" if "trade_date" in params else "range/static",
                    "status": "ok",
                    "rows": int(len(df)),
                    "seconds": round(elapsed, 3),
                    "columns": ",".join(map(str, df.columns)),
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "api_name": api_name,
                    "params_mode": "trade_date" if "trade_date" in params else "range/static",
                    "status": "failed",
                    "rows": 0,
                    "seconds": round(time.time() - started, 3),
                    "columns": "",
                    "error": redact_token(str(exc))[:1000],
                }
            )
    return pd.DataFrame(rows)


def coverage_for(open_dates: list[str], endpoints: list[EndpointSpec]) -> pd.DataFrame:
    rows = []
    expected = set(open_dates)
    for spec in endpoints:
        path = RAW_DIR / spec.api_name
        files = sorted(path.glob("trade_date=*.parquet")) if path.exists() else []
        got = {p.stem.split("=", 1)[1] for p in files}
        in_range = got & expected
        missing = sorted(expected - got)
        rows.append(
            {
                "api_name": spec.api_name,
                "purpose": spec.purpose,
                "expected_open_dates": len(expected),
                "available_dates": len(in_range),
                "missing_dates": len(missing),
                "first_missing": missing[0] if missing else "",
                "last_missing": missing[-1] if missing else "",
                "first_available": min(in_range) if in_range else "",
                "last_available": max(in_range) if in_range else "",
            }
        )
    return pd.DataFrame(rows)


def write_conclusion(out_dir: Path, start: str, end: str, smoke: pd.DataFrame, before: pd.DataFrame, after: pd.DataFrame, download: pd.DataFrame) -> None:
    lines = [
        "# v9 Swing Research - Batch 1 History Daily Data",
        "",
        "## Scope",
        "",
        f"- Requested range: {start} to {end}.",
        "- This batch only validates and downloads daily-level historical data needed for swing-model research.",
        "- It does not train a model, tune parameters, or modify v7_locked.",
        "",
        "## Interface Smoke Test",
        "",
        smoke.to_markdown(index=False),
        "",
        "## Coverage Before Download",
        "",
        before.to_markdown(index=False),
        "",
        "## Coverage After Download",
        "",
        after.to_markdown(index=False),
        "",
        "## Download Summary",
        "",
        download.groupby(["api_name", "status"], dropna=False)
        .agg(files=("path", "count"), rows=("rows", "sum"), seconds=("seconds", "sum"))
        .reset_index()
        .to_markdown(index=False),
        "",
        "## Data Mapping",
        "",
        "| Need | Endpoint | Local path pattern | Notes |",
        "|---|---|---|---|",
        "| Daily OHLCV | daily | data_tushare/raw/daily/trade_date=YYYYMMDD.parquet | Base daily label and return data |",
        "| Adjustment | adj_factor | data_tushare/raw/adj_factor/trade_date=YYYYMMDD.parquet | Used to build adjusted prices |",
        "| Market cap / turnover | daily_basic | data_tushare/raw/daily_basic/trade_date=YYYYMMDD.parquet | total_mv, circ_mv, turnover_rate, turnover_rate_f |",
        "| Suspension | suspend_d | data_tushare/raw/suspend_d/trade_date=YYYYMMDD.parquet | Empty file means no returned suspended records for that date |",
        "| Limit prices | stk_limit | data_tushare/raw/stk_limit/trade_date=YYYYMMDD.parquet | Used with OHLCV to infer limit-up/down blocks |",
        "| ST / name history | namechange | data_tushare/raw/bootstrap/namechange_START_END.parquet | Point-in-time ST proxy; must be converted to daily flags before modeling |",
        "",
        "## Important Constraints",
        "",
        "- Multi-day range calls for full-market daily-like endpoints can be row-capped by the provider. Batch 1 therefore downloads these endpoints by single trade_date.",
        "- ST status is not a single clean daily field here; it must be derived from namechange intervals and validated before use.",
        "- This completes raw daily-level data staging only. Feature joins, market-regime labels, and factor tests belong to later batches.",
    ]
    (out_dir / "batch1_conclusion.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="v9 Batch1: validate and download daily-level historical data for swing research.")
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20260522")
    parser.add_argument("--sample-date", default="20260522")
    parser.add_argument("--requests-per-minute", type=int, default=150)
    parser.add_argument("--max-requests", type=int, default=9800)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--apis", default=",".join(spec.api_name for spec in ENDPOINTS))
    args = parser.parse_args()

    start = ymd(args.start)
    end = ymd(args.end)
    out_dir = REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    token = read_token()
    base_url = get_proxy_url()
    client = TushareProxyClient(token, base_url, timeout=60)
    limiter = RateLimiter(args.requests_per_minute)
    selected = [spec for spec in ENDPOINTS if spec.api_name in {x.strip() for x in args.apis.split(",") if x.strip()}]

    smoke = smoke_test(client, limiter, args.sample_date, start, end, args.retries)
    smoke.to_csv(out_dir / "batch1_api_smoke_test.csv", index=False)

    bootstrap_rows = fetch_bootstrap(client, limiter, start, end, args.refresh, args.retries)
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(out_dir / "batch1_bootstrap_summary.csv", index=False)
    trade_cal_path = bootstrap_path("trade_cal", start, end)
    open_dates = open_dates_from_trade_cal(trade_cal_path)
    before = coverage_for(open_dates, selected)
    before.to_csv(out_dir / "batch1_coverage_before.csv", index=False)

    download_rows: list[dict[str, Any]] = []
    request_count = 0
    if not args.smoke_only:
        progress_path = out_dir / "batch1_download_progress.jsonl"
        progress_path.write_text("", encoding="utf-8")
        pending: list[tuple[EndpointSpec, str]] = []
        for trade_date in open_dates:
            for spec in selected:
                path = daily_path(spec.api_name, trade_date)
                if path.exists() and path.stat().st_size > 0 and not args.refresh:
                    download_rows.append(
                        {
                            "api_name": spec.api_name,
                            "trade_date": trade_date,
                            "status": "exists",
                            "rows": read_existing_rows(path),
                            "seconds": 0.0,
                            "path": str(path),
                            "error": "",
                        }
                    )
                    continue
                if len(pending) >= args.max_requests:
                    download_rows.append(
                        {
                            "api_name": spec.api_name,
                            "trade_date": trade_date,
                            "status": "skipped_max_requests",
                            "rows": 0,
                            "seconds": 0.0,
                            "path": str(path),
                            "error": f"max_requests={args.max_requests}",
                        }
                    )
                    continue
                pending.append((spec, trade_date))

        if pending:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                futures = {
                    executor.submit(download_daily_task, token, base_url, limiter, spec, trade_date, args.retries): (spec.api_name, trade_date)
                    for spec, trade_date in pending
                }
                done = 0
                for future in as_completed(futures):
                    row = future.result()
                    request_count += 1
                    done += 1
                    download_rows.append(row)
                    if done % 100 == 0 or row["status"] == "failed":
                        payload = {"ts": now_iso(), "done": done, "total": len(pending), **row}
                        with progress_path.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    download = pd.DataFrame(download_rows)
    if download.empty:
        download = pd.DataFrame(columns=["api_name", "trade_date", "status", "rows", "seconds", "path", "error"])
    download.to_csv(out_dir / "batch1_download_summary.csv", index=False)

    after = coverage_for(open_dates, selected)
    after.to_csv(out_dir / "batch1_coverage_after.csv", index=False)
    missing = []
    for spec in selected:
        expected = set(open_dates)
        got = {p.stem.split("=", 1)[1] for p in (RAW_DIR / spec.api_name).glob("trade_date=*.parquet")} if (RAW_DIR / spec.api_name).exists() else set()
        for trade_date in sorted(expected - got):
            missing.append({"api_name": spec.api_name, "trade_date": trade_date})
    pd.DataFrame(missing).to_csv(out_dir / "batch1_missing_after_download.csv", index=False)

    write_conclusion(out_dir, start, end, smoke, before, after, download)
    output_hashes = []
    for path in sorted(out_dir.glob("batch1_*")):
        if path.is_file() and path.name != "batch1_file_sha256.csv":
            output_hashes.append({"file": str(path), "sha256": sha256_file(path)})
    pd.DataFrame(output_hashes).to_csv(out_dir / "batch1_file_sha256.csv", index=False)

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "start": start,
                "end": end,
                "open_dates": len(open_dates),
                "requests_made": request_count,
                "missing_after": len(missing),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
