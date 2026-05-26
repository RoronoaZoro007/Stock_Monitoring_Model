#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import ssl
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tushare_data_pipeline import TushareError, TushareProxyClient, TushareRateLimit, get_proxy_url, redact_token  # noqa: E402


DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "tushare" / "v8_forward_shadow"
DEFAULT_RANK_FILE = ROOT / "data_tushare" / "manifests" / "liquid_top3000_20251120_20260213.csv"
LOCKED_RANK_FILE = ROOT / "locked_artifacts" / "v7_cap20_strong_label_003" / "config" / "liquid_top3000_20251120_20260213.csv"
DEFAULT_MINUTE_DIR = ROOT / "data_tushare" / "raw" / "stk_mins" / "freq=5min"
DEFAULT_TRADE_CAL = ROOT / "data_tushare" / "raw" / "bootstrap" / "trade_cal.parquet"
BEIJING_TZ = timezone(timedelta(hours=8))


def bj_now() -> str:
    return datetime.now(BEIJING_TZ).isoformat(timespec="seconds")


def bj_datetime() -> datetime:
    return datetime.now(BEIJING_TZ)


def today_ymd() -> str:
    return bj_datetime().strftime("%Y%m%d")


def ymd_to_iso(value: str) -> str:
    s = str(value).replace("-", "")[:8]
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def sanitize(text: str) -> str:
    out = redact_token(str(text or ""))
    wx_token = os.environ.get("WXPUSHER_APP_TOKEN", "").strip()
    if wx_token:
        out = out.replace(wx_token, f"{wx_token[:6]}...{wx_token[-4:]}")
    return out[:800]


def check_result(name: str, status: str, severity: str, started: float, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "severity": severity,
        "duration_seconds": round(time.monotonic() - started, 3),
        **extra,
    }


def load_open_dates_from_cache() -> list[str]:
    if not DEFAULT_TRADE_CAL.exists() or DEFAULT_TRADE_CAL.stat().st_size <= 0:
        return []
    try:
        cal = pd.read_parquet(DEFAULT_TRADE_CAL)
    except Exception:
        return []
    if cal.empty or "cal_date" not in cal.columns or "is_open" not in cal.columns:
        return []
    cal["cal_date"] = cal["cal_date"].astype(str)
    cal["is_open"] = pd.to_numeric(cal["is_open"], errors="coerce").fillna(0).astype(int)
    return sorted(cal[cal["is_open"].eq(1)]["cal_date"].tolist())


def fetch_open_dates_from_akshare() -> tuple[list[str], dict[str, Any]]:
    started = time.monotonic()
    try:
        import akshare as ak

        raw = ak.tool_trade_date_hist_sina()
        dates = (
            pd.to_datetime(raw["trade_date"], errors="coerce")
            .dropna()
            .dt.strftime("%Y%m%d")
            .tolist()
        )
        return sorted(set(dates)), check_result(
            "akshare_sina_trade_calendar",
            "success",
            "info",
            started,
            rows=len(dates),
            role="trade_cal_fallback",
        )
    except Exception as exc:
        return [], check_result(
            "akshare_sina_trade_calendar",
            "failed",
            "warning",
            started,
            error=sanitize(repr(exc)),
            role="trade_cal_fallback",
        )


def prior_trade_date(trade_date: str, open_dates: list[str]) -> str:
    prior = [d for d in open_dates if d < trade_date]
    return prior[-1] if prior else ""


def read_probe_codes(rank_file: Path, count: int = 5) -> list[str]:
    try:
        rank = pd.read_csv(rank_file, dtype={"ts_code": str})
    except Exception:
        return []
    if "liquidity_rank" in rank.columns:
        rank["liquidity_rank"] = pd.to_numeric(rank["liquidity_rank"], errors="coerce")
        rank = rank.sort_values("liquidity_rank")
    return rank["ts_code"].dropna().astype(str).drop_duplicates().head(count).tolist()


def resolve_probe_rank_file(rank_file: Path) -> tuple[Path, str]:
    if rank_file.exists():
        return rank_file, "requested_rank_file"
    if LOCKED_RANK_FILE.exists():
        return LOCKED_RANK_FILE, "locked_artifact_fallback"
    return rank_file, "missing"


def check_tcp_tls(url: str, name: str, severity_on_fail: str, timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    parsed = urlparse(url)
    host = parsed.hostname or url
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            if parsed.scheme == "https":
                ctx = ssl.create_default_context()
                with ctx.wrap_socket(sock, server_hostname=host):
                    pass
        return check_result(name, "success", "info", started, host=host, port=port, check="tcp_tls")
    except Exception as exc:
        return check_result(name, "failed", severity_on_fail, started, host=host, port=port, check="tcp_tls", error=sanitize(repr(exc)))


def check_tushare_trade_cal(token: str, trade_date: str, timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    client = TushareProxyClient(token, get_proxy_url(), timeout=timeout)
    try:
        df = client.call(
            "trade_cal",
            {"exchange": "SSE", "start_date": trade_date, "end_date": trade_date},
            fields="exchange,cal_date,is_open,pretrade_date",
        )
        return check_result(
            "tushare_proxy_trade_cal_api",
            "success",
            "info",
            started,
            base_url=get_proxy_url(),
            rows=int(len(df)),
            role="baseline_calendar",
        )
    except (TushareError, TushareRateLimit, SystemExit) as exc:
        return check_result(
            "tushare_proxy_trade_cal_api",
            "failed",
            "warning",
            started,
            base_url=get_proxy_url(),
            role="baseline_calendar",
            impact="will rely on local cache or AkShare/Sina fallback if available",
            error=sanitize(repr(exc)),
        )


def check_stk_mins_api(token: str, trade_date: str, prior: str, rank_file: Path, timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    probe_rank_file, rank_source = resolve_probe_rank_file(rank_file)
    codes = read_probe_codes(probe_rank_file)
    if not prior:
        return check_result(
            "tushare_proxy_stk_mins_api",
            "skipped",
            "fatal",
            started,
            role="open_exit_and_tail_realtime_bars",
            requested_rank_file=str(rank_file),
            probe_rank_file=str(probe_rank_file),
            rank_source=rank_source,
            impact="cannot verify required live minute provider because prior trade date is unavailable",
        )
    if not codes:
        return check_result(
            "tushare_proxy_stk_mins_api",
            "skipped",
            "fatal",
            started,
            role="open_exit_and_tail_realtime_bars",
            requested_rank_file=str(rank_file),
            probe_rank_file=str(probe_rank_file),
            rank_source=rank_source,
            impact="cannot verify required live minute provider because rank file is unavailable",
        )
    client = TushareProxyClient(token, get_proxy_url(), timeout=timeout)
    iso = ymd_to_iso(prior)
    params = {
        "ts_code": ",".join(codes),
        "start_date": f"{iso} 09:35:00",
        "end_date": f"{iso} 09:35:00",
        "freq": "5min",
    }
    try:
        df = client.call("stk_mins", params)
        status = "success" if len(df) > 0 else "empty_response"
        severity = "info" if len(df) > 0 else "fatal"
        return check_result(
            "tushare_proxy_stk_mins_api",
            status,
            severity,
            started,
            base_url=get_proxy_url(),
            probe_trade_date=prior,
            requested_rank_file=str(rank_file),
            probe_rank_file=str(probe_rank_file),
            rank_source=rank_source,
            probe_codes=len(codes),
            rows=int(len(df)),
            role="open_exit_and_tail_realtime_bars",
            impact="required for T+1 09:35-10:30 exit monitoring, T-day open auction dependency, and T-day 14:30-15:00 tail tracking",
        )
    except (TushareError, TushareRateLimit, SystemExit) as exc:
        return check_result(
            "tushare_proxy_stk_mins_api",
            "failed",
            "fatal",
            started,
            base_url=get_proxy_url(),
            probe_trade_date=prior,
            requested_rank_file=str(rank_file),
            probe_rank_file=str(probe_rank_file),
            rank_source=rank_source,
            probe_codes=len(codes),
            role="open_exit_and_tail_realtime_bars",
            impact="required provider is unavailable; stop today's paper runner and restart only after the provider recovers",
            error=sanitize(repr(exc)),
        )


def realtime_expected_checks(trade_date: str) -> tuple[bool, str | None, str]:
    current_day = today_ymd()
    if trade_date > current_day:
        return False, None, "future_trade_date"
    if trade_date < current_day:
        return True, "14:50:00", "historical_realtime_replay"
    now = bj_datetime()
    hhmm = now.hour * 100 + now.minute
    if hhmm < 925:
        return False, None, "before_open_auction_expected_time"
    if hhmm < 935:
        return True, None, "open_auction_due_minute_not_due"
    if hhmm >= 1450:
        return True, "14:50:00", "tail_signal_realtime_due"
    if hhmm >= 1030:
        return True, "10:30:00", "default_exit_realtime_due"
    return True, "09:35:00", "first_exit_realtime_due"


def check_today_realtime_data(token: str, trade_date: str, rank_file: Path, timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    expected, minute_time, phase = realtime_expected_checks(trade_date)
    if not expected:
        return check_result(
            "today_realtime_data_ok",
            "pending",
            "info",
            started,
            base_url=get_proxy_url(),
            phase=phase,
            role="today_open_auction_and_intraday_minutes",
            impact="today real-time data is not expected yet at current Beijing time",
        )

    probe_rank_file, rank_source = resolve_probe_rank_file(rank_file)
    codes = read_probe_codes(probe_rank_file)
    client = TushareProxyClient(token, get_proxy_url(), timeout=timeout)
    subchecks: list[dict[str, Any]] = []

    def call(api_name: str, params: dict[str, Any], expected_rows: bool = True) -> None:
        item_started = time.monotonic()
        try:
            df = client.call(api_name, params)
            rows = int(len(df))
            subchecks.append(
                {
                    "api_name": api_name,
                    "status": "success" if rows > 0 or not expected_rows else "empty_response",
                    "rows": rows,
                    "seconds": round(time.monotonic() - item_started, 3),
                    "params": {k: ("<codes>" if k == "ts_code" else v) for k, v in params.items()},
                }
            )
        except (TushareError, TushareRateLimit, SystemExit) as exc:
            subchecks.append(
                {
                    "api_name": api_name,
                    "status": "failed",
                    "rows": 0,
                    "seconds": round(time.monotonic() - item_started, 3),
                    "params": {k: ("<codes>" if k == "ts_code" else v) for k, v in params.items()},
                    "error": sanitize(repr(exc)),
                }
            )

    call("stk_auction_o", {"trade_date": trade_date})
    if minute_time:
        if codes:
            iso = ymd_to_iso(trade_date)
            call(
                "stk_mins",
                {
                    "ts_code": ",".join(codes),
                    "start_date": f"{iso} {minute_time}",
                    "end_date": f"{iso} {minute_time}",
                    "freq": "5min",
                },
            )
        else:
            subchecks.append(
                {
                    "api_name": "stk_mins",
                    "status": "skipped",
                    "rows": 0,
                    "seconds": 0.0,
                    "error": "rank file unavailable for minute probe",
                }
            )

    bad = [x for x in subchecks if x.get("status") != "success"]
    status = "success" if not bad else "empty_response" if all(x.get("status") == "empty_response" for x in bad) else "failed"
    if status == "success":
        impact = "today realtime feed returned rows for all currently due probes"
    elif status == "empty_response":
        empty_apis = ", ".join(sorted(set(str(x.get("api_name")) for x in bad if x.get("status") == "empty_response")))
        impact = (
            f"接口请求成功但今日实时数据返回 0 行: {empty_apis}；"
            "可能是供应商当日实时数据未刷新、未开放或实时服务异常，建议联系供应商核实今日 stk_mins/stk_auction_o 数据状态。"
        )
    else:
        failed_apis = ", ".join(sorted(set(str(x.get("api_name")) for x in bad)))
        impact = (
            f"今日实时数据探活失败: {failed_apis}；"
            "请先确认供应商接口服务、token 权限和当日实时数据发布状态。"
        )
    return check_result(
        "today_realtime_data_ok",
        status,
        "info" if status == "success" else "warning",
        started,
        base_url=get_proxy_url(),
        phase=phase,
        trade_date=trade_date,
        minute_probe_time=minute_time or "",
        probe_rank_file=str(probe_rank_file),
        rank_source=rank_source,
        probe_codes=len(codes),
        subchecks=subchecks,
        role="today_open_auction_and_intraday_minutes",
        impact=impact,
    )


def aggregate_check(name: str, ok: bool, started: float, *, status_if_false: str, severity_if_false: str, **extra: Any) -> dict[str, Any]:
    return check_result(
        name,
        "success" if ok else status_if_false,
        "info" if ok else severity_if_false,
        started,
        **extra,
    )


def check_local_cache(trade_date: str) -> dict[str, Any]:
    started = time.monotonic()
    open_dates = load_open_dates_from_cache()
    covers_trade_date = trade_date in open_dates
    return check_result(
        "local_trade_calendar_cache",
        "success" if covers_trade_date else "missing_trade_date",
        "info" if covers_trade_date else "warning",
        started,
        path=str(DEFAULT_TRADE_CAL),
        open_dates=int(len(open_dates)),
        covers_trade_date=bool(covers_trade_date),
    )


def summarize(checks: list[dict[str, Any]]) -> str:
    fatal = [c for c in checks if c.get("severity") == "fatal"]
    warnings = [c for c in checks if c.get("severity") == "warning"]
    if fatal:
        return "fatal"
    if warnings:
        return "warning"
    return "success"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight downstream connectivity checks for forward shadow paper tracking.")
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--rank-file", default=str(DEFAULT_RANK_FILE))
    parser.add_argument("--minute-dir", default=str(DEFAULT_MINUTE_DIR))
    parser.add_argument("--send-notifications", action="store_true")
    parser.add_argument("--wxpusher-endpoint", default="https://wxpusher.zjiecode.com/api/send/message")
    parser.add_argument("--timeout", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trade_date = str(args.trade_date)
    checks: list[dict[str, Any]] = []

    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    wx_token = os.environ.get("WXPUSHER_APP_TOKEN", "").strip()
    if not token:
        checks.append(
            {
                "name": "env_TUSHARE_TOKEN",
                "status": "missing",
                "severity": "fatal",
                "impact": "cannot call Tushare provider",
                "duration_seconds": 0.0,
            }
        )
    else:
        checks.append({"name": "env_TUSHARE_TOKEN", "status": "present", "severity": "info", "duration_seconds": 0.0})
    if args.send_notifications and not wx_token:
        checks.append(
            {
                "name": "env_WXPUSHER_APP_TOKEN",
                "status": "missing",
                "severity": "fatal",
                "impact": "WxPusher notifications cannot be sent",
                "duration_seconds": 0.0,
            }
        )
    elif args.send_notifications:
        checks.append({"name": "env_WXPUSHER_APP_TOKEN", "status": "present", "severity": "info", "duration_seconds": 0.0})
    else:
        checks.append({"name": "env_WXPUSHER_APP_TOKEN", "status": "skipped_notifications_disabled", "severity": "info", "duration_seconds": 0.0})

    cache_check = check_local_cache(trade_date)
    checks.append(cache_check)
    ak_open_dates, ak_check = fetch_open_dates_from_akshare()
    checks.append(ak_check)
    cache_open_dates = load_open_dates_from_cache()
    open_dates = sorted(set(cache_open_dates) | set(ak_open_dates))
    prior = prior_trade_date(trade_date, open_dates)

    if token:
        trade_cal_check = check_tushare_trade_cal(token, trade_date, int(args.timeout))
        historical_minute_check = check_stk_mins_api(token, trade_date, prior, Path(args.rank_file), int(args.timeout))
        checks.append(trade_cal_check)
        checks.append(historical_minute_check)
        checks.append(
            aggregate_check(
                "provider_up",
                trade_cal_check.get("status") == "success",
                time.monotonic(),
                status_if_false="failed",
                severity_if_false="warning",
                base_url=get_proxy_url(),
                role="provider_api_reachable",
                impact="provider API can accept requests and return a valid trade_cal response"
                if trade_cal_check.get("status") == "success"
                else "provider API did not return a valid trade_cal response; check domain, service availability, or token",
            )
        )
        checks.append(
            aggregate_check(
                "historical_data_ok",
                historical_minute_check.get("status") == "success",
                time.monotonic(),
                status_if_false=str(historical_minute_check.get("status") or "failed"),
                severity_if_false="fatal",
                base_url=get_proxy_url(),
                probe_trade_date=prior,
                role="historical_minute_data_probe",
                impact="historical stk_mins data returned rows for the prior trading day"
                if historical_minute_check.get("status") == "success"
                else "historical stk_mins data did not return usable rows; required for replay and baseline provider verification",
            )
        )
        checks.append(check_today_realtime_data(token, trade_date, Path(args.rank_file), int(args.timeout)))

    if args.send_notifications:
        checks.append(check_tcp_tls(args.wxpusher_endpoint, "wxpusher_domain_tcp_tls", "warning", int(args.timeout)))

    trade_cal_provider_ok = any(c["name"] == "tushare_proxy_trade_cal_api" and c["status"] == "success" for c in checks)
    trade_cal_fallback_ok = any(c["name"] == "akshare_sina_trade_calendar" and c["status"] == "success" for c in checks)
    local_trade_cal_ok = bool(cache_check.get("covers_trade_date"))
    if not (trade_cal_provider_ok or trade_cal_fallback_ok or local_trade_cal_ok):
        checks.append(
            {
                "name": "trade_calendar_resilience",
                "status": "failed",
                "severity": "fatal",
                "impact": f"cannot determine whether {trade_date} is an open trading day",
                "duration_seconds": 0.0,
            }
        )
    else:
        checks.append(
            {
                "name": "trade_calendar_resilience",
                "status": "success",
                "severity": "info",
                "provider_ok": trade_cal_provider_ok,
                "fallback_ok": trade_cal_fallback_ok,
                "local_cache_ok": local_trade_cal_ok,
                "duration_seconds": 0.0,
            }
        )

    overall = summarize(checks)
    output_root = Path(args.output_root)
    out_path = output_root / "preflight" / f"{trade_date}_preflight.json"
    payload = {
        "task": "preflight_downstream",
        "overall_status": overall,
        "generated_time_beijing": bj_now(),
        "trade_date": trade_date,
        "prior_trade_date_for_probe": prior,
        "checks": checks,
        "notes": [
            "WxPusher domain failure cannot be reliably reported through WxPusher itself; check this local JSON and dashboard status.",
            "stk_mins probe uses a prior-trading-day 09:35 bar to validate the same provider domain used by T+1 exit monitoring, T-day open auction, and T-day tail tracking.",
            "If the provider minute probe is fatal, the production runner should not continue because required real-time data is unavailable.",
            "This preflight does not modify v7 model, features, labels, TopN, or exit rules.",
        ],
    }
    write_json(out_path, payload)
    payload["report_path"] = str(out_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    if overall == "fatal":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
