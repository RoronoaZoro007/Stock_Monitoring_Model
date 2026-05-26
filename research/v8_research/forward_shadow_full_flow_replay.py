#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from build_forward_shadow_daily_features import (
    AUCTION_RAW_DIR,
    DAILY_FILE,
    MINUTE_DIR,
    RANK_PATH,
    ROOT,
    STOCK_BASIC_FILE,
    build_forward_features,
    vwap_from_amount_vol,
)
from build_forward_shadow_score_matrix import DEFAULT_MODEL_FILE, build_score_matrix
from tushare_data_pipeline import TushareError, TushareProxyClient, TushareRateLimit, get_proxy_url, get_token


DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "tushare" / "v8_forward_shadow" / "full_flow_replay"
DEFAULT_TOPIC_ID = 44635
ROUTE_ORDER = [
    "S0_v7_original_top10",
    "S1_U2_filter_only_no_refill",
    "S0_v7_original_top10_tail_down",
    "S1_U2_filter_only_no_refill_tail_down",
]
ROUTE_LABELS = {
    "S0_v7_original_top10": "Control 1 / S0 原始Top10",
    "S1_U2_filter_only_no_refill": "Control 2 / S1 U2保留",
    "S0_v7_original_top10_tail_down": "Candidate 1 / S0+tail_down",
    "S1_U2_filter_only_no_refill_tail_down": "Candidate 2 / S1+tail_down",
}
STEP_LABELS = {
    "startup_check": "启动检查",
    "auction_guard_0925": "竞价数据检查",
    "download_until_1430": "拉取至14:30",
    "incremental_download_1435": "拉取14:35",
    "incremental_download_1440": "拉取14:40",
    "incremental_download_1445": "拉取14:45",
    "tail_precheck_1445": "尾盘预检查",
    "download_1450_bar": "拉取14:50",
    "build_forward_features": "构建特征",
    "build_score_matrix": "生成Score",
    "freeze_signals": "冻结信号",
    "record_entry_1455_vwap": "记录Entry",
    "close_snapshot_1500": "收盘快照",
}


@dataclass
class StepRecord:
    step_id: str
    logical_time: str
    status: str
    duration_seconds: float
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def bj_now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ymd_to_iso(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def infer_prev_trade_date(trade_date: str) -> str:
    current = datetime.strptime(trade_date, "%Y%m%d")
    prior = current - timedelta(days=1)
    while prior.weekday() >= 5:
        prior -= timedelta(days=1)
    return prior.strftime("%Y%m%d")


def read_rank_codes(rank_file: Path, top_rank: int, trade_date: str) -> list[str]:
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


def source_minute_file(ts_code: str, trade_date: str, source_minute_dir: Path) -> Path:
    return source_minute_dir / f"ts_code={ts_code}" / f"{trade_date}_{trade_date}.parquet"


def staged_minute_file(ts_code: str, trade_date: str, staged_minute_dir: Path) -> Path:
    return staged_minute_dir / f"ts_code={ts_code}" / f"{trade_date}_{trade_date}.parquet"


def normalize_minute(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["dt"] = pd.to_datetime(out["trade_time"], errors="coerce")
    out["bar_time"] = out["dt"].dt.strftime("%H:%M")
    return out.dropna(subset=["dt"]).sort_values("dt")


def write_staged_symbol(
    ts_code: str,
    trade_date: str,
    source_minute_dir: Path,
    staged_minute_dir: Path,
    bar_time: str,
    mode: str,
) -> dict[str, Any]:
    src = source_minute_file(ts_code, trade_date, source_minute_dir)
    if not src.exists():
        return {"ts_code": ts_code, "status": "missing_source", "rows_written": 0, "new_rows": 0}
    df = normalize_minute(pd.read_parquet(src))
    if mode == "until":
        new = df[df["bar_time"].le(bar_time)].copy()
        combined = new
    elif mode == "bar":
        new = df[df["bar_time"].eq(bar_time)].copy()
        dst = staged_minute_file(ts_code, trade_date, staged_minute_dir)
        if dst.exists():
            old = normalize_minute(pd.read_parquet(dst))
            combined = pd.concat([old, new], ignore_index=True)
            combined = combined.drop_duplicates(["ts_code", "trade_time"], keep="last")
        else:
            combined = new
    else:
        raise ValueError(f"unsupported mode: {mode}")
    keep = [c for c in ["ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount"] if c in combined.columns]
    combined = combined[keep].sort_values("trade_time", ascending=False)
    dst = staged_minute_file(ts_code, trade_date, staged_minute_dir)
    dst.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(dst, index=False)
    return {
        "ts_code": ts_code,
        "status": "done",
        "rows_written": int(len(combined)),
        "new_rows": int(len(new)),
    }


def write_staged_frame(
    df: pd.DataFrame,
    codes: list[str],
    trade_date: str,
    staged_minute_dir: Path,
    mode: str,
) -> dict[str, Any]:
    if df.empty:
        df = pd.DataFrame(columns=["ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount"])
    df = df.copy()
    if "ts_code" not in df.columns:
        df["ts_code"] = ""
    df["ts_code"] = df["ts_code"].astype(str)
    returned_codes = set(df["ts_code"].dropna().astype(str).unique())
    rows_written = 0
    new_rows = 0
    for code in codes:
        new = df[df["ts_code"].eq(code)].copy()
        dst = staged_minute_file(code, trade_date, staged_minute_dir)
        if mode == "bar" and dst.exists():
            old = pd.read_parquet(dst)
            combined = pd.concat([old, new], ignore_index=True)
            combined = combined.drop_duplicates(["ts_code", "trade_time"], keep="last")
        else:
            combined = new
        keep = [c for c in ["ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount"] if c in combined.columns]
        combined = combined[keep].sort_values("trade_time", ascending=False) if keep else combined
        dst.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(dst, index=False)
        rows_written += int(len(combined))
        new_rows += int(len(new))
    return {
        "rows_written_total": rows_written,
        "new_rows": new_rows,
        "missing_return_codes": len(set(codes) - returned_codes),
    }


def stage_download(
    codes: list[str],
    trade_date: str,
    source_minute_dir: Path,
    staged_minute_dir: Path,
    bar_time: str,
    mode: str,
    workers: int,
) -> dict[str, Any]:
    rows_written = 0
    new_rows = 0
    missing_source = 0
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(write_staged_symbol, code, trade_date, source_minute_dir, staged_minute_dir, bar_time, mode)
            for code in codes
        ]
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            rows_written += int(res.get("rows_written", 0))
            new_rows += int(res.get("new_rows", 0))
            if res.get("status") == "missing_source":
                missing_source += 1
    return {
        "symbols": len(codes),
        "symbols_processed": done,
        "missing_source_symbols": missing_source,
        "rows_written_total": rows_written,
        "new_rows": new_rows,
        "bar_time": bar_time,
        "mode": mode,
    }


class ProviderRateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.interval = 60.0 / max(1, int(requests_per_minute))
        self.last = 0.0

    def wait(self) -> None:
        wait_seconds = self.interval - (time.monotonic() - self.last)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        self.last = time.monotonic()


def chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def provider_download(
    client: TushareProxyClient,
    limiter: ProviderRateLimiter,
    codes: list[str],
    trade_date: str,
    staged_minute_dir: Path,
    bar_time: str,
    mode: str,
    batch_size: int,
    max_retries: int,
) -> dict[str, Any]:
    if mode == "until":
        start_time = "09:30"
        end_time = bar_time
    elif mode == "bar":
        start_time = bar_time
        end_time = bar_time
    else:
        raise ValueError(f"unsupported provider download mode: {mode}")
    iso = ymd_to_iso(trade_date)
    request_count = 0
    rows_returned = 0
    rows_written = 0
    new_rows = 0
    missing_return_codes = 0
    rate_limit_retries = 0
    api_errors = 0
    request_seconds = 0.0
    for batch_index, batch_codes in enumerate(chunked(codes, batch_size), start=1):
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
            request_count += 1
            try:
                df = client.call("stk_mins", params)
                elapsed = time.monotonic() - started
                request_seconds += elapsed
                rows_returned += int(len(df))
                write = write_staged_frame(df, batch_codes, trade_date, staged_minute_dir, mode)
                rows_written += int(write["rows_written_total"])
                new_rows += int(write["new_rows"])
                missing_return_codes += int(write["missing_return_codes"])
                break
            except TushareRateLimit:
                elapsed = time.monotonic() - started
                request_seconds += elapsed
                rate_limit_retries += 1
                if attempt > max_retries:
                    raise
                time.sleep(min(120, 10 * attempt))
            except TushareError:
                api_errors += 1
                if attempt > max_retries:
                    raise
                time.sleep(min(60, 5 * attempt))
    return {
        "symbols": len(codes),
        "symbols_processed": len(codes),
        "missing_source_symbols": 0,
        "rows_written_total": rows_written,
        "new_rows": new_rows,
        "bar_time": bar_time,
        "mode": mode,
        "data_mode": "provider",
        "provider_api": "stk_mins",
        "provider_start_time": start_time,
        "provider_end_time": end_time,
        "provider_batch_size": batch_size,
        "provider_request_count": request_count,
        "provider_rows_returned": rows_returned,
        "provider_missing_return_codes": missing_return_codes,
        "provider_request_seconds": round(request_seconds, 3),
        "provider_rate_limit_retries": rate_limit_retries,
        "provider_api_errors": api_errors,
    }


def run_download_step(
    data_mode: str,
    codes: list[str],
    trade_date: str,
    source_minute_dir: Path,
    staged_minute_dir: Path,
    bar_time: str,
    mode: str,
    workers: int,
    client: TushareProxyClient | None,
    limiter: ProviderRateLimiter | None,
    provider_batch_size: int,
    provider_max_retries: int,
) -> dict[str, Any]:
    if data_mode == "local_snapshot":
        details = stage_download(codes, trade_date, source_minute_dir, staged_minute_dir, bar_time, mode, workers)
        details["data_mode"] = "local_snapshot"
        return details
    if data_mode == "provider":
        if client is None or limiter is None:
            raise ValueError("provider mode requires initialized Tushare client and limiter")
        return provider_download(client, limiter, codes, trade_date, staged_minute_dir, bar_time, mode, provider_batch_size, provider_max_retries)
    raise ValueError(f"unsupported data_mode: {data_mode}")


def precheck_staged(
    codes: list[str],
    trade_date: str,
    staged_minute_dir: Path,
    required_latest_bar: str,
) -> dict[str, Any]:
    missing_file = 0
    missing_latest_bar = 0
    abnormal_price = 0
    zero_amount_latest = 0
    for code in codes:
        path = staged_minute_file(code, trade_date, staged_minute_dir)
        if not path.exists():
            missing_file += 1
            continue
        df = normalize_minute(pd.read_parquet(path))
        if required_latest_bar not in set(df["bar_time"]):
            missing_latest_bar += 1
        bad = df[
            (pd.to_numeric(df["close"], errors="coerce") <= 0)
            | (pd.to_numeric(df["high"], errors="coerce") < pd.to_numeric(df["low"], errors="coerce"))
            | (pd.to_numeric(df["close"], errors="coerce") > pd.to_numeric(df["high"], errors="coerce") * 1.001)
            | (pd.to_numeric(df["close"], errors="coerce") < pd.to_numeric(df["low"], errors="coerce") * 0.999)
        ]
        abnormal_price += int(len(bad))
        latest = df[df["bar_time"].eq(required_latest_bar)]
        if latest.empty or float(pd.to_numeric(latest["amount"], errors="coerce").fillna(0).sum()) <= 0:
            zero_amount_latest += 1
    return {
        "symbols_checked": len(codes),
        "missing_file": missing_file,
        "missing_latest_bar": missing_latest_bar,
        "abnormal_price_rows": abnormal_price,
        "zero_amount_latest_bar_symbols": zero_amount_latest,
        "required_latest_bar": required_latest_bar,
    }


def run_command(argv: list[str]) -> dict[str, Any]:
    proc = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True)
    return {
        "return_code": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def send_wxpusher(title: str, content: str, topic_id: int, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"status": "dry_run", "title": title}
    token = os.environ.get("WXPUSHER_APP_TOKEN", "").strip()
    if not token:
        return {"status": "skipped_missing_token", "title": title}
    payload = {
        "appToken": token,
        "content": content,
        "summary": title[:100],
        "contentType": 3,
        "topicIds": [int(topic_id)],
        "verifyPay": False,
    }
    resp = requests.post("https://wxpusher.zjiecode.com/api/send/message", json=payload, timeout=20)
    return {"status": "success" if resp.status_code == 200 else "failed", "http_status": resp.status_code, "body": resp.text[:1000]}


def fmt_int(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def fmt_float(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_kv(rows: list[tuple[str, Any]]) -> str:
    out = ["| 指标 | 值 |", "|---|---:|"]
    for key, value in rows:
        out.append(f"| {key} | {value} |")
    return "\n".join(out)


def compact_step_message(step: StepRecord, trade_date: str, data_mode: str, extra_md: str = "") -> str:
    label = STEP_LABELS.get(step.step_id, step.step_id)
    rows = [
        ("交易日", trade_date),
        ("逻辑时间", step.logical_time),
        ("步骤", label),
        ("步骤ID", step.step_id),
        ("状态", step.status),
        ("耗时", f"{step.duration_seconds:.2f}s"),
        ("数据模式", data_mode),
    ]
    body = ["## 节点状态", markdown_kv(rows)]
    if extra_md:
        body.extend(["", extra_md])
    body.extend(["", "说明：paper-only，不下单，不构成交易建议。"])
    return "\n".join(body)


def download_message(details: dict[str, Any]) -> str:
    rows = [
        ("bar", details.get("bar_time", "-")),
        ("模式", details.get("mode", "-")),
        ("股票数", fmt_int(details.get("symbols"))),
        ("新增行数", fmt_int(details.get("new_rows"))),
        ("累计落地行数", fmt_int(details.get("rows_written_total"))),
        ("缺源股票", fmt_int(details.get("missing_source_symbols", 0))),
    ]
    if details.get("data_mode") == "provider":
        rows.extend(
            [
                ("接口请求数", fmt_int(details.get("provider_request_count"))),
                ("接口返回行数", fmt_int(details.get("provider_rows_returned"))),
                ("接口耗时", f"{fmt_float(details.get('provider_request_seconds'))}s"),
                ("限流重试", fmt_int(details.get("provider_rate_limit_retries"))),
                ("接口错误", fmt_int(details.get("provider_api_errors"))),
                ("缺返回代码", fmt_int(details.get("provider_missing_return_codes"))),
            ]
        )
    return "## 数据拉取\n" + markdown_kv(rows)


def precheck_message(details: dict[str, Any]) -> str:
    rows = [
        ("检查股票数", fmt_int(details.get("symbols_checked"))),
        ("缺文件", fmt_int(details.get("missing_file"))),
        ("缺最新bar", fmt_int(details.get("missing_latest_bar"))),
        ("异常价格行", fmt_int(details.get("abnormal_price_rows"))),
        ("最新bar零成交", fmt_int(details.get("zero_amount_latest_bar_symbols"))),
        ("要求bar", details.get("required_latest_bar", "-")),
    ]
    return "## 尾盘预检查\n" + markdown_kv(rows)


def feature_message(features: dict[str, Any]) -> str:
    skipped = features.get("skipped") or {}
    rows = [
        ("特征行数", fmt_int(features.get("rows"))),
        ("TopRank", fmt_int(features.get("top_rank"))),
        ("数据截止", features.get("data_max_timestamp", "-")),
        ("使用15:00/次日", str(features.get("uses_15_or_next_day_for_features", "-")).lower()),
        ("near_limit跳过", fmt_int(skipped.get("near_limit", 0))),
        ("missing_daily跳过", fmt_int(skipped.get("missing_daily", 0))),
    ]
    return "## 特征构建\n" + markdown_kv(rows) + f"\n\n文件：`{features.get('feature_path', '-')}`"


def score_message(score: dict[str, Any]) -> str:
    rows = [
        ("score行数", fmt_int(score.get("rows"))),
        ("特征数", fmt_int(score.get("feature_count"))),
        ("逻辑数据截止", score.get("logical_data_max_timestamp", "-")),
        ("使用15:00/次日", str(score.get("uses_15_or_next_day_for_score", "-")).lower()),
        ("重训模型", "false"),
    ]
    return "## 打分结果\n" + markdown_kv(rows) + f"\n\n文件：`{score.get('score_path', '-')}`"


def auction_message(details: dict[str, Any]) -> str:
    rows = [
        ("状态", details.get("status", "-")),
        ("交易日", details.get("trade_date", "-")),
        ("前一交易日", details.get("prior_trade_date", "-")),
        ("TopRank", fmt_int(details.get("top_rank"))),
        ("耗时", f"{fmt_float(details.get('duration_seconds'))}s"),
        ("目录", details.get("auction_raw_dir", "-")),
    ]
    body = ["## 集合竞价检查", markdown_kv(rows)]
    after = details.get("after") or []
    if after:
        table = ["| role | api | date | rows | present | missing |", "|---|---|---:|---:|---:|---:|"]
        for item in after:
            table.append(
                "| {role} | {api_name} | {trade_date} | {rows} | {present} | {missing} |".format(
                    role=item.get("role", ""),
                    api_name=item.get("api_name", ""),
                    trade_date=item.get("trade_date", ""),
                    rows=fmt_int(item.get("rows")),
                    present=fmt_int(item.get("top_symbols_present")),
                    missing=fmt_int(item.get("top_symbols_missing")),
                )
            )
        body.extend(["", "\n".join(table)])
    return "\n".join(body)


def format_step_table(records: list[StepRecord]) -> str:
    rows = ["| 时间 | 步骤 | 状态 | 耗时 | 说明 |", "|---|---|---:|---:|---|"]
    for r in records:
        rows.append(f"| {r.logical_time} | {r.step_id} | {r.status} | {r.duration_seconds:.3f} | {r.message} |")
    return "\n".join(rows)


def route_summary(output_root: Path, trade_date: str) -> pd.DataFrame:
    status_path = output_root / "forward_shadow_candidate_status.csv"
    if not status_path.exists():
        return pd.DataFrame()
    status = pd.read_csv(status_path)
    status["trade_date"] = status["trade_date"].astype(str)
    status = status[status["trade_date"].eq(trade_date)].copy()
    signals_path = output_root / "daily_signals" / f"{trade_date}_signals.csv"
    if signals_path.exists():
        signals = pd.read_csv(signals_path)
        signals["trade_date"] = signals["trade_date"].astype(str)
        sig = signals[signals["trade_date"].eq(trade_date)].copy()
        avg_score = sig.groupby("strategy_id")["score"].mean().rename("avg_score").reset_index()
        codes = sig.groupby("strategy_id")["code"].apply(lambda s: ",".join(s.astype(str).tolist())).rename("codes").reset_index()
        status = status.merge(avg_score, on="strategy_id", how="left").merge(codes, on="strategy_id", how="left")
    for col in ["avg_score", "codes"]:
        if col not in status.columns:
            status[col] = np.nan
    status["route_order"] = status["strategy_id"].map({v: i for i, v in enumerate(ROUTE_ORDER)})
    return status.sort_values("route_order")


def format_route_table(routes: pd.DataFrame) -> str:
    if routes.empty:
        return "No route status generated."
    rows = ["| 线路 | 启用 | 入选 | tail_down | 原因 | 平均score |", "|---|---:|---:|---:|---|---:|"]
    for row in routes.itertuples(index=False):
        selected = int(getattr(row, "selected_count", 0))
        raw_reason = getattr(row, "no_trade_reason", "")
        reason = "" if pd.isna(raw_reason) else str(raw_reason)
        active = selected > 0
        avg = getattr(row, "avg_score", np.nan)
        avg_text = "" if pd.isna(avg) else f"{float(avg):.6f}"
        label = ROUTE_LABELS.get(row.strategy_id, row.strategy_id)
        rows.append(
            f"| {label} | {str(active).lower()} | {selected} | {str(bool(row.tail_down_flag)).lower()} | {reason} | {avg_text} |"
        )
    return "\n".join(rows)


def format_route_stock_table(paper_root: Path, trade_date: str, max_rows: int) -> str:
    signals_path = paper_root / "daily_signals" / f"{trade_date}_signals.csv"
    if not signals_path.exists():
        return "No selected stock detail generated."
    signals = pd.read_csv(signals_path)
    if signals.empty:
        return "No selected stock detail generated."
    out = signals.copy()
    out["线路"] = out["strategy_id"].map(ROUTE_LABELS).fillna(out["strategy_id"])
    rename = {
        "code": "代码",
        "name": "名称",
        "original_v7_rank": "排名",
        "score": "score",
        "in_u2": "U2",
        "position_weight": "权重",
    }
    cols = ["线路", "code", "name", "original_v7_rank", "score", "in_u2", "position_weight"]
    cols = [c for c in cols if c in out.columns]
    out = out[cols].rename(columns=rename)
    if "score" in out.columns:
        out["score"] = pd.to_numeric(out["score"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    if "权重" in out.columns:
        out["权重"] = pd.to_numeric(out["权重"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    if len(out) > max_rows:
        shown = out.head(max_rows)
        return shown.to_markdown(index=False) + f"\n\n仅展示前 {max_rows} 行，完整明细见 CSV。"
    return out.to_markdown(index=False)


def entry_message(entry: pd.DataFrame, max_rows: int, entry_path: Path) -> str:
    if entry.empty:
        return "## Entry记录\n无 entry 行。"
    out = entry.copy()
    out["线路"] = out["strategy_id"].map(ROUTE_LABELS).fillna(out["strategy_id"])
    out = out.rename(
        columns={
            "code": "代码",
            "name": "名称",
            "paper_entry_status": "状态",
            "expected_entry_time": "时间",
            "entry_vwap": "VWAP",
            "weight": "权重",
        }
    )
    cols = [c for c in ["线路", "代码", "名称", "状态", "时间", "VWAP", "权重"] if c in out.columns]
    out = out[cols]
    if "VWAP" in out.columns:
        out["VWAP"] = pd.to_numeric(out["VWAP"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    if "权重" in out.columns:
        out["权重"] = pd.to_numeric(out["权重"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    if len(out) > max_rows:
        table = out.head(max_rows).to_markdown(index=False) + f"\n\n仅展示前 {max_rows} 行，完整明细见 CSV。"
    else:
        table = out.to_markdown(index=False)
    return "\n".join(["## 14:55 Entry记录", table, "", f"文件：`{entry_path}`"])


def write_entry_prices(paper_root: Path, staged_minute_dir: Path, trade_date: str) -> pd.DataFrame:
    signals_path = paper_root / "daily_signals" / f"{trade_date}_signals.csv"
    if not signals_path.exists():
        return pd.DataFrame()
    signals = pd.read_csv(signals_path)
    rows: list[dict[str, Any]] = []
    for row in signals.itertuples(index=False):
        code = str(row.code)
        path = staged_minute_file(code, trade_date, staged_minute_dir)
        status = "missing_1455_bar"
        entry_vwap = np.nan
        if path.exists():
            df = normalize_minute(pd.read_parquet(path))
            bar = df[df["bar_time"].eq("14:55")]
            if not bar.empty:
                b = bar.iloc[-1]
                entry_vwap = vwap_from_amount_vol(b.get("amount"), b.get("vol"), b.get("close"), b.get("low"), b.get("high"))
                status = "entry_recorded"
        rows.append(
            {
                "trade_date": trade_date,
                "strategy_id": row.strategy_id,
                "candidate_id": row.candidate_id,
                "code": code,
                "name": getattr(row, "name", ""),
                "original_v7_rank": getattr(row, "original_v7_rank", np.nan),
                "score": getattr(row, "score", np.nan),
                "paper_entry_status": status,
                "expected_entry_time": "14:55",
                "entry_vwap": entry_vwap,
                "entry_price_source": "staged 20260522 14:55 5min bar amount/vol VWAP",
                "weight": getattr(row, "position_weight", np.nan),
                "paper_tracking_only": True,
            }
        )
    out = pd.DataFrame(rows)
    out_dir = paper_root / "daily_entry_prices"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"{trade_date}_entry_prices.csv", index=False)
    return out


def add_record(records: list[StepRecord], step_id: str, logical_time: str, started: float, status: str, message: str, details: dict[str, Any]) -> StepRecord:
    rec = StepRecord(step_id, logical_time, status, time.monotonic() - started, message, details)
    records.append(rec)
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description="Full production-timeline replay for forward shadow paper tracking.")
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--top-rank", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--data-mode", choices=["local_snapshot", "provider"], default="local_snapshot")
    parser.add_argument("--source-minute-dir", default=str(MINUTE_DIR))
    parser.add_argument("--rank-file", default=str(RANK_PATH))
    parser.add_argument("--daily-file", default=str(DAILY_FILE))
    parser.add_argument("--model-file", default=str(DEFAULT_MODEL_FILE))
    parser.add_argument("--prior-trade-date")
    parser.add_argument("--auction-raw-dir")
    parser.add_argument("--topic-id", type=int, default=DEFAULT_TOPIC_ID)
    parser.add_argument("--dry-run-push", action="store_true")
    parser.add_argument("--provider-batch-size", type=int, default=160)
    parser.add_argument("--provider-requests-per-minute", type=int, default=120)
    parser.add_argument("--provider-timeout", type=int, default=90)
    parser.add_argument("--provider-max-retries", type=int, default=3)
    parser.add_argument("--max-push-stock-rows", type=int, default=24)
    args = parser.parse_args()

    trade_date = str(args.trade_date)
    root = Path(args.output_root) / trade_date
    staged_minute_dir = root / "staged_raw" / "stk_mins" / "freq=5min"
    auction_raw_dir = Path(args.auction_raw_dir) if args.auction_raw_dir else root / "staged_raw"
    paper_root = root / "paper_outputs"
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    paper_root.mkdir(parents=True, exist_ok=True)

    records: list[StepRecord] = []
    push_logs: list[dict[str, Any]] = []
    provider_client: TushareProxyClient | None = None
    provider_limiter: ProviderRateLimiter | None = None
    if args.data_mode == "provider":
        if args.provider_batch_size <= 0 or args.provider_batch_size > 180:
            raise SystemExit("--provider-batch-size must be in 1..180 for <=14:30 5min replay pulls")
        provider_client = TushareProxyClient(get_token(), get_proxy_url(), timeout=args.provider_timeout)
        provider_limiter = ProviderRateLimiter(args.provider_requests_per_minute)

    def finish_and_push(step: StepRecord, extra_md: str = "") -> None:
        step_label = STEP_LABELS.get(step.step_id, step.step_id)
        content = "\n".join([f"# Forward Shadow {trade_date}", "", compact_step_message(step, trade_date, args.data_mode, extra_md)])
        title = f"FS {trade_date} {step.logical_time} {step_label}: {step.status}"
        push = send_wxpusher(title, content, args.topic_id, args.dry_run_push)
        push_logs.append(
            {
                "step_id": step.step_id,
                "step_label": step_label,
                "logical_time": step.logical_time,
                "content_chars": len(content),
                "content_preview": content[:500],
                **push,
            }
        )

    started = time.monotonic()
    codes = read_rank_codes(Path(args.rank_file), args.top_rank, trade_date)
    rec = add_record(records, "startup_check", "14:30:00", started, "success", f"loaded {len(codes)} active Top{args.top_rank} symbols", {"symbols": len(codes), "data_mode": args.data_mode})
    finish_and_push(rec, markdown_kv([("股票池", fmt_int(len(codes))), ("TopRank", fmt_int(args.top_rank)), ("数据源模式", args.data_mode)]))

    started = time.monotonic()
    prior_trade_date = str(args.prior_trade_date or infer_prev_trade_date(trade_date))
    auction_cmd = [
        sys.executable,
        "research/v8_research/forward_shadow_data_guard.py",
        "auction-guard",
        "--trade-date",
        trade_date,
        "--prior-trade-date",
        prior_trade_date,
        "--output-root",
        str(root),
        "--auction-raw-dir",
        str(auction_raw_dir),
        "--rank-file",
        str(Path(args.rank_file)),
        "--top-rank",
        str(args.top_rank),
        "--requests-per-minute",
        str(args.provider_requests_per_minute),
        "--timeout",
        str(args.provider_timeout),
        "--max-retries",
        str(args.provider_max_retries),
    ]
    auction_run = run_command(auction_cmd)
    auction_report_path = root / "data_guards" / f"{trade_date}_auction_guard_0925.json"
    auction_details: dict[str, Any] = {"command": auction_cmd, **auction_run}
    if auction_report_path.exists():
        try:
            auction_details.update(json.loads(auction_report_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    auction_status = "success" if auction_run["return_code"] == 0 else "failed"
    rec = add_record(records, "auction_guard_0925", "09:25:30", started, auction_status, "checked/fetched T-1 close auction and T open auction", auction_details)
    finish_and_push(rec, auction_message(auction_details))
    if auction_status != "success":
        raise SystemExit(json.dumps(auction_details, ensure_ascii=False, indent=2, default=str))

    started = time.monotonic()
    initial = run_download_step(
        args.data_mode,
        codes,
        trade_date,
        Path(args.source_minute_dir),
        staged_minute_dir,
        "14:30",
        "until",
        args.workers,
        provider_client,
        provider_limiter,
        args.provider_batch_size,
        args.provider_max_retries,
    )
    rec = add_record(records, "download_until_1430", "14:30:05", started, "success", "staged all <=14:30 minute bars from empty replay cache", initial)
    finish_and_push(rec, download_message(initial))

    for bar in ["14:35", "14:40", "14:45"]:
        started = time.monotonic()
        details = run_download_step(
            args.data_mode,
            codes,
            trade_date,
            Path(args.source_minute_dir),
            staged_minute_dir,
            bar,
            "bar",
            args.workers,
            provider_client,
            provider_limiter,
            args.provider_batch_size,
            args.provider_max_retries,
        )
        rec = add_record(records, f"incremental_download_{bar.replace(':', '')}", f"{bar}:05", started, "success", f"staged incremental {bar} bar", details)
        finish_and_push(rec, download_message(details))

    started = time.monotonic()
    pre = precheck_staged(codes, trade_date, staged_minute_dir, "14:45")
    pre_path = report_dir / f"{trade_date}_tail_precheck_1445.json"
    pre_path.write_text(json.dumps(pre, ensure_ascii=False, indent=2), encoding="utf-8")
    pre_status = "success" if pre["missing_latest_bar"] == 0 and pre["abnormal_price_rows"] == 0 else "warning"
    rec = add_record(records, "tail_precheck_1445", "14:45:00", started, pre_status, "checked missing bar, abnormal price and zero amount", pre)
    finish_and_push(rec, precheck_message(pre))

    started = time.monotonic()
    details = run_download_step(
        args.data_mode,
        codes,
        trade_date,
        Path(args.source_minute_dir),
        staged_minute_dir,
        "14:50",
        "bar",
        args.workers,
        provider_client,
        provider_limiter,
        args.provider_batch_size,
        args.provider_max_retries,
    )
    rec = add_record(records, "download_1450_bar", "14:50:05", started, "success", "staged critical 14:50 bar", details)
    finish_and_push(rec, download_message(details))

    started = time.monotonic()
    features = build_forward_features(
        trade_date=trade_date,
        top_rank=args.top_rank,
        min_amount_sofar=20_000_000,
        output_root=paper_root,
        rank_file=Path(args.rank_file),
        daily_file=Path(args.daily_file),
        minute_dir=staged_minute_dir,
        auction_raw_dir=auction_raw_dir,
    )
    rec = add_record(records, "build_forward_features", "14:50:40", started, "success", f"built <=14:50 features rows={features['rows']}", features)
    finish_and_push(rec, feature_message(features))

    started = time.monotonic()
    score = build_score_matrix(
        trade_date=trade_date,
        feature_file=Path(features["feature_path"]),
        model_file=Path(args.model_file),
        output_root=paper_root,
        data_max_timestamp=f"{ymd_to_iso(trade_date)} 14:50:00",
        paper_reconstruction=True,
    )
    rec = add_record(records, "build_score_matrix", "14:51:20", started, "success", f"built score matrix rows={score['rows']}", score)
    finish_and_push(rec, score_message(score))

    started = time.monotonic()
    cmd = [
        sys.executable,
        "research/v8_research/forward_shadow_runner.py",
        "--trade-date",
        trade_date,
        "--score-file",
        str(score["score_path"]),
        "--daily-file",
        str(Path(args.daily_file)),
        "--output-root",
        str(paper_root),
    ]
    freeze = run_command(cmd)
    status = "success" if freeze["return_code"] == 0 else "failed"
    routes = route_summary(paper_root, trade_date)
    route_path = report_dir / f"{trade_date}_route_summary.csv"
    routes.to_csv(route_path, index=False)
    rec = add_record(records, "freeze_signals", "14:51:50", started, status, "froze four paper tracking routes", {"command": cmd, **freeze})
    freeze_msg = "\n\n".join(["## 四条线路", format_route_table(routes), "## 推荐明细", format_route_stock_table(paper_root, trade_date, args.max_push_stock_rows), f"完整文件：`{route_path}`"])
    finish_and_push(rec, freeze_msg)

    started = time.monotonic()
    details = run_download_step(
        args.data_mode,
        codes,
        trade_date,
        Path(args.source_minute_dir),
        staged_minute_dir,
        "14:55",
        "bar",
        args.workers,
        provider_client,
        provider_limiter,
        args.provider_batch_size,
        args.provider_max_retries,
    )
    entry = write_entry_prices(paper_root, staged_minute_dir, trade_date)
    entry_path = paper_root / "daily_entry_prices" / f"{trade_date}_entry_prices.csv"
    rec = add_record(records, "record_entry_1455_vwap", "14:55:05", started, "success", f"recorded paper entry VWAP rows={len(entry)}", {"download": details, "entry_rows": len(entry), "entry_path": str(entry_path)})
    finish_and_push(rec, "\n\n".join([download_message(details), entry_message(entry, args.max_push_stock_rows, entry_path)]))

    started = time.monotonic()
    details = run_download_step(
        args.data_mode,
        codes,
        trade_date,
        Path(args.source_minute_dir),
        staged_minute_dir,
        "15:00",
        "bar",
        args.workers,
        provider_client,
        provider_limiter,
        args.provider_batch_size,
        args.provider_max_retries,
    )
    rec = add_record(records, "close_snapshot_1500", "15:00:05", started, "success", "staged 15:00 close bar for end-of-day replay completeness", details)
    finish_and_push(rec, download_message(details))

    timings = pd.DataFrame([{"generated_time_beijing": bj_now(), **r.__dict__, "details": json.dumps(r.details, ensure_ascii=False, default=str)} for r in records])
    timings_path = report_dir / f"{trade_date}_full_flow_step_timings.csv"
    timings.to_csv(timings_path, index=False)
    push_path = report_dir / f"{trade_date}_wxpusher_push_logs.csv"
    pd.DataFrame(push_logs).to_csv(push_path, index=False)
    files = [timings_path, push_path, route_path, pre_path, Path(features["feature_path"]), Path(score["score_path"]), entry_path]
    sha_rows = [{"file": str(p), "sha256": sha256_file(p)} for p in files if p.exists()]
    sha_path = report_dir / f"{trade_date}_full_flow_sha256.csv"
    pd.DataFrame(sha_rows).to_csv(sha_path, index=False)

    final_md = "\n\n".join(
        [
            "## 步骤耗时",
            format_step_table(records),
            "## 四条线路",
            format_route_table(routes),
            "## 输出文件",
            "\n".join(f"- `{p}`" for p in files if p.exists()),
        ]
    )
    final_push = send_wxpusher(f"FS {trade_date} 15:00:30 全流程完成", final_md, args.topic_id, args.dry_run_push)
    push_logs.append(
        {
            "step_id": "final_summary",
            "step_label": "最终汇总",
            "logical_time": "15:00:30",
            "content_chars": len(final_md),
            "content_preview": final_md[:500],
            **final_push,
        }
    )
    pd.DataFrame(push_logs).to_csv(push_path, index=False)

    summary = {
        "trade_date": trade_date,
        "paper_only": True,
        "source_data_mode": "real provider stk_mins requests into isolated staged replay cache" if args.data_mode == "provider" else "isolated staged replay from existing local raw minute snapshot",
        "data_mode": args.data_mode,
        "output_root": str(root),
        "paper_output_root": str(paper_root),
        "auction_raw_dir": str(auction_raw_dir),
        "step_timings": str(timings_path),
        "route_summary": str(route_path),
        "entry_prices": str(entry_path),
        "push_logs": str(push_path),
        "sha256": str(sha_path),
        "final_push": final_push,
        "records": [r.__dict__ for r in records],
    }
    summary_path = report_dir / f"{trade_date}_full_flow_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
