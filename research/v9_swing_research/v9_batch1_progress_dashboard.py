#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import subprocess
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data_tushare" / "raw"
REPORT_DIR = ROOT / "reports" / "tushare" / "v9_swing_research" / "batch1_history_daily"
PROGRESS_PATH = REPORT_DIR / "batch1_download_progress.jsonl"
TRADE_CAL_PATH = RAW_DIR / "bootstrap" / "trade_cal_20180101_20260522.parquet"
ENDPOINTS = ["daily", "adj_factor", "daily_basic", "suspend_d", "stk_limit"]


def safe_read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return rows


def load_open_dates() -> list[str]:
    if not TRADE_CAL_PATH.exists():
        return []
    try:
        cal = pd.read_parquet(TRADE_CAL_PATH)
        cal["is_open"] = pd.to_numeric(cal["is_open"], errors="coerce").fillna(0).astype(int)
        return sorted(cal[cal["is_open"].eq(1)]["cal_date"].astype(str).unique().tolist())
    except Exception:
        return []


def endpoint_counts(open_dates: list[str]) -> list[dict[str, Any]]:
    expected = set(open_dates)
    rows = []
    for api in ENDPOINTS:
        files = sorted((RAW_DIR / api).glob("trade_date=*.parquet")) if (RAW_DIR / api).exists() else []
        got = {p.stem.split("=", 1)[1] for p in files}
        in_range = got & expected
        missing = sorted(expected - got)
        rows.append(
            {
                "api": api,
                "available": len(in_range),
                "expected": len(expected),
                "missing": len(missing),
                "first_missing": missing[0] if missing else "",
                "last_missing": missing[-1] if missing else "",
            }
        )
    return rows


def running_processes() -> list[str]:
    try:
        out = subprocess.check_output(["ps", "-ax", "-o", "pid,etime,command"], text=True)
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        if "v9_batch1_download_daily_history.py" in line and "rg " not in line:
            rows.append(line.strip())
    return rows


def parse_ts(value: str) -> float | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def build_status() -> dict[str, Any]:
    open_dates = load_open_dates()
    counts = endpoint_counts(open_dates)
    total_expected = sum(row["expected"] for row in counts)
    total_available = sum(row["available"] for row in counts)
    total_missing = sum(row["missing"] for row in counts)
    progress = safe_read_jsonl(PROGRESS_PATH)
    latest = progress[-1] if progress else {}
    failures = [row for row in progress if row.get("status") == "failed"]
    rate_limited = [row for row in failures if "429" in str(row.get("error", "")) or "Too Many Requests" in str(row.get("error", ""))]
    rate_per_min = None
    eta_minutes = None
    if len(progress) >= 2:
        first = progress[0]
        last = progress[-1]
        t0 = parse_ts(str(first.get("ts", "")))
        t1 = parse_ts(str(last.get("ts", "")))
        d0 = float(first.get("done", 0) or 0)
        d1 = float(last.get("done", 0) or 0)
        if t0 and t1 and t1 > t0 and d1 > d0:
            rate_per_min = (d1 - d0) / ((t1 - t0) / 60.0)
            remaining = max(0.0, float(last.get("total", 0) or 0) - d1)
            eta_minutes = remaining / rate_per_min if rate_per_min > 0 else None
    now = time.time()
    latest_ts = parse_ts(str(latest.get("ts", ""))) if latest else None
    if running_processes():
        if latest_ts is None:
            state = "running"
        elif now - latest_ts > 180:
            state = "running_or_cooling_down"
        else:
            state = "running"
    else:
        state = "not_running"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "state": state,
        "processes": running_processes(),
        "open_dates": len(open_dates),
        "coverage": {
            "available": total_available,
            "expected": total_expected,
            "missing": total_missing,
            "pct": total_available / total_expected if total_expected else 0.0,
        },
        "endpoint_counts": counts,
        "progress_file": str(PROGRESS_PATH),
        "latest_progress": latest,
        "progress_rows": len(progress),
        "failures": len(failures),
        "rate_limited_failures": len(rate_limited),
        "rate_per_min": rate_per_min,
        "eta_minutes": eta_minutes,
        "recent": progress[-20:],
    }


def render_html(status: dict[str, Any]) -> str:
    coverage = status["coverage"]
    rate = status["rate_per_min"]
    eta = status["eta_minutes"]
    rate_text = f"{rate:.1f}/min" if rate is not None else "N/A"
    eta_text = f"{eta:.1f} min" if eta is not None else "N/A"
    endpoint_rows = "\n".join(
        f"<tr><td>{html.escape(row['api'])}</td><td>{row['available']}</td><td>{row['expected']}</td><td>{row['missing']}</td><td>{html.escape(row['first_missing'])}</td><td>{html.escape(row['last_missing'])}</td></tr>"
        for row in status["endpoint_counts"]
    )
    recent_rows = "\n".join(
        f"<tr><td>{html.escape(str(row.get('ts','')))}</td><td>{row.get('done','')}</td><td>{row.get('total','')}</td><td>{html.escape(str(row.get('api_name','')))}</td><td>{html.escape(str(row.get('trade_date','')))}</td><td>{html.escape(str(row.get('status','')))}</td><td>{html.escape(str(row.get('error',''))[:120])}</td></tr>"
        for row in reversed(status["recent"])
    )
    process_text = "<br>".join(html.escape(x) for x in status["processes"]) or "未检测到下载进程"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="10">
  <title>v9 Batch1 下载进度</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #18212f; background: #f6f7f9; }}
    h1 {{ margin: 0 0 12px; font-size: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin: 16px 0; }}
    .card {{ background: #fff; border: 1px solid #d9dee7; border-radius: 8px; padding: 14px; }}
    .label {{ color: #667085; font-size: 12px; margin-bottom: 6px; }}
    .value {{ font-size: 22px; font-weight: 650; }}
    .bar {{ height: 10px; background: #e6e9ef; border-radius: 999px; overflow: hidden; margin-top: 10px; }}
    .bar span {{ display: block; height: 100%; background: #2266cc; width: {coverage['pct'] * 100:.2f}%; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; margin-top: 12px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ color: #475467; background: #f9fafb; font-weight: 600; }}
    .muted {{ color: #667085; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>v9 Batch1 历史日频数据下载进度</h1>
  <div class="muted">自动每 10 秒刷新；只读监控，不修改下载任务。更新时间：{html.escape(status['generated_at'])}</div>
  <div class="grid">
    <div class="card"><div class="label">状态</div><div class="value">{html.escape(status['state'])}</div></div>
    <div class="card"><div class="label">总覆盖</div><div class="value">{coverage['available']} / {coverage['expected']}</div><div class="bar"><span></span></div><div class="muted">{coverage['pct']:.2%}</div></div>
    <div class="card"><div class="label">当前下载速率</div><div class="value">{html.escape(rate_text)}</div></div>
    <div class="card"><div class="label">预计剩余</div><div class="value">{html.escape(eta_text)}</div></div>
  </div>
  <div class="grid">
    <div class="card"><div class="label">缺失文件</div><div class="value">{coverage['missing']}</div></div>
    <div class="card"><div class="label">失败记录</div><div class="value">{status['failures']}</div></div>
    <div class="card"><div class="label">429/限速失败</div><div class="value">{status['rate_limited_failures']}</div></div>
    <div class="card"><div class="label">交易日数量</div><div class="value">{status['open_dates']}</div></div>
  </div>
  <h2>接口覆盖率</h2>
  <table><thead><tr><th>接口</th><th>已落地</th><th>应有</th><th>缺失</th><th>首个缺失</th><th>最后缺失</th></tr></thead><tbody>{endpoint_rows}</tbody></table>
  <h2>最近进度</h2>
  <table><thead><tr><th>时间</th><th>done</th><th>total</th><th>接口</th><th>日期</th><th>状态</th><th>错误</th></tr></thead><tbody>{recent_rows}</tbody></table>
  <h2>下载进程</h2>
  <div class="card mono">{process_text}</div>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.startswith("/api/status"):
            payload = json.dumps(build_status(), ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        body = render_html(build_status()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only v9 Batch1 daily data download dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"v9 Batch1 dashboard: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
