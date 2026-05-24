#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "tushare" / "v8_forward_shadow"
DEFAULT_LOG_DIR = ROOT / "logs" / "forward_shadow_dashboard"
BEIJING_TZ = timezone(timedelta(hours=8))
MIN_TRADE_DATE = "20260521"


def bj_now() -> datetime:
    return datetime.now(BEIJING_TZ)


def today_ymd() -> str:
    return bj_now().strftime("%Y%m%d")


def ymd_to_date_input(value: str) -> str:
    v = normalize_trade_date(value)
    return f"{v[:4]}-{v[4:6]}-{v[6:8]}"


def normalize_trade_date(value: str | None) -> str:
    raw = (value or "").strip()
    if not raw:
        return today_ymd()
    raw = raw.replace("-", "")
    if len(raw) != 8 or not raw.isdigit():
        raise ValueError("trade_date must be YYYYMMDD or YYYY-MM-DD")
    datetime.strptime(raw, "%Y%m%d")
    return raw


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": repr(exc)}


def read_csv_rows(path: Path, max_rows: int = 200) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except Exception:
        return []
    if max_rows and len(rows) > max_rows:
        return rows[-max_rows:]
    return rows


def write_json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def write_text_response(handler: BaseHTTPRequestHandler, status: int, body: str, content_type: str = "text/html; charset=utf-8") -> None:
    data = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def env_health(send_notifications: bool) -> dict[str, Any]:
    missing = []
    if not os.environ.get("TUSHARE_TOKEN", "").strip():
        missing.append("TUSHARE_TOKEN")
    if send_notifications and not os.environ.get("WXPUSHER_APP_TOKEN", "").strip():
        missing.append("WXPUSHER_APP_TOKEN")
    return {
        "tushare_token_present": bool(os.environ.get("TUSHARE_TOKEN", "").strip()),
        "wxpusher_token_present": bool(os.environ.get("WXPUSHER_APP_TOKEN", "").strip()),
        "missing_required": missing,
    }


class DashboardState:
    def __init__(self, output_root: Path, topic_id: int) -> None:
        self.output_root = output_root
        self.topic_id = topic_id
        self.lock = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.current_job_id: str | None = None

    def current_job(self) -> dict[str, Any] | None:
        with self.lock:
            return self.jobs.get(self.current_job_id or "")

    def has_running_job(self) -> bool:
        job = self.current_job()
        if not job:
            return False
        proc = job.get("process")
        return bool(proc and proc.poll() is None)


def command_for_job(payload: dict[str, Any], default_output_root: Path, default_topic_id: int) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    trade_date = normalize_trade_date(payload.get("trade_date"))
    if trade_date < MIN_TRADE_DATE:
        raise ValueError(f"trade_date must be >= {MIN_TRADE_DATE}")
    mode = str(payload.get("mode") or "live_time")
    if mode not in {"fast_replay", "live_time"}:
        raise ValueError("mode must be fast_replay or live_time")
    send_notifications = bool(payload.get("send_notifications", True))
    output_root = Path(str(payload.get("output_root") or default_output_root))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    requests_per_minute = int(payload.get("requests_per_minute") or 120)
    batch_size = int(payload.get("batch_size") or 160)
    lookback_days = int(payload.get("lookback_trading_days") or 90)
    topic_id = int(payload.get("topic_id") or default_topic_id)
    skip_moneyflow = bool(payload.get("skip_moneyflow", False))

    health = env_health(send_notifications)
    if health["missing_required"]:
        raise ValueError("missing environment variables: " + ", ".join(health["missing_required"]))

    cmd = [
        sys.executable,
        "research/v8_research/forward_shadow_live_orchestrator.py",
        "--trade-date",
        trade_date,
        "--output-root",
        str(output_root),
        "--requests-per-minute",
        str(requests_per_minute),
        "--batch-size",
        str(batch_size),
        "--lookback-trading-days",
        str(lookback_days),
        "--topic-id",
        str(topic_id),
    ]
    if send_notifications:
        cmd.append("--send-notifications")
    if mode == "fast_replay":
        cmd.append("--no-wait")
    if skip_moneyflow:
        cmd.append("--skip-moneyflow")

    env = os.environ.copy()
    env.setdefault("TUSHARE_PROXY_URL", "http://tsy.xiaodefa.cn")

    warnings = []
    if mode == "live_time" and trade_date != today_ymd():
        warnings.append("真实时间模式建议使用当天北京时间交易日；历史日期会按当前时钟执行已过节点。")
    meta = {
        "trade_date": trade_date,
        "mode": mode,
        "send_notifications": send_notifications,
        "output_root": str(output_root),
        "requests_per_minute": requests_per_minute,
        "batch_size": batch_size,
        "lookback_trading_days": lookback_days,
        "topic_id": topic_id,
        "skip_moneyflow": skip_moneyflow,
        "warnings": warnings,
    }
    return cmd, env, meta


def consume_process(job: dict[str, Any], state: DashboardState) -> None:
    proc: subprocess.Popen[str] = job["process"]
    log_path = Path(job["log_path"])
    status = "completed"
    try:
        with log_path.open("a", encoding="utf-8") as log:
            assert proc.stdout is not None
            for line in proc.stdout:
                clean = line.rstrip("\n")
                log.write(clean + "\n")
                log.flush()
                with state.lock:
                    job["log_tail"].append(clean)
            return_code = proc.wait()
        with state.lock:
            if return_code != 0:
                status = "stopped" if job.get("status") == "stopping" else "failed"
            job["status"] = status
            job["return_code"] = return_code
            job["finished_at_beijing"] = bj_now().isoformat(timespec="seconds")
    except Exception as exc:
        with state.lock:
            job["status"] = "failed"
            job["error"] = repr(exc)
            job["finished_at_beijing"] = bj_now().isoformat(timespec="seconds")


def start_job(state: DashboardState, payload: dict[str, Any]) -> dict[str, Any]:
    with state.lock:
        if state.has_running_job():
            raise RuntimeError("a dashboard job is already running")
    cmd, env, meta = command_for_job(payload, state.output_root, state.topic_id)
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:12]
    log_path = DEFAULT_LOG_DIR / f"{meta['trade_date']}_{job_id}.log"
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    job = {
        "job_id": job_id,
        "status": "running",
        "process": proc,
        "pid": proc.pid,
        "command": cmd,
        "safe_command": cmd,
        "meta": meta,
        "started_at_beijing": bj_now().isoformat(timespec="seconds"),
        "finished_at_beijing": None,
        "return_code": None,
        "log_path": str(log_path),
        "log_tail": deque(maxlen=300),
    }
    with state.lock:
        state.jobs[job_id] = job
        state.current_job_id = job_id
    thread = threading.Thread(target=consume_process, args=(job, state), daemon=True)
    thread.start()
    return job


def stop_job(state: DashboardState) -> dict[str, Any]:
    with state.lock:
        job = state.current_job()
        if not job:
            return {"stopped": False, "message": "no current job"}
        proc = job.get("process")
        if not proc or proc.poll() is not None:
            return {"stopped": False, "message": "current job is not running"}
        job["status"] = "stopping"
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        return {"stopped": True, "message": "terminate signal sent"}
    except Exception as exc:
        with state.lock:
            job["status"] = "failed"
            job["error"] = repr(exc)
        return {"stopped": False, "message": repr(exc)}


def numeric(value: str) -> float | None:
    try:
        if value in {"", "nan", "NaN", "None"}:
            return None
        return float(value)
    except Exception:
        return None


def summarize_by_strategy(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        strategy = row.get("strategy_id") or row.get("line_id") or ""
        if not strategy:
            continue
        item = grouped.setdefault(strategy, {"strategy_id": strategy, "rows": 0, "selected": 0, "score_sum": 0.0, "score_count": 0})
        item["rows"] += 1
        selected = str(row.get("final_selected_flag", "1")).lower() in {"1", "true", "yes"}
        if selected:
            item["selected"] += 1
        score = numeric(str(row.get("score", "")))
        if score is not None:
            item["score_sum"] += score
            item["score_count"] += 1
    out = []
    for item in grouped.values():
        score_count = item.pop("score_count")
        score_sum = item.pop("score_sum")
        item["avg_score"] = round(score_sum / score_count, 6) if score_count else None
        out.append(item)
    return sorted(out, key=lambda x: x["strategy_id"])


def dashboard_artifacts(output_root: Path, trade_date: str) -> dict[str, Any]:
    live_dir = output_root / "live_runner"
    status_path = live_dir / f"{trade_date}_live_runner_status.json"
    steps_path = live_dir / f"{trade_date}_live_runner_steps.csv"
    candidate_path = output_root / "forward_shadow_candidate_status.csv"
    signals_path = output_root / "daily_signals" / f"{trade_date}_signals.csv"
    entry_path = output_root / "daily_entry_prices" / f"{trade_date}_entry_prices.csv"

    status_doc = read_json(status_path)
    steps = read_csv_rows(steps_path, max_rows=120)
    candidates = [r for r in read_csv_rows(candidate_path, max_rows=10000) if str(r.get("trade_date", "")) == trade_date]
    signals = read_csv_rows(signals_path, max_rows=1000)
    entries = read_csv_rows(entry_path, max_rows=1000)

    entry_counts: dict[str, int] = {}
    for row in entries:
        key = row.get("paper_entry_status") or "unknown"
        entry_counts[key] = entry_counts.get(key, 0) + 1

    files = {
        "status": str(status_path),
        "steps": str(steps_path),
        "candidate_status": str(candidate_path),
        "daily_signals": str(signals_path),
        "entry_prices": str(entry_path),
    }
    exists = {name: Path(path).exists() for name, path in files.items()}
    return {
        "live_status": status_doc,
        "steps": steps,
        "candidate_status": candidates,
        "signal_summary": summarize_by_strategy(signals),
        "entry_counts": entry_counts,
        "files": files,
        "file_exists": exists,
    }


def public_job(job: dict[str, Any] | None) -> dict[str, Any] | None:
    if not job:
        return None
    proc = job.get("process")
    running = bool(proc and proc.poll() is None)
    return {
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "running": running,
        "pid": job.get("pid"),
        "meta": job.get("meta"),
        "started_at_beijing": job.get("started_at_beijing"),
        "finished_at_beijing": job.get("finished_at_beijing"),
        "return_code": job.get("return_code"),
        "log_path": job.get("log_path"),
        "log_tail": list(job.get("log_tail") or [])[-120:],
        "command": job.get("safe_command"),
        "error": job.get("error"),
    }


def index_html(default_output_root: Path, topic_id: int) -> str:
    default_date = ymd_to_date_input(today_ymd())
    min_date = ymd_to_date_input(MIN_TRADE_DATE)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Forward Shadow 控制台</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --line: #d8dee9;
      --text: #202733;
      --muted: #667085;
      --accent: #1769aa;
      --bad: #b42318;
      --ok: #067647;
      --warn: #b54708;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }}
    header {{
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{ margin: 0; font-size: 20px; }}
    main {{ padding: 20px 24px 32px; max-width: 1440px; margin: 0 auto; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(170px, 1fr)); gap: 12px; align-items: end; }}
    label {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 5px; }}
    input, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
      background: white;
    }}
    .checks {{ display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }}
    .checks label {{ display: inline-flex; align-items: center; gap: 6px; margin: 0; font-size: 13px; color: var(--text); }}
    input[type="checkbox"] {{ width: auto; }}
    button {{
      border: 0;
      border-radius: 6px;
      padding: 10px 14px;
      font: inherit;
      cursor: pointer;
      background: var(--accent);
      color: white;
      min-width: 96px;
    }}
    button.secondary {{ background: #344054; }}
    button.danger {{ background: var(--bad); }}
    button:disabled {{ opacity: .5; cursor: not-allowed; }}
    .bar {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 9px;
      background: #eef4ff;
      color: #194185;
      font-size: 12px;
      border: 1px solid #c7d7fe;
    }}
    .pill.ok {{ background: #ecfdf3; color: var(--ok); border-color: #abefc6; }}
    .pill.bad {{ background: #fef3f2; color: var(--bad); border-color: #fecdca; }}
    .pill.warn {{ background: #fffaeb; color: var(--warn); border-color: #fedf89; }}
    .metrics {{ display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 10px; }}
    .metric {{ border: 1px solid var(--line); border-radius: 6px; padding: 10px; background: #fcfcfd; }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; }}
    .metric strong {{ font-size: 18px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 7px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; background: #f8fafc; }}
    pre {{
      white-space: pre-wrap;
      overflow: auto;
      max-height: 260px;
      padding: 12px;
      background: #101828;
      color: #e4e7ec;
      border-radius: 6px;
      font-size: 12px;
    }}
    .muted {{ color: var(--muted); }}
    .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    @media (max-width: 920px) {{
      .grid, .metrics, .two {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Forward Shadow Paper Tracking 控制台</h1>
    <div class="muted">本地 paper-only 页面：启动任务、查看进度和输出文件；不下单，不重训，不改 v7_locked。</div>
  </header>
  <main>
    <section>
      <div class="grid">
        <div>
          <label for="tradeDate">执行日期</label>
          <input id="tradeDate" type="date" min="{min_date}" value="{default_date}">
        </div>
        <div>
          <label for="mode">执行模式</label>
          <select id="mode">
            <option value="live_time" selected>模式2：真实时间点模式</option>
            <option value="fast_replay">模式1：模拟时间点快跑</option>
          </select>
        </div>
        <div>
          <label for="rpm">请求速率 / 分钟</label>
          <input id="rpm" type="number" min="1" max="500" value="120">
        </div>
        <div>
          <label for="batchSize">批量股票数</label>
          <input id="batchSize" type="number" min="1" max="800" value="160">
        </div>
        <div>
          <label for="outputRoot">输出目录</label>
          <input id="outputRoot" value="{html.escape(rel(default_output_root))}">
        </div>
        <div>
          <label for="topicId">WxPusher topic</label>
          <input id="topicId" type="number" value="{topic_id}">
        </div>
        <div class="checks">
          <label><input id="sendNotifications" type="checkbox" checked> 推送 WxPusher</label>
          <label><input id="skipMoneyflow" type="checkbox"> 跳过 moneyflow</label>
        </div>
        <div class="bar">
          <button id="startBtn">启动</button>
          <button id="stopBtn" class="danger">停止</button>
          <button id="refreshBtn" class="secondary">刷新</button>
        </div>
      </div>
      <p class="muted">模式1仍走真实接口和真实数据处理，只是不等待 09:25/14:50 等墙上时间；模式2按北京时间等待，已过节点会立即补执行。</p>
      <div id="message" class="muted"></div>
    </section>

    <section>
      <div class="bar">
        <span id="jobStatus" class="pill">未启动</span>
        <span id="modePill" class="pill">mode: -</span>
        <span id="datePill" class="pill">date: -</span>
        <span id="pidPill" class="pill">pid: -</span>
      </div>
      <div class="metrics" style="margin-top:12px">
        <div class="metric"><span>当前节点</span><strong id="currentStep">-</strong></div>
        <div class="metric"><span>节点状态</span><strong id="runnerState">-</strong></div>
        <div class="metric"><span>已完成</span><strong id="completedSteps">0</strong></div>
        <div class="metric"><span>总节点</span><strong id="totalSteps">-</strong></div>
        <div class="metric"><span>更新时间</span><strong id="updatedAt">-</strong></div>
      </div>
    </section>

    <section class="two">
      <div>
        <h2>四线路状态</h2>
        <div id="candidateTable"></div>
      </div>
      <div>
        <h2>信号与入场统计</h2>
        <div id="signalSummary"></div>
        <div id="entryCounts" style="margin-top:10px"></div>
      </div>
    </section>

    <section>
      <h2>步骤进度</h2>
      <div id="stepsTable"></div>
    </section>

    <section>
      <h2>输出文件</h2>
      <div id="filesTable"></div>
    </section>

    <section>
      <h2>运行日志</h2>
      <pre id="logTail"></pre>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);

    function dateToYmd(v) {{ return (v || '').replaceAll('-', ''); }}
    function setMessage(text, bad=false) {{
      $('message').textContent = text || '';
      $('message').style.color = bad ? '#b42318' : '#667085';
    }}
    function esc(v) {{
      return String(v ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
    }}
    function table(rows, cols) {{
      if (!rows || !rows.length) return '<div class="muted">暂无数据</div>';
      return '<table><thead><tr>' + cols.map(c => '<th>'+esc(c.label || c)+'</th>').join('') + '</tr></thead><tbody>' +
        rows.map(r => '<tr>' + cols.map(c => '<td>'+esc(r[c.key || c])+'</td>').join('') + '</tr>').join('') +
        '</tbody></table>';
    }}
    async function startJob() {{
      const payload = {{
        trade_date: dateToYmd($('tradeDate').value),
        mode: $('mode').value,
        requests_per_minute: Number($('rpm').value || 120),
        batch_size: Number($('batchSize').value || 160),
        output_root: $('outputRoot').value,
        topic_id: Number($('topicId').value || {topic_id}),
        send_notifications: $('sendNotifications').checked,
        skip_moneyflow: $('skipMoneyflow').checked
      }};
      const resp = await fetch('/api/start', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(payload)}});
      const data = await resp.json();
      if (!resp.ok) {{
        setMessage(data.error || '启动失败', true);
        return;
      }}
      setMessage('任务已启动：' + data.job.job_id);
      refresh();
    }}
    async function stopJob() {{
      const resp = await fetch('/api/stop', {{method:'POST'}});
      const data = await resp.json();
      setMessage(data.message || '停止请求已发送', !data.stopped);
      refresh();
    }}
    function render(data) {{
      const job = data.job;
      const artifacts = data.artifacts || {{}};
      const live = artifacts.live_status || {{}};
      const jobStatus = job ? job.status : 'idle';
      $('jobStatus').textContent = jobStatus;
      $('jobStatus').className = 'pill ' + (jobStatus === 'completed' ? 'ok' : (jobStatus === 'failed' ? 'bad' : (jobStatus === 'running' || jobStatus === 'stopping' ? 'warn' : '')));
      $('modePill').textContent = 'mode: ' + (job?.meta?.mode || '-');
      $('datePill').textContent = 'date: ' + (job?.meta?.trade_date || data.default_trade_date || '-');
      $('pidPill').textContent = 'pid: ' + (job?.pid || '-');
      $('currentStep').textContent = live.current_step_id || '-';
      $('runnerState').textContent = live.status || '-';
      $('completedSteps').textContent = live.completed_steps ?? 0;
      $('totalSteps').textContent = live.total_steps ?? '-';
      $('updatedAt').textContent = live.updated_at_beijing || '-';
      $('candidateTable').innerHTML = table(artifacts.candidate_status || [], [
        {{key:'strategy_id', label:'line_id'}}, {{key:'tail_down_flag', label:'tail_down'}}, {{key:'selected_count', label:'selected'}}, {{key:'no_trade_reason', label:'no_trade'}}
      ]);
      $('signalSummary').innerHTML = table(artifacts.signal_summary || [], [
        {{key:'strategy_id', label:'line_id'}}, {{key:'rows', label:'rows'}}, {{key:'selected', label:'selected'}}, {{key:'avg_score', label:'avg_score'}}
      ]);
      const entryRows = Object.entries(artifacts.entry_counts || {{}}).map(([status, count]) => ({{status, count}}));
      $('entryCounts').innerHTML = table(entryRows, ['status','count']);
      $('stepsTable').innerHTML = table(artifacts.steps || [], [
        {{key:'step_id', label:'step'}}, {{key:'scheduled_time', label:'scheduled'}}, {{key:'status', label:'status'}}, {{key:'duration_seconds', label:'seconds'}}, {{key:'return_code', label:'rc'}}
      ]);
      const files = Object.entries(artifacts.files || {{}}).map(([name, path]) => ({{name, path, exists: artifacts.file_exists?.[name] ? 'yes' : 'no'}}));
      $('filesTable').innerHTML = table(files, ['name','exists','path']);
      $('logTail').textContent = (job?.log_tail || []).join('\\n');
      $('startBtn').disabled = !!job?.running;
      $('stopBtn').disabled = !job?.running;
    }}
    async function refresh() {{
      try {{
        const resp = await fetch('/api/status');
        render(await resp.json());
      }} catch (e) {{
        setMessage(String(e), true);
      }}
    }}
    $('startBtn').addEventListener('click', startJob);
    $('stopBtn').addEventListener('click', stopJob);
    $('refreshBtn').addEventListener('click', refresh);
    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>"""


def make_handler(state: DashboardState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ForwardShadowDashboard/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

        def read_body_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                write_text_response(self, HTTPStatus.OK, index_html(state.output_root, state.topic_id))
                return
            if parsed.path == "/api/status":
                params = parse_qs(parsed.query)
                with state.lock:
                    job = state.current_job()
                    public = public_job(job)
                trade_date = None
                output_root = state.output_root
                if public and public.get("meta"):
                    trade_date = public["meta"].get("trade_date")
                    output_root = Path(public["meta"].get("output_root") or state.output_root)
                if params.get("trade_date"):
                    trade_date = normalize_trade_date(params["trade_date"][0])
                trade_date = trade_date or today_ymd()
                payload = {
                    "default_trade_date": today_ymd(),
                    "min_trade_date": MIN_TRADE_DATE,
                    "beijing_now": bj_now().isoformat(timespec="seconds"),
                    "git": {
                        "branch": git_value(["branch", "--show-current"]),
                        "head": git_value(["rev-parse", "HEAD"]),
                        "v7_locked": git_value(["rev-parse", "v7_locked"]),
                    },
                    "env": env_health(send_notifications=True),
                    "job": public,
                    "artifacts": dashboard_artifacts(output_root, trade_date),
                }
                write_json_response(self, HTTPStatus.OK, payload)
                return
            write_text_response(self, HTTPStatus.NOT_FOUND, "not found", "text/plain; charset=utf-8")

        def do_POST(self) -> None:
            if self.path == "/api/start":
                try:
                    job = start_job(state, self.read_body_json())
                    public = public_job(job)
                    write_json_response(self, HTTPStatus.OK, {"job": public})
                except Exception as exc:
                    write_json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            if self.path == "/api/stop":
                write_json_response(self, HTTPStatus.OK, stop_job(state))
                return
            write_json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Local visual dashboard for forward shadow paper tracking.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--topic-id", type=int, default=44635)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    state = DashboardState(output_root, int(args.topic_id))
    server = ThreadingHTTPServer((args.host, int(args.port)), make_handler(state))
    print(f"Forward Shadow dashboard: http://{args.host}:{args.port}/")
    print("paper-only; no broker API; no live order action.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down dashboard")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
