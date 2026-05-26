#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import signal
import shutil
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
STALE_RUNNING_SECONDS = 30 * 60
DEFAULT_STEP_PLAN: list[tuple[str, str]] = [
    ("preflight_downstream", "startup"),
    ("prepare_baseline", "07:30:00"),
    ("auction_guard_0925", "09:25:30"),
    ("fetch_exit_0935_bar", "09:35:05"),
    ("exit_check_0935", "09:35:50"),
    ("fetch_exit_0940_bar", "09:40:05"),
    ("exit_exec_0940", "09:40:50"),
    ("fetch_exit_0945_bar", "09:45:05"),
    ("exit_check_0945", "09:45:50"),
    ("fetch_exit_0950_bar", "09:50:05"),
    ("exit_exec_0950", "09:50:50"),
    ("fetch_exit_1000_bar", "10:00:05"),
    ("exit_check_1000", "10:00:50"),
    ("fetch_exit_1005_bar", "10:05:05"),
    ("exit_exec_1005", "10:05:50"),
    ("fetch_exit_1025_bar", "10:25:05"),
    ("exit_prealert_1025", "10:25:50"),
    ("fetch_exit_1030_bar", "10:30:05"),
    ("exit_default_1030", "10:30:50"),
    ("auction_guard_1031_refresh", "10:31:30"),
    ("auction_guard_1300_refresh", "13:00:30"),
    ("fetch_tail_until_1430", "14:30:00"),
    ("fetch_tail_1435_bar", "14:35:05"),
    ("fetch_tail_1440_bar", "14:40:05"),
    ("fetch_tail_1445_bar", "14:45:05"),
    ("auction_guard_1445_refresh", "14:45:20"),
    ("fetch_tail_1450_bar", "14:50:05"),
    ("auction_guard_1450_required", "14:50:20"),
    ("build_forward_features", "14:50:40"),
    ("build_score_matrix", "14:51:20"),
    ("freeze_signals", "14:51:50"),
    ("fetch_tail_1455_bar", "14:55:05"),
    ("record_entry_1455_vwap", "14:55:50"),
    ("fetch_tail_1500_bar", "15:00:05"),
]
OPTIONAL_PRIOR_STEP_PLAN: list[tuple[str, str]] = [
    ("seed_prior_artifacts", "07:35:00"),
    ("prior_settlement_skipped", "07:36:00"),
    ("prior_reconstruction_skipped", "07:35:00"),
    ("prior_auction_guard", "07:40:00"),
    ("prior_fetch_until_1430", "07:41:00"),
    ("prior_fetch_1435", "07:42:00"),
    ("prior_fetch_1440", "07:42:00"),
    ("prior_fetch_1445", "07:42:00"),
    ("prior_fetch_1450", "07:42:00"),
    ("prior_fetch_1455", "07:42:00"),
    ("prior_build_features", "07:45:00"),
    ("prior_build_score", "07:46:00"),
    ("prior_freeze_signals", "07:47:00"),
    ("prior_record_entry", "07:48:00"),
]


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


def resolve_output_root(value: str | None, default_output_root: Path) -> Path:
    output_root = Path(str(value or default_output_root))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    return output_root


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


def parse_bj_datetime(value: str | None) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(BEIJING_TZ)
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=BEIJING_TZ)
        return dt.astimezone(BEIJING_TZ)
    except Exception:
        return None


def annotate_stale_live_status(status_doc: dict[str, Any]) -> dict[str, Any]:
    if not status_doc or str(status_doc.get("status") or "") != "running":
        return status_doc
    updated = parse_bj_datetime(str(status_doc.get("updated_at_beijing") or ""))
    if not updated:
        return status_doc
    stale_seconds = max(0.0, (bj_now() - updated).total_seconds())
    if stale_seconds < STALE_RUNNING_SECONDS:
        return status_doc
    out = dict(status_doc)
    out["original_status"] = "running"
    out["status"] = "stale_running"
    out["stale_seconds"] = round(stale_seconds, 1)
    out["stale_threshold_seconds"] = STALE_RUNNING_SECONDS
    out["stale_note"] = (
        "状态文件仍标记 running，但更新时间已超过阈值；这通常是旧任务异常退出后遗留的状态，"
        "不代表当前仍在执行。请切换到“最近一次运行结果”查看最新 run_id 输出。"
    )
    return out


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


def env_health(send_notifications: bool, env: dict[str, str] | None = None) -> dict[str, Any]:
    env_map = env or os.environ
    missing = []
    if not env_map.get("TUSHARE_TOKEN", "").strip():
        missing.append("TUSHARE_TOKEN")
    if send_notifications and not env_map.get("WXPUSHER_APP_TOKEN", "").strip():
        missing.append("WXPUSHER_APP_TOKEN")
    return {
        "tushare_token_present": bool(env_map.get("TUSHARE_TOKEN", "").strip()),
        "wxpusher_token_present": bool(env_map.get("WXPUSHER_APP_TOKEN", "").strip()),
        "missing_required": missing,
    }


class DashboardState:
    def __init__(self, output_root: Path, topic_id: int) -> None:
        self.output_root = output_root
        self.topic_id = topic_id
        self.lock = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.current_job_id: str | None = None
        self.dependency_jobs: dict[str, dict[str, Any]] = {}
        self.current_dependency_job_id: str | None = None

    def current_job(self) -> dict[str, Any] | None:
        with self.lock:
            return self.jobs.get(self.current_job_id or "")

    def current_dependency_job(self) -> dict[str, Any] | None:
        with self.lock:
            return self.dependency_jobs.get(self.current_dependency_job_id or "")

    def has_running_job(self) -> bool:
        job = self.current_job()
        if not job:
            return False
        proc = job.get("process")
        return bool(proc and proc.poll() is None)

    def has_running_dependency_job(self) -> bool:
        job = self.current_dependency_job()
        if not job:
            return False
        proc = job.get("process")
        return bool(proc and proc.poll() is None)


def safe_label(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    out = "".join(ch if ch in allowed else "_" for ch in value.strip())
    return out[:80]


def command_for_job(payload: dict[str, Any], default_output_root: Path, default_topic_id: int) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    trade_date = normalize_trade_date(payload.get("trade_date"))
    if trade_date < MIN_TRADE_DATE:
        raise ValueError(f"trade_date must be >= {MIN_TRADE_DATE}")
    mode = str(payload.get("mode") or "live_time")
    if mode not in {"fast_replay", "live_time"}:
        raise ValueError("mode must be fast_replay or live_time")
    send_notifications = bool(payload.get("send_notifications", True))
    output_root = resolve_output_root(str(payload.get("output_root") or ""), default_output_root)
    requests_per_minute = int(payload.get("requests_per_minute") or 120)
    batch_size = int(payload.get("batch_size") or 160)
    lookback_days = int(payload.get("lookback_trading_days") or 90)
    topic_id = int(payload.get("topic_id") or payload.get("group_id") or default_topic_id)
    notification_policy = str(payload.get("notification_policy") or "key_events")
    if notification_policy not in {"all_steps", "key_events", "trade_only", "failures_only", "none"}:
        raise ValueError("notification_policy must be all_steps, key_events, trade_only, failures_only, or none")
    skip_moneyflow = bool(payload.get("skip_moneyflow", False))
    preserve_snapshot = bool(payload.get("preserve_run_snapshot", False))
    use_run_id_output_dir = bool(payload.get("use_run_id_output_dir", False))
    force_refresh_minutes = bool(payload.get("force_refresh_minutes", False))
    prior_input_policy = str(payload.get("prior_input_policy") or "reuse_only")
    if prior_input_policy not in {"reuse_or_rebuild", "reuse_only", "force_rebuild"}:
        raise ValueError("prior_input_policy must be reuse_or_rebuild, reuse_only, or force_rebuild")
    if "reset_settlement_outputs" in payload:
        reset_settlement_outputs = bool(payload.get("reset_settlement_outputs"))
    else:
        reset_settlement_outputs = mode == "fast_replay"
    job_id = safe_label(str(payload.get("_job_id") or uuid.uuid4().hex[:12]))
    requested_run_id = safe_label(str(payload.get("run_id") or ""))
    run_id = requested_run_id or safe_label(f"{trade_date}_{bj_now().strftime('%H%M%S')}_{job_id}")
    base_output_root = output_root
    if use_run_id_output_dir:
        output_root = base_output_root / "runs" / run_id

    env = os.environ.copy()
    env.setdefault("TUSHARE_PROXY_URL", "https://tt.xiaodefa.cn")
    page_tushare_token = str(payload.get("tushare_token") or "").strip()
    page_wxpusher_token = str(payload.get("wxpusher_app_token") or "").strip()
    if page_tushare_token:
        env["TUSHARE_TOKEN"] = page_tushare_token
    if page_wxpusher_token:
        env["WXPUSHER_APP_TOKEN"] = page_wxpusher_token

    effective_notifications = send_notifications and notification_policy != "none"
    health = env_health(effective_notifications, env)
    if health["missing_required"]:
        raise ValueError("missing credentials: " + ", ".join(health["missing_required"]) + ". Set env vars or enter them on the page.")

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
        "--notification-policy",
        notification_policy,
        "--prior-input-policy",
        prior_input_policy,
        "--prior-seed-root",
        str(base_output_root),
    ]
    if effective_notifications:
        cmd.append("--send-notifications")
    if mode == "fast_replay":
        cmd.append("--no-wait")
    if skip_moneyflow:
        cmd.append("--skip-moneyflow")
    if force_refresh_minutes:
        cmd.append("--force-refresh-minutes")
    if reset_settlement_outputs:
        cmd.append("--reset-settlement-outputs")

    warnings = []
    if mode == "live_time" and trade_date != today_ymd():
        warnings.append("真实时间模式建议使用当天北京时间交易日；历史日期会按当前时钟执行已过节点。")
    meta = {
        "trade_date": trade_date,
        "mode": mode,
        "send_notifications": send_notifications,
        "effective_notifications": effective_notifications,
        "output_root": str(output_root),
        "base_output_root": str(base_output_root),
        "run_id": run_id,
        "use_run_id_output_dir": use_run_id_output_dir,
        "preserve_run_snapshot": preserve_snapshot,
        "force_refresh_minutes": force_refresh_minutes,
        "reset_settlement_outputs": reset_settlement_outputs,
        "requests_per_minute": requests_per_minute,
        "batch_size": batch_size,
        "lookback_trading_days": lookback_days,
        "topic_id": topic_id,
        "notification_policy": notification_policy,
        "skip_moneyflow": skip_moneyflow,
        "prior_input_policy": prior_input_policy,
        "credential_source": {
            "tushare_token": "page_input" if page_tushare_token else "environment",
            "wxpusher_app_token": "page_input" if page_wxpusher_token else "environment",
        },
        "warnings": warnings,
    }
    return cmd, env, meta


def command_for_dependency_probe(payload: dict[str, Any], default_output_root: Path) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    trade_date = normalize_trade_date(payload.get("trade_date"))
    output_root = resolve_output_root(str(payload.get("output_root") or ""), default_output_root)
    timeout = int(payload.get("timeout") or 8)
    env = os.environ.copy()
    env.setdefault("TUSHARE_PROXY_URL", "https://tt.xiaodefa.cn")
    page_tushare_token = str(payload.get("tushare_token") or "").strip()
    page_wxpusher_token = str(payload.get("wxpusher_app_token") or "").strip()
    if page_tushare_token:
        env["TUSHARE_TOKEN"] = page_tushare_token
    if page_wxpusher_token:
        env["WXPUSHER_APP_TOKEN"] = page_wxpusher_token
    health = env_health(send_notifications=False, env=env)
    if health["missing_required"]:
        raise ValueError("missing credentials: " + ", ".join(health["missing_required"]) + ". Set env vars or enter them on the page.")
    cmd = [
        sys.executable,
        "research/v8_research/forward_shadow_preflight.py",
        "--trade-date",
        trade_date,
        "--output-root",
        str(output_root),
        "--rank-file",
        str(ROOT / "data_tushare" / "manifests" / "liquid_top3000_20251120_20260213.csv"),
        "--minute-dir",
        str(ROOT / "data_tushare" / "raw" / "stk_mins" / "freq=5min"),
        "--timeout",
        str(timeout),
    ]
    meta = {
        "trade_date": trade_date,
        "output_root": str(output_root),
        "mode": "dependency_probe",
        "timeout": timeout,
        "credential_source": {
            "tushare_token": "page_input" if page_tushare_token else "environment",
            "wxpusher_app_token": "page_input" if page_wxpusher_token else "environment",
        },
    }
    return cmd, env, meta


def copy_if_exists(src: Path, dst_root: Path, base: Path) -> None:
    if not src.exists():
        return
    rel_path = src.relative_to(base) if src.is_relative_to(base) else Path(src.name)
    dst = dst_root / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def preserve_run_snapshot(job: dict[str, Any]) -> Path:
    meta = job.get("meta", {})
    trade_date = str(meta.get("trade_date"))
    output_root = Path(str(meta.get("output_root")))
    base_output_root = Path(str(meta.get("base_output_root") or output_root))
    run_id = safe_label(str(meta.get("run_id") or job.get("job_id") or trade_date))
    snapshot_root = base_output_root / "run_snapshots" / run_id
    snapshot_root.mkdir(parents=True, exist_ok=True)
    files = [
        output_root / "live_runner" / f"{trade_date}_live_runner_status.json",
        output_root / "live_runner" / f"{trade_date}_live_runner_steps.csv",
        output_root / "test_features" / f"{trade_date}_forward_features.csv",
        output_root / "test_features" / f"{trade_date}_forward_features_meta.json",
        output_root / "score_matrices" / f"{trade_date}_score_matrix.csv",
        output_root / "score_matrices" / f"{trade_date}_score_matrix_meta.json",
        output_root / "daily_signals" / f"{trade_date}_signals.csv",
        output_root / "daily_entry_prices" / f"{trade_date}_entry_prices.csv",
        output_root / "daily_ledgers" / f"{trade_date}_ledgers.csv",
        output_root / "daily_execution_quality" / f"{trade_date}_execution_quality.csv",
        output_root / "forward_shadow_candidate_status.csv",
        output_root / "forward_shadow_daily_summary.csv",
        output_root / "forward_shadow_trade_details.csv",
        output_root / "forward_shadow_execution_quality.csv",
        Path(str(job.get("log_path") or "")),
    ]
    for path in sorted((output_root / "data_guards").glob(f"{trade_date}_*.json")):
        files.append(path)
    for src in files:
        if src and str(src) != ".":
            copy_if_exists(src, snapshot_root, output_root)
    manifest = {
        "job_id": job.get("job_id"),
        "trade_date": trade_date,
        "run_id": run_id,
        "created_at_beijing": bj_now().isoformat(timespec="seconds"),
        "status": job.get("status"),
        "return_code": job.get("return_code"),
        "meta": meta,
        "log_path": job.get("log_path"),
        "note": "Snapshot contains lightweight run outputs only; raw minute parquet files are not copied.",
    }
    (snapshot_root / "snapshot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return snapshot_root


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
                if status == "failed":
                    job["error"] = f"runner exited with code {return_code}"
                    job["last_error_lines"] = list(job.get("log_tail") or [])[-30:]
            if job.get("meta", {}).get("preserve_run_snapshot"):
                try:
                    snapshot_path = preserve_run_snapshot(job)
                    job["snapshot_path"] = str(snapshot_path)
                except Exception as exc:
                    job["snapshot_error"] = repr(exc)
            job["status"] = status
            job["return_code"] = return_code
            job["finished_at_beijing"] = bj_now().isoformat(timespec="seconds")
    except Exception as exc:
        with state.lock:
            job["status"] = "failed"
            job["error"] = repr(exc)
            job["last_error_lines"] = list(job.get("log_tail") or [])[-30:]
            job["finished_at_beijing"] = bj_now().isoformat(timespec="seconds")


def start_job(state: DashboardState, payload: dict[str, Any]) -> dict[str, Any]:
    with state.lock:
        if state.has_running_job():
            raise RuntimeError("a dashboard job is already running")
    job_id = uuid.uuid4().hex[:12]
    payload = dict(payload)
    payload["_job_id"] = job_id
    cmd, env, meta = command_for_job(payload, state.output_root, state.topic_id)
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
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
        "snapshot_path": None,
        "snapshot_error": None,
    }
    with state.lock:
        state.jobs[job_id] = job
        state.current_job_id = job_id
    thread = threading.Thread(target=consume_process, args=(job, state), daemon=True)
    thread.start()
    return job


def start_dependency_probe(state: DashboardState, payload: dict[str, Any]) -> dict[str, Any]:
    with state.lock:
        if state.has_running_dependency_job():
            raise RuntimeError("dependency probe is already running")
    job_id = uuid.uuid4().hex[:12]
    payload = dict(payload)
    payload["_job_id"] = job_id
    cmd, env, meta = command_for_dependency_probe(payload, state.output_root)
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = DEFAULT_LOG_DIR / f"{meta['trade_date']}_dependency_{job_id}.log"
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
        "snapshot_path": None,
        "snapshot_error": None,
    }
    with state.lock:
        state.dependency_jobs[job_id] = job
        state.current_dependency_job_id = job_id
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


def latest_run_context(
    base_output_root: Path,
    public_job_doc: dict[str, Any] | None,
    trade_date: str | None = None,
    include_base: bool = True,
) -> dict[str, Any] | None:
    if public_job_doc and public_job_doc.get("meta"):
        meta = public_job_doc["meta"]
        job_trade_date = str(meta.get("trade_date") or "")
        output_root = resolve_output_root(str(meta.get("output_root") or ""), base_output_root)
        if job_trade_date and (not trade_date or job_trade_date == trade_date):
            return {
                "trade_date": job_trade_date,
                "output_root": output_root,
                "source": "current_dashboard_job",
                "run_id": meta.get("run_id") or public_job_doc.get("job_id") or "",
                "status": public_job_doc.get("status") or "",
            }

    candidates: list[dict[str, Any]] = []
    roots = [base_output_root] if include_base else []
    runs_root = base_output_root / "runs"
    if runs_root.exists():
        roots.extend([p for p in runs_root.iterdir() if p.is_dir()])
    for root in roots:
        live_dir = root / "live_runner"
        if not live_dir.exists():
            continue
        for path in live_dir.glob("*_live_runner_status.json"):
            path_trade_date = path.name.split("_", 1)[0]
            if not (path_trade_date.isdigit() and len(path_trade_date) == 8):
                continue
            if trade_date and path_trade_date != trade_date:
                continue
            candidates.append(
                {
                    "trade_date": path_trade_date,
                    "output_root": root,
                    "source": "latest_status_file",
                    "run_id": root.name if root.parent.name == "runs" else "",
                    "mtime": path.stat().st_mtime,
                    "status_path": str(path),
                }
            )
    if not candidates:
        return None
    return max(candidates, key=lambda x: float(x.get("mtime") or 0.0))


def current_job_context(
    base_output_root: Path,
    public_job_doc: dict[str, Any] | None,
    trade_date: str,
    active_job_id: str,
) -> dict[str, Any] | None:
    if not active_job_id or not public_job_doc or not public_job_doc.get("meta"):
        return None
    if str(public_job_doc.get("job_id") or "") != active_job_id:
        return None
    meta = public_job_doc["meta"]
    job_trade_date = str(meta.get("trade_date") or "")
    if job_trade_date != trade_date:
        return None
    return {
        "trade_date": job_trade_date,
        "output_root": resolve_output_root(str(meta.get("output_root") or ""), base_output_root),
        "source": "active_dashboard_job",
        "run_id": meta.get("run_id") or public_job_doc.get("job_id") or "",
        "status": public_job_doc.get("status") or "",
    }


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


def synthesize_step_progress(
    status_doc: dict[str, Any],
    completed_rows: list[dict[str, str]],
    plan_hint: str = "auto",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    completed_by_step: dict[str, dict[str, str]] = {}
    extra_rows: list[dict[str, str]] = []
    current_step = str(status_doc.get("current_step_id") or "")
    optional_ids = {step_id for step_id, _ in OPTIONAL_PRIOR_STEP_PLAN}
    optional_by_id = {step_id: (step_id, scheduled) for step_id, scheduled in OPTIONAL_PRIOR_STEP_PLAN}
    completed_ids = {row.get("step_id", "") for row in completed_rows}
    prior_seed_plan = [optional_by_id["seed_prior_artifacts"]]
    prior_rebuild_plan = [
        item
        for item in OPTIONAL_PRIOR_STEP_PLAN
        if item[0] not in {"seed_prior_artifacts", "prior_reconstruction_skipped"}
    ]
    use_prior_plan = bool(optional_ids.intersection(completed_ids) or current_step in optional_ids or plan_hint in {"seed", "rebuild"})
    if plan_hint == "seed":
        plan = [*DEFAULT_STEP_PLAN[:2], *prior_seed_plan, *DEFAULT_STEP_PLAN[2:]]
    elif plan_hint == "rebuild":
        plan = [*DEFAULT_STEP_PLAN[:2], *prior_rebuild_plan, *DEFAULT_STEP_PLAN[2:]]
    elif use_prior_plan:
        if "prior_settlement_skipped" in completed_ids or current_step == "prior_settlement_skipped":
            active_prior_plan = [optional_by_id["seed_prior_artifacts"], optional_by_id["prior_settlement_skipped"]]
        elif "prior_reconstruction_skipped" in completed_ids or current_step == "prior_reconstruction_skipped":
            active_prior_plan = [optional_by_id["prior_reconstruction_skipped"]]
        elif "seed_prior_artifacts" in completed_ids and not any(step.startswith("prior_") for step in completed_ids):
            active_prior_plan = prior_seed_plan
        elif "seed_prior_artifacts" in completed_ids:
            active_prior_plan = [*prior_seed_plan, *prior_rebuild_plan]
        else:
            active_prior_plan = prior_rebuild_plan
        plan = [*DEFAULT_STEP_PLAN[:2], *active_prior_plan, *DEFAULT_STEP_PLAN[2:]]
    else:
        plan = DEFAULT_STEP_PLAN
    plan_ids = {step_id for step_id, _ in plan}
    for row in completed_rows:
        step_id = row.get("step_id", "")
        if not step_id:
            continue
        if step_id in plan_ids and step_id not in completed_by_step:
            completed_by_step[step_id] = row
        elif step_id not in plan_ids:
            extra_rows.append(row)

    current_status = str(status_doc.get("status") or "")
    status_updated = parse_bj_datetime(str(status_doc.get("updated_at_beijing") or ""))
    current_elapsed = None
    if current_step and current_status in {"running", "stale_running"} and status_updated:
        current_elapsed = max(0.0, round((bj_now() - status_updated).total_seconds(), 1))

    rows: list[dict[str, Any]] = []
    for idx, (step_id, scheduled_time) in enumerate(plan, start=1):
        completed = completed_by_step.get(step_id)
        row: dict[str, Any] = {
            "index": idx,
            "step_id": step_id,
            "scheduled_time": scheduled_time,
            "status": "pending",
            "duration_seconds": "",
            "running_elapsed_seconds": "",
            "return_code": "",
            "message": "",
        }
        if completed:
            row.update(
                {
                    "status": completed.get("status", "success"),
                    "duration_seconds": completed.get("duration_seconds", ""),
                    "return_code": completed.get("return_code", ""),
                    "message": completed.get("message", ""),
                }
            )
        elif step_id == current_step:
            row["status"] = "waiting" if current_status == "waiting" else ("stale" if current_status == "stale_running" else "running")
            row["running_elapsed_seconds"] = current_elapsed if current_elapsed is not None else ""
        rows.append(row)

    for row in extra_rows:
        rows.append(
            {
                "index": len(rows) + 1,
                "step_id": row.get("step_id", ""),
                "scheduled_time": row.get("scheduled_time", ""),
                "status": row.get("status", "success"),
                "duration_seconds": row.get("duration_seconds", ""),
                "running_elapsed_seconds": "",
                "return_code": row.get("return_code", ""),
                "message": row.get("message", ""),
            }
        )

    if current_step and current_step not in {str(row.get("step_id")) for row in rows}:
        rows.append(
            {
                "index": len(rows) + 1,
                "step_id": current_step,
                "scheduled_time": status_doc.get("current_scheduled_time", ""),
                "status": "waiting" if current_status == "waiting" else ("stale" if current_status == "stale_running" else "running"),
                "duration_seconds": "",
                "running_elapsed_seconds": current_elapsed if current_elapsed is not None else "",
                "return_code": "",
                "message": status_doc.get("message", ""),
            }
        )

    completed_steps = int(status_doc.get("completed_steps") or len([r for r in completed_rows if r.get("status") == "success"]))
    status_total = status_doc.get("total_steps")
    try:
        total_steps = int(status_total) if status_total not in {None, ""} else len(plan)
    except Exception:
        total_steps = len(plan)
    if completed_steps > total_steps:
        total_steps = completed_steps
    pct = round(completed_steps / total_steps * 100, 1) if total_steps else 0.0
    progress = {
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "progress_pct": pct,
        "current_step_elapsed_seconds": current_elapsed,
        "current_step_id": current_step,
        "stale_status": current_status == "stale_running",
        "stale_seconds": status_doc.get("stale_seconds", ""),
    }
    return rows, progress


def line_short(value: str) -> str:
    mapping = {
        "S0_v7_original_top10": "S0",
        "S1_U2_filter_only_no_refill": "S1",
        "S0_v7_original_top10_tail_down": "S0+R1",
        "S1_U2_filter_only_no_refill_tail_down": "S1+R1",
    }
    return mapping.get(str(value), str(value))


def expected_prior_trade_date(trade_date: str) -> str | None:
    summary_path = DEFAULT_OUTPUT_ROOT / "data_preparation" / f"{trade_date}_baseline_prepare_summary.json"
    summary = read_json(summary_path)
    prior = str(summary.get("prior_trade_date") or "").strip()
    if prior.isdigit() and len(prior) == 8:
        return prior
    cal_path = ROOT / "data_tushare" / "raw" / "bootstrap" / "trade_cal.parquet"
    if not cal_path.exists():
        return None
    try:
        import pandas as pd

        cal = pd.read_parquet(cal_path)
        cal["cal_date"] = cal["cal_date"].astype(str)
        rows = cal[cal["cal_date"].eq(trade_date)]
        if not rows.empty:
            open_rows = rows[pd.to_numeric(rows.get("is_open", 0), errors="coerce").fillna(0).astype(int).eq(1)]
            if not open_rows.empty:
                value = str(open_rows.iloc[0].get("pretrade_date") or "").strip()
                if value.isdigit() and len(value) == 8:
                    return value
        open_days = cal[pd.to_numeric(cal.get("is_open", 0), errors="coerce").fillna(0).astype(int).eq(1)]["cal_date"]
        candidates = sorted([d for d in open_days.astype(str).tolist() if d.isdigit() and len(d) == 8 and d < trade_date])
        return candidates[-1] if candidates else None
    except Exception:
        return None


def infer_prior_signal_date(output_root: Path, trade_date: str) -> str | None:
    expected_prior = expected_prior_trade_date(trade_date)
    if expected_prior:
        return expected_prior
    candidates: set[str] = set()
    entry_dir = output_root / "daily_entry_prices"
    for path in entry_dir.glob("*_entry_prices.csv"):
        day = path.name.split("_", 1)[0]
        if day.isdigit() and len(day) == 8 and day < trade_date:
            candidates.add(day)
    for folder in ["daily_exit_recommendations", "daily_exit_execution", "daily_exit_settlement"]:
        for path in (output_root / folder).glob(f"*_{trade_date}_*.csv"):
            day = path.name.split("_", 1)[0]
            if day.isdigit() and len(day) == 8 and day < trade_date:
                candidates.add(day)
    return max(candidates) if candidates else None


def select_columns(rows: list[dict[str, str]], cols: list[str], limit: int = 80) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows[:limit]:
        item: dict[str, Any] = {}
        for col in cols:
            item[col] = row.get(col, "")
        if "strategy_id" in row:
            item["line"] = line_short(row.get("strategy_id", ""))
        out.append(item)
    return out


def enrich_with_signal_rank_score(rows: list[dict[str, str]], signal_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    signal_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for signal in signal_rows:
        key = (signal.get("strategy_id", ""), signal.get("code", ""))
        if key[0] and key[1]:
            signal_by_key[key] = signal
    enriched: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        signal = signal_by_key.get((row.get("strategy_id", ""), row.get("code", "")), {})
        if signal:
            item.setdefault("original_v7_rank", signal.get("original_v7_rank", ""))
            item.setdefault("score", signal.get("score", ""))
            item.setdefault("in_u2", signal.get("in_u2", ""))
            item.setdefault("tail_down_flag", signal.get("tail_down_flag", ""))
        enriched.append(item)
    return enriched


def load_prior_context(output_root: Path, trade_date: str, prior_entry_root: Path | None = None) -> dict[str, Any]:
    entry_root = prior_entry_root or output_root
    prior = infer_prior_signal_date(entry_root, trade_date)
    if not prior:
        return {
            "prior_signal_date": None,
            "status": "missing_prior_signal_date",
            "message": "未找到前一交易日纸面选股/入场文件。",
            "entries": [],
            "sell_recommendations": [],
            "sell_execution": [],
            "settlement_summary": [],
        }
    entry_path = entry_root / "daily_entry_prices" / f"{prior}_entry_prices.csv"
    rec_path = output_root / "daily_exit_recommendations" / f"{prior}_{trade_date}_exit_recommendations.csv"
    exec_path = output_root / "daily_exit_execution" / f"{prior}_{trade_date}_exit_execution.csv"
    settle_path = output_root / "daily_exit_settlement" / f"{prior}_{trade_date}_exit_settlement.csv"
    signals_path = entry_root / "daily_signals" / f"{prior}_signals.csv"
    signal_rows = read_csv_rows(signals_path, max_rows=2000)
    entry_rows = read_csv_rows(entry_path, max_rows=1000)
    rec_rows = read_csv_rows(rec_path, max_rows=1000)
    exec_rows = read_csv_rows(exec_path, max_rows=1000)
    settle_rows = read_csv_rows(settle_path, max_rows=1000)
    entry_rows = enrich_with_signal_rank_score(entry_rows, signal_rows)
    rec_rows = enrich_with_signal_rank_score(rec_rows, signal_rows)
    exec_rows = enrich_with_signal_rank_score(exec_rows, signal_rows)
    settle_rows = enrich_with_signal_rank_score(settle_rows, signal_rows)
    summary: list[dict[str, Any]] = []
    if settle_rows:
        grouped: dict[str, dict[str, Any]] = {}
        for row in settle_rows:
            sid = row.get("strategy_id", "")
            item = grouped.setdefault(sid, {"line": line_short(sid), "positions": 0, "ret_5bp_sum": 0.0, "ret_10bp_sum": 0.0, "ret_count": 0})
            item["positions"] += 1
            r5 = numeric(str(row.get("return_5bp", "")))
            r10 = numeric(str(row.get("return_10bp_impact", "")))
            w = numeric(str(row.get("weight", ""))) or 0.0
            if r5 is not None:
                item["ret_5bp_sum"] += r5 * w
            if r10 is not None:
                item["ret_10bp_sum"] += r10 * w
                item["ret_count"] += 1
        for item in grouped.values():
            item["daily_return_5bp"] = item.pop("ret_5bp_sum")
            item["daily_return_10bp_impact"] = item.pop("ret_10bp_sum")
            item.pop("ret_count", None)
            summary.append(item)
    return {
        "prior_signal_date": prior,
        "status": "available" if entry_rows else "missing_entry_file",
        "message": "" if entry_rows else f"未找到 {prior} 的纸面入场文件。",
        "entries": select_columns(entry_rows, ["line", "strategy_id", "code", "name", "original_v7_rank", "score", "expected_entry_time", "entry_vwap", "weight", "paper_entry_status"], limit=80),
        "sell_recommendations": select_columns(rec_rows, ["line", "checkpoint", "decision_time", "expected_exit_time", "strategy_id", "code", "name", "original_v7_rank", "score", "entry_vwap", "recommended_sell_price", "exit_reason", "recommendation_status"], limit=120),
        "sell_execution": select_columns(exec_rows, ["line", "actual_exit_time", "strategy_id", "code", "name", "original_v7_rank", "score", "entry_vwap", "exit_vwap", "exit_reason", "return_5bp", "return_10bp_impact", "paper_exit_status"], limit=120),
        "settlement_summary": summary,
        "files": {
            "prior_entry": str(entry_path),
            "sell_recommendations": str(rec_path),
            "sell_execution": str(exec_path),
            "sell_settlement": str(settle_path),
        },
    }


def load_today_context(output_root: Path, trade_date: str) -> dict[str, Any]:
    signals_path = output_root / "daily_signals" / f"{trade_date}_signals.csv"
    entry_path = output_root / "daily_entry_prices" / f"{trade_date}_entry_prices.csv"
    signals = read_csv_rows(signals_path, max_rows=1000)
    entries = read_csv_rows(entry_path, max_rows=1000)
    entry_by_key = {(row.get("strategy_id", ""), row.get("code", "")): row for row in entries}
    buy_rows: list[dict[str, Any]] = []
    for row in signals:
        if str(row.get("final_selected_flag", "")).lower() not in {"true", "1", "yes"}:
            continue
        entry = entry_by_key.get((row.get("strategy_id", ""), row.get("code", "")), {})
        buy_rows.append(
            {
                "line": line_short(row.get("strategy_id", "")),
                "strategy_id": row.get("strategy_id", ""),
                "code": row.get("code", ""),
                "name": row.get("name", ""),
                "rank": row.get("original_v7_rank", ""),
                "score": row.get("score", ""),
                "in_u2": row.get("in_u2", ""),
                "tail_down": row.get("tail_down_flag", ""),
                "expected_entry_time": entry.get("expected_entry_time") or row.get("entry_time", "14:55"),
                "entry_vwap": entry.get("entry_vwap", ""),
                "entry_status": entry.get("paper_entry_status", "pending_entry_price"),
                "weight": entry.get("weight") or row.get("position_weight", ""),
            }
        )
    return {
        "buy_signals": buy_rows,
        "signals_generated": bool(signals),
        "entries_recorded": bool(entries),
        "files": {
            "signals": str(signals_path),
            "entry_prices": str(entry_path),
        },
    }


DEPENDENCY_LABELS = {
    "provider_up": ("Provider/API 可请求", "行情服务"),
    "historical_data_ok": ("历史行情数据", "行情服务"),
    "today_realtime_data_ok": ("今日实时行情数据", "行情服务"),
    "env_TUSHARE_TOKEN": ("Tushare Token", "凭证"),
    "env_WXPUSHER_APP_TOKEN": ("WxPusher Token", "凭证"),
    "local_trade_calendar_cache": ("本地交易日历缓存", "日历"),
    "akshare_sina_trade_calendar": ("AkShare/Sina 日历备用源", "日历"),
    "tushare_proxy_trade_cal_api": ("Tushare 日历接口", "行情服务"),
    "tushare_proxy_stk_mins_api": ("Tushare 分钟线接口", "行情服务"),
    "tushare_proxy_open_auction_api": ("Tushare 开盘竞价接口", "行情服务"),
    "wxpusher_domain_tcp_tls": ("WxPusher 推送域名", "通知"),
    "trade_calendar_resilience": ("交易日历容错", "日历"),
}


def dependency_display(name: str) -> tuple[str, str]:
    return DEPENDENCY_LABELS.get(name, (name, "其他"))


def dependency_state(row: dict[str, Any]) -> str:
    status = str(row.get("status") or "").lower()
    severity = str(row.get("severity") or "").lower()
    if severity == "fatal" or status in {"failed", "missing", "empty_response", "missing_trade_date"}:
        return "bad"
    if severity == "warning" or status in {"warning", "partial", "skipped"}:
        return "warn"
    if status in {"success", "present", "disabled", "skipped_notifications_disabled"}:
        return "ok"
    return "neutral"


def dependency_state_label(state: str) -> str:
    return {
        "bad": "不可用",
        "warn": "告警",
        "ok": "正常",
        "neutral": "未知",
    }.get(state, "未知")


def severity_rank(row: dict[str, Any]) -> tuple[int, str]:
    return {"bad": 0, "warn": 1, "neutral": 2, "ok": 3}.get(dependency_state(row), 2), str(row.get("name") or "")


def summarize_auction_guard_dependency(output_root: Path, trade_date: str) -> dict[str, Any] | None:
    path = output_root / "data_guards" / f"{trade_date}_auction_guard_0925.json"
    if not path.exists():
        return None
    data = read_json(path)
    status = str(data.get("status") or "unknown")
    missing_required = data.get("missing_required") or []
    missing_optional = data.get("missing_optional") or []
    fetch_results = data.get("fetch_results") or []
    failed_fetch = [x for x in fetch_results if str(x.get("status") or "") == "failed"]
    if status == "failed" or missing_required:
        severity = "fatal"
    elif status == "warning" or missing_optional or failed_fetch:
        severity = "warning"
    else:
        severity = "info"
    impact_parts: list[str] = []
    if missing_required:
        roles = sorted(set(str(x.get("role") or x.get("api_name") or "") for x in missing_required))
        impact_parts.append("缺失 required: " + ", ".join([x for x in roles if x]))
    if missing_optional:
        roles = sorted(set(str(x.get("role") or x.get("api_name") or "") for x in missing_optional))
        impact_parts.append("缺失 optional: " + ", ".join([x for x in roles if x]))
    if failed_fetch:
        errors = [str(x.get("error") or "") for x in failed_fetch if x.get("error")]
        if errors:
            impact_parts.append(errors[0])
    return {
        "name": "tushare_proxy_open_auction_api",
        "display_name": "Tushare 开盘竞价接口",
        "category": "行情服务",
        "status": status,
        "severity": severity,
        "role": "T 日开盘竞价 stk_auction_o",
        "impact": "；".join(impact_parts) or "开盘竞价检查完成",
        "duration_seconds": data.get("duration_seconds", ""),
        "file": str(path),
    }


def build_dependency_cards(rows: list[dict[str, Any]], overall_status: str) -> list[dict[str, Any]]:
    by_name = {str(row.get("name") or ""): row for row in rows}

    def card(card_id: str, title: str, names: list[str], purpose: str) -> dict[str, Any]:
        matched = [by_name[name] for name in names if name in by_name]
        if not matched:
            state = "neutral"
            summary = "未检查"
        else:
            state = dependency_state(sorted(matched, key=severity_rank)[0])
            bad_or_warn = [row for row in matched if dependency_state(row) in {"bad", "warn"}]
            summary_row = bad_or_warn[0] if bad_or_warn else matched[0]
            summary = str(summary_row.get("impact") or summary_row.get("status") or "")
            if not summary:
                summary = "检查通过"
        return {
            "id": card_id,
            "title": title,
            "state": state,
            "state_label": dependency_state_label(state),
            "purpose": purpose,
            "summary": summary,
        }

    cards = [
        card(
            "provider_up",
            "Provider/API 可请求",
            ["provider_up", "tushare_proxy_trade_cal_api"],
            "域名、token、基础 API 请求是否可用",
        ),
        card(
            "historical_data",
            "历史数据可返回",
            ["historical_data_ok", "tushare_proxy_stk_mins_api"],
            "历史分钟线/回放数据是否可用",
        ),
        card(
            "today_realtime",
            "今日实时数据可返回",
            ["today_realtime_data_ok", "tushare_proxy_open_auction_api"],
            "今日分钟线、开盘竞价等实时数据是否已返回",
        ),
        card(
            "notification",
            "消息推送链路",
            ["env_WXPUSHER_APP_TOKEN", "wxpusher_domain_tcp_tls"],
            "买入/卖出提醒与异常告警",
        ),
        card(
            "calendar",
            "交易日历链路",
            ["local_trade_calendar_cache", "akshare_sina_trade_calendar", "tushare_proxy_trade_cal_api", "trade_calendar_resilience"],
            "判断交易日、T-1/T-2 日期",
        ),
        card(
            "credentials",
            "凭证配置",
            ["env_TUSHARE_TOKEN", "env_WXPUSHER_APP_TOKEN"],
            "本次运行所需 token 是否存在",
        ),
    ]
    if overall_status == "fatal" and all(item["state"] != "bad" for item in cards):
        cards.insert(
            0,
            {
                "id": "overall",
                "title": "整体状态",
                "state": "bad",
                "state_label": "不可用",
                "purpose": "启动预检",
                "summary": "存在 fatal 依赖，但未能归入具体服务。",
            },
        )
    return cards


def empty_dependency_status(source_label: str, source_note: str) -> dict[str, Any]:
    return {
        "overall_status": "not_started",
        "ui_state": "neutral",
        "headline": "尚未启动本次依赖检查",
        "action": "启动任务后会显示本次预检结果；当前不会混入历史依赖状态。",
        "generated_time_beijing": "",
        "prior_trade_date_for_probe": "",
        "checks": [],
        "issues": [],
        "cards": [],
        "file": "",
        "exists": False,
        "source_label": source_label,
        "source_note": source_note,
    }


def load_dependency_status(
    output_root: Path,
    trade_date: str,
    source_label: str = "历史已有数据",
    source_note: str = "读取已落地的 preflight/data_guards 文件，不代表当前刚刚重新探活。",
    suppress_history: bool = False,
) -> dict[str, Any]:
    if suppress_history:
        return empty_dependency_status(source_label, source_note)
    preflight_path = output_root / "preflight" / f"{trade_date}_preflight.json"
    preflight = read_json(preflight_path)
    checks = preflight.get("checks") or []
    rows: list[dict[str, Any]] = []
    for item in checks:
        display_name, category = dependency_display(str(item.get("name") or ""))
        rows.append(
            {
                "name": item.get("name", ""),
                "display_name": display_name,
                "category": category,
                "status": item.get("status", ""),
                "severity": item.get("severity", ""),
                "role": item.get("role", ""),
                "impact": item.get("impact") or item.get("error") or "",
                "duration_seconds": item.get("duration_seconds", ""),
                "state": "",
            }
        )
    auction_row = summarize_auction_guard_dependency(output_root, trade_date)
    if auction_row:
        rows.append(auction_row)
    for row in rows:
        row["state"] = dependency_state(row)
    rows = sorted(rows, key=severity_rank)
    issues = [row for row in rows if row["state"] in {"bad", "warn"}]
    overall_status = str(preflight.get("overall_status") or ("missing" if not preflight else "unknown"))
    if any(row["state"] == "bad" for row in rows):
        ui_state = "bad"
    elif any(row["state"] == "warn" for row in rows) or overall_status in {"warning", "missing", "unknown"}:
        ui_state = "warn"
    else:
        ui_state = "ok"
    today_realtime_issue = next((row for row in issues if row.get("name") == "today_realtime_data_ok"), None)
    if today_realtime_issue and today_realtime_issue.get("state") == "bad":
        headline = "今日实时行情数据不可用"
        action = str(today_realtime_issue.get("impact") or "今日分钟线/开盘竞价未返回可用数据，请联系供应商核实。")
    elif ui_state == "bad":
        headline = "依赖不可用，今日流程应暂停"
        action = "等待服务商恢复后重新启动；不要用迟到数据事后生成实时信号。"
    elif ui_state == "warn":
        headline = "依赖存在告警，需要确认"
        action = "查看异常项；关键行情链路告警时不要继续生成有效买卖信号。"
    else:
        headline = "依赖检查通过"
        action = "行情、通知和日历链路当前可用。"
    return {
        "overall_status": overall_status,
        "ui_state": ui_state,
        "headline": headline,
        "action": action,
        "generated_time_beijing": preflight.get("generated_time_beijing", ""),
        "prior_trade_date_for_probe": preflight.get("prior_trade_date_for_probe", ""),
        "checks": rows,
        "issues": issues,
        "cards": build_dependency_cards(rows, overall_status),
        "file": str(preflight_path),
        "exists": preflight_path.exists(),
        "source_label": source_label,
        "source_note": source_note,
    }


def dashboard_artifacts(
    output_root: Path,
    trade_date: str,
    prior_entry_root: Path | None = None,
    plan_hint: str = "auto",
    dependency_source_label: str = "历史已有数据",
    dependency_source_note: str = "读取已落地的 preflight/data_guards 文件，不代表当前刚刚重新探活。",
    suppress_dependency_history: bool = False,
) -> dict[str, Any]:
    live_dir = output_root / "live_runner"
    status_path = live_dir / f"{trade_date}_live_runner_status.json"
    steps_path = live_dir / f"{trade_date}_live_runner_steps.csv"
    candidate_path = output_root / "forward_shadow_candidate_status.csv"
    signals_path = output_root / "daily_signals" / f"{trade_date}_signals.csv"
    entry_path = output_root / "daily_entry_prices" / f"{trade_date}_entry_prices.csv"
    preflight_path = output_root / "preflight" / f"{trade_date}_preflight.json"

    status_doc = annotate_stale_live_status(read_json(status_path))
    steps = read_csv_rows(steps_path, max_rows=120)
    step_progress, progress = synthesize_step_progress(status_doc, steps, plan_hint=plan_hint)
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
        "preflight": str(preflight_path),
    }
    exists = {name: Path(path).exists() for name, path in files.items()}
    return {
        "live_status": status_doc,
        "steps": step_progress,
        "raw_steps": steps,
        "progress": progress,
        "candidate_status": candidates,
        "signal_summary": summarize_by_strategy(signals),
        "entry_counts": entry_counts,
        "prior_context": load_prior_context(output_root, trade_date, prior_entry_root=prior_entry_root),
        "today_context": load_today_context(output_root, trade_date),
        "dependency_status": load_dependency_status(
            output_root,
            trade_date,
            source_label=dependency_source_label,
            source_note=dependency_source_note,
            suppress_history=suppress_dependency_history,
        ),
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
        "last_error_lines": list(job.get("last_error_lines") or []),
        "snapshot_path": job.get("snapshot_path"),
        "snapshot_error": job.get("snapshot_error"),
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
      --bg: #f3f5f8;
      --panel: #ffffff;
      --line: #d9dee7;
      --text: #1f2937;
      --muted: #667085;
      --soft: #f8fafc;
      --accent: #175cd3;
      --bad: #b42318;
      --ok: #067647;
      --warn: #b54708;
      --shadow: 0 1px 2px rgba(16, 24, 40, .06);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }}
    header {{
      padding: 18px 28px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{ margin: 0; font-size: 20px; letter-spacing: 0; }}
    h2 {{ font-size: 16px; margin: 0 0 10px; }}
    h3 {{ font-size: 14px; margin: 0 0 8px; }}
    main {{ padding: 18px 24px 32px; max-width: 1480px; margin: 0 auto; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: var(--shadow);
    }}
    .control-head {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 12px;
    }}
    .control-title {{ display: flex; flex-direction: column; gap: 3px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(170px, 1fr)); gap: 12px; align-items: end; }}
    .compact-grid {{ display: grid; grid-template-columns: repeat(3, minmax(160px, 1fr)); gap: 12px; align-items: end; }}
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
    .stack {{ display: flex; flex-direction: column; gap: 12px; }}
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
    .metrics {{ display: grid; grid-template-columns: 1.2fr 1.8fr repeat(5, minmax(110px, 1fr)); gap: 10px; }}
    .metric {{ border: 1px solid var(--line); border-radius: 6px; padding: 10px; background: #fcfcfd; min-height: 70px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; }}
    .metric strong {{ display: block; font-size: 17px; line-height: 1.3; margin-top: 4px; word-break: break-word; }}
    .progress-track {{ height: 8px; background: #eaecf0; border-radius: 999px; overflow: hidden; }}
    .progress-fill {{ height: 100%; width: 0%; background: var(--accent); transition: width .25s ease; }}
    .hero {{
      display: grid;
      grid-template-columns: minmax(260px, 1fr) 1.2fr 1.2fr;
      gap: 14px;
    }}
    .action-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }}
    .action-card header {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--soft);
    }}
    .action-card .body {{ padding: 12px 14px; }}
    .action-card h2 {{ margin: 0; }}
    .action-meta {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
    .value-big {{ font-size: 24px; font-weight: 700; line-height: 1.2; }}
    .dependency-box {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fcfcfd;
      padding: 12px;
    }}
    .dependency-toolbar {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }}
    .dependency-monitor {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
    }}
    .dependency-toolbar button {{ min-width: 120px; padding: 8px 12px; }}
    .dependency-banner {{
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 10px;
      align-items: start;
      border-radius: 8px;
      padding: 12px;
      border: 1px solid var(--line);
      background: #f8fafc;
      margin: 8px 0 12px;
    }}
    .dependency-banner.bad {{ background: #fff6f5; border-color: #fecdca; color: var(--bad); }}
    .dependency-banner.warn {{ background: #fffaeb; border-color: #fedf89; color: var(--warn); }}
    .dependency-banner.ok {{ background: #ecfdf3; border-color: #abefc6; color: var(--ok); }}
    .dependency-icon {{
      display: inline-flex;
      width: 30px;
      height: 30px;
      border-radius: 999px;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      background: rgba(255,255,255,.72);
      border: 1px solid currentColor;
    }}
    .dependency-title {{ font-size: 15px; font-weight: 700; color: inherit; }}
    .dependency-action {{ color: var(--text); margin-top: 3px; }}
    .dependency-cards {{
      display: grid;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .dependency-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      padding: 10px;
      min-height: 108px;
    }}
    .dependency-card.bad {{ border-color: #fecdca; background: #fff6f5; }}
    .dependency-card.warn {{ border-color: #fedf89; background: #fffaeb; }}
    .dependency-card.ok {{ border-color: #abefc6; background: #ecfdf3; }}
    .dependency-card.neutral {{ background: #f8fafc; }}
    .dependency-card-head {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; }}
    .dependency-card-title {{ font-weight: 700; }}
    .dependency-card-purpose {{ color: var(--muted); font-size: 12px; margin-top: 6px; }}
    .dependency-card-summary {{ margin-top: 8px; color: var(--text); font-size: 12px; }}
    .dependency-section-title {{ font-size: 13px; color: var(--muted); font-weight: 700; margin: 12px 0 4px; }}
    .source-badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 3px 8px;
      background: #eef4ff;
      color: #194185;
      border: 1px solid #c7d7fe;
      font-size: 12px;
      margin-right: 8px;
    }}
    .source-note {{
      color: var(--muted);
      font-size: 12px;
    }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 7px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; background: #f8fafc; }}
    td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .line-tags {{ display: flex; flex-wrap: wrap; gap: 4px; }}
    .line-tag {{ display: inline-flex; padding: 2px 6px; border-radius: 999px; background: #eef4ff; color: #194185; border: 1px solid #c7d7fe; font-size: 12px; }}
    .status-chip {{ display: inline-flex; border-radius: 999px; padding: 2px 8px; font-size: 12px; border: 1px solid #d0d5dd; background: #f2f4f7; color: #344054; }}
    .status-chip.success {{ background: #ecfdf3; color: var(--ok); border-color: #abefc6; }}
    .status-chip.running {{ background: #eff8ff; color: #175cd3; border-color: #b2ddff; }}
    .status-chip.waiting {{ background: #fffaeb; color: var(--warn); border-color: #fedf89; }}
    .status-chip.warning, .status-chip.partial, .status-chip.stale, .status-chip.stale_running {{ background: #fffaeb; color: var(--warn); border-color: #fedf89; }}
    .status-chip.fatal, .status-chip.bad, .status-chip.empty_response, .status-chip.missing, .status-chip.missing_trade_date {{ background: #fef3f2; color: var(--bad); border-color: #fecdca; }}
    .status-chip.info, .status-chip.ok, .status-chip.present {{ background: #ecfdf3; color: var(--ok); border-color: #abefc6; }}
    .status-chip.pending {{ background: #f2f4f7; color: #475467; border-color: #d0d5dd; }}
    .status-chip.skipped {{ background: #fffaeb; color: var(--warn); border-color: #fedf89; }}
    .status-chip.failed {{ background: #fef3f2; color: var(--bad); border-color: #fecdca; }}
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
    .three {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
    .notice {{ padding: 10px 12px; border: 1px solid #fedf89; background: #fffaeb; color: #93370d; border-radius: 6px; }}
    .secret-row {{ grid-column: 1 / -1; display: grid; grid-template-columns: 1fr 1fr 220px; gap: 12px; align-items: end; }}
    .error-panel {{
      display: none;
      border-color: #fecdca;
      background: #fff6f5;
    }}
    .error-panel h2 {{ color: var(--bad); }}
    .error-panel pre {{ background: #7a271a; color: #fff1f0; }}
    details {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fcfcfd;
      padding: 10px 12px;
      margin-top: 12px;
    }}
    summary {{ cursor: pointer; font-weight: 600; }}
    .tabs {{ display: flex; gap: 8px; border-bottom: 1px solid var(--line); margin-bottom: 14px; }}
    .tab-btn {{
      border: 0;
      background: transparent;
      color: var(--muted);
      border-radius: 0;
      min-width: 0;
      padding: 10px 4px;
      border-bottom: 2px solid transparent;
    }}
    .tab-btn.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .empty {{
      padding: 14px;
      border: 1px dashed var(--line);
      color: var(--muted);
      border-radius: 6px;
      background: #fcfcfd;
    }}
    .status-line {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
    .nowrap {{ white-space: nowrap; }}
    @media (max-width: 920px) {{
      .grid, .compact-grid, .metrics, .hero, .two, .three, .secret-row, .dependency-cards {{ grid-template-columns: 1fr; }}
      main {{ padding: 14px; }}
      .control-head {{ flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Forward Shadow Paper Tracking 控制台</h1>
    <div class="muted">本地 paper-only 页面：只做前向纸面跟踪；不下单，不重训，不改 v7_locked。</div>
  </header>
  <main>
    <section>
      <div class="control-head">
        <div class="control-title">
          <h2>任务启动</h2>
          <div class="muted">默认真实时间点运行；历史补跑可切换为模拟时间点快跑。token 只注入本次子进程，不落盘。</div>
        </div>
        <div class="bar">
          <button id="startBtn">启动</button>
          <button id="stopBtn" class="danger">停止</button>
          <button id="refreshBtn" class="secondary">刷新页面数据</button>
        </div>
      </div>
      <div class="compact-grid">
        <div>
          <label for="tradeDate">执行日期</label>
          <input id="tradeDate" type="date" min="{min_date}" value="{default_date}">
        </div>
        <div>
          <label for="mode">执行模式</label>
          <select id="mode">
            <option value="live_time" selected>真实时间点模式</option>
            <option value="fast_replay">模拟时间点快跑</option>
          </select>
        </div>
        <div>
          <label for="dataSourceMode">数据展示口径</label>
          <select id="dataSourceMode">
            <option value="history" selected>历史已有数据（所选日期）</option>
            <option value="latest_run">最近一次运行结果</option>
            <option value="rerun_clean">重跑清爽视图（仅 T-1 历史 + 本次输出）</option>
          </select>
        </div>
        <div>
          <label for="notificationPolicy">消息推送策略</label>
          <select id="notificationPolicy">
            <option value="key_events" selected>关键交易 + 失败</option>
            <option value="trade_only">只推买卖提示 + 错误</option>
            <option value="failures_only">只推失败</option>
            <option value="all_steps">所有节点</option>
            <option value="none">不推送</option>
          </select>
        </div>
      </div>
      <details>
        <summary>高级设置</summary>
        <div class="grid" style="margin-top:12px">
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
            <label for="runId">run_id（可选）</label>
            <input id="runId" placeholder="留空自动生成">
          </div>
          <div>
            <label for="priorInputPolicy">T-1 输入策略</label>
            <select id="priorInputPolicy">
              <option value="reuse_only" selected>只复用已冻结 T-1，缺失跳过结算（推荐）</option>
              <option value="reuse_or_rebuild">复用已冻结 T-1，缺失自动重建（诊断）</option>
              <option value="force_rebuild">强制重建 T-1</option>
            </select>
          </div>
          <div>
            <label for="topicId">WxPusher Topic / GroupId</label>
            <input id="topicId" type="number" value="{topic_id}">
          </div>
          <div class="checks">
            <label><input id="sendNotifications" type="checkbox" checked> 推送 WxPusher</label>
            <label><input id="skipMoneyflow" type="checkbox"> 跳过 moneyflow</label>
          </div>
          <div class="secret-row">
            <div>
              <label for="tushareToken">TUSHARE_TOKEN（可选，留空使用环境变量）</label>
              <input id="tushareToken" type="password" autocomplete="new-password" placeholder="只用于本次启动，不落盘">
            </div>
            <div>
              <label for="wxpusherToken">WXPUSHER_APP_TOKEN（可选，留空使用环境变量）</label>
              <input id="wxpusherToken" type="password" autocomplete="new-password" placeholder="只用于本次启动，不落盘">
            </div>
            <div class="checks">
              <label><input id="showTokens" type="checkbox"> 显示输入</label>
            </div>
          </div>
          <div class="checks" style="grid-column:1/-1">
            <label><input id="useRunIdOutputDir" type="checkbox"> run_id 输出目录</label>
            <label><input id="preserveRunSnapshot" type="checkbox"> 保留本次 run 快照</label>
            <label><input id="forceRefreshMinutes" type="checkbox"> 强制重新下载分钟线</label>
          </div>
        </div>
        <p class="muted">模拟时间点快跑仍走真实接口和真实数据处理，只是不等待墙上时间。快跑启动时会先清理本次 T-1/T 对应的卖出提示、退出记录、结算摘要等派生产物，避免和旧补跑结果混合；run_id 输出目录写入 `输出目录/runs/run_id`；T-1 默认从输出目录基线复用已冻结的 `daily_signals` 和 `daily_entry_prices`；强制重新下载分钟线会重新请求分钟 bar 并按时间键覆盖去重。</p>
      </details>
      <div id="message" class="muted"></div>
    </section>

    <section class="stack">
      <div class="status-line">
        <span id="jobStatus" class="pill">未启动</span>
        <span id="modePill" class="pill">mode: -</span>
        <span id="datePill" class="pill">date: -</span>
        <span id="dataSourcePill" class="pill">source: history</span>
        <span id="runIdPill" class="pill">run_id: -</span>
        <span id="pidPill" class="pill">pid: -</span>
      </div>
      <div class="metrics">
        <div class="metric"><span>当前节点</span><strong id="currentStep">-</strong></div>
        <div class="metric"><span>正在做什么</span><strong id="currentStepMeaning">-</strong></div>
        <div class="metric"><span>节点状态</span><strong id="runnerState">-</strong></div>
        <div class="metric"><span>完成进度</span><strong id="completedSteps">0 / 30</strong></div>
        <div class="metric"><span>当前耗时</span><strong id="currentElapsed">-</strong></div>
        <div class="metric"><span>总节点</span><strong id="totalSteps">-</strong></div>
        <div class="metric"><span>更新时间</span><strong id="updatedAt">-</strong></div>
      </div>
      <div class="progress-track" aria-label="任务进度">
        <div id="progressBar" class="progress-fill"></div>
      </div>
    </section>

    <section id="errorPanel" class="error-panel">
      <h2>运行错误</h2>
      <div id="errorText"></div>
      <pre id="errorTail"></pre>
    </section>

    <section>
      <div class="tabs">
        <button class="tab-btn active" data-tab="overview">今日看板</button>
        <button class="tab-btn" data-tab="details">买卖明细</button>
        <button class="tab-btn" data-tab="ops">运行日志</button>
      </div>

      <div id="tab-overview" class="tab-panel active">
        <div class="hero">
          <div class="action-card">
            <header>
              <h2>今日状态</h2>
              <div class="action-meta" id="overviewDate">-</div>
            </header>
            <div class="body">
              <div class="value-big" id="overviewHeadline">-</div>
              <div class="muted" id="overviewSubline" style="margin-top:8px">-</div>
            </div>
          </div>
          <div class="action-card">
            <header>
              <h2>今日卖出提示</h2>
              <div class="action-meta" id="sellActionMeta">-</div>
            </header>
            <div class="body">
              <div id="sellActionTable"></div>
            </div>
          </div>
          <div class="action-card">
            <header>
              <h2>今日尾盘买入</h2>
              <div class="action-meta" id="buyActionMeta">-</div>
            </header>
            <div class="body">
              <div id="buyActionTable"></div>
            </div>
          </div>
        </div>
        <div class="two" style="margin-top:16px">
          <div>
            <h2>依赖连通性</h2>
            <div id="dependencyPanel" class="dependency-box">
              <div class="dependency-toolbar">
                <div class="dependency-monitor">
                  <span id="dependencyMonitorStatus" class="status-chip pending">监测状态：空闲</span>
                  <span id="dependencyMonitorMeta">未手动刷新</span>
                </div>
                <button id="dependencyRefreshBtn" class="secondary" disabled>刷新依赖连通性</button>
              </div>
              <div id="dependencyMeta" class="muted"></div>
              <div id="dependencyBanner"></div>
              <div id="dependencyCards"></div>
              <div class="dependency-section-title">异常项</div>
              <div id="dependencyIssues"></div>
              <details>
                <summary>完整检查明细</summary>
                <div id="dependencyTable"></div>
              </details>
            </div>
          </div>
          <div>
            <h2>四线路状态</h2>
            <div id="candidateTable"></div>
          </div>
        </div>
        <div class="two" style="margin-top:16px">
          <div>
            <h2>信号与入场统计</h2>
            <div id="signalSummary"></div>
            <div id="entryCounts" style="margin-top:10px"></div>
          </div>
        </div>
      </div>

      <div id="tab-details" class="tab-panel">
        <h2>T-1 纸面持仓与今日卖出</h2>
        <div id="priorNotice" class="notice"></div>
        <div class="two" style="margin-top:12px">
          <div>
            <h2>前一交易日纸面买入</h2>
            <div id="priorEntriesTable"></div>
          </div>
          <div>
            <h2>今日卖出提示</h2>
            <div id="sellRecommendationsTable"></div>
          </div>
        </div>
        <div class="two" style="margin-top:12px">
          <div>
            <h2>今日退出记录</h2>
            <div id="sellExecutionTable"></div>
          </div>
          <div>
            <h2>线路结算摘要</h2>
            <div id="settlementSummaryTable"></div>
          </div>
        </div>

        <h2 style="margin-top:16px">今日尾盘选股与买入价</h2>
        <div id="todayBuyNotice" class="muted"></div>
        <div id="todayBuyTable"></div>
      </div>

      <div id="tab-ops" class="tab-panel">
        <div class="two">
          <div>
            <h2>步骤进度</h2>
            <div id="stepsTable"></div>
          </div>
          <div>
            <h2>输出文件</h2>
            <div id="filesTable"></div>
          </div>
        </div>
        <h2 style="margin-top:16px">运行日志</h2>
        <pre id="logTail"></pre>
      </div>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    let activeRunJobId = localStorage.getItem('forwardShadowActiveJobId') || '';
    let latestStatusData = null;

    function dateToYmd(v) {{ return (v || '').replaceAll('-', ''); }}
    function setMessage(text, bad=false) {{
      $('message').textContent = text || '';
      $('message').style.color = bad ? '#b42318' : '#667085';
    }}
    function esc(v) {{
      return String(v ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
    }}
    function asNumber(v) {{
      if (v === null || v === undefined || v === '') return null;
      const n = Number(v);
      return Number.isFinite(n) ? n : null;
    }}
    function fmtPrice(v) {{
      const n = asNumber(v);
      return n === null ? '-' : n.toFixed(2);
    }}
    function fmtScore(v) {{
      const n = asNumber(v);
      return n === null ? '-' : n.toFixed(4);
    }}
    function fmtPct(v) {{
      const n = asNumber(v);
      return n === null ? '-' : (n * 100).toFixed(2) + '%';
    }}
    function fmtWeight(v) {{
      const n = asNumber(v);
      return n === null ? '-' : (n * 100).toFixed(1) + '%';
    }}
    function fmtNumber(v, digits=0) {{
      const n = asNumber(v);
      return n === null ? '-' : n.toFixed(digits);
    }}
    function fmtDuration(v) {{
      const n = asNumber(v);
      if (n === null) return '-';
      if (n < 60) return n.toFixed(1) + 's';
      const minutes = Math.floor(n / 60);
      const seconds = Math.round(n % 60);
      return minutes + 'm ' + seconds + 's';
    }}
    function compactText(v, maxLen=110) {{
      const text = String(v ?? '').replace(/\\s+/g, ' ').trim();
      if (!text) return '-';
      return text.length > maxLen ? text.slice(0, maxLen - 1) + '...' : text;
    }}
    function isTrue(v) {{
      return String(v ?? '').toLowerCase() === 'true' || String(v ?? '') === '1';
    }}
    function formatCell(value, key, col) {{
      const type = col.type || '';
      const lower = String(key || '').toLowerCase();
      if (type === 'lines') return renderLineTags(value || []);
      if (type === 'price' || lower.includes('vwap') || lower.includes('price')) return fmtPrice(value);
      if (type === 'pct' || lower.includes('return')) return fmtPct(value);
      if (type === 'score' || lower.includes('score')) return fmtScore(value);
      if (type === 'weight' || lower === 'weight') return fmtWeight(value);
      if (type === 'seconds') return fmtDuration(value);
      if (type === 'status') return '<span class="status-chip ' + esc(String(value || 'pending')) + '">' + esc(value || 'pending') + '</span>';
      if (type === 'message') return esc(compactText(value));
      if (type === 'number') return fmtNumber(value, col.digits || 0);
      return esc(value === undefined || value === null || value === '' ? '-' : value);
    }}
    function numericClass(col, key) {{
      const type = col.type || '';
      const lower = String(key || '').toLowerCase();
      return ['price','pct','score','weight','seconds','number'].includes(type) || lower.includes('vwap') || lower.includes('return') || lower.includes('score') || lower.includes('price') ? 'num' : '';
    }}
    function renderLineTags(lines) {{
      const arr = Array.isArray(lines) ? lines : String(lines || '').split(/[，,\\/]+/).filter(Boolean);
      if (!arr.length) return '-';
      return '<div class="line-tags">' + arr.map(x => '<span class="line-tag">' + esc(x) + '</span>').join('') + '</div>';
    }}
    function table(rows, cols) {{
      if (!rows || !rows.length) return '<div class="empty">暂无数据</div>';
      const head = cols.map(c => {{
        const key = c.key || c;
        const cls = numericClass(c, key);
        return '<th class="' + cls + '">' + esc(c.label || c) + '</th>';
      }}).join('');
      const body = rows.map(r => '<tr>' + cols.map(c => {{
        const key = c.key || c;
        const cls = numericClass(c, key);
        return '<td class="' + cls + '">' + formatCell(r[key], key, c) + '</td>';
      }}).join('') + '</tr>').join('');
      return '<div class="table-wrap"><table><thead><tr>' + head + '</tr></thead><tbody>' + body + '</tbody></table></div>';
    }}
    function compactBuyRows(rows) {{
      const map = new Map();
      for (const r of rows || []) {{
        const key = (r.code || '') + '|' + (r.expected_entry_time || '') + '|' + (r.entry_vwap || '');
        if (!map.has(key)) {{
          map.set(key, {{
            lines: [],
            code: r.code || '',
            name: r.name || '',
            rank: r.rank || '',
            score: r.score || '',
            expected_entry_time: r.expected_entry_time || '14:55',
            entry_vwap: r.entry_vwap || '',
            weight: r.weight || '',
            entry_status: r.entry_status || ''
          }});
        }}
        const item = map.get(key);
        if (r.line && !item.lines.includes(r.line)) item.lines.push(r.line);
        const oldScore = asNumber(item.score);
        const newScore = asNumber(r.score);
        if (newScore !== null && (oldScore === null || newScore > oldScore)) item.score = r.score;
      }}
      return Array.from(map.values());
    }}
    function compactSellRows(rows) {{
      const map = new Map();
      for (const r of rows || []) {{
        const time = r.expected_exit_time || r.actual_exit_time || '';
        const price = r.recommended_sell_price || r.exit_vwap || '';
        const key = (r.code || '') + '|' + time + '|' + price + '|' + (r.exit_reason || '');
        if (!map.has(key)) {{
          map.set(key, {{
            lines: [],
            decision_time: r.decision_time || '',
            exit_time: time,
            code: r.code || '',
            name: r.name || '',
            rank: r.original_v7_rank || '',
            score: r.score || '',
            sell_price: price,
            exit_reason: r.exit_reason || '',
            status: r.recommendation_status || r.paper_exit_status || ''
          }});
        }}
        const item = map.get(key);
        if (r.line && !item.lines.includes(r.line)) item.lines.push(r.line);
        const oldScore = asNumber(item.score);
        const newScore = asNumber(r.score);
        if (newScore !== null && (oldScore === null || newScore > oldScore)) item.score = r.score;
      }}
      return Array.from(map.values());
    }}
    function dependencyIcon(state) {{
      if (state === 'bad') return '!';
      if (state === 'warn') return '?';
      if (state === 'ok') return '✓';
      return '-';
    }}
    function dependencyStatusClass(state) {{
      if (state === 'bad') return 'failed';
      if (state === 'warn') return 'warning';
      if (state === 'ok') return 'success';
      return 'pending';
    }}
    function renderDependencyBanner(deps) {{
      const state = deps.ui_state || 'neutral';
      const cls = state === 'bad' ? 'bad' : (state === 'warn' ? 'warn' : (state === 'ok' ? 'ok' : 'neutral'));
      return '<div class="dependency-banner ' + cls + '">' +
        '<div class="dependency-icon">' + esc(dependencyIcon(state)) + '</div>' +
        '<div><div class="dependency-title">' + esc(deps.headline || '依赖状态未知') + '</div>' +
        '<div class="dependency-action">' + esc(deps.action || '') + '</div></div>' +
        '</div>';
    }}
    function renderDependencyCards(cards) {{
      if (!cards || !cards.length) return '<div class="empty">暂无依赖分组数据</div>';
      return '<div class="dependency-cards">' + cards.map(card => {{
        const state = card.state || 'neutral';
        return '<div class="dependency-card ' + esc(state) + '">' +
          '<div class="dependency-card-head"><span class="dependency-card-title">' + esc(card.title || '-') + '</span>' +
          '<span class="status-chip ' + dependencyStatusClass(state) + '">' + esc(card.state_label || '-') + '</span></div>' +
          '<div class="dependency-card-purpose">' + esc(card.purpose || '-') + '</div>' +
          '<div class="dependency-card-summary">' + esc(compactText(card.summary || '-', 96)) + '</div>' +
          '</div>';
      }}).join('') + '</div>';
    }}
    function stepDescription(step) {{
      const map = {{
        preflight_downstream: '启动预检：检查 Tushare、分钟线接口、WxPusher 与日历 fallback 连通性',
        prepare_baseline: '检查交易日历、股票池、T-1 日线与基础数据',
        seed_prior_artifacts: '检查 T-1 冻结信号/入场文件',
        prior_settlement_skipped: 'T-1 冻结文件缺失，跳过前日结算',
        prior_reconstruction_skipped: '前一交易日纸面买入记录已存在，跳过重建',
        prior_auction_guard: '补齐前一交易日集合竞价侧数据',
        prior_fetch_until_1430: '补齐前一交易日 14:30 前分钟线',
        prior_fetch_1435: '补齐前一交易日 14:35 增量 bar',
        prior_fetch_1440: '补齐前一交易日 14:40 增量 bar',
        prior_fetch_1445: '补齐前一交易日 14:45 增量 bar',
        prior_fetch_1450: '补齐前一交易日 14:50 增量 bar',
        prior_fetch_1455: '补齐前一交易日 14:55 入场 bar',
        prior_build_features: '重建前一交易日尾盘特征',
        prior_build_score: '重建前一交易日 v7 score',
        prior_freeze_signals: '重建前一交易日四线路纸面信号',
        prior_record_entry: '记录前一交易日 14:55 纸面买入 VWAP',
        auction_guard_0925: '检查 T-1 收盘竞价与 T 日开盘竞价数据',
        fetch_exit_0935_bar: '拉取 09:35 退出判断 bar',
        exit_check_0935: '检查第一档止盈/止损卖出条件',
        fetch_exit_0940_bar: '拉取 09:40 纸面卖出执行 bar',
        exit_exec_0940: '记录第一档纸面卖出 VWAP',
        fetch_exit_0945_bar: '拉取 09:45 退出判断 bar',
        exit_check_0945: '检查第二档卖出条件',
        fetch_exit_0950_bar: '拉取 09:50 纸面卖出执行 bar',
        exit_exec_0950: '记录第二档纸面卖出 VWAP',
        fetch_exit_1000_bar: '拉取 10:00 退出判断 bar',
        exit_check_1000: '检查第三档卖出条件',
        fetch_exit_1005_bar: '拉取 10:05 纸面卖出执行 bar',
        exit_exec_1005: '记录第三档纸面卖出 VWAP',
        fetch_exit_1025_bar: '拉取 10:25 兜底卖出预警 bar',
        exit_prealert_1025: '生成 10:30 兜底卖出预提示',
        fetch_exit_1030_bar: '拉取 10:30 默认退出 bar',
        exit_default_1030: '记录未触发提前退出股票的默认卖出',
        auction_guard_1031_refresh: '盘中补拉 T 日开盘竞价（不阻断）',
        auction_guard_1300_refresh: '午后补拉 T 日开盘竞价（不阻断）',
        fetch_tail_until_1430: '拉取 T 日 14:30 前分钟线',
        fetch_tail_1435_bar: '拉取 14:35 增量 bar',
        fetch_tail_1440_bar: '拉取 14:40 增量 bar',
        fetch_tail_1445_bar: '拉取 14:45 增量 bar',
        auction_guard_1445_refresh: '尾盘前补拉 T 日开盘竞价（不阻断）',
        fetch_tail_1450_bar: '拉取 14:50 关键特征 bar',
        auction_guard_1450_required: '尾盘信号前强校验 T 日开盘竞价',
        build_forward_features: '只使用 <=14:50 数据构建今日特征',
        build_score_matrix: '用 v7 locked 模型生成今日 score matrix',
        freeze_signals: '冻结四条线路的今日尾盘纸面买入候选',
        fetch_tail_1455_bar: '拉取 14:55 纸面买入 VWAP bar',
        record_entry_1455_vwap: '记录今日 14:55 纸面买入价',
        fetch_tail_1500_bar: '拉取 15:00 归档 bar'
      }};
      return map[step] || '等待或执行 paper-only 跟踪节点';
    }}
    async function startJob() {{
      const dataMode = $('dataSourceMode').value;
      const payload = {{
        trade_date: dateToYmd($('tradeDate').value),
        mode: $('mode').value,
        requests_per_minute: Number($('rpm').value || 120),
        batch_size: Number($('batchSize').value || 160),
        output_root: $('outputRoot').value,
        run_id: $('runId').value.trim(),
        topic_id: Number($('topicId').value || {topic_id}),
        notification_policy: $('notificationPolicy').value,
        prior_input_policy: $('priorInputPolicy').value,
        send_notifications: $('sendNotifications').checked,
        skip_moneyflow: $('skipMoneyflow').checked,
        use_run_id_output_dir: $('useRunIdOutputDir').checked || dataMode === 'rerun_clean',
        preserve_run_snapshot: $('preserveRunSnapshot').checked,
        force_refresh_minutes: $('forceRefreshMinutes').checked,
        tushare_token: $('tushareToken').value.trim(),
        wxpusher_app_token: $('wxpusherToken').value.trim()
      }};
      const resp = await fetch('/api/start', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(payload)}});
      const data = await resp.json();
      if (!resp.ok) {{
        setMessage(data.error || '启动失败', true);
        return;
      }}
      activeRunJobId = data.job.job_id || '';
      if (activeRunJobId) localStorage.setItem('forwardShadowActiveJobId', activeRunJobId);
      setMessage('任务已启动：' + data.job.job_id + '；run_id=' + (data.job?.meta?.run_id || '-'));
      refresh();
    }}
    async function stopJob() {{
      const resp = await fetch('/api/stop', {{method:'POST'}});
      const data = await resp.json();
      setMessage(data.message || '停止请求已发送', !data.stopped);
      refresh();
    }}
    async function refreshDependencies() {{
      const hasTushareCredential = !!$('tushareToken').value.trim() || !!latestStatusData?.env?.tushare_token_present;
      if (!hasTushareCredential) {{
        setMessage('缺少 TUSHARE_TOKEN：请输入 token 或在启动 dashboard 前配置环境变量后再刷新依赖连通性。', true);
        updateDependencyRefreshAvailability();
        return;
      }}
      const payload = {{
        trade_date: dateToYmd($('tradeDate').value),
        output_root: $('outputRoot').value,
        tushare_token: $('tushareToken').value.trim(),
        wxpusher_app_token: $('wxpusherToken').value.trim(),
        timeout: 8
      }};
      const resp = await fetch('/api/dependency-refresh', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(payload)
      }});
      const data = await resp.json();
      if (!resp.ok) {{
        setMessage(data.error || '依赖探活启动失败', true);
        return;
      }}
      setMessage('依赖连通性刷新已启动：' + (data.job?.job_id || '-'));
      refresh();
    }}
    function updateDependencyRefreshAvailability() {{
      const depJob = latestStatusData?.dependency_monitor || null;
      const depRunning = !!depJob?.running;
      const hasTushareCredential = !!$('tushareToken').value.trim() || !!latestStatusData?.env?.tushare_token_present;
      $('dependencyRefreshBtn').disabled = depRunning || !hasTushareCredential;
      if (!hasTushareCredential) {{
        $('dependencyMonitorStatus').textContent = '监测状态：缺少 token';
        $('dependencyMonitorStatus').className = 'status-chip failed';
        $('dependencyMonitorMeta').textContent = '请输入 TUSHARE_TOKEN，或使用带环境变量的方式启动 dashboard；未带 token 的刷新结果无效。';
      }}
    }}
    function render(data) {{
      latestStatusData = data;
      const job = data.job;
      const depJob = data.dependency_monitor || null;
      const artifacts = data.artifacts || {{}};
      const live = artifacts.live_status || {{}};
      const progress = artifacts.progress || {{}};
      const prior = artifacts.prior_context || {{}};
      const today = artifacts.today_context || {{}};
      const deps = artifacts.dependency_status || {{}};
      const jobStatus = job ? job.status : 'idle';
      const selectedDate = data.display_trade_date || data.requested_trade_date || job?.meta?.trade_date || data.default_trade_date || '-';
      $('jobStatus').textContent = jobStatus;
      $('jobStatus').className = 'pill ' + (jobStatus === 'completed' ? 'ok' : (jobStatus === 'failed' ? 'bad' : (jobStatus === 'running' || jobStatus === 'stopping' ? 'warn' : '')));
      $('modePill').textContent = 'mode: ' + (job?.meta?.mode || '-');
      $('datePill').textContent = 'date: ' + selectedDate;
      $('dataSourcePill').textContent = 'source: ' + (data.display_source || data.data_source_mode || 'history');
      $('runIdPill').textContent = 'run_id: ' + (data.display_run_id || job?.meta?.run_id || '-');
      $('pidPill').textContent = 'pid: ' + (job?.pid || '-');
      const depRunning = !!depJob?.running;
      const depStatus = depJob ? (depJob.status || 'unknown') : 'idle';
      $('dependencyMonitorStatus').textContent = '监测状态：' + (depRunning ? '运行中' : (depJob ? depStatus : '空闲'));
      $('dependencyMonitorStatus').className = 'status-chip ' + (depRunning ? 'running' : (depStatus === 'completed' ? 'success' : (depStatus === 'failed' ? 'failed' : 'pending')));
      $('dependencyMonitorMeta').textContent = depJob
        ? ('job_id=' + (depJob.job_id || '-') + '；开始=' + (depJob.started_at_beijing || '-') + '；结束=' + (depJob.finished_at_beijing || '-'))
        : '未手动刷新';
      updateDependencyRefreshAvailability();
      if (job?.snapshot_path) setMessage('任务快照已保存：' + job.snapshot_path, false);
      if (job?.snapshot_error) setMessage('任务快照保存失败：' + job.snapshot_error, true);
      $('currentStep').textContent = live.current_step_id || '-';
      $('currentStepMeaning').textContent = stepDescription(live.current_step_id || '');
      $('runnerState').textContent = live.status || '-';
      const completed = progress.completed_steps ?? live.completed_steps ?? 0;
      const total = progress.total_steps ?? live.total_steps ?? 29;
      const pct = progress.progress_pct ?? (total ? Math.round(completed / total * 1000) / 10 : 0);
      $('completedSteps').textContent = completed + ' / ' + total + ' (' + pct + '%)';
      $('currentElapsed').textContent = fmtDuration(progress.current_step_elapsed_seconds);
      $('totalSteps').textContent = total;
      $('progressBar').style.width = Math.max(0, Math.min(100, Number(pct) || 0)) + '%';
      $('updatedAt').textContent = live.updated_at_beijing || '-';

      const sellRows = prior.sell_recommendations?.length ? prior.sell_recommendations : (prior.sell_execution || []);
      const compactSell = compactSellRows(sellRows);
      const compactBuy = compactBuyRows(today.buy_signals || []);
      const sellCount = compactSell.length;
      const buyCount = compactBuy.length;
      $('overviewDate').textContent = '执行日期 ' + selectedDate;
      $('overviewHeadline').textContent = jobStatus === 'running'
        ? '运行中'
        : (jobStatus === 'failed' ? '有错误' : (jobStatus === 'completed' ? '已完成' : '待启动 / 可查看历史'));
      $('overviewSubline').textContent = '卖出提示 ' + sellCount + ' 条；尾盘买入候选 ' + buyCount + ' 条；当前节点：' + stepDescription(live.current_step_id || '');
      if (live.status === 'stale_running') {{
        $('overviewHeadline').textContent = '历史状态已过期';
        $('overviewSubline').textContent = live.stale_note || '当前展示的是旧 running 状态文件，不代表任务仍在执行；请切换“最近一次运行结果”查看最新 run_id。';
      }}
      if (data.data_source_mode === 'rerun_clean' && data.display_source === 'rerun_clean_empty') {{
        $('overviewSubline').textContent = data.display_note || '重跑清爽视图：T 日结果等待本次 run 产生。';
      }}
      $('sellActionMeta').textContent = prior.prior_signal_date
        ? ('来自 T-1 信号 ' + prior.prior_signal_date + '，价格展示保留 2 位小数')
        : '未找到 T-1 信号';
      $('buyActionMeta').textContent = today.signals_generated
        ? (today.entries_recorded ? '已记录 14:55 paper entry VWAP' : '已冻结候选，等待 14:55 VWAP')
        : '今日尾盘信号尚未生成';
      $('sellActionTable').innerHTML = table(compactSell, [
        {{key:'lines', label:'线路', type:'lines'}},
        {{key:'exit_time', label:'建议/实际卖出'}},
        {{key:'rank', label:'rank', type:'number'}},
        {{key:'score', label:'score', type:'score'}},
        {{key:'code', label:'股票'}},
        {{key:'name', label:'名称'}},
        {{key:'sell_price', label:'推荐/执行价', type:'price'}},
        {{key:'exit_reason', label:'原因'}},
        {{key:'status', label:'状态'}}
      ]);
      $('buyActionTable').innerHTML = table(compactBuy, [
        {{key:'lines', label:'线路', type:'lines'}},
        {{key:'rank', label:'rank', type:'number'}},
        {{key:'code', label:'股票'}},
        {{key:'name', label:'名称'}},
        {{key:'score', label:'score', type:'score'}},
        {{key:'expected_entry_time', label:'建议买入'}},
        {{key:'entry_vwap', label:'参考买价', type:'price'}},
        {{key:'weight', label:'权重', type:'weight'}}
      ]);

      $('candidateTable').innerHTML = table(artifacts.candidate_status || [], [
        {{key:'strategy_id', label:'line_id'}}, {{key:'tail_down_flag', label:'tail_down'}}, {{key:'selected_count', label:'selected', type:'number'}}, {{key:'no_trade_reason', label:'no_trade'}}
      ]);
      const depSource = deps.source_label || '未标注来源';
      const depSourceNote = deps.source_note || '';
      const depTime = deps.exists
        ? ('最近预检：' + (deps.generated_time_beijing || '-') + '；探活参考交易日：' + (deps.prior_trade_date_for_probe || '-') + '；原始 overall=' + (deps.overall_status || '-'))
        : '尚未生成启动预检文件。';
      $('dependencyMeta').innerHTML = '<span class="source-badge">数据来源：' + esc(depSource) + '</span>' +
        '<span class="source-note">' + esc(depTime + (depSourceNote ? '；' + depSourceNote : '')) + '</span>';
      $('dependencyBanner').innerHTML = renderDependencyBanner(deps);
      $('dependencyCards').innerHTML = renderDependencyCards(deps.cards || []);
      $('dependencyIssues').innerHTML = table(deps.issues || [], [
        {{key:'display_name', label:'异常依赖'}},
        {{key:'status', label:'状态', type:'status'}},
        {{key:'severity', label:'级别', type:'status'}},
        {{key:'role', label:'用途'}},
        {{key:'impact', label:'影响/错误', type:'message'}},
        {{key:'duration_seconds', label:'耗时', type:'seconds'}}
      ]);
      $('dependencyTable').innerHTML = table(deps.checks || [], [
        {{key:'category', label:'类别'}},
        {{key:'display_name', label:'依赖/检查项'}},
        {{key:'status', label:'状态', type:'status'}},
        {{key:'severity', label:'级别', type:'status'}},
        {{key:'role', label:'用途'}},
        {{key:'duration_seconds', label:'耗时', type:'seconds'}},
        {{key:'impact', label:'影响/错误', type:'message'}}
      ]);
      $('signalSummary').innerHTML = table(artifacts.signal_summary || [], [
        {{key:'strategy_id', label:'line_id'}}, {{key:'rows', label:'rows', type:'number'}}, {{key:'selected', label:'selected', type:'number'}}, {{key:'avg_score', label:'avg_score', type:'score'}}
      ]);
      const entryRows = Object.entries(artifacts.entry_counts || {{}}).map(([status, count]) => ({{status, count}}));
      $('entryCounts').innerHTML = table(entryRows, [{{key:'status', label:'状态'}}, {{key:'count', label:'数量', type:'number'}}]);
      $('priorNotice').textContent = prior.prior_signal_date
        ? ('T-1 signal_date: ' + prior.prior_signal_date + (prior.message ? '；' + prior.message : ''))
        : (prior.message || '未找到前一交易日纸面选股信息。');
      $('priorEntriesTable').innerHTML = table(prior.entries || [], [
        {{key:'line', label:'线路'}}, {{key:'original_v7_rank', label:'rank', type:'number'}}, {{key:'score', label:'score', type:'score'}}, {{key:'code', label:'股票'}}, {{key:'name', label:'名称'}}, {{key:'entry_vwap', label:'买入VWAP', type:'price'}}, {{key:'weight', label:'权重', type:'weight'}}, {{key:'paper_entry_status', label:'状态'}}
      ]);
      $('sellRecommendationsTable').innerHTML = table(prior.sell_recommendations || [], [
        {{key:'line', label:'线路'}}, {{key:'decision_time', label:'判断'}}, {{key:'expected_exit_time', label:'建议卖出'}}, {{key:'original_v7_rank', label:'rank', type:'number'}}, {{key:'score', label:'score', type:'score'}}, {{key:'code', label:'股票'}}, {{key:'name', label:'名称'}}, {{key:'recommended_sell_price', label:'推荐卖价', type:'price'}}, {{key:'exit_reason', label:'原因'}}, {{key:'recommendation_status', label:'状态'}}
      ]);
      $('sellExecutionTable').innerHTML = table(prior.sell_execution || [], [
        {{key:'line', label:'线路'}}, {{key:'actual_exit_time', label:'卖出时间'}}, {{key:'original_v7_rank', label:'rank', type:'number'}}, {{key:'score', label:'score', type:'score'}}, {{key:'code', label:'股票'}}, {{key:'name', label:'名称'}}, {{key:'entry_vwap', label:'买入', type:'price'}}, {{key:'exit_vwap', label:'卖出', type:'price'}}, {{key:'return_10bp_impact', label:'10bp+impact', type:'pct'}}, {{key:'paper_exit_status', label:'状态'}}
      ]);
      $('settlementSummaryTable').innerHTML = table(prior.settlement_summary || [], [
        {{key:'line', label:'线路'}}, {{key:'positions', label:'笔数', type:'number'}}, {{key:'daily_return_5bp', label:'5bp日收益', type:'pct'}}, {{key:'daily_return_10bp_impact', label:'10bp+impact日收益', type:'pct'}}
      ]);
      $('todayBuyNotice').textContent = today.signals_generated
        ? (today.entries_recorded ? '今日已生成信号并记录 14:55 纸面买入价。' : '今日已冻结尾盘候选，14:55 买入价尚未记录。')
        : '今日尾盘选股信号尚未生成。';
      $('todayBuyTable').innerHTML = table(today.buy_signals || [], [
        {{key:'line', label:'线路'}}, {{key:'rank', label:'rank', type:'number'}}, {{key:'code', label:'股票'}}, {{key:'name', label:'名称'}}, {{key:'score', label:'score', type:'score'}}, {{key:'expected_entry_time', label:'建议买入'}}, {{key:'entry_vwap', label:'买入VWAP', type:'price'}}, {{key:'entry_status', label:'状态'}}, {{key:'weight', label:'权重', type:'weight'}}
      ]);
      const stepRows = (artifacts.steps || []).map(r => Object.assign({{}}, r, {{
        step_description: stepDescription(r.step_id || '')
      }}));
      $('stepsTable').innerHTML = table(stepRows, [
        {{key:'index', label:'#', type:'number'}},
        {{key:'scheduled_time', label:'scheduled'}},
        {{key:'status', label:'status', type:'status'}},
        {{key:'step_description', label:'任务说明'}},
        {{key:'step_id', label:'step_id'}},
        {{key:'running_elapsed_seconds', label:'running', type:'seconds'}},
        {{key:'duration_seconds', label:'finished', type:'seconds'}},
        {{key:'return_code', label:'rc', type:'number'}},
        {{key:'message', label:'校验摘要/说明', type:'message'}}
      ]);
      const files = Object.entries(artifacts.files || {{}}).map(([name, path]) => ({{name, path, exists: artifacts.file_exists?.[name] ? 'yes' : 'no'}}));
      $('filesTable').innerHTML = table(files, ['name','exists','path']);
      $('logTail').textContent = (job?.log_tail || []).join('\\n');
      const hasError = !!(job?.error || job?.status === 'failed');
      $('errorPanel').style.display = hasError ? 'block' : 'none';
      $('errorText').textContent = hasError ? ((job?.error || 'runner failed') + '；return_code=' + (job?.return_code ?? '-')) : '';
      $('errorTail').textContent = hasError ? ((job?.last_error_lines || job?.log_tail || []).join('\\n')) : '';
      $('startBtn').disabled = !!job?.running;
      $('stopBtn').disabled = !job?.running;
    }}
    async function refresh() {{
      try {{
        const params = new URLSearchParams();
        params.set('trade_date', dateToYmd($('tradeDate').value));
        params.set('output_root', $('outputRoot').value);
        params.set('data_source_mode', $('dataSourceMode').value);
        params.set('prior_input_policy', $('priorInputPolicy').value);
        if ($('dataSourceMode').value === 'rerun_clean' && activeRunJobId) params.set('active_job_id', activeRunJobId);
        const resp = await fetch('/api/status?' + params.toString());
        render(await resp.json());
      }} catch (e) {{
        setMessage(String(e), true);
      }}
    }}
    $('startBtn').addEventListener('click', startJob);
    document.querySelectorAll('.tab-btn').forEach(btn => {{
      btn.addEventListener('click', () => {{
        const tab = btn.dataset.tab;
        document.querySelectorAll('.tab-btn').forEach(x => x.classList.toggle('active', x === btn));
        document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.toggle('active', panel.id === 'tab-' + tab));
      }});
    }});
    $('showTokens').addEventListener('change', () => {{
      const typ = $('showTokens').checked ? 'text' : 'password';
      $('tushareToken').type = typ;
      $('wxpusherToken').type = typ;
    }});
    $('stopBtn').addEventListener('click', stopJob);
    $('refreshBtn').addEventListener('click', refresh);
    $('dependencyRefreshBtn').addEventListener('click', refreshDependencies);
    $('tushareToken').addEventListener('input', updateDependencyRefreshAvailability);
    $('wxpusherToken').addEventListener('input', updateDependencyRefreshAvailability);
    function clearActiveRunAndRefresh() {{
      activeRunJobId = '';
      localStorage.removeItem('forwardShadowActiveJobId');
      refresh();
    }}
    $('tradeDate').addEventListener('change', clearActiveRunAndRefresh);
    $('outputRoot').addEventListener('change', clearActiveRunAndRefresh);
    $('dataSourceMode').addEventListener('change', () => {{
      if ($('dataSourceMode').value === 'rerun_clean') {{
        activeRunJobId = '';
        localStorage.removeItem('forwardShadowActiveJobId');
        $('useRunIdOutputDir').checked = true;
        setMessage('重跑清爽视图会自动使用新的 run_id 输出目录；启动前不会读取同日期旧 run 结果。');
      }}
      refresh();
    }});
    $('priorInputPolicy').addEventListener('change', refresh);
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
                    dependency_job = state.current_dependency_job()
                    dependency_public = public_job(dependency_job)
                trade_date = None
                base_output_root = state.output_root
                if public and public.get("meta"):
                    trade_date = public["meta"].get("trade_date")
                if params.get("trade_date"):
                    trade_date = normalize_trade_date(params["trade_date"][0])
                if params.get("output_root"):
                    base_output_root = resolve_output_root(params["output_root"][0], state.output_root)
                trade_date = trade_date or today_ymd()
                data_source_mode = str((params.get("data_source_mode") or ["history"])[0] or "history")
                if data_source_mode not in {"history", "latest_run", "rerun_clean"}:
                    data_source_mode = "history"
                active_job_id = str((params.get("active_job_id") or [""])[0] or "")
                prior_input_policy = str((params.get("prior_input_policy") or ["reuse_only"])[0] or "reuse_only")
                if prior_input_policy not in {"reuse_or_rebuild", "reuse_only", "force_rebuild"}:
                    prior_input_policy = "reuse_only"
                display_trade_date = trade_date
                display_output_root = base_output_root
                prior_entry_root: Path | None = None
                plan_hint = "auto"
                display_source = "history"
                display_note = "展示所选日期在当前输出目录下已有的历史结果。"
                display_run_id = ""
                dependency_source_label = "历史落地结果"
                dependency_source_note = "读取所选日期已保存的 preflight/data_guards 文件；不是当前刚刚重新探活。"
                suppress_dependency_history = False
                if data_source_mode == "latest_run":
                    ctx = latest_run_context(base_output_root, public, include_base=True)
                    if ctx:
                        display_trade_date = str(ctx["trade_date"])
                        display_output_root = Path(ctx["output_root"])
                        display_source = str(ctx.get("source") or "latest_run")
                        display_run_id = str(ctx.get("run_id") or "")
                        display_note = "展示最近一次 dashboard run 或最近状态文件对应的结果。"
                        dependency_source_label = "最近一次运行结果"
                        dependency_source_note = "读取最近一次运行目录里的依赖检查结果；是否实时取决于该 run 的启动时间。"
                elif data_source_mode == "rerun_clean":
                    ctx = current_job_context(base_output_root, public, trade_date=trade_date, active_job_id=active_job_id)
                    plan_hint = "rebuild" if prior_input_policy == "force_rebuild" else "seed"
                    prior_entry_root = None if prior_input_policy == "force_rebuild" else base_output_root
                    if ctx:
                        display_output_root = Path(ctx["output_root"])
                        display_source = str(ctx.get("source") or "rerun_clean_run")
                        display_run_id = str(ctx.get("run_id") or "")
                        dependency_source_label = "本次重跑输出"
                        dependency_source_note = "只读取当前 run_id 输出目录里的依赖检查结果。"
                        if prior_input_policy == "force_rebuild":
                            display_note = "强制重建 T-1：T-1 入场和 T 日结果都只读取本次 run 输出目录。"
                        else:
                            display_note = "T-1 入场只从已冻结历史基线读取；缺失则跳过前日结算，T 日结果只读取本次 run 输出目录。"
                    else:
                        display_output_root = base_output_root / "runs" / "__rerun_clean_waiting__"
                        display_source = "rerun_clean_empty"
                        dependency_source_label = "本次重跑尚未启动"
                        dependency_source_note = "清爽视图启动前不读取历史依赖结果。"
                        suppress_dependency_history = True
                        if prior_input_policy == "force_rebuild":
                            display_note = "强制重建 T-1：尚未启动本次 run；T-1 入场、T 日卖出和尾盘选股都保持空白。"
                        else:
                            display_note = "尚未启动本次重跑或当前页面没有本次 job_id；只展示已冻结 T-1 历史入场，缺失不会事后重建。"
                elif public and public.get("meta") and str(public["meta"].get("trade_date") or "") == trade_date:
                    meta_output_root = resolve_output_root(str(public["meta"].get("output_root") or ""), base_output_root)
                    if meta_output_root == display_output_root:
                        dependency_source_label = "当前页面任务输出"
                        dependency_source_note = "读取当前 dashboard 任务输出目录；任务启动后会刷新为本次预检结果。"
                if dependency_public and dependency_public.get("meta"):
                    dep_meta = dependency_public["meta"]
                    dep_output_root = resolve_output_root(str(dep_meta.get("output_root") or ""), base_output_root)
                    if str(dep_meta.get("trade_date") or "") == display_trade_date and dep_output_root == display_output_root:
                        if dependency_public.get("running"):
                            dependency_source_label = "手动探活运行中"
                            dependency_source_note = "正在重新执行依赖连通性检查；完成后会刷新本区域结果。"
                        elif dependency_public.get("status") == "completed":
                            dependency_source_label = "刚刚手动刷新"
                            dependency_source_note = "读取最近一次手动刷新生成的 preflight 文件。"
                        elif dependency_public.get("status") == "failed":
                            dependency_source_label = "手动探活失败"
                            dependency_source_note = "最近一次手动刷新失败；请查看监测状态和日志。"
                payload = {
                    "default_trade_date": today_ymd(),
                    "requested_trade_date": trade_date,
                    "requested_output_root": str(base_output_root),
                    "display_trade_date": display_trade_date,
                    "display_output_root": str(display_output_root),
                    "display_source": display_source,
                    "display_run_id": display_run_id,
                    "display_note": display_note,
                    "data_source_mode": data_source_mode,
                    "prior_input_policy": prior_input_policy,
                    "min_trade_date": MIN_TRADE_DATE,
                    "beijing_now": bj_now().isoformat(timespec="seconds"),
                    "git": {
                        "branch": git_value(["branch", "--show-current"]),
                        "head": git_value(["rev-parse", "HEAD"]),
                        "v7_locked": git_value(["rev-parse", "v7_locked"]),
                    },
                    "env": env_health(send_notifications=True),
                    "job": public,
                    "dependency_monitor": dependency_public,
                    "artifacts": dashboard_artifacts(
                        display_output_root,
                        display_trade_date,
                        prior_entry_root=prior_entry_root,
                        plan_hint=plan_hint,
                        dependency_source_label=dependency_source_label,
                        dependency_source_note=dependency_source_note,
                        suppress_dependency_history=suppress_dependency_history,
                    ),
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
            if self.path == "/api/dependency-refresh":
                try:
                    job = start_dependency_probe(state, self.read_body_json())
                    write_json_response(self, HTTPStatus.OK, {"job": public_job(job)})
                except Exception as exc:
                    write_json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
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
