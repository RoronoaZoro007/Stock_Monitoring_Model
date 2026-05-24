#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "tushare" / "v8_forward_shadow" / "provider_chain_20260521_20260522"
DEFAULT_TOPIC_ID = 44635


@dataclass
class ChainStep:
    step_id: str
    logical_time: str
    status: str
    duration_seconds: float
    return_code: int
    stdout_tail: str
    stderr_tail: str
    command: list[str]


def bj_now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def redact_env_text(text: str) -> str:
    out = text or ""
    for key in ["TUSHARE_TOKEN", "WXPUSHER_APP_TOKEN"]:
        token = os.environ.get(key, "").strip()
        if token:
            out = out.replace(token, f"{token[:6]}...{token[-4:]}")
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_step(step_id: str, logical_time: str, cmd: list[str], cwd: Path) -> ChainStep:
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    elapsed = time.monotonic() - started
    return ChainStep(
        step_id=step_id,
        logical_time=logical_time,
        status="success" if proc.returncode == 0 else "failed",
        duration_seconds=round(elapsed, 3),
        return_code=proc.returncode,
        stdout_tail=redact_env_text(proc.stdout[-4000:]),
        stderr_tail=redact_env_text(proc.stderr[-4000:]),
        command=cmd,
    )


def send_wxpusher(title: str, content: str, topic_id: int, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"status": "dry_run", "title": title}
    token = os.environ.get("WXPUSHER_APP_TOKEN", "").strip()
    if not token:
        return {"status": "skipped_missing_wxpusher_token", "title": title}
    payload = {
        "appToken": token,
        "content": content,
        "summary": title[:100],
        "contentType": 3,
        "topicIds": [int(topic_id)],
        "verifyPay": False,
    }
    resp = requests.post("https://wxpusher.zjiecode.com/api/send/message", json=payload, timeout=20)
    return {
        "status": "success" if resp.status_code == 200 else "failed",
        "http_status": resp.status_code,
        "body": resp.text[:1000],
    }


def ensure_env() -> None:
    missing = [key for key in ["TUSHARE_TOKEN", "WXPUSHER_APP_TOKEN"] if not os.environ.get(key, "").strip()]
    if missing:
        raise SystemExit(
            "missing required environment variables: "
            + ", ".join(missing)
            + ". Do not write tokens into code or command logs; export them in the shell before running."
        )


def step_message(step: ChainStep) -> str:
    rows = [
        "| 字段 | 值 |",
        "|---|---:|",
        f"| step | {step.step_id} |",
        f"| logical_time | {step.logical_time} |",
        f"| status | {step.status} |",
        f"| duration_seconds | {step.duration_seconds:.3f} |",
        f"| return_code | {step.return_code} |",
    ]
    body = ["## Forward Shadow Provider Chain", "\n".join(rows), "", "paper-only，不下单，不构成交易建议。"]
    if step.stderr_tail:
        body.extend(["", "### stderr tail", "```text", step.stderr_tail[-1200:], "```"])
    return "\n".join(body)


def append_step(steps: list[ChainStep], pushes: list[dict[str, Any]], step: ChainStep, topic_id: int, dry_run_push: bool) -> None:
    steps.append(step)
    push = send_wxpusher(
        f"FS provider chain {step.logical_time} {step.step_id}: {step.status}",
        step_message(step),
        topic_id,
        dry_run_push,
    )
    pushes.append({"step_id": step.step_id, "logical_time": step.logical_time, **push})
    if step.status != "success":
        raise SystemExit(json.dumps(asdict(step), ensure_ascii=False, indent=2))


def tail_replay_cmd(
    trade_date: str,
    prior_trade_date: str,
    output_root: Path,
    topic_id: int,
    dry_run_push: bool,
    requests_per_minute: int,
    batch_size: int,
) -> list[str]:
    cmd = [
        sys.executable,
        "research/v8_research/forward_shadow_full_flow_replay.py",
        "--trade-date",
        trade_date,
        "--prior-trade-date",
        prior_trade_date,
        "--output-root",
        str(output_root),
        "--data-mode",
        "provider",
        "--provider-requests-per-minute",
        str(requests_per_minute),
        "--provider-batch-size",
        str(batch_size),
        "--topic-id",
        str(topic_id),
    ]
    if dry_run_push:
        cmd.append("--dry-run-push")
    return cmd


def minute_fetch_cmd(trade_date: str, signal_date: str, output_root: Path, minute_dir: Path, bar_time: str, requests_per_minute: int) -> list[str]:
    return [
        sys.executable,
        "research/v8_research/forward_shadow_data_guard.py",
        "minute-fetch",
        "--trade-date",
        trade_date,
        "--signal-date",
        signal_date,
        "--output-root",
        str(output_root),
        "--minute-dir",
        str(minute_dir),
        "--codes-source",
        "prior_entries",
        "--bar-time",
        bar_time,
        "--mode",
        "bar",
        "--requests-per-minute",
        str(requests_per_minute),
        "--retry-until-complete-seconds",
        "40",
    ]


def exit_monitor_cmd(signal_date: str, settlement_date: str, output_root: Path, minute_dir: Path, checkpoint: str) -> list[str]:
    return [
        sys.executable,
        "research/v8_research/forward_shadow_exit_monitor.py",
        "--signal-date",
        signal_date,
        "--settlement-date",
        settlement_date,
        "--output-root",
        str(output_root),
        "--minute-dir",
        str(minute_dir),
        "--checkpoint",
        checkpoint,
        "--send-notifications",
    ]


def file_hashes(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        if path.exists() and path.is_file():
            rows.append({"file": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 20260521/20260522 provider-backed paper tracking chain.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--topic-id", type=int, default=DEFAULT_TOPIC_ID)
    parser.add_argument("--dry-run-push", action="store_true")
    parser.add_argument("--requests-per-minute", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=160)
    args = parser.parse_args()

    ensure_env()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tail_root = output_root / "tail_sessions"
    paper_20260521 = tail_root / "20260521" / "paper_outputs"
    minute_20260522 = tail_root / "20260522" / "staged_raw" / "stk_mins" / "freq=5min"
    report_dir = output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    steps: list[ChainStep] = []
    pushes: list[dict[str, Any]] = []

    append_step(
        steps,
        pushes,
        run_step(
            "20260521_tail_full_flow",
            "20260521 14:30-15:00",
            tail_replay_cmd("20260521", "20260520", tail_root, args.topic_id, args.dry_run_push, args.requests_per_minute, args.batch_size),
            ROOT,
        ),
        args.topic_id,
        args.dry_run_push,
    )

    exit_pairs = [
        ("fetch_exit_0935_bar", "20260522 09:35:05", minute_fetch_cmd("20260522", "20260521", paper_20260521, minute_20260522, "09:35", args.requests_per_minute)),
        ("exit_check_0935", "20260522 09:35:50", exit_monitor_cmd("20260521", "20260522", paper_20260521, minute_20260522, "check_0935")),
        ("fetch_exit_0940_bar", "20260522 09:40:05", minute_fetch_cmd("20260522", "20260521", paper_20260521, minute_20260522, "09:40", args.requests_per_minute)),
        ("exit_exec_0940", "20260522 09:40:50", exit_monitor_cmd("20260521", "20260522", paper_20260521, minute_20260522, "exec_0940")),
        ("fetch_exit_0945_bar", "20260522 09:45:05", minute_fetch_cmd("20260522", "20260521", paper_20260521, minute_20260522, "09:45", args.requests_per_minute)),
        ("exit_check_0945", "20260522 09:45:50", exit_monitor_cmd("20260521", "20260522", paper_20260521, minute_20260522, "check_0945")),
        ("fetch_exit_0950_bar", "20260522 09:50:05", minute_fetch_cmd("20260522", "20260521", paper_20260521, minute_20260522, "09:50", args.requests_per_minute)),
        ("exit_exec_0950", "20260522 09:50:50", exit_monitor_cmd("20260521", "20260522", paper_20260521, minute_20260522, "exec_0950")),
        ("fetch_exit_1000_bar", "20260522 10:00:05", minute_fetch_cmd("20260522", "20260521", paper_20260521, minute_20260522, "10:00", args.requests_per_minute)),
        ("exit_check_1000", "20260522 10:00:50", exit_monitor_cmd("20260521", "20260522", paper_20260521, minute_20260522, "check_1000")),
        ("fetch_exit_1005_bar", "20260522 10:05:05", minute_fetch_cmd("20260522", "20260521", paper_20260521, minute_20260522, "10:05", args.requests_per_minute)),
        ("exit_exec_1005", "20260522 10:05:50", exit_monitor_cmd("20260521", "20260522", paper_20260521, minute_20260522, "exec_1005")),
        ("fetch_exit_1025_bar", "20260522 10:25:05", minute_fetch_cmd("20260522", "20260521", paper_20260521, minute_20260522, "10:25", args.requests_per_minute)),
        ("exit_prealert_1025", "20260522 10:25:50", exit_monitor_cmd("20260521", "20260522", paper_20260521, minute_20260522, "prealert_1025")),
        ("fetch_exit_1030_bar", "20260522 10:30:05", minute_fetch_cmd("20260522", "20260521", paper_20260521, minute_20260522, "10:30", args.requests_per_minute)),
        ("exit_default_1030", "20260522 10:30:50", exit_monitor_cmd("20260521", "20260522", paper_20260521, minute_20260522, "default_1030")),
    ]
    for step_id, logical_time, cmd in exit_pairs:
        append_step(steps, pushes, run_step(step_id, logical_time, cmd, ROOT), args.topic_id, args.dry_run_push)

    append_step(
        steps,
        pushes,
        run_step(
            "20260522_tail_full_flow",
            "20260522 14:30-15:00",
            tail_replay_cmd("20260522", "20260521", tail_root, args.topic_id, args.dry_run_push, args.requests_per_minute, args.batch_size),
            ROOT,
        ),
        args.topic_id,
        args.dry_run_push,
    )

    steps_path = report_dir / "provider_chain_step_timings.csv"
    pd.DataFrame([asdict(step) | {"generated_time_beijing": bj_now()} for step in steps]).to_csv(steps_path, index=False)
    pushes_path = report_dir / "provider_chain_push_logs.csv"
    pd.DataFrame(pushes).to_csv(pushes_path, index=False)
    expected_outputs = [
        steps_path,
        pushes_path,
        tail_root / "20260521" / "reports" / "20260521_full_flow_summary.json",
        tail_root / "20260521" / "paper_outputs" / "daily_signals" / "20260521_signals.csv",
        tail_root / "20260521" / "paper_outputs" / "daily_entry_prices" / "20260521_entry_prices.csv",
        tail_root / "20260521" / "paper_outputs" / "daily_exit_execution" / "20260521_20260522_exit_execution.csv",
        tail_root / "20260521" / "paper_outputs" / "daily_ledgers" / "20260521_20260522_exit_ledgers.csv",
        tail_root / "20260522" / "reports" / "20260522_full_flow_summary.json",
        tail_root / "20260522" / "paper_outputs" / "daily_signals" / "20260522_signals.csv",
        tail_root / "20260522" / "paper_outputs" / "daily_entry_prices" / "20260522_entry_prices.csv",
    ]
    sha_path = report_dir / "provider_chain_sha256.csv"
    file_hashes(expected_outputs).to_csv(sha_path, index=False)
    summary = {
        "generated_time_beijing": bj_now(),
        "paper_only": True,
        "uses_real_provider": True,
        "does_not_use_existing_20260521_20260522_outputs": True,
        "output_root": str(output_root),
        "step_timings": str(steps_path),
        "push_logs": str(pushes_path),
        "sha256": str(sha_path),
        "steps": [asdict(step) for step in steps],
    }
    summary_path = report_dir / "provider_chain_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    send_wxpusher(
        "FS provider chain 20260521-20260522 completed",
        "## Provider chain completed\n\n"
        + f"- output_root: `{output_root}`\n"
        + f"- steps: `{steps_path}`\n"
        + f"- sha256: `{sha_path}`\n\npaper-only，不下单，不构成交易建议。",
        args.topic_id,
        args.dry_run_push,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
