#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "tushare" / "v8_forward_shadow"
DEFAULT_RANK_FILE = ROOT / "data_tushare" / "manifests" / "liquid_top3000_20251120_20260213.csv"
DEFAULT_DAILY_FILE = ROOT / "data_tushare" / "clean" / "daily_repaired_top3000.parquet"
DEFAULT_MINUTE_DIR = ROOT / "data_tushare" / "raw" / "stk_mins" / "freq=5min"
DEFAULT_AUCTION_RAW_DIR = ROOT / "data_tushare" / "raw"
DEFAULT_MODEL_FILE = ROOT / "locked_artifacts" / "v7_cap20_strong_label_003" / "model" / "model.pkl"
BEIJING_TZ = timezone(timedelta(hours=8))


@dataclass
class StepResult:
    step_id: str
    scheduled_time: str
    status: str
    duration_seconds: float
    return_code: int
    stdout_tail: str
    stderr_tail: str
    command: list[str]
    message: str = ""


def bj_now() -> datetime:
    return datetime.now(BEIJING_TZ)


def today_ymd() -> str:
    return bj_now().strftime("%Y%m%d")


def ymd_iso(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def parse_hms(value: str) -> datetime:
    now = bj_now()
    h, m, s = [int(x) for x in value.split(":")]
    return now.replace(hour=h, minute=m, second=s, microsecond=0)


def wait_until(hms: str, no_wait: bool) -> None:
    if no_wait:
        return
    target = parse_hms(hms)
    delay = (target - bj_now()).total_seconds()
    if delay > 0:
        time.sleep(delay)


def redact(text: str) -> str:
    out = text or ""
    for key in ["TUSHARE_TOKEN", "WXPUSHER_APP_TOKEN"]:
        token = os.environ.get(key, "").strip()
        if token:
            out = out.replace(token, f"{token[:6]}...{token[-4:]}")
    return out


def send_wxpusher(title: str, content: str, topic_id: int, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"status": "disabled"}
    token = os.environ.get("WXPUSHER_APP_TOKEN", "").strip()
    if not token:
        return {"status": "skipped_missing_wxpusher_token"}
    payload = {
        "appToken": token,
        "content": content,
        "summary": title[:100],
        "contentType": 3,
        "topicIds": [int(topic_id)],
        "verifyPay": False,
    }
    try:
        resp = requests.post("https://wxpusher.zjiecode.com/api/send/message", json=payload, timeout=20)
        return {"status": "success" if resp.status_code == 200 else "failed", "http_status": resp.status_code, "body": resp.text[:1000]}
    except requests.RequestException as exc:
        return {"status": "failed", "error": repr(exc)[:500]}


def command_result_message(step: StepResult, extra: str = "") -> str:
    rows = [
        "| 字段 | 值 |",
        "|---|---:|",
        f"| step | {step.step_id} |",
        f"| scheduled_time | {step.scheduled_time} |",
        f"| status | {step.status} |",
        f"| duration_seconds | {step.duration_seconds:.3f} |",
        f"| return_code | {step.return_code} |",
    ]
    body = ["# Forward Shadow Live Runner", "\n".join(rows)]
    if step.message:
        body.extend(["", step.message])
    if extra:
        body.extend(["", extra])
    body.extend(["", "paper-only，不下单，不构成交易建议。"])
    if step.stderr_tail and step.status != "success":
        body.extend(["", "```text", step.stderr_tail[-1500:], "```"])
    return "\n".join(body)


def run_cmd(step_id: str, scheduled_time: str, cmd: list[str], push: bool, topic_id: int, extra_summary: str = "") -> StepResult:
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    result = StepResult(
        step_id=step_id,
        scheduled_time=scheduled_time,
        status="success" if proc.returncode == 0 else "failed",
        duration_seconds=round(time.monotonic() - started, 3),
        return_code=proc.returncode,
        stdout_tail=redact(proc.stdout[-5000:]),
        stderr_tail=redact(proc.stderr[-5000:]),
        command=cmd,
        message=extra_summary,
    )
    send_wxpusher(
        f"FS live {scheduled_time} {step_id}: {result.status}",
        command_result_message(result),
        topic_id,
        push,
    )
    if result.status != "success":
        raise SystemExit(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return result


def python_cmd(script: str, *args: str) -> list[str]:
    return [sys.executable, script, *args]


def env_check(send_notifications: bool) -> None:
    missing = [k for k in ["TUSHARE_TOKEN"] if not os.environ.get(k, "").strip()]
    if send_notifications and not os.environ.get("WXPUSHER_APP_TOKEN", "").strip():
        missing.append("WXPUSHER_APP_TOKEN")
    if missing:
        raise SystemExit(
            "Missing environment variables: "
            + ", ".join(missing)
            + ". Export them in shell; never commit tokens."
        )
    os.environ.setdefault("TUSHARE_PROXY_URL", "http://tsy.xiaodefa.cn")


def parse_prepare_summary(stdout: str) -> dict[str, Any]:
    try:
        return json.loads(stdout[stdout.find("{") :])
    except Exception:
        return {}


def entry_file_ready(output_root: Path, trade_date: str) -> bool:
    path = output_root / "daily_entry_prices" / f"{trade_date}_entry_prices.csv"
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if df.empty or "paper_entry_status" not in df.columns:
        return False
    return bool(df["paper_entry_status"].astype(str).eq("entry_recorded").any())


def route_summary(output_root: Path, trade_date: str) -> str:
    path = output_root / "forward_shadow_candidate_status.csv"
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    df["trade_date"] = df["trade_date"].astype(str)
    day = df[df["trade_date"].eq(trade_date)].copy()
    if day.empty:
        return ""
    cols = ["strategy_id", "tail_down_flag", "selected_count", "no_trade_reason"]
    cols = [c for c in cols if c in day.columns]
    return "## 四线路状态\n\n" + day[cols].to_markdown(index=False)


def entry_summary(output_root: Path, trade_date: str) -> str:
    path = output_root / "daily_entry_prices" / f"{trade_date}_entry_prices.csv"
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    counts = df.get("paper_entry_status", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
    return "## Entry 记录\n\n" + "\n".join([f"- {k}: {v}" for k, v in counts.items()])


def write_live_status(
    output_root: Path,
    trade_date: str,
    status: str,
    current_step_id: str = "",
    scheduled_time: str = "",
    completed_steps: int = 0,
    total_steps: int | None = None,
    message: str = "",
    no_wait: bool = False,
) -> Path:
    out_dir = output_root / "live_runner"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{trade_date}_live_runner_status.json"
    payload = {
        "trade_date": trade_date,
        "status": status,
        "current_step_id": current_step_id,
        "current_scheduled_time": scheduled_time,
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "message": message,
        "no_wait": bool(no_wait),
        "updated_at_beijing": bj_now().isoformat(timespec="seconds"),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def reconstruct_prior_if_needed(args: argparse.Namespace, trade_date: str, prior: str, prior2: str, results: list[StepResult]) -> None:
    output_root = Path(args.output_root)
    if entry_file_ready(output_root, prior):
        write_live_status(
            output_root,
            trade_date,
            "running",
            current_step_id="prior_reconstruction_skipped",
            completed_steps=len(results),
            message=f"{prior} prior entry file already exists; reconstruction skipped.",
            no_wait=bool(args.no_wait),
        )
        send_wxpusher(
            f"FS prior {prior} entry ready",
            f"# Forward Shadow\n\n{prior} prior entry file already exists; reconstruction skipped.\n\npaper-only。",
            args.topic_id,
            bool(args.send_notifications),
        )
        return
    steps = [
        (
            "prior_auction_guard",
            "07:40:00",
            python_cmd(
                "research/v8_research/forward_shadow_data_guard.py",
                "auction-guard",
                "--trade-date",
                prior,
                "--prior-trade-date",
                prior2,
                "--output-root",
                str(output_root),
                "--auction-raw-dir",
                str(args.auction_raw_dir),
                "--rank-file",
                str(args.rank_file),
                "--top-rank",
                "3000",
                "--requests-per-minute",
                str(args.requests_per_minute),
            ),
        ),
        (
            "prior_fetch_until_1430",
            "07:41:00",
            minute_fetch_cmd(prior, None, output_root, Path(args.minute_dir), "top3000", "14:30", "until", args),
        ),
    ]
    for bar in ["14:35", "14:40", "14:45", "14:50", "14:55"]:
        steps.append((f"prior_fetch_{bar.replace(':', '')}", "07:42:00", minute_fetch_cmd(prior, None, output_root, Path(args.minute_dir), "top3000", bar, "bar", args)))
    steps.extend(
        [
            (
                "prior_build_features",
                "07:45:00",
                build_features_cmd(prior, output_root, args),
            ),
            (
                "prior_build_score",
                "07:46:00",
                build_score_cmd(prior, output_root, args),
            ),
            (
                "prior_freeze_signals",
                "07:47:00",
                freeze_cmd(prior, output_root, args),
            ),
            (
                "prior_record_entry",
                "07:48:00",
                record_entry_cmd(prior, output_root, args),
            ),
        ]
    )
    for step_id, scheduled, cmd in steps:
        write_live_status(
            output_root,
            trade_date,
            "running",
            current_step_id=step_id,
            scheduled_time=scheduled,
            completed_steps=len(results),
            message=f"Preparing prior entry state for {prior}.",
            no_wait=bool(args.no_wait),
        )
        result = run_cmd(step_id, scheduled, cmd, bool(args.send_notifications), int(args.topic_id))
        results.append(result)
        write_results(results, output_root, trade_date)
        write_live_status(
            output_root,
            trade_date,
            "running",
            current_step_id=step_id,
            scheduled_time=scheduled,
            completed_steps=len(results),
            message=f"Completed prior reconstruction step {step_id}.",
            no_wait=bool(args.no_wait),
        )
    send_wxpusher(
        f"FS prior {prior} reconstruction completed",
        f"# Forward Shadow prior reconstruction\n\n{prior} entry is prepared.\n\n{route_summary(output_root, prior)}\n\n{entry_summary(output_root, prior)}",
        args.topic_id,
        bool(args.send_notifications),
    )


def minute_fetch_cmd(trade_date: str, signal_date: str | None, output_root: Path, minute_dir: Path, codes_source: str, bar_time: str, mode: str, args: argparse.Namespace) -> list[str]:
    cmd = python_cmd(
        "research/v8_research/forward_shadow_data_guard.py",
        "minute-fetch",
        "--trade-date",
        trade_date,
        "--output-root",
        str(output_root),
        "--minute-dir",
        str(minute_dir),
        "--codes-source",
        codes_source,
        "--bar-time",
        bar_time,
        "--mode",
        mode,
        "--requests-per-minute",
        str(args.requests_per_minute),
        "--batch-size",
        str(args.batch_size),
        "--retry-until-complete-seconds",
        "40",
    )
    if codes_source == "top3000":
        cmd.extend(["--rank-file", str(args.rank_file), "--top-rank", "3000"])
    if signal_date:
        cmd.extend(["--signal-date", signal_date])
    return cmd


def build_features_cmd(trade_date: str, output_root: Path, args: argparse.Namespace) -> list[str]:
    return python_cmd(
        "research/v8_research/build_forward_shadow_daily_features.py",
        "--trade-date",
        trade_date,
        "--output-root",
        str(output_root),
        "--rank-file",
        str(args.rank_file),
        "--daily-file",
        str(args.daily_file),
        "--minute-dir",
        str(args.minute_dir),
        "--auction-raw-dir",
        str(args.auction_raw_dir),
    )


def build_score_cmd(trade_date: str, output_root: Path, args: argparse.Namespace) -> list[str]:
    return python_cmd(
        "research/v8_research/build_forward_shadow_score_matrix.py",
        "--trade-date",
        trade_date,
        "--feature-file",
        str(output_root / "test_features" / f"{trade_date}_forward_features.csv"),
        "--model-file",
        str(args.model_file),
        "--output-root",
        str(output_root),
        "--data-max-timestamp",
        f"{ymd_iso(trade_date)} 14:50:00",
    )


def freeze_cmd(trade_date: str, output_root: Path, args: argparse.Namespace) -> list[str]:
    return python_cmd(
        "research/v8_research/forward_shadow_runner.py",
        "--trade-date",
        trade_date,
        "--score-file",
        str(output_root / "score_matrices" / f"{trade_date}_score_matrix.csv"),
        "--daily-file",
        str(args.daily_file),
        "--output-root",
        str(output_root),
    )


def record_entry_cmd(trade_date: str, output_root: Path, args: argparse.Namespace) -> list[str]:
    return python_cmd(
        "research/v8_research/forward_shadow_data_guard.py",
        "record-entry",
        "--trade-date",
        trade_date,
        "--output-root",
        str(output_root),
        "--minute-dir",
        str(args.minute_dir),
    )


def exit_monitor_cmd(signal_date: str, settlement_date: str, checkpoint: str, args: argparse.Namespace) -> list[str]:
    return python_cmd(
        "research/v8_research/forward_shadow_exit_monitor.py",
        "--signal-date",
        signal_date,
        "--settlement-date",
        settlement_date,
        "--output-root",
        str(args.output_root),
        "--minute-dir",
        str(args.minute_dir),
        "--checkpoint",
        checkpoint,
        "--send-notifications",
    )


def write_results(results: list[StepResult], output_root: Path, trade_date: str) -> Path:
    out_dir = output_root / "live_runner"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{trade_date}_live_runner_steps.csv"
    pd.DataFrame([asdict(r) for r in results]).to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Portable live paper-tracking orchestrator for the locked forward shadow routes.")
    parser.add_argument("--trade-date", default="auto", help="YYYYMMDD, default today in Asia/Shanghai")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--rank-file", default=str(DEFAULT_RANK_FILE))
    parser.add_argument("--daily-file", default=str(DEFAULT_DAILY_FILE))
    parser.add_argument("--minute-dir", default=str(DEFAULT_MINUTE_DIR))
    parser.add_argument("--auction-raw-dir", default=str(DEFAULT_AUCTION_RAW_DIR))
    parser.add_argument("--model-file", default=str(DEFAULT_MODEL_FILE))
    parser.add_argument("--topic-id", type=int, default=44635)
    parser.add_argument("--send-notifications", action="store_true")
    parser.add_argument("--no-wait", action="store_true", help="Execute all scheduled steps immediately; intended only for operational drills.")
    parser.add_argument("--requests-per-minute", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--lookback-trading-days", type=int, default=90)
    parser.add_argument("--skip-moneyflow", action="store_true")
    args = parser.parse_args()

    env_check(bool(args.send_notifications))
    trade_date = today_ymd() if args.trade_date == "auto" else str(args.trade_date)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[StepResult] = []
    write_live_status(
        output_root,
        trade_date,
        "running",
        current_step_id="prepare_baseline",
        scheduled_time="07:30:00",
        completed_steps=0,
        message="Starting forward shadow paper-only runner.",
        no_wait=bool(args.no_wait),
    )

    prepare_cmd = python_cmd(
        "research/v8_research/prepare_forward_shadow_baseline.py",
        "--trade-date",
        trade_date,
        "--rank-file",
        str(args.rank_file),
        "--daily-file",
        str(args.daily_file),
        "--lookback-trading-days",
        str(args.lookback_trading_days),
        "--requests-per-minute",
        str(args.requests_per_minute),
    )
    if args.skip_moneyflow:
        prepare_cmd.append("--skip-moneyflow")
    prepare = run_cmd("prepare_baseline", "07:30:00", prepare_cmd, bool(args.send_notifications), int(args.topic_id))
    results.append(prepare)
    write_results(results, output_root, trade_date)
    write_live_status(
        output_root,
        trade_date,
        "running",
        current_step_id="prepare_baseline",
        scheduled_time="07:30:00",
        completed_steps=len(results),
        message="Baseline preparation completed.",
        no_wait=bool(args.no_wait),
    )
    summary = parse_prepare_summary(prepare.stdout_tail)
    prior = str(summary.get("prior_trade_date") or "")
    prior2 = str(summary.get("prior2_trade_date") or "")
    if not prior or not prior2:
        raise SystemExit("prepare_baseline did not return prior trade dates")

    reconstruct_prior_if_needed(args, trade_date, prior, prior2, results)

    scheduled_steps: list[tuple[str, str, list[str], str]] = [
        (
            "auction_guard_0925",
            "09:25:30",
            python_cmd(
                "research/v8_research/forward_shadow_data_guard.py",
                "auction-guard",
                "--trade-date",
                trade_date,
                "--prior-trade-date",
                prior,
                "--output-root",
                str(output_root),
                "--auction-raw-dir",
                str(args.auction_raw_dir),
                "--rank-file",
                str(args.rank_file),
                "--top-rank",
                "3000",
                "--requests-per-minute",
                str(args.requests_per_minute),
            ),
            "",
        ),
    ]
    for step_id, hms, bar, checkpoint in [
        ("fetch_exit_0935_bar", "09:35:05", "09:35", ""),
        ("exit_check_0935", "09:35:50", "", "check_0935"),
        ("fetch_exit_0940_bar", "09:40:05", "09:40", ""),
        ("exit_exec_0940", "09:40:50", "", "exec_0940"),
        ("fetch_exit_0945_bar", "09:45:05", "09:45", ""),
        ("exit_check_0945", "09:45:50", "", "check_0945"),
        ("fetch_exit_0950_bar", "09:50:05", "09:50", ""),
        ("exit_exec_0950", "09:50:50", "", "exec_0950"),
        ("fetch_exit_1000_bar", "10:00:05", "10:00", ""),
        ("exit_check_1000", "10:00:50", "", "check_1000"),
        ("fetch_exit_1005_bar", "10:05:05", "10:05", ""),
        ("exit_exec_1005", "10:05:50", "", "exec_1005"),
        ("fetch_exit_1025_bar", "10:25:05", "10:25", ""),
        ("exit_prealert_1025", "10:25:50", "", "prealert_1025"),
        ("fetch_exit_1030_bar", "10:30:05", "10:30", ""),
        ("exit_default_1030", "10:30:50", "", "default_1030"),
    ]:
        cmd = minute_fetch_cmd(trade_date, prior, output_root, Path(args.minute_dir), "prior_entries", bar, "bar", args) if bar else exit_monitor_cmd(prior, trade_date, checkpoint, args)
        scheduled_steps.append((step_id, hms, cmd, ""))

    scheduled_steps.extend(
        [
            ("fetch_tail_until_1430", "14:30:00", minute_fetch_cmd(trade_date, None, output_root, Path(args.minute_dir), "top3000", "14:30", "until", args), ""),
            ("fetch_tail_1435_bar", "14:35:05", minute_fetch_cmd(trade_date, None, output_root, Path(args.minute_dir), "top3000", "14:35", "bar", args), ""),
            ("fetch_tail_1440_bar", "14:40:05", minute_fetch_cmd(trade_date, None, output_root, Path(args.minute_dir), "top3000", "14:40", "bar", args), ""),
            ("fetch_tail_1445_bar", "14:45:05", minute_fetch_cmd(trade_date, None, output_root, Path(args.minute_dir), "top3000", "14:45", "bar", args), ""),
            ("fetch_tail_1450_bar", "14:50:05", minute_fetch_cmd(trade_date, None, output_root, Path(args.minute_dir), "top3000", "14:50", "bar", args), ""),
            ("build_forward_features", "14:50:40", build_features_cmd(trade_date, output_root, args), ""),
            ("build_score_matrix", "14:51:20", build_score_cmd(trade_date, output_root, args), ""),
            ("freeze_signals", "14:51:50", freeze_cmd(trade_date, output_root, args), "routes"),
            ("fetch_tail_1455_bar", "14:55:05", minute_fetch_cmd(trade_date, None, output_root, Path(args.minute_dir), "top3000", "14:55", "bar", args), ""),
            ("record_entry_1455_vwap", "14:55:50", record_entry_cmd(trade_date, output_root, args), "entry"),
            ("fetch_tail_1500_bar", "15:00:05", minute_fetch_cmd(trade_date, None, output_root, Path(args.minute_dir), "top3000", "15:00", "bar", args), ""),
        ]
    )

    total_steps = len(results) + len(scheduled_steps)
    for step_id, hms, cmd, extra_kind in scheduled_steps:
        write_live_status(
            output_root,
            trade_date,
            "waiting" if not args.no_wait else "running",
            current_step_id=step_id,
            scheduled_time=hms,
            completed_steps=len(results),
            total_steps=total_steps,
            message=f"Waiting for {hms} Beijing time." if not args.no_wait else "No-wait mode: executing scheduled step immediately.",
            no_wait=bool(args.no_wait),
        )
        wait_until(hms, bool(args.no_wait))
        write_live_status(
            output_root,
            trade_date,
            "running",
            current_step_id=step_id,
            scheduled_time=hms,
            completed_steps=len(results),
            total_steps=total_steps,
            message=f"Running step {step_id}.",
            no_wait=bool(args.no_wait),
        )
        extra = ""
        if extra_kind == "routes":
            extra = route_summary(output_root, trade_date)
        elif extra_kind == "entry":
            extra = entry_summary(output_root, trade_date)
        result = run_cmd(step_id, hms, cmd, bool(args.send_notifications), int(args.topic_id), extra_summary=extra)
        results.append(result)
        write_results(results, output_root, trade_date)
        write_live_status(
            output_root,
            trade_date,
            "running",
            current_step_id=step_id,
            scheduled_time=hms,
            completed_steps=len(results),
            total_steps=total_steps,
            message=f"Completed step {step_id}.",
            no_wait=bool(args.no_wait),
        )
        if extra_kind == "routes":
            send_wxpusher(f"FS {trade_date} frozen routes", f"# Forward Shadow {trade_date}\n\n{route_summary(output_root, trade_date)}\n\npaper-only。", args.topic_id, bool(args.send_notifications))
        if extra_kind == "entry":
            send_wxpusher(f"FS {trade_date} entry recorded", f"# Forward Shadow {trade_date}\n\n{entry_summary(output_root, trade_date)}\n\npaper-only。", args.topic_id, bool(args.send_notifications))

    path = write_results(results, output_root, trade_date)
    write_live_status(
        output_root,
        trade_date,
        "completed",
        completed_steps=len(results),
        total_steps=len(results),
        message=f"Live paper tracking completed. Steps file: {path}",
        no_wait=bool(args.no_wait),
    )
    send_wxpusher(
        f"FS {trade_date} live paper tracking completed",
        f"# Forward Shadow {trade_date}\n\nLive paper tracking completed.\n\n- steps: `{path}`\n\n{route_summary(output_root, trade_date)}\n\n{entry_summary(output_root, trade_date)}\n\npaper-only，不下单。",
        args.topic_id,
        bool(args.send_notifications),
    )
    print(json.dumps({"trade_date": trade_date, "prior_trade_date": prior, "steps": str(path), "status": "success"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
