#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "research" / "v8_research" / "forward_shadow_runtime_config.example.json"
DEFAULT_V7_LOCKED_COMMIT = "f688eec575af4667681d87b5b2c1fca72754e399"
FORBIDDEN_ACTIONS = (
    "broker",
    "order",
    "trade-live",
    "simulation-live",
)


@dataclass(frozen=True)
class TaskResult:
    checkpoint: str
    status: str
    message: str
    started_at: str
    finished_at: str
    duration_seconds: float
    command: list[str] | None = None
    return_code: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    details: dict[str, Any] | None = None


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    if not cfg.get("runtime", {}).get("paper_only", True):
        raise SystemExit("runtime.paper_only must remain true for forward shadow tracking")
    return cfg


def bj_now(tz_name: str = "Asia/Shanghai") -> datetime:
    return datetime.now(ZoneInfo(tz_name))


def iso_now(tz_name: str = "Asia/Shanghai") -> str:
    return bj_now(tz_name).isoformat(timespec="seconds")


def today_ymd(tz_name: str = "Asia/Shanghai") -> str:
    return bj_now(tz_name).strftime("%Y%m%d")


def ymd_to_iso(value: str) -> str:
    s = value.replace("-", "")[:8]
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def parse_hms(value: str) -> dt_time:
    return datetime.strptime(value, "%H:%M:%S").time()


def tail(text: str, max_chars: int = 4000) -> str:
    if not text:
        return ""
    return text[-max_chars:]


def redact_sensitive(text: str) -> str:
    out = text or ""
    for key in ("TUSHARE_TOKEN", "WXPUSHER_APP_TOKEN"):
        token = os.environ.get(key, "")
        if token:
            out = out.replace(token, f"{token[:6]}...{token[-4:]}")
    return out


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def infer_prior_trade_date(trade_date: str, cfg: dict[str, Any]) -> str:
    bootstrap = ROOT / "data_tushare" / "raw" / "bootstrap" / "trade_cal.parquet"
    if bootstrap.exists():
        cal = pd.read_parquet(bootstrap)
        if "cal_date" in cal.columns and "is_open" in cal.columns:
            cal["cal_date"] = cal["cal_date"].astype(str)
            cal["is_open"] = pd.to_numeric(cal["is_open"], errors="coerce").fillna(0).astype(int)
            days = sorted(cal[(cal["is_open"].eq(1)) & (cal["cal_date"] < trade_date)]["cal_date"].tolist())
            if days:
                return str(days[-1])
    daily_file = ROOT / cfg.get("paths", {}).get("daily_file", "data_tushare/clean/daily_repaired_top3000.parquet")
    if daily_file.exists():
        daily = pd.read_parquet(daily_file, columns=["trade_date"])
        days = sorted(set(daily["trade_date"].astype(str)))
        days = [d for d in days if d < trade_date]
        if days:
            return days[-1]
    current = datetime.strptime(trade_date, "%Y%m%d").date()
    prior = current - timedelta(days=1)
    while prior.weekday() >= 5:
        prior -= timedelta(days=1)
    return prior.strftime("%Y%m%d")


def ensure_no_forbidden_command(argv: list[str]) -> None:
    raw = " ".join(argv).lower()
    found = [word for word in FORBIDDEN_ACTIONS if word in raw]
    if found:
        raise ValueError(f"forbidden command word for paper tracking: {found}")


class JsonLogger:
    def __init__(self, log_dir: Path, trade_date: str, tz_name: str) -> None:
        self.log_dir = log_dir
        self.trade_date = trade_date
        self.tz_name = tz_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"{trade_date}.jsonl"

    def write(self, event: str, **payload: Any) -> None:
        row = {
            "ts": iso_now(self.tz_name),
            "event": event,
            "trade_date": self.trade_date,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


class RuntimeStore:
    def __init__(self, db_path: Path, tz_name: str) -> None:
        self.db_path = db_path
        self.tz_name = tz_name
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.db_path)
        self.con.row_factory = sqlite3.Row
        self.init_schema()

    def init_schema(self) -> None:
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS task_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date TEXT NOT NULL,
                checkpoint TEXT NOT NULL,
                scheduled_time TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                duration_seconds REAL,
                command_json TEXT,
                return_code INTEGER,
                stdout_tail TEXT,
                stderr_tail TEXT,
                error_msg TEXT,
                details_json TEXT
            )
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoint_state (
                trade_date TEXT NOT NULL,
                checkpoint TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                message TEXT,
                details_json TEXT,
                PRIMARY KEY (trade_date, checkpoint)
            )
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS push_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date TEXT NOT NULL,
                checkpoint TEXT,
                sent_at TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT,
                response_code INTEGER,
                response_text TEXT
            )
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date TEXT NOT NULL,
                checkpoint TEXT NOT NULL,
                detected_at TEXT NOT NULL,
                severity TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                message TEXT NOT NULL,
                details_json TEXT
            )
            """
        )
        self.con.commit()

    def start_run(self, trade_date: str, checkpoint: str, scheduled_time: str | None, command: list[str] | None) -> int:
        cur = self.con.execute(
            """
            INSERT INTO task_runs(trade_date, checkpoint, scheduled_time, started_at, status, command_json)
            VALUES (?, ?, ?, ?, 'running', ?)
            """,
            (trade_date, checkpoint, scheduled_time, iso_now(self.tz_name), json.dumps(command, ensure_ascii=False) if command else None),
        )
        self.con.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, result: TaskResult) -> None:
        self.con.execute(
            """
            UPDATE task_runs
            SET finished_at=?, status=?, duration_seconds=?, return_code=?,
                stdout_tail=?, stderr_tail=?, error_msg=?, details_json=?
            WHERE id=?
            """,
            (
                result.finished_at,
                result.status,
                result.duration_seconds,
                result.return_code,
                result.stdout_tail,
                result.stderr_tail,
                result.message if result.status not in {"success", "skipped", "dry_run"} else None,
                json.dumps(result.details or {}, ensure_ascii=False, default=str),
                run_id,
            ),
        )
        self.con.execute(
            """
            INSERT INTO checkpoint_state(trade_date, checkpoint, status, updated_at, message, details_json)
            SELECT trade_date, checkpoint, ?, ?, ?, ? FROM task_runs WHERE id=?
            ON CONFLICT(trade_date, checkpoint)
            DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at,
                          message=excluded.message, details_json=excluded.details_json
            """,
            (
                result.status,
                result.finished_at,
                result.message,
                json.dumps(result.details or {}, ensure_ascii=False, default=str),
                run_id,
            ),
        )
        self.con.commit()

    def add_push_log(self, trade_date: str, checkpoint: str, status: str, summary: str, response_code: int | None, response_text: str) -> None:
        self.con.execute(
            """
            INSERT INTO push_logs(trade_date, checkpoint, sent_at, status, summary, response_code, response_text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (trade_date, checkpoint, iso_now(self.tz_name), status, summary, response_code, response_text[:2000]),
        )
        self.con.commit()

    def add_violation(self, trade_date: str, checkpoint: str, severity: str, violation_type: str, message: str, details: dict[str, Any] | None = None) -> None:
        self.con.execute(
            """
            INSERT INTO violations(trade_date, checkpoint, detected_at, severity, violation_type, message, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (trade_date, checkpoint, iso_now(self.tz_name), severity, violation_type, message, json.dumps(details or {}, ensure_ascii=False, default=str)),
        )
        self.con.commit()

    def checkpoint_status(self, trade_date: str, checkpoint: str) -> str | None:
        row = self.con.execute(
            "SELECT status FROM checkpoint_state WHERE trade_date=? AND checkpoint=?",
            (trade_date, checkpoint),
        ).fetchone()
        return str(row["status"]) if row else None

    def latest_status(self, trade_date: str | None = None) -> pd.DataFrame:
        if trade_date:
            return pd.read_sql_query(
                "SELECT * FROM checkpoint_state WHERE trade_date=? ORDER BY checkpoint",
                self.con,
                params=(trade_date,),
            )
        return pd.read_sql_query("SELECT * FROM checkpoint_state ORDER BY updated_at DESC LIMIT 50", self.con)


class WxPusherNotifier:
    def __init__(self, cfg: dict[str, Any], store: RuntimeStore, tz_name: str, force_send: bool = False) -> None:
        self.cfg = cfg
        self.store = store
        self.tz_name = tz_name
        self.enabled = bool(cfg.get("enabled", False))
        self.dry_run = bool(cfg.get("dry_run", True)) and not force_send
        self.endpoint = str(cfg.get("endpoint", "https://wxpusher.zjiecode.com/api/send/message"))
        self.app_token_env = str(cfg.get("app_token_env", "WXPUSHER_APP_TOKEN"))
        self.topic_ids = [int(x) for x in cfg.get("topic_ids", [])]
        self.content_type = int(cfg.get("content_type", 3))
        self.verify_pay = bool(cfg.get("verify_pay", False))

    def send(self, trade_date: str, checkpoint: str, summary: str, content: str) -> None:
        if not self.enabled:
            self.store.add_push_log(trade_date, checkpoint, "disabled", summary, None, "")
            return
        if self.dry_run:
            self.store.add_push_log(trade_date, checkpoint, "dry_run", summary, None, "notification dry-run")
            return
        token = os.environ.get(self.app_token_env, "").strip()
        if not token:
            self.store.add_push_log(trade_date, checkpoint, "skipped_missing_token", summary, None, f"missing {self.app_token_env}")
            return
        payload = {
            "appToken": token,
            "content": content,
            "summary": summary[:100],
            "contentType": self.content_type,
            "topicIds": self.topic_ids,
            "verifyPay": self.verify_pay,
        }
        try:
            resp = requests.post(self.endpoint, json=payload, timeout=20)
            body = redact_sensitive(resp.text[:2000])
            status = "success" if resp.status_code == 200 else "failed"
            self.store.add_push_log(trade_date, checkpoint, status, summary, resp.status_code, body)
        except requests.RequestException as exc:
            self.store.add_push_log(trade_date, checkpoint, "failed", summary, None, redact_sensitive(repr(exc)))


class ForwardShadowRuntime:
    def __init__(
        self,
        cfg: dict[str, Any],
        trade_date: str,
        prior_trade_date: str | None,
        dry_run: bool = False,
        force_send: bool = False,
        feature_file_override: str | None = None,
        replay_mode: bool = False,
    ) -> None:
        self.cfg = cfg
        self.trade_date = trade_date
        self.prior_trade_date = prior_trade_date or infer_prior_trade_date(trade_date, cfg)
        self.dry_run = dry_run
        self.feature_file_override = feature_file_override
        self.replay_mode = replay_mode
        runtime_cfg = cfg.get("runtime", {})
        self.tz_name = runtime_cfg.get("timezone", "Asia/Shanghai")
        self.output_root = ROOT / runtime_cfg.get("output_root", "reports/tushare/v8_forward_shadow")
        self.log = JsonLogger(ROOT / runtime_cfg.get("log_dir", "logs/forward_shadow"), trade_date, self.tz_name)
        self.store = RuntimeStore(ROOT / runtime_cfg.get("state_db", "data_tushare/forward_shadow_runtime.sqlite3"), self.tz_name)
        self.notifier = WxPusherNotifier(cfg.get("wxpusher", {}), self.store, self.tz_name, force_send=force_send)

    def schedule_by_checkpoint(self) -> dict[str, dict[str, Any]]:
        return {str(item["checkpoint"]): item for item in self.cfg.get("schedule", [])}

    def placeholders(self) -> dict[str, str]:
        paths_cfg = self.cfg.get("paths", {})
        runtime_cfg = self.cfg.get("runtime", {})
        forward_feature_file = self.output_root / "test_features" / f"{self.trade_date}_forward_features.csv"
        default_feature_file = str(paths_cfg.get("default_feature_file", ""))
        if self.feature_file_override:
            feature_file = self.feature_file_override
        elif default_feature_file in {"", "__forward_feature_file__"}:
            feature_file = str(forward_feature_file)
        else:
            feature_file = default_feature_file
        return {
            "python": runtime_cfg.get("python", sys.executable),
            "trade_date": self.trade_date,
            "trade_date_iso": ymd_to_iso(self.trade_date),
            "prior_trade_date": self.prior_trade_date,
            "prior_trade_date_iso": ymd_to_iso(self.prior_trade_date),
            "output_root": str(self.output_root),
            "score_file": str(self.output_root / "score_matrices" / f"{self.trade_date}_score_matrix.csv"),
            "score_meta_file": str(self.output_root / "score_matrices" / f"{self.trade_date}_score_matrix_meta.json"),
            "signals_file": str(self.output_root / "daily_signals" / f"{self.trade_date}_signals.csv"),
            "forward_feature_file": str(forward_feature_file),
            "daily_file": str(ROOT / paths_cfg.get("daily_file", "data_tushare/clean/daily_repaired_top3000.parquet")),
            "minute_dir": str(ROOT / paths_cfg.get("minute_dir", "data_tushare/raw/stk_mins/freq=5min")),
            "auction_raw_dir": str(ROOT / paths_cfg.get("auction_raw_dir", "data_tushare/raw")),
            "feature_file": str(ROOT / feature_file) if feature_file and not str(feature_file).startswith("/") else str(feature_file),
            "model_file": str(ROOT / paths_cfg.get("model_file", "locked_artifacts/v7_cap20_strong_label_003/model/model.pkl")),
            "paper_reconstruction_flag": "--paper-reconstruction" if self.replay_mode else "",
        }

    def render_argv(self, argv: list[str]) -> list[str]:
        values = self.placeholders()
        out = [str(part).format(**values) for part in argv]
        out = [part for part in out if part != ""]
        ensure_no_forbidden_command(out)
        return out

    def run_checkpoint(self, checkpoint: str, notify_override: str | None = None) -> TaskResult:
        schedule = self.schedule_by_checkpoint().get(checkpoint, {"checkpoint": checkpoint, "kind": "internal", "notify": "always"})
        kind = schedule.get("kind", "internal")
        command: list[str] | None = None
        if kind == "command":
            command_name = str(schedule.get("command") or "")
            command_cfg = self.cfg.get("commands", {}).get(command_name)
            if not command_cfg:
                result = self._result(checkpoint, "failed", f"missing command config: {command_name}", None, None, None)
                self._record_and_notify(schedule, result, notify_override)
                return result
            if not bool(command_cfg.get("enabled", True)):
                result = self._result(checkpoint, "skipped", f"command disabled: {command_name}", None, None, {"command_name": command_name})
                self._record_and_notify(schedule, result, notify_override)
                return result
            command = self.render_argv(list(command_cfg.get("argv") or []))
        run_id = self.store.start_run(self.trade_date, checkpoint, schedule.get("time"), command)
        self.log.write("checkpoint_started", checkpoint=checkpoint, kind=kind, command=command, dry_run=self.dry_run)
        started = time.monotonic()
        try:
            if kind == "command":
                result = self._run_command(checkpoint, command or [], started)
            else:
                result = self._run_internal(checkpoint, started)
        except Exception as exc:
            result = self._result(
                checkpoint,
                "failed",
                redact_sensitive(f"{type(exc).__name__}: {exc}"),
                started,
                None,
                {"traceback": redact_sensitive(traceback.format_exc())},
            )
        self.store.finish_run(run_id, result)
        self.log.write(
            "checkpoint_finished",
            checkpoint=checkpoint,
            status=result.status,
            duration_seconds=result.duration_seconds,
            message=result.message,
            return_code=result.return_code,
        )
        self._notify_if_needed(schedule, result, notify_override)
        return result

    def _record_and_notify(self, schedule: dict[str, Any], result: TaskResult, notify_override: str | None) -> None:
        run_id = self.store.start_run(self.trade_date, result.checkpoint, schedule.get("time"), result.command)
        self.store.finish_run(run_id, result)
        self.log.write("checkpoint_finished", checkpoint=result.checkpoint, status=result.status, message=result.message)
        self._notify_if_needed(schedule, result, notify_override)

    def _result(
        self,
        checkpoint: str,
        status: str,
        message: str,
        started_monotonic: float | None,
        command: list[str] | None,
        details: dict[str, Any] | None,
        return_code: int | None = None,
        stdout_tail: str = "",
        stderr_tail: str = "",
    ) -> TaskResult:
        finished = iso_now(self.tz_name)
        duration = float(time.monotonic() - started_monotonic) if started_monotonic else 0.0
        return TaskResult(
            checkpoint=checkpoint,
            status=status,
            message=message,
            started_at="",
            finished_at=finished,
            duration_seconds=duration,
            command=command,
            return_code=return_code,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            details=details or {},
        )

    def _run_command(self, checkpoint: str, command: list[str], started: float) -> TaskResult:
        if self.dry_run:
            return self._result(checkpoint, "dry_run", "command not executed because --dry-run is active", started, command, {"command": command})
        timeout = int(self.cfg.get("runtime", {}).get("command_timeout_seconds", 900))
        proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        status = "success" if proc.returncode == 0 else "failed"
        message = "command completed" if proc.returncode == 0 else f"command failed with code {proc.returncode}"
        return self._result(
            checkpoint,
            status,
            message,
            started,
            command,
            {"command": command},
            return_code=proc.returncode,
            stdout_tail=redact_sensitive(tail(proc.stdout)),
            stderr_tail=redact_sensitive(tail(proc.stderr)),
        )

    def _run_internal(self, checkpoint: str, started: float) -> TaskResult:
        if checkpoint == "startup_check":
            return self._startup_check(started)
        if checkpoint == "premarket_check":
            return self._premarket_check(started)
        if checkpoint == "latency_guard":
            return self._latency_guard(started)
        if checkpoint == "entry_observation_check":
            return self._entry_observation_check(started)
        return self._result(checkpoint, "success", "internal checkpoint acknowledged", started, None, {})

    def _startup_check(self, started: float) -> TaskResult:
        branch = run_git(["branch", "--show-current"])
        head = run_git(["rev-parse", "HEAD"])
        v7_locked = run_git(["rev-parse", "v7_locked"])
        expected = self.cfg.get("guards", {}).get("expected_v7_locked_commit", DEFAULT_V7_LOCKED_COMMIT)
        details = {
            "branch": branch,
            "head": head,
            "v7_locked": v7_locked,
            "expected_v7_locked_commit": expected,
            "paper_only": self.cfg.get("runtime", {}).get("paper_only", True),
            "prior_trade_date": self.prior_trade_date,
            "replay_mode": self.replay_mode,
        }
        if v7_locked != expected:
            self.store.add_violation(self.trade_date, "startup_check", "P0", "v7_locked_moved", "v7_locked commit does not match expected", details)
            return self._result("startup_check", "failed", "v7_locked commit mismatch", started, None, details)
        return self._result("startup_check", "success", "startup lock checks passed", started, None, details)

    def _premarket_check(self, started: float) -> TaskResult:
        values = self.placeholders()
        required = {
            "daily_file": Path(values["daily_file"]),
            "model_file": Path(values["model_file"]),
            "tail_down_definition": ROOT / "reports/tushare/v8_research/forward_shadow_setup/tail_down_definition.md",
            "U2_definition": ROOT / "reports/tushare/v8_research/forward_shadow_setup/U2_definition.md",
        }
        missing = {name: str(path) for name, path in required.items() if not path.exists()}
        details = {"required": {k: str(v) for k, v in required.items()}, "missing": missing}
        if missing:
            self.store.add_violation(self.trade_date, "premarket_check", "P0", "missing_required_file", "required runtime file missing", details)
            return self._result("premarket_check", "failed", "missing required files", started, None, details)
        return self._result("premarket_check", "success", "required files present", started, None, details)

    def _latency_guard(self, started: float) -> TaskResult:
        guards = self.cfg.get("guards", {})
        meta_path = Path(self.placeholders()["score_meta_file"])
        status_path = self.output_root / "forward_shadow_candidate_status.csv"
        details: dict[str, Any] = {"score_meta_file": str(meta_path), "status_file": str(status_path)}
        violations: list[str] = []
        if not meta_path.exists():
            violations.append("missing_score_meta")
        else:
            meta = json.loads(meta_path.read_text())
            details["score_meta"] = meta
            logical_ts = str(meta.get("logical_data_max_timestamp", ""))
            max_allowed = f"{ymd_to_iso(self.trade_date)} {guards.get('max_signal_data_time', '14:50:00')}"
            details["max_allowed_data_timestamp"] = max_allowed
            if logical_ts > max_allowed:
                violations.append("data_timestamp_violation")
        if status_path.exists():
            status = pd.read_csv(status_path)
            status["trade_date"] = status["trade_date"].astype(str)
            day = status[status["trade_date"].eq(self.trade_date)].copy()
            if not day.empty and "freeze_time_beijing" in day.columns:
                freeze_time = str(day["freeze_time_beijing"].dropna().astype(str).max())
                details["freeze_time_beijing"] = freeze_time
                allow_replay = bool(guards.get("allow_replay_after_close", False)) or self.replay_mode
                details["allow_replay_after_close"] = allow_replay
                details["replay_mode"] = self.replay_mode
                latest = guards.get("latest_signal_freeze_time", "14:55:00")
                if not allow_replay and "T" in freeze_time:
                    clock = freeze_time.split("T", 1)[1][:8]
                    if clock > latest:
                        violations.append("late_freeze_time")
        else:
            violations.append("missing_candidate_status")
        if violations:
            self.store.add_violation(self.trade_date, "latency_guard", "P0", ",".join(violations), "latency or timestamp guard failed", details)
            return self._result("latency_guard", "failed", "latency guard failed: " + ",".join(violations), started, None, details)
        return self._result("latency_guard", "success", "latency guard passed", started, None, details)

    def _entry_observation_check(self, started: float) -> TaskResult:
        signals_file = Path(self.placeholders()["signals_file"])
        entry_file = self.output_root / "daily_entry_prices" / f"{self.trade_date}_entry_prices.csv"
        details = {"signals_file": str(signals_file), "entry_file": str(entry_file)}
        if not signals_file.exists():
            self.store.add_violation(self.trade_date, "entry_observation_check", "P1", "missing_signals_file", "signals file missing at entry observation", details)
            return self._result("entry_observation_check", "failed", "signals file missing", started, None, details)
        signals = pd.read_csv(signals_file)
        details["signal_rows"] = int(len(signals))
        details["selected_strategies"] = sorted(signals.get("strategy_id", pd.Series(dtype=str)).astype(str).unique().tolist())
        if not entry_file.exists():
            self.store.add_violation(self.trade_date, "entry_observation_check", "P1", "missing_entry_file", "entry price file missing at entry observation", details)
            return self._result("entry_observation_check", "failed", "entry price file missing", started, None, details)
        entry = pd.read_csv(entry_file)
        details["entry_rows"] = int(len(entry))
        details["entry_recorded_rows"] = int(entry.get("paper_entry_status", pd.Series(dtype=str)).astype(str).eq("entry_recorded").sum())
        if details["entry_recorded_rows"] <= 0:
            self.store.add_violation(self.trade_date, "entry_observation_check", "P1", "no_entry_recorded", "no paper entry VWAP recorded", details)
            return self._result("entry_observation_check", "failed", "no paper entry VWAP recorded", started, None, details)
        return self._result("entry_observation_check", "success", "signals and entry price files exist", started, None, details)

    def _notify_if_needed(self, schedule: dict[str, Any], result: TaskResult, notify_override: str | None) -> None:
        policy = notify_override or schedule.get("notify", "on_failure")
        should = policy == "always" or (policy == "on_failure" and result.status not in {"success", "skipped", "dry_run"})
        if not should:
            return
        summary = f"ForwardShadow {self.trade_date} {result.checkpoint}: {result.status}"
        content = "\n".join(
            [
                f"# Forward Shadow Paper Tracking",
                "",
                f"- trade_date: `{self.trade_date}`",
                f"- checkpoint: `{result.checkpoint}`",
                f"- status: `{result.status}`",
                f"- message: `{result.message}`",
                f"- duration_seconds: `{result.duration_seconds:.3f}`",
                f"- paper_only: `true`",
                f"- replay_mode: `{str(self.replay_mode).lower()}`",
                "",
                "No live/simulation order action is performed.",
            ]
        )
        self.notifier.send(self.trade_date, result.checkpoint, summary, content)

    def run_replay_day(self, include_download: bool = False, include_settlement: bool = False) -> list[TaskResult]:
        """Run a historical day through the forward-shadow pipeline without changing locked rules."""
        checkpoints = ["startup_check", "premarket_check"]
        if include_download:
            checkpoints.append("download_1450_bar")
        checkpoints.extend(
            [
                "build_forward_features",
                "build_score_matrix",
                "freeze_signals",
                "latency_guard",
                "entry_observation_check",
            ]
        )
        if include_settlement:
            checkpoints.append("prior_trade_settlement")
        self.log.write(
            "replay_day_started",
            trade_date=self.trade_date,
            prior_trade_date=self.prior_trade_date,
            include_download=include_download,
            include_settlement=include_settlement,
            dry_run=self.dry_run,
        )
        results: list[TaskResult] = []
        for checkpoint in checkpoints:
            result = self.run_checkpoint(checkpoint)
            results.append(result)
            if result.status == "failed":
                self.log.write("replay_day_stopped", failed_checkpoint=checkpoint, message=result.message)
                break
        self.log.write("replay_day_finished", result_count=len(results), statuses={r.checkpoint: r.status for r in results})
        return results

    def due_items(self, now_dt: datetime) -> list[dict[str, Any]]:
        today_date = now_dt.date()
        due = []
        for item in self.cfg.get("schedule", []):
            scheduled = datetime.combine(today_date, parse_hms(item["time"]), tzinfo=now_dt.tzinfo)
            if scheduled <= now_dt and self.store.checkpoint_status(self.trade_date, str(item["checkpoint"])) is None:
                due.append(item)
        return sorted(due, key=lambda item: str(item["time"]))

    def run_daemon(self) -> None:
        poll = int(self.cfg.get("runtime", {}).get("poll_seconds", 5))
        self.log.write("daemon_started", trade_date=self.trade_date, prior_trade_date=self.prior_trade_date, dry_run=self.dry_run)
        while True:
            now_dt = bj_now(self.tz_name)
            for item in self.due_items(now_dt):
                self.run_checkpoint(str(item["checkpoint"]))
            time.sleep(poll)


def command_init_db(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    tz_name = cfg.get("runtime", {}).get("timezone", "Asia/Shanghai")
    store = RuntimeStore(ROOT / cfg.get("runtime", {}).get("state_db", "data_tushare/forward_shadow_runtime.sqlite3"), tz_name)
    print(json.dumps({"state_db": str(store.db_path), "status": "initialized"}, ensure_ascii=False, indent=2))


def command_run_once(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    trade_date = args.trade_date or today_ymd(cfg.get("runtime", {}).get("timezone", "Asia/Shanghai"))
    rt = ForwardShadowRuntime(
        cfg,
        trade_date=trade_date,
        prior_trade_date=args.prior_trade_date,
        dry_run=args.dry_run,
        force_send=args.send_notifications,
        feature_file_override=args.feature_file,
        replay_mode=args.replay_mode,
    )
    result = rt.run_checkpoint(args.checkpoint, notify_override=args.notify)
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2, default=str))


def command_daemon(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    trade_date = args.trade_date or today_ymd(cfg.get("runtime", {}).get("timezone", "Asia/Shanghai"))
    rt = ForwardShadowRuntime(
        cfg,
        trade_date=trade_date,
        prior_trade_date=args.prior_trade_date,
        dry_run=args.dry_run,
        force_send=args.send_notifications,
        feature_file_override=args.feature_file,
        replay_mode=False,
    )
    rt.run_daemon()


def command_replay_day(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    trade_date = args.trade_date
    rt = ForwardShadowRuntime(
        cfg,
        trade_date=trade_date,
        prior_trade_date=args.prior_trade_date,
        dry_run=args.dry_run,
        force_send=args.send_notifications,
        feature_file_override=args.feature_file,
        replay_mode=True,
    )
    results = rt.run_replay_day(include_download=args.include_download, include_settlement=args.include_settlement)
    print(json.dumps([r.__dict__ for r in results], ensure_ascii=False, indent=2, default=str))


def command_status(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    tz_name = cfg.get("runtime", {}).get("timezone", "Asia/Shanghai")
    store = RuntimeStore(ROOT / cfg.get("runtime", {}).get("state_db", "data_tushare/forward_shadow_runtime.sqlite3"), tz_name)
    df = store.latest_status(args.trade_date)
    print(df.to_string(index=False) if not df.empty else "no runtime checkpoint state")


def command_send_test(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    trade_date = args.trade_date or today_ymd(cfg.get("runtime", {}).get("timezone", "Asia/Shanghai"))
    rt = ForwardShadowRuntime(
        cfg,
        trade_date=trade_date,
        prior_trade_date=args.prior_trade_date,
        dry_run=True,
        force_send=args.send_notifications,
        feature_file_override=args.feature_file,
        replay_mode=bool(getattr(args, "replay_mode", False)),
    )
    rt.notifier.send(
        trade_date,
        "send_test",
        "ForwardShadow Wxpusher test",
        "# Forward Shadow Wxpusher Test\n\npaper-only runtime notification test.",
    )
    print(json.dumps({"trade_date": trade_date, "status": "queued_or_dry_run", "send_notifications": bool(args.send_notifications)}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-only forward shadow runtime scheduler.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init-db")
    init.set_defaults(func=command_init_db)

    run_once = sub.add_parser("run-once")
    run_once.add_argument("--checkpoint", required=True)
    run_once.add_argument("--trade-date")
    run_once.add_argument("--prior-trade-date")
    run_once.add_argument("--feature-file")
    run_once.add_argument("--dry-run", action="store_true")
    run_once.add_argument("--replay-mode", action="store_true", help="Allow after-close replay while keeping data timestamp guards.")
    run_once.add_argument("--send-notifications", action="store_true", help="Actually send WxPusher messages if token env is present.")
    run_once.add_argument("--notify", choices=["always", "on_failure", "never"])
    run_once.set_defaults(func=command_run_once)

    daemon = sub.add_parser("daemon")
    daemon.add_argument("--trade-date")
    daemon.add_argument("--prior-trade-date")
    daemon.add_argument("--feature-file")
    daemon.add_argument("--dry-run", action="store_true")
    daemon.add_argument("--send-notifications", action="store_true", help="Actually send WxPusher messages if token env is present.")
    daemon.set_defaults(func=command_daemon)

    replay = sub.add_parser("replay-day", help="Run a historical trade date through the paper-only pipeline for flow validation.")
    replay.add_argument("--trade-date", required=True, help="YYYYMMDD logical trade date, e.g. 20260522")
    replay.add_argument("--prior-trade-date")
    replay.add_argument("--feature-file", help="Optional prebuilt feature file. Default uses test_features/{trade_date}_forward_features.csv.")
    replay.add_argument("--include-download", action="store_true", help="Also run the optional single-day Top3000 downloader checkpoint.")
    replay.add_argument("--include-settlement", action="store_true", help="Also run prior-trade settlement if exit data exists.")
    replay.add_argument("--dry-run", action="store_true")
    replay.add_argument("--send-notifications", action="store_true", help="Actually send WxPusher messages if token env is present.")
    replay.set_defaults(func=command_replay_day)

    status = sub.add_parser("status")
    status.add_argument("--trade-date")
    status.set_defaults(func=command_status)

    send_test = sub.add_parser("send-test")
    send_test.add_argument("--trade-date")
    send_test.add_argument("--prior-trade-date")
    send_test.add_argument("--feature-file")
    send_test.add_argument("--replay-mode", action="store_true")
    send_test.add_argument("--send-notifications", action="store_true", help="Actually send WxPusher messages if token env is present.")
    send_test.set_defaults(func=command_send_test)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
