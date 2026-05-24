#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
STORE_DIR = ROOT / "data_tushare"
RAW_DIR = STORE_DIR / "raw"
MANIFEST_DIR = STORE_DIR / "manifests"
LOG_DIR = STORE_DIR / "logs"
DB_PATH = MANIFEST_DIR / "download_tasks.sqlite3"
BEIJING_TZ = timezone(timedelta(hours=8))


class TushareError(RuntimeError):
    pass


class TushareRateLimit(TushareError):
    pass


@dataclass(frozen=True)
class ApiTask:
    task_id: str
    api_name: str
    params: dict[str, Any]
    fields: str | None
    output_path: str
    priority: int
    group_name: str
    ts_code: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    freq: str | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def beijing_day_utc_bounds() -> tuple[str, str]:
    now_local = datetime.now(BEIJING_TZ)
    start_local = datetime(now_local.year, now_local.month, now_local.day, tzinfo=BEIJING_TZ)
    end_local = start_local + timedelta(days=1)
    return (
        start_local.astimezone(timezone.utc).isoformat(timespec="seconds"),
        end_local.astimezone(timezone.utc).isoformat(timespec="seconds"),
    )


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_token() -> str:
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise SystemExit("缺少 TUSHARE_TOKEN 环境变量。不要把 token 写进代码或命令输出。")
    if len(token) < 20:
        raise SystemExit("TUSHARE_TOKEN 看起来不完整。")
    return token


def get_proxy_url() -> str:
    return os.environ.get("TUSHARE_PROXY_URL", "http://tsy.xiaodefa.cn").rstrip("/")


def redact_token(text: str, token: str | None = None) -> str:
    token = token or os.environ.get("TUSHARE_TOKEN", "")
    if token and token in text:
        return text.replace(token, f"{token[:6]}...{token[-4:]}")
    return text


def task_hash(api_name: str, params: dict[str, Any], fields: str | None) -> str:
    raw = json.dumps(
        {"api_name": api_name, "params": params, "fields": fields or ""},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def output_path_for(api_name: str, params: dict[str, Any], freq: str | None = None) -> str:
    base = RAW_DIR / api_name
    if api_name == "stk_mins":
        ts_code = params["ts_code"]
        start = params["start_date"][:10].replace("-", "")
        end = params["end_date"][:10].replace("-", "")
        return str(base / f"freq={freq or params.get('freq', '5min')}" / f"ts_code={ts_code}" / f"{start}_{end}.parquet")
    if "trade_date" in params:
        return str(base / f"trade_date={params['trade_date']}.parquet")
    if "list_status" in params:
        return str(base / f"list_status={params['list_status']}.parquet")
    if api_name in {"concept_detail"} and "ts_code" in params:
        return str(base / f"ts_code={params['ts_code']}.parquet")
    if api_name in {"ths_member"} and "ts_code" in params:
        return str(base / f"ths_code={params['ts_code']}.parquet")
    suffix = task_hash(api_name, params, None)[:12]
    return str(base / f"{suffix}.parquet")


def date_priority_map(dates: list[str], direction: str, offset: int, step: int = 10) -> dict[str, int]:
    ordered = sorted(dates, reverse=(direction == "near-first"))
    return {date_value: offset + idx * step for idx, date_value in enumerate(ordered)}


def connect_db() -> sqlite3.Connection:
    ensure_dirs()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            api_name TEXT NOT NULL,
            group_name TEXT NOT NULL,
            ts_code TEXT,
            start_date TEXT,
            end_date TEXT,
            freq TEXT,
            params_json TEXT NOT NULL,
            fields TEXT,
            output_path TEXT NOT NULL,
            priority INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            row_count INTEGER,
            checksum TEXT,
            request_seconds REAL,
            not_before REAL NOT NULL DEFAULT 0,
            error_msg TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS request_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT,
            api_name TEXT,
            requested_at TEXT NOT NULL,
            status TEXT NOT NULL,
            row_count INTEGER,
            request_seconds REAL,
            error_msg TEXT
        )
        """
    )
    con.commit()
    return con


def insert_tasks(con: sqlite3.Connection, tasks: list[ApiTask]) -> tuple[int, int]:
    inserted = 0
    existing = 0
    ts = now_iso()
    for task in tasks:
        cur = con.execute(
            """
            INSERT OR IGNORE INTO tasks (
                task_id, api_name, group_name, ts_code, start_date, end_date, freq,
                params_json, fields, output_path, priority, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
            (
                task.task_id,
                task.api_name,
                task.group_name,
                task.ts_code,
                task.start_date,
                task.end_date,
                task.freq,
                json.dumps(task.params, ensure_ascii=False, sort_keys=True),
                task.fields,
                task.output_path,
                task.priority,
                ts,
                ts,
            ),
        )
        if cur.rowcount:
            inserted += 1
        else:
            existing += 1
    con.commit()
    return inserted, existing


class TushareProxyClient:
    def __init__(self, token: str, base_url: str, timeout: int = 60) -> None:
        self.token = token
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept-Encoding": "gzip",
                "Content-Type": "application/json",
                "User-Agent": "tail-alpha-research/1.0",
            }
        )

    def call(self, api_name: str, params: dict[str, Any], fields: str | None = None) -> pd.DataFrame:
        payload: dict[str, Any] = {"api_name": api_name, "token": self.token, "params": params}
        if fields:
            payload["fields"] = fields
        started = time.time()
        try:
            resp = self.session.post(self.base_url, json=payload, timeout=self.timeout)
        except requests.Timeout as exc:
            raise TushareRateLimit(f"request timeout after {self.timeout}s") from exc
        except requests.RequestException as exc:
            raise TushareError(str(exc)) from exc

        elapsed = time.time() - started
        if resp.status_code >= 500:
            raise TushareError(f"http {resp.status_code}, elapsed={elapsed:.2f}s")
        if resp.status_code != 200:
            raise TushareError(f"http {resp.status_code}: {resp.text[:300]}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise TushareError(f"invalid json: {resp.text[:300]}") from exc

        code = data.get("code")
        msg = str(data.get("msg") or "")
        if code != 0:
            lowered = msg.lower()
            if any(word in msg for word in ["超速", "过快", "冷却", "频率", "限速", "timeout"]) or "timeout" in lowered:
                raise TushareRateLimit(f"api={api_name} code={code} msg={msg}")
            raise TushareError(f"api={api_name} code={code} msg={msg}")

        table = data.get("data") or {}
        cols = table.get("fields") or []
        rows = table.get("items") or []
        return pd.DataFrame(rows, columns=cols)


class RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.interval = 60.0 / max(requests_per_minute, 1)
        self.last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        wait_s = self.interval - (now - self.last)
        if wait_s > 0:
            time.sleep(wait_s)
        self.last = time.monotonic()


def write_parquet_atomic(df: pd.DataFrame, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)
    return sha1_file(path)


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_bootstrap_table(name: str) -> pd.DataFrame:
    path = RAW_DIR / "bootstrap" / f"{name}.parquet"
    if not path.exists():
        raise SystemExit(f"缺少 bootstrap 数据: {path}，请先运行 init")
    return pd.read_parquet(path)


def bootstrap_metadata(client: TushareProxyClient, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    bootstrap_dir = RAW_DIR / "bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    statuses = []
    for status in ["L", "D", "P"]:
        df = client.call(
            "stock_basic",
            {"exchange": "", "list_status": status},
            "ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date",
        )
        if not df.empty:
            statuses.append(df)
        time.sleep(0.4)
    stock_basic = pd.concat(statuses, ignore_index=True) if statuses else pd.DataFrame()
    stock_basic.to_parquet(bootstrap_dir / "stock_basic.parquet", index=False, compression="zstd")

    trade_cal = client.call("trade_cal", {"exchange": "SSE", "start_date": start, "end_date": end})
    trade_cal.to_parquet(bootstrap_dir / "trade_cal.parquet", index=False, compression="zstd")
    return stock_basic, trade_cal


def ymd(date_text: str) -> str:
    return date_text.replace("-", "")[:8]


def iso_date(date_text: str) -> str:
    s = ymd(date_text)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def active_universe(stock_basic: pd.DataFrame, start: str, end: str, include_bj: bool) -> pd.DataFrame:
    df = stock_basic.copy()
    df = df[df["ts_code"].astype(str).str.endswith((".SH", ".SZ") if not include_bj else (".SH", ".SZ", ".BJ"))]
    df["list_date"] = df["list_date"].fillna("")
    df["delist_date"] = df["delist_date"].fillna("")
    start_ymd = ymd(start)
    end_ymd = ymd(end)
    df = df[df["list_date"].le(end_ymd)]
    df = df[(df["delist_date"].eq("")) | (df["delist_date"].ge(start_ymd))]
    return df.sort_values("ts_code").drop_duplicates("ts_code").reset_index(drop=True)


def open_dates(trade_cal: pd.DataFrame) -> list[str]:
    df = trade_cal.copy()
    df["is_open"] = pd.to_numeric(df["is_open"], errors="coerce").fillna(0).astype(int)
    dates = df[df["is_open"].eq(1)]["cal_date"].astype(str).tolist()
    return sorted(set(dates))


def chunk_dates(dates: list[str], chunk_size: int, direction: str) -> list[tuple[str, str]]:
    chunks = [(dates[i], dates[min(i + chunk_size - 1, len(dates) - 1)]) for i in range(0, len(dates), chunk_size)]
    if direction == "near-first":
        chunks.reverse()
    return chunks


def create_task(api_name: str, params: dict[str, Any], fields: str | None, priority: int, group_name: str, **meta: Any) -> ApiTask:
    task_id = task_hash(api_name, params, fields)
    return ApiTask(
        task_id=task_id,
        api_name=api_name,
        params=params,
        fields=fields,
        output_path=output_path_for(api_name, params, meta.get("freq")),
        priority=priority,
        group_name=group_name,
        ts_code=meta.get("ts_code"),
        start_date=meta.get("start_date"),
        end_date=meta.get("end_date"),
        freq=meta.get("freq"),
    )


def build_tasks(
    stock_basic: pd.DataFrame,
    trade_cal: pd.DataFrame,
    start: str,
    end: str,
    freq: str,
    direction: str,
    include_enhanced: bool,
    include_memberships: bool,
    include_bj: bool,
    max_symbols: int | None,
) -> list[ApiTask]:
    tasks: list[ApiTask] = []
    universe = active_universe(stock_basic, start, end, include_bj)
    if max_symbols:
        universe = universe.head(max_symbols).copy()
    dates = open_dates(trade_cal)
    if not dates:
        raise SystemExit("交易日历为空，无法创建任务")

    daily_priority = date_priority_map(dates, direction, offset=100, step=10)
    minute_priority = date_priority_map(dates, direction, offset=10_000, step=10)

    # Small daily/cross-sectional endpoints: fetch by trade_date to minimize requests.
    enhanced_apis = ["stk_auction_o", "stk_auction", "stk_auction_c", "moneyflow", "moneyflow_ths", "top_list", "top_inst"]
    for d in sorted(dates, reverse=(direction == "near-first")):
        p = daily_priority[d]
        tasks.append(create_task("daily", {"trade_date": d}, None, p, "daily", start_date=d, end_date=d))
        if include_enhanced:
            for idx, api in enumerate(enhanced_apis, start=1):
                tasks.append(create_task(api, {"trade_date": d}, None, p + idx, "enhanced_daily", start_date=d, end_date=d))

    # Historical minute endpoint: fetch by symbol and date window.
    # 5min has ~49 bars/trading day, so 162 open days uses ~7938 rows/request,
    # close to the documented 8000-row response cap without exceeding it.
    # 1min uses monthly-ish windows to stay under the same cap.
    chunk_size = 162 if freq == "5min" else 25
    for _, row in universe.iterrows():
        ts_code = row["ts_code"]
        list_date = str(row.get("list_date") or "")
        delist_date = str(row.get("delist_date") or "")
        stock_dates = [d for d in dates if (not list_date or d >= list_date) and (not delist_date or d <= delist_date)]
        for s, e in chunk_dates(stock_dates, chunk_size, direction):
            params = {
                "ts_code": ts_code,
                "start_date": f"{iso_date(s)} 09:30:00",
                "end_date": f"{iso_date(e)} 15:00:00",
                "freq": freq,
            }
            tasks.append(
                create_task(
                    "stk_mins",
                    params,
                    None,
                    minute_priority.get(e, 10_000),
                    "minutes",
                    ts_code=ts_code,
                    start_date=s,
                    end_date=e,
                    freq=freq,
                )
            )

    if include_enhanced:
        tasks.append(create_task("concept", {}, None, 0, "static"))
        tasks.append(create_task("ths_index", {"exchange": "A", "type": "N"}, None, 1, "static"))
        if include_memberships:
            # Build these after a first pass has downloaded concept/ths_index if possible.
            # For now we can create concept_detail by stock and ths_member by known bootstrap fetches if present.
            for _, row in universe.iterrows():
                ts_code = row["ts_code"]
                tasks.append(create_task("concept_detail", {"ts_code": ts_code}, None, 900_000, "membership", ts_code=ts_code))
            ths_path = RAW_DIR / "ths_index" / f"{task_hash('ths_index', {'exchange': 'A', 'type': 'N'}, None)[:12]}.parquet"
            if ths_path.exists():
                ths_index = pd.read_parquet(ths_path)
                for idx, code in enumerate(ths_index.get("ts_code", pd.Series(dtype=str)).dropna().astype(str).unique()):
                    tasks.append(create_task("ths_member", {"ts_code": code}, None, 910_000 + idx, "membership", ts_code=code))

    return tasks


def command_init(args: argparse.Namespace) -> None:
    ensure_dirs()
    client = TushareProxyClient(get_token(), get_proxy_url(), timeout=args.timeout)
    stock_basic, trade_cal = bootstrap_metadata(client, ymd(args.start), ymd(args.end))
    tasks = build_tasks(
        stock_basic=stock_basic,
        trade_cal=trade_cal,
        start=args.start,
        end=args.end,
        freq=args.freq,
        direction=args.direction,
        include_enhanced=args.include_enhanced,
        include_memberships=args.include_memberships,
        include_bj=args.include_bj,
        max_symbols=args.max_symbols,
    )
    con = connect_db()
    inserted, existing = insert_tasks(con, tasks)
    con.close()
    universe_count = len(active_universe(stock_basic, args.start, args.end, args.include_bj))
    print(
        json.dumps(
            {
                "db_path": str(DB_PATH),
                "start": args.start,
                "end": args.end,
                "freq": args.freq,
                "direction": args.direction,
                "universe_count": universe_count,
                "tasks_inserted": inserted,
                "tasks_existing": existing,
                "tasks_total_planned": len(tasks),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_status(args: argparse.Namespace) -> None:
    con = connect_db()
    summary = pd.read_sql_query(
        """
        SELECT status, COUNT(*) AS tasks, COALESCE(SUM(row_count), 0) AS rows
        FROM tasks GROUP BY status ORDER BY status
        """,
        con,
    )
    by_api = pd.read_sql_query(
        """
        SELECT api_name, status, COUNT(*) AS tasks, COALESCE(SUM(row_count), 0) AS rows
        FROM tasks GROUP BY api_name, status ORDER BY api_name, status
        """,
        con,
    )
    errors = pd.read_sql_query(
        """
        SELECT api_name, ts_code, start_date, end_date, attempts, error_msg, updated_at
        FROM tasks
        WHERE status='failed'
        ORDER BY updated_at DESC
        LIMIT 10
        """,
        con,
    )
    con.close()
    print("== status ==")
    print(summary.to_string(index=False) if not summary.empty else "no tasks")
    print("\n== by api ==")
    print(by_api.to_string(index=False) if not by_api.empty else "no tasks")
    if not errors.empty:
        print("\n== recent errors ==")
        print(errors.to_string(index=False))


def reset_running(con: sqlite3.Connection) -> None:
    con.execute(
        """
        UPDATE tasks
        SET status='pending', updated_at=?, error_msg='reset stale running task'
        WHERE status='running'
        """,
        (now_iso(),),
    )
    con.commit()


def today_request_count(con: sqlite3.Connection) -> int:
    start_utc, end_utc = beijing_day_utc_bounds()
    row = con.execute(
        "SELECT COUNT(*) AS n FROM request_log WHERE requested_at >= ? AND requested_at < ?",
        (start_utc, end_utc),
    ).fetchone()
    return int(row["n"] if row else 0)


def fetch_next_task(con: sqlite3.Connection) -> sqlite3.Row | None:
    return con.execute(
        """
        SELECT * FROM tasks
        WHERE status IN ('pending', 'failed') AND not_before <= ? AND attempts < 8
        ORDER BY
            CASE status WHEN 'pending' THEN 0 ELSE 1 END ASC,
            priority ASC,
            attempts ASC,
            task_id ASC
        LIMIT 1
        """,
        (time.time(),),
    ).fetchone()


def mark_running(con: sqlite3.Connection, task_id: str) -> None:
    con.execute(
        "UPDATE tasks SET status='running', attempts=attempts+1, updated_at=? WHERE task_id=?",
        (now_iso(), task_id),
    )
    con.commit()


def mark_done(con: sqlite3.Connection, task_id: str, row_count: int, checksum: str, seconds: float) -> None:
    ts = now_iso()
    con.execute(
        """
        UPDATE tasks
        SET status='done', row_count=?, checksum=?, request_seconds=?, error_msg=NULL, updated_at=?
        WHERE task_id=?
        """,
        (row_count, checksum, seconds, ts, task_id),
    )
    con.execute(
        """
        INSERT INTO request_log(task_id, api_name, requested_at, status, row_count, request_seconds)
        SELECT task_id, api_name, ?, 'done', ?, ? FROM tasks WHERE task_id=?
        """,
        (ts, row_count, seconds, task_id),
    )
    con.commit()


def mark_failed(con: sqlite3.Connection, task_id: str, error: str, seconds: float, cooldown: int) -> None:
    ts = now_iso()
    con.execute(
        """
        UPDATE tasks
        SET status='failed', error_msg=?, request_seconds=?, not_before=?, updated_at=?
        WHERE task_id=?
        """,
        (error[:1000], seconds, time.time() + cooldown, ts, task_id),
    )
    con.execute(
        """
        INSERT INTO request_log(task_id, api_name, requested_at, status, row_count, request_seconds, error_msg)
        SELECT task_id, api_name, ?, 'failed', 0, ?, ? FROM tasks WHERE task_id=?
        """,
        (ts, seconds, error[:1000], task_id),
    )
    con.commit()


def command_run(args: argparse.Namespace) -> None:
    token = get_token()
    client = TushareProxyClient(token, get_proxy_url(), timeout=args.timeout)
    con = connect_db()
    if args.reset_running:
        reset_running(con)
    limiter = RateLimiter(args.requests_per_minute)
    completed = 0
    failed = 0
    start_count = today_request_count(con)
    print(f"starting downloader: already_used_today={start_count}, max_requests={args.max_requests}")

    try:
        while True:
            used_today = today_request_count(con)
            if used_today >= args.max_requests:
                print(f"stop: reached max_requests={args.max_requests} for today")
                break
            if args.max_tasks and completed + failed >= args.max_tasks:
                print(f"stop: reached max_tasks={args.max_tasks}")
                break

            task = fetch_next_task(con)
            if task is None:
                print("stop: no runnable tasks")
                break

            path = Path(task["output_path"])
            if path.exists() and path.stat().st_size > 0 and not args.refresh_done:
                checksum = sha1_file(path)
                mark_done(con, task["task_id"], int(pd.read_parquet(path).shape[0]), checksum, 0.0)
                completed += 1
                continue

            mark_running(con, task["task_id"])
            params = json.loads(task["params_json"])
            limiter.wait()
            started = time.time()
            try:
                df = client.call(task["api_name"], params, task["fields"])
                checksum = write_parquet_atomic(df, task["output_path"])
                seconds = time.time() - started
                mark_done(con, task["task_id"], len(df), checksum, seconds)
                completed += 1
                print(
                    f"done api={task['api_name']} rows={len(df)} "
                    f"task={task['task_id'][:8]} seconds={seconds:.2f} path={task['output_path']}",
                    flush=True,
                )
            except TushareRateLimit as exc:
                seconds = time.time() - started
                error = redact_token(repr(exc), token)
                mark_failed(con, task["task_id"], error, seconds, args.cooldown_seconds)
                failed += 1
                print(f"cooldown api={task['api_name']} task={task['task_id'][:8]} error={error}", file=sys.stderr)
                if args.stop_on_cooldown:
                    break
            except Exception as exc:
                seconds = time.time() - started
                error = redact_token(repr(exc), token)
                cooldown = min(1800, 60 * max(1, int(task["attempts"]) + 1))
                mark_failed(con, task["task_id"], error, seconds, cooldown)
                failed += 1
                print(f"failed api={task['api_name']} task={task['task_id'][:8]} error={error}", file=sys.stderr)
    finally:
        con.close()

    print(json.dumps({"completed": completed, "failed": failed}, ensure_ascii=False, indent=2))


def command_audit_files(args: argparse.Namespace) -> None:
    con = connect_db()
    rows = con.execute(
        """
        SELECT task_id, output_path, row_count, checksum FROM tasks
        WHERE status='done'
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (args.limit,),
    ).fetchall()
    checked = 0
    bad: list[dict[str, Any]] = []
    for row in rows:
        path = Path(row["output_path"])
        checked += 1
        if not path.exists():
            bad.append({"task_id": row["task_id"], "error": "missing_file", "path": str(path)})
            continue
        checksum = sha1_file(path)
        if checksum != row["checksum"]:
            bad.append({"task_id": row["task_id"], "error": "checksum_mismatch", "path": str(path)})
    con.close()
    print(json.dumps({"checked": checked, "bad": bad}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tushare proxy downloader with resumable task manifest")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="bootstrap metadata and create resumable download tasks")
    init.add_argument("--start", default="2024-05-21")
    init.add_argument("--end", default="2026-05-21")
    init.add_argument("--freq", choices=["5min", "1min"], default="5min")
    init.add_argument("--direction", choices=["near-first", "far-first"], default="near-first")
    init.add_argument("--include-enhanced", action="store_true", help="include auctions, moneyflow, top list, concepts")
    init.add_argument("--include-memberships", action="store_true", help="include concept_detail per stock; ths_member if ths_index exists")
    init.add_argument("--include-bj", action="store_true", help="include Beijing Stock Exchange symbols")
    init.add_argument("--max-symbols", type=int, default=None, help="small-sample task creation for testing")
    init.add_argument("--timeout", type=int, default=60)
    init.set_defaults(func=command_init)

    run = sub.add_parser("run", help="run pending/failed tasks with rate limit and retry support")
    run.add_argument("--requests-per-minute", type=int, default=120)
    run.add_argument("--max-requests", type=int, default=9000)
    run.add_argument("--max-tasks", type=int, default=0, help="0 means no task-count cap")
    run.add_argument("--timeout", type=int, default=60)
    run.add_argument("--cooldown-seconds", type=int, default=600)
    run.add_argument("--stop-on-cooldown", action="store_true", default=True)
    run.add_argument("--no-stop-on-cooldown", dest="stop_on_cooldown", action="store_false")
    run.add_argument("--reset-running", action="store_true", default=True)
    run.add_argument("--refresh-done", action="store_true", help="re-download even if output file exists")
    run.set_defaults(func=command_run)

    status = sub.add_parser("status", help="show task progress")
    status.set_defaults(func=command_status)

    audit = sub.add_parser("audit-files", help="sample-check done task files")
    audit.add_argument("--limit", type=int, default=200)
    audit.set_defaults(func=command_audit_files)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
