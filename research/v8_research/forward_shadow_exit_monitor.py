#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "tushare" / "v8_forward_shadow"
DEFAULT_MINUTE_DIR = ROOT / "data_tushare" / "raw" / "stk_mins" / "freq=5min"
DEFAULT_TOPIC_ID = 44635

BUY_COMMISSION = 0.00025
SELL_COMMISSION = 0.00025
STAMP_TAX = 0.00050

TAKE_PROFIT_0935 = 0.015
TAKE_PROFIT_0945 = 0.012
STOP_LOSS = -0.018
WEAK_LOSS = -0.004

CHECKPOINT_RULES = {
    "check_0935": {"check_time": "09:35", "exec_time": "09:40"},
    "check_0945": {"check_time": "09:45", "exec_time": "09:50"},
    "check_1000": {"check_time": "10:00", "exec_time": "10:05"},
}
EXEC_CHECKPOINTS = {
    "exec_0940": "09:40",
    "exec_0950": "09:50",
    "exec_1005": "10:05",
}
ALL_CHECKPOINTS = [
    *CHECKPOINT_RULES.keys(),
    *EXEC_CHECKPOINTS.keys(),
    "prealert_1025",
    "default_1030",
    "settle_all",
]


@dataclass(frozen=True)
class ExitDecision:
    should_sell: bool
    reason: str
    check_time: str
    expected_exit_time: str
    ret_now: float
    check_price: float
    recommended_sell_price: float
    price_source: str


def bj_now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def num(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def pct(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"{value * 100:.2f}%"


def price(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"{value:.3f}"


def short_strategy_id(value: str) -> str:
    mapping = {
        "S0_v7_original_top10": "S0",
        "S1_U2_filter_only_no_refill": "S1",
        "S0_v7_original_top10_tail_down": "S0+TD",
        "S1_U2_filter_only_no_refill_tail_down": "S1+TD",
    }
    return mapping.get(value, value)


def board_label(code: Any) -> str:
    text = str(code or "").upper()
    raw = text.split(".", 1)[0]
    suffix = text.split(".", 1)[1] if "." in text else ""
    if suffix == "BJ" or raw.startswith(("43", "83", "87", "88", "92")):
        return "北交所"
    if raw.startswith(("688", "689")):
        return "科创板"
    if raw.startswith(("300", "301")):
        return "创业板"
    if raw.startswith(("900", "200")):
        return "B股"
    if suffix == "SH" or raw.startswith(("600", "601", "603", "605", "609")):
        return "沪主板"
    if suffix == "SZ" or raw.startswith(("000", "001", "002", "003")):
        return "深主板"
    return "未知"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def vwap_from_amount_vol(
    amount_value: Any,
    vol_value: Any,
    reference_price: Any,
    low_price: Any = np.nan,
    high_price: Any = np.nan,
) -> float:
    amount = num(amount_value)
    vol = num(vol_value)
    reference = num(reference_price)
    fallback = reference if np.isfinite(reference) and reference > 0 else np.nan
    if not (np.isfinite(amount) and amount > 0 and np.isfinite(vol) and vol > 0):
        return fallback
    base = amount / vol
    candidates = [base, base * 100.0, base / 100.0]
    candidates = [c for c in candidates if np.isfinite(c) and c > 0]
    if not candidates:
        return fallback
    low = num(low_price)
    high = num(high_price)
    if np.isfinite(low) and np.isfinite(high) and low > 0 and high >= low:
        lower = low * 0.98
        upper = high * 1.02
        bounded = [c for c in candidates if lower <= c <= upper]
        if bounded:
            return float(min(bounded, key=lambda c: abs(c / reference - 1.0) if np.isfinite(fallback) else 0.0))
    if np.isfinite(fallback) and fallback > 0:
        closest = min(candidates, key=lambda c: abs(c / reference - 1.0))
        if abs(closest / reference - 1.0) <= 0.10:
            return float(closest)
    return fallback


def bar_vwap(row: pd.Series | dict[str, Any] | None) -> float:
    if row is None:
        return np.nan
    return vwap_from_amount_vol(
        row.get("amount"),
        row.get("vol"),
        row.get("close"),
        row.get("low"),
        row.get("high"),
    )


def net_return(gross: float, slippage_bp: float, buy_impact: float = 0.0, sell_impact: float = 0.0) -> float:
    if not np.isfinite(gross):
        return np.nan
    slip = slippage_bp / 10000.0
    return (1.0 + gross) * (1.0 - slip - sell_impact - SELL_COMMISSION - STAMP_TAX) / (
        1.0 + slip + buy_impact + BUY_COMMISSION
    ) - 1.0


def impact_cost(participation: float) -> float:
    if not np.isfinite(participation) or participation <= 0:
        return 0.0
    return float(min(0.0050, 0.0015 * np.sqrt(participation)))


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_无新增记录_"
    text = df.loc[:, [c for c in columns if c in df.columns]].head(max_rows).copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        else:
            text[col] = text[col].fillna("").astype(str)
    rows = text.astype(str).to_numpy().tolist()
    widths = [len(str(c)) for c in text.columns]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    header = "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(text.columns)) + " |"
    sep = "| " + " | ".join("---".ljust(widths[i]) for i in range(len(text.columns))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |" for row in rows]
    if len(df) > max_rows:
        body.append(f"| ... | 共 {len(df)} 行，仅展示前 {max_rows} 行 |")
    return "\n".join([header, sep, *body])


def send_wxpusher(title: str, content: str, topic_id: int, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"status": "dry_run", "summary": title, "response_code": None, "response_text": "dry-run"}
    token = os.environ.get("WXPUSHER_APP_TOKEN", "").strip()
    if not token:
        return {"status": "skipped_missing_token", "summary": title, "response_code": None, "response_text": "missing WXPUSHER_APP_TOKEN"}
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
        return {
            "status": "success" if resp.status_code == 200 else "failed",
            "summary": title,
            "response_code": resp.status_code,
            "response_text": resp.text[:1000],
        }
    except requests.RequestException as exc:
        return {"status": "failed", "summary": title, "response_code": None, "response_text": repr(exc)}


class MinuteStore:
    def __init__(self, minute_dir: Path) -> None:
        self.minute_dir = minute_dir

    def _candidate_files(self, code: str, trade_date: str) -> list[Path]:
        symbol_dir = self.minute_dir / f"ts_code={code}"
        if not symbol_dir.exists():
            return []
        files: list[Path] = []
        for path in sorted(symbol_dir.glob("*.parquet")):
            stem = path.stem
            parts = stem.split("_")
            if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
                start = parts[-2]
                end = parts[-1]
                if start <= trade_date <= end:
                    files.append(path)
            elif trade_date in stem:
                files.append(path)
        return files

    @lru_cache(maxsize=4096)
    def symbol_day(self, code: str, trade_date: str) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for path in self._candidate_files(code, trade_date):
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            if df.empty:
                continue
            df = df.copy()
            if "ts_code" not in df.columns:
                df["ts_code"] = code
            if "trade_date" in df.columns:
                df["trade_date"] = df["trade_date"].astype(str)
                df = df[df["trade_date"].eq(trade_date)]
            if "dt" in df.columns:
                dt = pd.to_datetime(df["dt"], errors="coerce")
            elif "trade_time" in df.columns:
                raw_time = df["trade_time"].astype(str)
                dt = pd.to_datetime(trade_date + " " + raw_time.str[-8:], errors="coerce")
            elif "datetime" in df.columns:
                dt = pd.to_datetime(df["datetime"], errors="coerce")
            else:
                continue
            df["dt"] = dt
            df = df[df["dt"].notna()]
            df = df[df["dt"].dt.strftime("%Y%m%d").eq(trade_date)]
            if df.empty:
                continue
            df["bar_time"] = df["dt"].dt.strftime("%H:%M")
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates(["ts_code", "bar_time"], keep="last").sort_values("dt")
        for col in ["open", "high", "low", "close", "vol", "amount"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    def bar(self, code: str, trade_date: str, bar_time: str) -> pd.Series | None:
        df = self.symbol_day(code, trade_date)
        if df.empty or "bar_time" not in df.columns:
            return None
        rows = df[df["bar_time"].eq(bar_time)]
        if rows.empty:
            return None
        return rows.iloc[-1]

    def next_tradeable_bar(self, code: str, trade_date: str, start_time: str, latest_time: str | None = None) -> pd.Series | None:
        df = self.symbol_day(code, trade_date)
        if df.empty:
            return None
        rows = df[df["bar_time"].ge(start_time)].copy()
        if latest_time:
            rows = rows[rows["bar_time"].le(latest_time)]
        if rows.empty:
            return None
        amount = pd.to_numeric(rows.get("amount"), errors="coerce").fillna(0)
        vol = pd.to_numeric(rows.get("vol"), errors="coerce").fillna(0)
        rows = rows[(amount > 0) & (vol > 0)]
        if rows.empty:
            return None
        return rows.iloc[0]


def load_entry_file(output_root: Path, signal_date: str) -> pd.DataFrame:
    path = output_root / "daily_entry_prices" / f"{signal_date}_entry_prices.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing entry file: {path}")
    df = pd.read_csv(path)
    if "strategy_id" not in df.columns and "line_id" in df.columns:
        df["strategy_id"] = df["line_id"]
    if "line_id" not in df.columns:
        df["line_id"] = df["strategy_id"]
    if "weight" not in df.columns and "position_weight" in df.columns:
        df["weight"] = df["position_weight"]
    df["signal_date"] = signal_date
    df["code"] = df["code"].astype(str)
    df["strategy_id"] = df["strategy_id"].astype(str)
    df["line_id"] = df["line_id"].astype(str)
    df["entry_vwap"] = pd.to_numeric(df.get("entry_vwap"), errors="coerce")
    df["weight"] = pd.to_numeric(df.get("weight", 1.0), errors="coerce").fillna(0.0)
    if "paper_entry_status" not in df.columns:
        df["paper_entry_status"] = np.where(df["entry_vwap"].notna(), "entry_recorded", "missing_entry")
    return df


def output_paths(output_root: Path, signal_date: str, settlement_date: str) -> dict[str, Path]:
    return {
        "recommendations": output_root / "daily_exit_recommendations" / f"{signal_date}_{settlement_date}_exit_recommendations.csv",
        "execution": output_root / "daily_exit_execution" / f"{signal_date}_{settlement_date}_exit_execution.csv",
        "settlement": output_root / "daily_exit_settlement" / f"{signal_date}_{settlement_date}_exit_settlement.csv",
        "ledger": output_root / "daily_ledgers" / f"{signal_date}_{settlement_date}_exit_ledgers.csv",
        "quality": output_root / "daily_execution_quality" / f"{signal_date}_{settlement_date}_exit_quality.csv",
        "push_logs": output_root / "push_logs" / f"{signal_date}_{settlement_date}_exit_push_logs.csv",
        "sha256": output_root / "sha256" / f"{signal_date}_{settlement_date}_exit_monitor_sha256.csv",
    }


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def write_replace(df: pd.DataFrame, path: Path, keys: list[str]) -> None:
    ensure_dir(path.parent)
    if path.exists() and not df.empty:
        old = pd.read_csv(path)
        combined = pd.concat([old, df], ignore_index=True, sort=False)
    else:
        combined = df.copy()
    if not combined.empty and keys:
        combined = combined.drop_duplicates(keys, keep="last")
    combined.to_csv(path, index=False)


def decide_exit(entry_vwap: float, check_time: str, exec_time: str, bars: dict[str, pd.Series | None]) -> ExitDecision:
    check = bars.get(check_time)
    if check is None or not np.isfinite(entry_vwap) or entry_vwap <= 0:
        return ExitDecision(False, "missing_check_bar", check_time, exec_time, np.nan, np.nan, np.nan, "")
    check_close = num(check.get("close"))
    ret_now = check_close / entry_vwap - 1.0 if np.isfinite(check_close) and entry_vwap > 0 else np.nan
    should_sell = False
    reason = ""
    if np.isfinite(ret_now) and ret_now <= STOP_LOSS:
        should_sell = True
        reason = "stop_loss"
    elif check_time == "09:35" and np.isfinite(ret_now) and ret_now >= TAKE_PROFIT_0935:
        should_sell = True
        reason = "take_profit_0935"
    elif check_time == "09:45" and np.isfinite(ret_now) and ret_now >= TAKE_PROFIT_0945:
        prev = bars.get("09:40")
        prev_close = num(prev.get("close")) if prev is not None else np.nan
        should_sell = bool(not np.isfinite(prev_close) or check_close <= prev_close)
        reason = "take_profit_0945" if should_sell else ""
    elif check_time == "09:45" and np.isfinite(ret_now) and ret_now <= WEAK_LOSS:
        open_bar = bars.get("09:35")
        open_0935 = num(open_bar.get("open")) if open_bar is not None else np.nan
        should_sell = bool(np.isfinite(open_0935) and check_close < open_0935)
        reason = "weak_open" if should_sell else ""
    elif check_time == "10:00" and np.isfinite(ret_now) and ret_now < 0:
        should_sell = True
        reason = "not_recovered_1000"
    if not should_sell:
        return ExitDecision(False, "", check_time, exec_time, ret_now, check_close, np.nan, "")
    return ExitDecision(
        True,
        reason,
        check_time,
        exec_time,
        ret_now,
        check_close,
        check_close,
        f"check_bar_{check_time}_close_reference",
    )


def already_recommended(existing: pd.DataFrame, strategy_id: str, code: str) -> bool:
    if existing.empty:
        return False
    key = existing["strategy_id"].astype(str).eq(strategy_id) & existing["code"].astype(str).eq(code)
    return bool(key.any())


def build_decisions(
    entries: pd.DataFrame,
    existing_recs: pd.DataFrame,
    minute_store: MinuteStore,
    signal_date: str,
    settlement_date: str,
    checkpoint: str,
) -> pd.DataFrame:
    rule = CHECKPOINT_RULES[checkpoint]
    check_time = rule["check_time"]
    exec_time = rule["exec_time"]
    rows: list[dict[str, Any]] = []
    for _, entry in entries.iterrows():
        strategy_id = str(entry["strategy_id"])
        code = str(entry["code"])
        if str(entry.get("paper_entry_status", "")) != "entry_recorded":
            continue
        if already_recommended(existing_recs, strategy_id, code):
            continue
        bars = {
            "09:35": minute_store.bar(code, settlement_date, "09:35"),
            "09:40": minute_store.bar(code, settlement_date, "09:40"),
            "09:45": minute_store.bar(code, settlement_date, "09:45"),
            "10:00": minute_store.bar(code, settlement_date, "10:00"),
        }
        decision = decide_exit(float(entry["entry_vwap"]), check_time, exec_time, bars)
        if not decision.should_sell:
            continue
        rows.append(
            {
                "signal_date": signal_date,
                "settlement_date": settlement_date,
                "checkpoint": checkpoint,
                "decision_time": check_time,
                "expected_exit_time": exec_time,
                "strategy_id": strategy_id,
                "line_id": str(entry.get("line_id", strategy_id)),
                "code": code,
                "name": str(entry.get("name", "")),
                "weight": float(entry.get("weight", 0.0)),
                "entry_vwap": float(entry["entry_vwap"]),
                "check_price": decision.check_price,
                "return_at_check": decision.ret_now,
                "exit_reason": decision.reason,
                "recommended_sell_price": decision.recommended_sell_price,
                "recommendation_price_source": decision.price_source,
                "paper_only": True,
                "created_at_beijing": bj_now(),
                "recommendation_status": "recommended",
            }
        )
    return pd.DataFrame(rows)


def default_recommendations(
    entries: pd.DataFrame,
    existing_recs: pd.DataFrame,
    minute_store: MinuteStore,
    signal_date: str,
    settlement_date: str,
    checkpoint: str = "default_1030",
    decision_time: str = "10:30",
    reference_time: str = "10:30",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, entry in entries.iterrows():
        strategy_id = str(entry["strategy_id"])
        code = str(entry["code"])
        if str(entry.get("paper_entry_status", "")) != "entry_recorded":
            continue
        if already_recommended(existing_recs, strategy_id, code):
            continue
        bar = minute_store.bar(code, settlement_date, reference_time)
        ref = bar_vwap(bar) if bar is not None else np.nan
        source = f"check_bar_{reference_time}_vwap_reference" if np.isfinite(ref) else f"missing_{reference_time}_bar"
        rows.append(
            {
                "signal_date": signal_date,
                "settlement_date": settlement_date,
                "checkpoint": checkpoint,
                "decision_time": decision_time,
                "expected_exit_time": "10:30",
                "strategy_id": strategy_id,
                "line_id": str(entry.get("line_id", strategy_id)),
                "code": code,
                "name": str(entry.get("name", "")),
                "weight": float(entry.get("weight", 0.0)),
                "entry_vwap": float(entry["entry_vwap"]),
                "check_price": ref,
                "return_at_check": ref / float(entry["entry_vwap"]) - 1.0 if np.isfinite(ref) and float(entry["entry_vwap"]) > 0 else np.nan,
                "exit_reason": "time_exit",
                "recommended_sell_price": ref,
                "recommendation_price_source": source,
                "paper_only": True,
                "created_at_beijing": bj_now(),
                "recommendation_status": "recommended" if np.isfinite(ref) else "pending_missing_reference_bar",
            }
        )
    return pd.DataFrame(rows)


def execution_rows_for_time(
    recs: pd.DataFrame,
    existing_exec: pd.DataFrame,
    entries: pd.DataFrame,
    minute_store: MinuteStore,
    signal_date: str,
    settlement_date: str,
    exec_time: str,
) -> pd.DataFrame:
    if recs.empty:
        return pd.DataFrame()
    due = recs[recs["expected_exit_time"].astype(str).eq(exec_time)].copy()
    if due.empty:
        return pd.DataFrame()
    done_keys = set()
    if not existing_exec.empty:
        for _, row in existing_exec.iterrows():
            done_keys.add((str(row.get("strategy_id")), str(row.get("code"))))
    entry_map = {
        (str(row["strategy_id"]), str(row["code"])): row
        for _, row in entries.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for _, rec in due.iterrows():
        strategy_id = str(rec["strategy_id"])
        code = str(rec["code"])
        if (strategy_id, code) in done_keys:
            continue
        bar = minute_store.bar(code, settlement_date, exec_time)
        status = "exit_recorded"
        actual_time = exec_time
        delay_note = ""
        if bar is None or num(bar.get("amount"), 0.0) <= 0 or num(bar.get("vol"), 0.0) <= 0:
            next_bar = minute_store.next_tradeable_bar(code, settlement_date, exec_time, latest_time="10:30")
            if next_bar is None:
                status = "pending_missing_or_zero_amount_bar"
                actual_time = ""
                exit_vwap = np.nan
                exit_amount = np.nan
                delay_note = "no_tradeable_bar_until_1030"
            else:
                bar = next_bar
                actual_time = str(bar["bar_time"])
                exit_vwap = bar_vwap(bar)
                exit_amount = num(bar.get("amount"))
                status = "exit_recorded_delayed"
                delay_note = f"delayed_from_{exec_time}_to_{actual_time}"
        else:
            exit_vwap = bar_vwap(bar)
            exit_amount = num(bar.get("amount"))
        entry = entry_map.get((strategy_id, code))
        entry_vwap = num(rec.get("entry_vwap"))
        weight = num(rec.get("weight"), 0.0)
        entry_amount = num(entry.get("entry_amount")) if entry is not None else np.nan
        buy_participation = 0.0
        sell_participation = 0.0
        if np.isfinite(entry_amount) and entry_amount > 0 and weight > 0:
            notional = 100000.0 * weight
            buy_participation = min(1.0, notional / entry_amount)
        if np.isfinite(exit_amount) and exit_amount > 0 and weight > 0:
            notional = 100000.0 * weight
            sell_participation = min(1.0, notional / exit_amount)
        buy_impact = impact_cost(buy_participation)
        sell_impact = impact_cost(sell_participation)
        gross = exit_vwap / entry_vwap - 1.0 if np.isfinite(exit_vwap) and np.isfinite(entry_vwap) and entry_vwap > 0 else np.nan
        rows.append(
            {
                "signal_date": signal_date,
                "settlement_date": settlement_date,
                "strategy_id": strategy_id,
                "line_id": str(rec.get("line_id", strategy_id)),
                "code": code,
                "name": str(rec.get("name", "")),
                "weight": weight,
                "entry_vwap": entry_vwap,
                "paper_exit_status": status,
                "expected_exit_time": exec_time,
                "actual_exit_time": actual_time,
                "exit_vwap": exit_vwap,
                "exit_amount": exit_amount,
                "exit_reason": str(rec.get("exit_reason", "")),
                "exit_price_source": f"raw_stk_mins_5min_bar_{actual_time}_vwap" if actual_time else "",
                "delay_note": delay_note,
                "gross_return": gross,
                "return_5bp": net_return(gross, 5.0),
                "return_10bp_impact": net_return(gross, 10.0, buy_impact, sell_impact),
                "buy_participation": buy_participation,
                "sell_participation": sell_participation,
                "buy_impact_cost": buy_impact,
                "sell_impact_cost": sell_impact,
                "estimated_impact_cost": buy_impact + sell_impact,
                "fill_ratio": 1.0 if status.startswith("exit_recorded") else 0.0,
                "partial_fill": False,
                "zero_fill": not status.startswith("exit_recorded"),
                "paper_only": True,
                "created_at_beijing": bj_now(),
            }
        )
    return pd.DataFrame(rows)


def build_strategy_ledgers(execution: pd.DataFrame) -> pd.DataFrame:
    if execution.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    settled = execution[execution["paper_exit_status"].astype(str).str.startswith("exit_recorded")].copy()
    if settled.empty:
        return pd.DataFrame()
    for (strategy_id, ledger_type), col in [
        ((None, "theoretical_5bp_ledger"), "return_5bp"),
        ((None, "execution_10bp_impact_ledger"), "return_10bp_impact"),
    ]:
        for sid, group in settled.groupby("strategy_id"):
            weights = pd.to_numeric(group["weight"], errors="coerce").fillna(0.0)
            returns = pd.to_numeric(group[col], errors="coerce")
            valid = group[returns.notna()].copy()
            if valid.empty:
                daily_return = np.nan
            else:
                w = pd.to_numeric(valid["weight"], errors="coerce").fillna(0.0)
                if w.sum() > 0:
                    w = w / w.sum()
                daily_return = float((pd.to_numeric(valid[col], errors="coerce") * w).sum())
            wins = int((returns > 0).sum())
            losses = int((returns <= 0).sum())
            gain = float(returns[returns > 0].sum()) if (returns > 0).any() else 0.0
            loss = float(-returns[returns < 0].sum()) if (returns < 0).any() else 0.0
            rows.append(
                {
                    "signal_date": str(group["signal_date"].iloc[0]),
                    "settlement_date": str(group["settlement_date"].iloc[0]),
                    "strategy_id": sid,
                    "ledger_type": ledger_type,
                    "num_positions": int(len(group)),
                    "num_winners": wins,
                    "num_losers": losses,
                    "daily_return": daily_return,
                    "avg_win": float(returns[returns > 0].mean()) if (returns > 0).any() else 0.0,
                    "avg_loss": float(returns[returns < 0].mean()) if (returns < 0).any() else 0.0,
                    "profit_factor": gain / loss if loss > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_quality(execution: pd.DataFrame, path: Path, signal_date: str, settlement_date: str) -> pd.DataFrame:
    if execution.empty:
        quality = pd.DataFrame(
            [
                {
                    "signal_date": signal_date,
                    "settlement_date": settlement_date,
                    "total_exit_records": 0,
                    "recorded": 0,
                    "delayed": 0,
                    "zero_fill": 0,
                    "avg_fill_ratio": np.nan,
                }
            ]
        )
    else:
        status = execution["paper_exit_status"].astype(str)
        quality = pd.DataFrame(
            [
                {
                    "signal_date": signal_date,
                    "settlement_date": settlement_date,
                    "total_exit_records": int(len(execution)),
                    "recorded": int(status.str.startswith("exit_recorded").sum()),
                    "delayed": int(status.eq("exit_recorded_delayed").sum()),
                    "zero_fill": int(status.str.contains("pending|zero", case=False, regex=True).sum()),
                    "avg_fill_ratio": float(pd.to_numeric(execution.get("fill_ratio"), errors="coerce").mean()),
                }
            ]
        )
    ensure_dir(path.parent)
    quality.to_csv(path, index=False)
    return quality


def write_sha(paths: dict[str, Path]) -> pd.DataFrame:
    sha_path = paths["sha256"]
    rows = []
    for key, path in paths.items():
        if key == "sha256" or not path.exists():
            continue
        rows.append({"file_key": key, "path": str(path), "sha256": sha256_file(path)})
    out = pd.DataFrame(rows)
    ensure_dir(sha_path.parent)
    out.to_csv(sha_path, index=False)
    return out


def push_new_recommendations(
    recs: pd.DataFrame,
    checkpoint: str,
    settlement_date: str,
    signal_date: str,
    topic_id: int,
    dry_run: bool,
    paths: dict[str, Path],
) -> None:
    if recs.empty:
        return
    display = recs.copy()
    display["line"] = display["strategy_id"].map(short_strategy_id)
    display["板块"] = display["code"].map(board_label)
    display["ret_now"] = display["return_at_check"].map(pct)
    display["entry"] = display["entry_vwap"].map(price)
    display["ref_sell"] = display["recommended_sell_price"].map(price)
    title = f"FS {settlement_date} {checkpoint} 纸面卖出提示: {len(display)}只"
    content = "\n".join(
        [
            f"# {title}",
            "",
            f"- signal_date: `{signal_date}`",
            "- 类型: `paper tracking only`",
            "- 说明: 推荐价为当前观察点可得参考价，实际 paper exit VWAP 在执行 bar 完成后记录。",
            "",
            md_table(display, ["line", "code", "板块", "name", "exit_reason", "expected_exit_time", "entry", "ref_sell", "ret_now"], max_rows=30),
        ]
    )
    result = send_wxpusher(title, content, topic_id, dry_run)
    log = pd.DataFrame([{**result, "signal_date": signal_date, "settlement_date": settlement_date, "checkpoint": checkpoint, "sent_at_beijing": bj_now()}])
    write_replace(log, paths["push_logs"], ["signal_date", "settlement_date", "checkpoint", "summary"])


def push_new_execution(
    execution: pd.DataFrame,
    checkpoint: str,
    settlement_date: str,
    signal_date: str,
    topic_id: int,
    dry_run: bool,
    paths: dict[str, Path],
) -> None:
    if execution.empty:
        return
    display = execution.copy()
    display["line"] = display["strategy_id"].map(short_strategy_id)
    display["板块"] = display["code"].map(board_label)
    display["entry"] = display["entry_vwap"].map(price)
    display["exit"] = display["exit_vwap"].map(price)
    display["ret_5bp"] = display["return_5bp"].map(pct)
    display["ret_10bp_impact"] = display["return_10bp_impact"].map(pct)
    title = f"FS {settlement_date} {checkpoint} 纸面退出记录: {len(display)}只"
    content = "\n".join(
        [
            f"# {title}",
            "",
            f"- signal_date: `{signal_date}`",
            "- 类型: `paper tracking only`",
            "- 说明: 仅记录纸面账本，不代表实盘、模拟盘或交易建议。",
            "",
            md_table(display, ["line", "code", "板块", "name", "actual_exit_time", "exit_reason", "entry", "exit", "ret_5bp", "ret_10bp_impact"], max_rows=30),
        ]
    )
    result = send_wxpusher(title, content, topic_id, dry_run)
    log = pd.DataFrame([{**result, "signal_date": signal_date, "settlement_date": settlement_date, "checkpoint": checkpoint, "sent_at_beijing": bj_now()}])
    write_replace(log, paths["push_logs"], ["signal_date", "settlement_date", "checkpoint", "summary"])


def ledger_type_text(value: str) -> str:
    mapping = {
        "theoretical_5bp_ledger": "5bp理论",
        "execution_10bp_impact_ledger": "10bp+impact",
    }
    return mapping.get(value, value)


def push_log_sent(paths: dict[str, Path], checkpoint: str, summary: str, dry_run: bool) -> bool:
    log = read_csv_if_exists(paths["push_logs"])
    if log.empty:
        return False
    if "checkpoint" not in log.columns or "summary" not in log.columns:
        return False
    rows = log[
        log["checkpoint"].astype(str).eq(checkpoint)
        & log["summary"].astype(str).eq(summary)
    ].copy()
    if rows.empty:
        return False
    statuses = rows.get("status", pd.Series(dtype=str)).astype(str)
    if dry_run:
        return bool(statuses.isin(["success", "dry_run"]).any())
    return bool(statuses.eq("success").any())


def send_logged_wxpusher(
    title: str,
    content: str,
    checkpoint: str,
    signal_date: str,
    settlement_date: str,
    topic_id: int,
    dry_run: bool,
    paths: dict[str, Path],
) -> None:
    if push_log_sent(paths, checkpoint, title, dry_run):
        return
    result = send_wxpusher(title, content, topic_id, dry_run)
    log = pd.DataFrame(
        [
            {
                **result,
                "signal_date": signal_date,
                "settlement_date": settlement_date,
                "checkpoint": checkpoint,
                "sent_at_beijing": bj_now(),
            }
        ]
    )
    write_replace(log, paths["push_logs"], ["signal_date", "settlement_date", "checkpoint", "summary"])


def strategy_daily_return(group: pd.DataFrame, return_col: str) -> float:
    returns = pd.to_numeric(group.get(return_col), errors="coerce")
    valid = group[returns.notna()].copy()
    if valid.empty:
        return np.nan
    weights = pd.to_numeric(valid.get("weight"), errors="coerce").fillna(0.0)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = pd.Series(np.full(len(valid), 1.0 / len(valid)), index=valid.index)
    return float((pd.to_numeric(valid[return_col], errors="coerce") * weights).sum())


def push_final_settlement_summary(
    execution: pd.DataFrame,
    ledgers: pd.DataFrame,
    quality: pd.DataFrame,
    signal_date: str,
    settlement_date: str,
    topic_id: int,
    dry_run: bool,
    paths: dict[str, Path],
) -> None:
    total = int(len(execution))
    quality_row = quality.iloc[0].to_dict() if not quality.empty else {}
    recorded = int(num(quality_row.get("recorded"), 0))
    delayed = int(num(quality_row.get("delayed"), 0))
    zero_fill = int(num(quality_row.get("zero_fill"), 0))
    avg_fill_ratio = num(quality_row.get("avg_fill_ratio"))
    headline = (
        f"- signal_date: `{signal_date}`\n"
        f"- settlement_date: `{settlement_date}`\n"
        "- 类型: `paper tracking only`\n"
        "- 说明: 仅为纸面跟踪结算汇总，不代表实盘、模拟盘或交易建议。\n"
        f"- 退出记录: `{total}`；已记录: `{recorded}`；延迟成交: `{delayed}`；零/缺失成交: `{zero_fill}`；"
        f"平均成交比例: `{pct(avg_fill_ratio) if np.isfinite(avg_fill_ratio) else ''}`"
    )

    if execution.empty:
        title = f"FS {settlement_date} 今日退出汇总: 无退出记录"
        content = "\n".join([f"# {title}", "", headline])
        send_logged_wxpusher(title, content, "settlement_summary_trades_01", signal_date, settlement_date, topic_id, dry_run, paths)
    else:
        display = execution.copy()
        display["line"] = display["strategy_id"].map(short_strategy_id)
        display["板块"] = display["code"].map(board_label)
        display["entry"] = display["entry_vwap"].map(price)
        display["exit"] = display["exit_vwap"].map(price)
        display["ret_5bp"] = display["return_5bp"].map(pct)
        display["ret_10bp_impact"] = display["return_10bp_impact"].map(pct)
        display["fill"] = display["fill_ratio"].map(pct)
        reason_counts = (
            display.groupby(["actual_exit_time", "exit_reason"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["actual_exit_time", "exit_reason"])
        )
        chunk_size = 18
        chunks = [display.iloc[i : i + chunk_size].copy() for i in range(0, len(display), chunk_size)]
        for idx, chunk in enumerate(chunks, start=1):
            suffix = f" {idx}/{len(chunks)}" if len(chunks) > 1 else ""
            title = f"FS {settlement_date} 今日退出汇总{suffix}: {total}条"
            content_parts = [
                f"# {title}",
                "",
                headline,
            ]
            if idx == 1:
                content_parts.extend(
                    [
                        "",
                        "## 退出时间/原因分布",
                        md_table(reason_counts, ["actual_exit_time", "exit_reason", "count"], max_rows=20),
                    ]
                )
            content_parts.extend(
                [
                    "",
                    "## 逐笔退出记录",
                    md_table(
                        chunk,
                        ["line", "code", "板块", "name", "actual_exit_time", "exit_reason", "entry", "exit", "ret_5bp", "ret_10bp_impact", "fill"],
                        max_rows=chunk_size,
                    ),
                ]
            )
            send_logged_wxpusher(
                title,
                "\n".join(content_parts),
                f"settlement_summary_trades_{idx:02d}",
                signal_date,
                settlement_date,
                topic_id,
                dry_run,
                paths,
            )

    if ledgers.empty:
        ledger_display = pd.DataFrame()
    else:
        ledger_display = ledgers.copy()
        ledger_display["line"] = ledger_display["strategy_id"].map(short_strategy_id)
        ledger_display["ledger"] = ledger_display["ledger_type"].map(ledger_type_text)
        ledger_display["daily_ret"] = ledger_display["daily_return"].map(pct)
        ledger_display["avg_win_pct"] = ledger_display["avg_win"].map(pct)
        ledger_display["avg_loss_pct"] = ledger_display["avg_loss"].map(pct)
        ledger_display["PF"] = ledger_display["profit_factor"].map(lambda x: "" if not np.isfinite(num(x)) else f"{float(x):.2f}")
        ledger_display = ledger_display.sort_values(["line", "ledger"])

    per_line_rows: list[dict[str, Any]] = []
    if not execution.empty:
        for sid, group in execution.groupby("strategy_id"):
            per_line_rows.append(
                {
                    "line": short_strategy_id(str(sid)),
                    "positions": int(len(group)),
                    "ret_5bp": pct(strategy_daily_return(group, "return_5bp")),
                    "ret_10bp_impact": pct(strategy_daily_return(group, "return_10bp_impact")),
                    "winners": int((pd.to_numeric(group.get("return_10bp_impact"), errors="coerce") > 0).sum()),
                    "losers": int((pd.to_numeric(group.get("return_10bp_impact"), errors="coerce") <= 0).sum()),
                }
            )
    per_line = pd.DataFrame(per_line_rows)

    title = f"FS {settlement_date} 线路结算摘要"
    content = "\n".join(
        [
            f"# {title}",
            "",
            headline,
            "",
            "## 线路收益概览",
            md_table(per_line, ["line", "positions", "ret_5bp", "ret_10bp_impact", "winners", "losers"], max_rows=20),
            "",
            "## 双账本结算",
            md_table(
                ledger_display,
                ["line", "ledger", "num_positions", "num_winners", "num_losers", "daily_ret", "avg_win_pct", "avg_loss_pct", "PF"],
                max_rows=20,
            ),
        ]
    )
    send_logged_wxpusher(title, content, "settlement_summary_ledgers", signal_date, settlement_date, topic_id, dry_run, paths)


def run_monitor(args: argparse.Namespace) -> dict[str, Any]:
    signal_date = args.signal_date
    settlement_date = args.settlement_date
    output_root = Path(args.output_root)
    minute_dir = Path(args.minute_dir)
    paths = output_paths(output_root, signal_date, settlement_date)
    entries = load_entry_file(output_root, signal_date)
    entries = entries[entries["paper_entry_status"].astype(str).eq("entry_recorded")].copy()
    minute_store = MinuteStore(minute_dir)
    recs_before = read_csv_if_exists(paths["recommendations"])
    exec_before = read_csv_if_exists(paths["execution"])
    new_recs = pd.DataFrame()
    new_exec = pd.DataFrame()

    if args.checkpoint in CHECKPOINT_RULES:
        new_recs = build_decisions(entries, recs_before, minute_store, signal_date, settlement_date, args.checkpoint)
        if not new_recs.empty:
            write_replace(new_recs, paths["recommendations"], ["signal_date", "settlement_date", "strategy_id", "code"])
            if args.send_notifications:
                push_new_recommendations(new_recs, args.checkpoint, settlement_date, signal_date, args.topic_id, args.dry_run_push, paths)

    if args.checkpoint in EXEC_CHECKPOINTS:
        recs_after = read_csv_if_exists(paths["recommendations"])
        exec_time = EXEC_CHECKPOINTS[args.checkpoint]
        new_exec = execution_rows_for_time(recs_after, exec_before, entries, minute_store, signal_date, settlement_date, exec_time)
        if not new_exec.empty:
            write_replace(new_exec, paths["execution"], ["signal_date", "settlement_date", "strategy_id", "code"])
            if args.send_notifications:
                push_new_execution(new_exec, args.checkpoint, settlement_date, signal_date, args.topic_id, args.dry_run_push, paths)

    if args.checkpoint == "prealert_1025":
        new_recs = default_recommendations(
            entries,
            recs_before,
            minute_store,
            signal_date,
            settlement_date,
            checkpoint="prealert_1025",
            decision_time="10:25",
            reference_time="10:25",
        )
        if not new_recs.empty:
            write_replace(new_recs, paths["recommendations"], ["signal_date", "settlement_date", "strategy_id", "code"])
            if args.send_notifications:
                push_new_recommendations(new_recs, args.checkpoint, settlement_date, signal_date, args.topic_id, args.dry_run_push, paths)

    if args.checkpoint == "default_1030":
        new_recs = default_recommendations(entries, recs_before, minute_store, signal_date, settlement_date)
        if not new_recs.empty:
            write_replace(new_recs, paths["recommendations"], ["signal_date", "settlement_date", "strategy_id", "code"])
        recs_after = read_csv_if_exists(paths["recommendations"])
        new_exec = execution_rows_for_time(recs_after, exec_before, entries, minute_store, signal_date, settlement_date, "10:30")
        if not new_exec.empty:
            write_replace(new_exec, paths["execution"], ["signal_date", "settlement_date", "strategy_id", "code"])
        if args.send_notifications:
            if not new_recs.empty:
                push_new_recommendations(new_recs, args.checkpoint, settlement_date, signal_date, args.topic_id, args.dry_run_push, paths)
            if not new_exec.empty:
                push_new_execution(new_exec, args.checkpoint, settlement_date, signal_date, args.topic_id, args.dry_run_push, paths)

    if args.checkpoint == "settle_all":
        for cp in ["check_0935", "exec_0940", "check_0945", "exec_0950", "check_1000", "exec_1005", "prealert_1025", "default_1030"]:
            sub = argparse.Namespace(**{**vars(args), "checkpoint": cp, "send_notifications": False})
            run_monitor(sub)

    execution = read_csv_if_exists(paths["execution"])
    ledgers = pd.DataFrame()
    if not execution.empty:
        write_replace(execution, paths["settlement"], ["signal_date", "settlement_date", "strategy_id", "code"])
        ledgers = build_strategy_ledgers(execution)
        if not ledgers.empty:
            ensure_dir(paths["ledger"].parent)
            ledgers.to_csv(paths["ledger"], index=False)
    quality = write_quality(execution, paths["quality"], signal_date, settlement_date)
    if args.send_notifications and args.checkpoint in {"default_1030", "settle_all"}:
        push_final_settlement_summary(execution, ledgers, quality, signal_date, settlement_date, args.topic_id, args.dry_run_push, paths)
    sha = write_sha(paths)

    return {
        "signal_date": signal_date,
        "settlement_date": settlement_date,
        "checkpoint": args.checkpoint,
        "active_entries": int(len(entries)),
        "new_recommendations": int(len(new_recs)),
        "new_execution_records": int(len(new_exec)),
        "total_recommendations": int(len(read_csv_if_exists(paths["recommendations"]))),
        "total_execution_records": int(len(read_csv_if_exists(paths["execution"]))),
        "quality_rows": int(len(quality)),
        "sha_rows": int(len(sha)),
        "paths": {k: str(v) for k, v in paths.items() if v.exists()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forward shadow T+1 exit monitor. Paper tracking only.")
    parser.add_argument("--signal-date", required=True, help="T date of frozen forward-shadow signal, YYYYMMDD.")
    parser.add_argument("--settlement-date", required=True, help="T+1 date used for exit monitoring, YYYYMMDD.")
    parser.add_argument("--checkpoint", required=True, choices=ALL_CHECKPOINTS)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--minute-dir", default=str(DEFAULT_MINUTE_DIR))
    parser.add_argument("--send-notifications", action="store_true")
    parser.add_argument("--dry-run-push", action="store_true")
    parser.add_argument("--topic-id", type=int, default=DEFAULT_TOPIC_ID)
    return parser.parse_args()


def main() -> None:
    started = time.monotonic()
    args = parse_args()
    result = run_monitor(args)
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
