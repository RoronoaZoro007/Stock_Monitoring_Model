#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "tushare" / "v8_forward_shadow"
DEFAULT_DAILY_FILE = ROOT / "data_tushare" / "clean" / "daily_repaired_top3000.parquet"
TOP_N = 10
V7_LOCKED_COMMIT = "f688eec575af4667681d87b5b2c1fca72754e399"

STRATEGIES = [
    {
        "strategy_id": "S0_v7_original_top10",
        "candidate_id": "control_1",
        "use_u2_filter": False,
        "use_tail_down_gate": False,
        "description": "Control 1: v7 original Top10",
    },
    {
        "strategy_id": "S1_U2_filter_only_no_refill",
        "candidate_id": "control_2",
        "use_u2_filter": True,
        "use_tail_down_gate": False,
        "description": "Control 2: v7 Top10 kept only when in U2, no refill",
    },
    {
        "strategy_id": "S0_v7_original_top10_tail_down",
        "candidate_id": "candidate_1",
        "use_u2_filter": False,
        "use_tail_down_gate": True,
        "description": "Candidate 1: v7 original Top10 gated by tail_down",
    },
    {
        "strategy_id": "S1_U2_filter_only_no_refill_tail_down",
        "candidate_id": "candidate_2",
        "use_u2_filter": True,
        "use_tail_down_gate": True,
        "description": "Candidate 2: U2 filter-only gated by tail_down",
    },
]


def beijing_now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def load_score_file(path: Path, trade_date: str) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, compression="infer")
    df["trade_date"] = df["trade_date"].astype(str)
    df = df[df["trade_date"] == str(trade_date)].copy()
    required = {"trade_date", "ts_code", "score", "market_tail_ret_median"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"score file missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"no score rows found for trade_date={trade_date}")
    return df


def compute_tail_down(score_df: pd.DataFrame) -> tuple[bool, float]:
    value = float(pd.to_numeric(score_df["market_tail_ret_median"], errors="coerce").dropna().iloc[0])
    return bool(value < 0.0), value


def compute_u2_membership(score_df: pd.DataFrame, daily_file: Path, trade_date: str) -> pd.DataFrame:
    if not daily_file.exists():
        raise FileNotFoundError(f"daily file not found for U2 membership: {daily_file}")
    daily = pd.read_parquet(daily_file, columns=["ts_code", "trade_date", "amount"])
    daily["trade_date"] = daily["trade_date"].astype(str)
    daily["amount"] = pd.to_numeric(daily["amount"], errors="coerce")
    hist = daily[daily["trade_date"] <= str(trade_date)].sort_values(["ts_code", "trade_date"]).copy()
    hist["avg_amount_60d"] = hist.groupby("ts_code")["amount"].transform(
        lambda s: s.shift(1).rolling(60, min_periods=20).mean()
    )
    current = hist[hist["trade_date"] == str(trade_date)][["ts_code", "trade_date", "avg_amount_60d"]]
    base = score_df[["ts_code", "trade_date"]].merge(current, on=["ts_code", "trade_date"], how="left")
    base["rank_amount_60d"] = base["avg_amount_60d"].rank(method="first", ascending=False)
    base["in_u2"] = base["rank_amount_60d"].le(2500).fillna(False)
    return base[["trade_date", "ts_code", "avg_amount_60d", "rank_amount_60d", "in_u2"]]


def top10_with_membership(score_df: pd.DataFrame, u2: pd.DataFrame) -> pd.DataFrame:
    base = score_df.merge(u2, on=["trade_date", "ts_code"], how="left")
    base["in_u2"] = base["in_u2"].fillna(False).astype(bool)
    base = base.sort_values(["trade_date", "score"], ascending=[True, False]).copy()
    base["original_v7_rank"] = base.groupby("trade_date").cumcount() + 1
    return base[base["original_v7_rank"] <= TOP_N].copy()


def select_for_strategy(top10: pd.DataFrame, strategy: dict[str, Any], tail_down: bool) -> tuple[pd.DataFrame, str]:
    if strategy["use_tail_down_gate"] and not tail_down:
        return top10.iloc[0:0].copy(), "tail_down_not_triggered"
    selected = top10.copy()
    if strategy["use_u2_filter"]:
        selected = selected[selected["in_u2"]].copy()
        if selected.empty:
            return selected, "u2_filter_no_remaining_v7_top10"
    if selected.empty:
        return selected, "no_selected_candidate"
    return selected, ""


def signal_rows(
    top10: pd.DataFrame,
    trade_date: str,
    tail_down: bool,
    market_tail_ret_median: float,
    freeze_time: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        selected, no_trade_reason = select_for_strategy(top10, strategy, tail_down)
        status_rows.append(
            {
                "trade_date": trade_date,
                "strategy_id": strategy["strategy_id"],
                "candidate_id": strategy["candidate_id"],
                "tail_down_flag": tail_down,
                "market_tail_ret_median": market_tail_ret_median,
                "U2_filter_flag": strategy["use_u2_filter"],
                "tail_down_gate_flag": strategy["use_tail_down_gate"],
                "selected_count": len(selected),
                "no_trade_reason": no_trade_reason,
                "freeze_time_beijing": freeze_time,
                "v7_locked_commit": V7_LOCKED_COMMIT,
                "paper_tracking_only": True,
            }
        )
        if selected.empty:
            continue
        weight = 1.0 / len(selected)
        for row in selected.itertuples(index=False):
            rows.append(
                {
                    "trade_date": trade_date,
                    "strategy_id": strategy["strategy_id"],
                    "candidate_id": strategy["candidate_id"],
                    "tail_down_flag": tail_down,
                    "U2_filter_flag": strategy["use_u2_filter"],
                    "tail_down_gate_flag": strategy["use_tail_down_gate"],
                    "original_v7_rank": int(row.original_v7_rank),
                    "final_selected_flag": True,
                    "no_trade_reason": "",
                    "code": row.ts_code,
                    "name": getattr(row, "name", ""),
                    "score": float(row.score),
                    "entry_time": "14:55",
                    "expected_entry_vwap": getattr(row, "entry_vwap", np.nan),
                    "exit_rule": "v7_locked_exit_rule",
                    "expected_exit_time": getattr(row, "exit_time", ""),
                    "expected_exit_vwap": np.nan,
                    "theoretical_5bp_return": np.nan,
                    "execution_10bp_impact_return": np.nan,
                    "estimated_impact_cost": np.nan,
                    "estimated_fill_ratio": np.nan,
                    "partial_fill_flag": False,
                    "zero_fill_flag": False,
                    "limit_up_block_flag": False,
                    "limit_down_block_flag": False,
                    "suspend_flag": False,
                    "daily_strategy_return": np.nan,
                    "cumulative_strategy_return": np.nan,
                    "position_weight": weight,
                    "avg_amount_60d": getattr(row, "avg_amount_60d", np.nan),
                    "rank_amount_60d": getattr(row, "rank_amount_60d", np.nan),
                    "in_u2": bool(getattr(row, "in_u2", False)),
                    "market_tail_ret_median": market_tail_ret_median,
                    "freeze_time_beijing": freeze_time,
                    "paper_tracking_only": True,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(status_rows)


def empty_ledger_from_signals(signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame()
    ledgers = []
    quality = []
    for ledger_id in ["theoretical_5bp_ledger", "execution_10bp_impact_ledger"]:
        tmp = signals.copy()
        tmp["ledger_id"] = ledger_id
        tmp["realized_return"] = np.nan
        tmp["ledger_status"] = "pending_exit_settlement"
        ledgers.append(tmp)
    for key, group in signals.groupby(["trade_date", "strategy_id", "candidate_id"], dropna=False):
        quality.append(
            {
                "trade_date": key[0],
                "strategy_id": key[1],
                "candidate_id": key[2],
                "selected_count": len(group),
                "estimated_avg_fill_ratio": np.nan,
                "partial_fill_count": 0,
                "zero_fill_count": 0,
                "limit_up_block_count": 0,
                "limit_down_block_count": 0,
                "suspend_count": 0,
                "status": "pending_execution_observation",
            }
        )
    return pd.concat(ledgers, ignore_index=True), pd.DataFrame(quality)


def write_outputs(output_root: Path, trade_date: str, signals: pd.DataFrame, ledgers: pd.DataFrame, quality: pd.DataFrame, status: pd.DataFrame) -> None:
    daily_signals = output_root / "daily_signals"
    daily_ledgers = output_root / "daily_ledgers"
    daily_quality = output_root / "daily_execution_quality"
    for path in [daily_signals, daily_ledgers, daily_quality]:
        path.mkdir(parents=True, exist_ok=True)
    signals.to_csv(daily_signals / f"{trade_date}_signals.csv", index=False)
    ledgers.to_csv(daily_ledgers / f"{trade_date}_ledgers.csv", index=False)
    quality.to_csv(daily_quality / f"{trade_date}_execution_quality.csv", index=False)

    append_or_replace(output_root / "forward_shadow_candidate_status.csv", status, ["trade_date", "strategy_id"])
    if not signals.empty:
        append_or_replace(output_root / "forward_shadow_trade_details.csv", signals, ["trade_date", "strategy_id", "code"])
    if not quality.empty:
        append_or_replace(output_root / "forward_shadow_execution_quality.csv", quality, ["trade_date", "strategy_id"])
    no_trade = status[status["selected_count"].eq(0)].copy()
    append_or_replace(output_root / "forward_shadow_no_trade_days.csv", no_trade, ["trade_date", "strategy_id"])

    summary = status[
        [
            "trade_date",
            "strategy_id",
            "candidate_id",
            "tail_down_flag",
            "U2_filter_flag",
            "selected_count",
            "no_trade_reason",
            "paper_tracking_only",
        ]
    ].copy()
    summary["theoretical_5bp_daily_return"] = np.nan
    summary["execution_10bp_impact_daily_return"] = np.nan
    summary["theoretical_5bp_cumulative_return"] = np.nan
    summary["execution_10bp_impact_cumulative_return"] = np.nan
    append_or_replace(output_root / "forward_shadow_daily_summary.csv", summary, ["trade_date", "strategy_id"])


def append_or_replace(path: Path, new_rows: pd.DataFrame, keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if new_rows.empty:
        if not path.exists():
            new_rows.to_csv(path, index=False)
        return
    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(keys, keep="last")
    else:
        combined = new_rows.copy()
    combined.to_csv(path, index=False)


def run_signal_mode(args: argparse.Namespace) -> None:
    trade_date = str(args.trade_date)
    score_df = load_score_file(Path(args.score_file), trade_date)
    tail_down, market_tail_ret_median = compute_tail_down(score_df)
    u2 = compute_u2_membership(score_df, Path(args.daily_file), trade_date)
    top10 = top10_with_membership(score_df, u2)
    freeze_time = beijing_now()
    signals, status = signal_rows(top10, trade_date, tail_down, market_tail_ret_median, freeze_time)
    ledgers, quality = empty_ledger_from_signals(signals)
    write_outputs(Path(args.output_root), trade_date, signals, ledgers, quality, status)
    print(
        json.dumps(
            {
                "trade_date": trade_date,
                "tail_down_flag": tail_down,
                "market_tail_ret_median": market_tail_ret_median,
                "signals": len(signals),
                "output_root": str(Path(args.output_root)),
                "paper_tracking_only": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward shadow paper-tracking runner. Does not place orders.")
    parser.add_argument("--mode", choices=["signal"], default="signal")
    parser.add_argument("--trade-date", required=True, help="YYYYMMDD")
    parser.add_argument("--score-file", required=True, help="Daily v7 locked score matrix with <=14:50 features.")
    parser.add_argument("--daily-file", default=str(DEFAULT_DAILY_FILE), help="Daily data used only for t-1 U2 membership.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    if args.mode == "signal":
        run_signal_mode(args)


if __name__ == "__main__":
    main()
