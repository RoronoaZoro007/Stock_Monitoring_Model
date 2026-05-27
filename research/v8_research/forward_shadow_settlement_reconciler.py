#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "tushare" / "v8_forward_shadow"

LINE_ID_MAP = {
    "S0_v7_original_top10": "S0_v7_original_top10",
    "S1_U2_filter_only_no_refill": "S1_U2_filter_only_no_refill",
    "S0_v7_original_top10_tail_down": "S0_tail_down",
    "S1_U2_filter_only_no_refill_tail_down": "S1_U2_tail_down",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def line_id(strategy_id: str) -> str:
    return LINE_ID_MAP.get(str(strategy_id), str(strategy_id))


def series_or_default(df: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def load_settlement_frames(output_root: Path, directory: str, suffix: str) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted((output_root / directory).glob(f"*{suffix}")):
        date_key = path.name[:8]
        frames[date_key] = pd.read_csv(path)
    return frames


def load_exit_pair_frames(output_root: Path, directory: str, suffix: str) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted((output_root / directory).glob(f"*{suffix}")):
        signal_date = path.name[:8]
        frames[signal_date] = pd.read_csv(path)
    return frames


def build_trade_details(
    status: pd.DataFrame,
    signal_frames: dict[str, pd.DataFrame],
    entry_frames: dict[str, pd.DataFrame],
    settlement_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for trade_date in sorted(signal_frames):
        signals = signal_frames[trade_date].copy()
        if signals.empty:
            continue
        signals["trade_date"] = signals["trade_date"].astype(str)
        signals = signals[signals["final_selected_flag"].fillna(False)].copy()
        if signals.empty:
            continue
        entries = entry_frames.get(trade_date, pd.DataFrame()).copy()
        settlement = settlement_frames.get(trade_date, pd.DataFrame()).copy()

        if not entries.empty:
            if "trade_date" not in entries.columns:
                entries["trade_date"] = trade_date
            entries["trade_date"] = entries["trade_date"].astype(str)
            if "candidate_id" not in entries.columns:
                entries["candidate_id"] = np.nan
            entries = entries.rename(
                columns={
                    "weight": "weight",
                    "generated_time_beijing": "entry_generated_time_beijing",
                }
            )
            keep_cols = [
                "trade_date",
                "strategy_id",
                "candidate_id",
                "code",
                "paper_entry_status",
                "entry_vwap",
                "entry_amount",
                "entry_price_source",
                "weight",
                "generated_time_beijing",
            ]
            keep_cols = [c for c in keep_cols if c in entries.columns]
            merge_keys = ["trade_date", "strategy_id", "code"]
            signals = signals.merge(entries[keep_cols], on=merge_keys, how="left", suffixes=("", "_entry"))
            if "candidate_id_entry" in signals.columns:
                signals["candidate_id"] = signals["candidate_id"].fillna(signals["candidate_id_entry"])
                signals = signals.drop(columns=["candidate_id_entry"])
        else:
            signals["paper_entry_status"] = np.nan
            signals["entry_vwap"] = np.nan
            signals["entry_amount"] = np.nan
            signals["entry_price_source"] = np.nan
            signals["weight"] = signals.get("position_weight", np.nan)

        if not settlement.empty:
            settlement["signal_date"] = settlement["signal_date"].astype(str)
            settlement["settlement_date"] = settlement["settlement_date"].astype(str)
            signals = signals.merge(
                settlement,
                left_on=["trade_date", "strategy_id", "code"],
                right_on=["signal_date", "strategy_id", "code"],
                how="left",
                suffixes=("", "_settlement"),
            )
        else:
            signals["settlement_date"] = np.nan
            signals["paper_exit_status"] = np.nan

        signals["line_id"] = signals["strategy_id"].map(line_id)
        signals["signal_date"] = signals["trade_date"]
        signals["settlement_status"] = signals.get("paper_exit_status", pd.Series([np.nan] * len(signals), index=signals.index)).astype(object)
        missing_settlement_mask = signals["settlement_status"].isna()
        if missing_settlement_mask.any():
            entry_status_subset = signals.loc[missing_settlement_mask, "paper_entry_status"]
            signals.loc[missing_settlement_mask, "settlement_status"] = np.where(
                entry_status_subset.astype(str).eq("entry_recorded"),
                "pending_exit_settlement",
                entry_status_subset.fillna("missing_entry"),
            )
        signals["weight"] = pd.to_numeric(signals.get("weight"), errors="coerce").fillna(
            pd.to_numeric(signals.get("position_weight"), errors="coerce")
        )
        signals["theoretical_5bp_return"] = pd.to_numeric(signals.get("return_5bp"), errors="coerce")
        signals["execution_10bp_impact_return"] = pd.to_numeric(signals.get("return_10bp_impact"), errors="coerce")
        signals["estimated_fill_ratio"] = pd.to_numeric(signals.get("fill_ratio"), errors="coerce")
        signals["partial_fill_flag"] = series_or_default(signals, "partial_fill", False).fillna(False).astype(bool)
        signals["zero_fill_flag"] = series_or_default(signals, "zero_fill", False).fillna(False).astype(bool)
        signals["limit_up_block_flag"] = False
        signals["limit_down_block_flag"] = False
        signals["suspend_flag"] = signals["settlement_status"].astype(str).str.contains("missing|pending", case=False, regex=True)
        signals["paper_tracking_only"] = True
        rows.append(signals)

    if not rows:
        return pd.DataFrame()

    merged = pd.concat(rows, ignore_index=True, sort=False)
    ordered_cols = [
        "signal_date",
        "settlement_date",
        "line_id",
        "code",
        "name",
        "weight",
        "entry_vwap",
        "exit_vwap",
        "gross_return",
        "return_5bp",
        "return_10bp_impact",
        "estimated_impact_cost",
        "buy_impact_cost",
        "sell_impact_cost",
        "fill_ratio",
        "partial_fill",
        "zero_fill",
        "limit_up_block",
        "limit_down_block",
        "entry_amount",
        "exit_amount",
        "settlement_status",
        "strategy_id",
        "paper_tracking_only",
        "trade_date",
        "candidate_id",
        "tail_down_flag",
        "U2_filter_flag",
        "tail_down_gate_flag",
        "original_v7_rank",
        "final_selected_flag",
        "no_trade_reason",
        "score",
        "entry_time",
        "expected_entry_vwap",
        "exit_rule",
        "expected_exit_time",
        "expected_exit_vwap",
        "theoretical_5bp_return",
        "execution_10bp_impact_return",
        "estimated_fill_ratio",
        "partial_fill_flag",
        "zero_fill_flag",
        "limit_up_block_flag",
        "limit_down_block_flag",
        "suspend_flag",
        "daily_strategy_return",
        "cumulative_strategy_return",
        "position_weight",
        "avg_amount_60d",
        "rank_amount_60d",
        "in_u2",
        "market_tail_ret_median",
        "freeze_time_beijing",
    ]
    for col in ordered_cols:
        if col not in merged.columns:
            merged[col] = np.nan
    merged = merged[ordered_cols]
    merged = merged.sort_values(["trade_date", "strategy_id", "original_v7_rank", "code"], kind="stable").reset_index(drop=True)
    return merged


def build_daily_summary(status: pd.DataFrame, ledger_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summary = status.copy()
    summary["trade_date"] = summary["trade_date"].astype(str)

    ledger_rows: list[dict[str, Any]] = []
    for signal_date, ledger in sorted(ledger_frames.items()):
        if ledger.empty:
            continue
        for _, row in ledger.iterrows():
            ledger_rows.append(
                {
                    "trade_date": str(row["signal_date"]),
                    "strategy_id": str(row["strategy_id"]),
                    "ledger_type": str(row["ledger_type"]),
                    "daily_return": pd.to_numeric(row["daily_return"], errors="coerce"),
                }
            )
    ledger_df = pd.DataFrame(ledger_rows)
    if ledger_df.empty:
        summary["theoretical_5bp_daily_return"] = np.nan
        summary["execution_10bp_impact_daily_return"] = np.nan
        summary["theoretical_5bp_cumulative_return"] = np.nan
        summary["execution_10bp_impact_cumulative_return"] = np.nan
        return summary.sort_values(["trade_date", "strategy_id"], kind="stable").reset_index(drop=True)

    pivot = (
        ledger_df.pivot_table(index=["trade_date", "strategy_id"], columns="ledger_type", values="daily_return", aggfunc="last")
        .reset_index()
        .rename(
            columns={
                "theoretical_5bp_ledger": "theoretical_5bp_daily_return",
                "execution_10bp_impact_ledger": "execution_10bp_impact_daily_return",
            }
        )
    )
    summary = summary.merge(pivot, on=["trade_date", "strategy_id"], how="left")
    for daily_col, cum_col in [
        ("theoretical_5bp_daily_return", "theoretical_5bp_cumulative_return"),
        ("execution_10bp_impact_daily_return", "execution_10bp_impact_cumulative_return"),
    ]:
        summary[cum_col] = np.nan
        for strategy_id, group in summary.groupby("strategy_id", sort=False):
            group = group.sort_values("trade_date", kind="stable")
            realized = pd.to_numeric(group[daily_col], errors="coerce")
            wealth = (1.0 + realized.fillna(0.0)).cumprod() - 1.0
            mask = realized.notna()
            summary.loc[group.index[mask], cum_col] = wealth[mask].to_numpy()
    return summary.sort_values(["trade_date", "strategy_id"], kind="stable").reset_index(drop=True)


def build_execution_quality(status: pd.DataFrame, settlement_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, item in status.iterrows():
        trade_date = str(item["trade_date"])
        strategy_id = str(item["strategy_id"])
        selected_count = int(item["selected_count"])
        if selected_count <= 0:
            continue
        settlement = settlement_frames.get(trade_date, pd.DataFrame()).copy()
        group = settlement[settlement["strategy_id"].astype(str).eq(strategy_id)].copy() if not settlement.empty else pd.DataFrame()
        if group.empty:
            quality_status = "pending_exit_settlement"
            avg_fill_ratio = np.nan
            partial_fill_count = 0
            zero_fill_count = 0
            limit_up_block_count = 0
            limit_down_block_count = 0
            suspend_count = 0
        else:
            fill_ratio = pd.to_numeric(series_or_default(group, "fill_ratio", np.nan), errors="coerce")
            exit_status = group["paper_exit_status"].astype(str)
            avg_fill_ratio = float(fill_ratio.mean()) if fill_ratio.notna().any() else np.nan
            partial_fill_count = int(series_or_default(group, "partial_fill", False).fillna(False).astype(bool).sum())
            zero_fill_count = int(series_or_default(group, "zero_fill", False).fillna(False).astype(bool).sum())
            limit_up_block_count = 0
            limit_down_block_count = 0
            suspend_count = int(exit_status.str.contains("missing|pending", case=False, regex=True).sum())
            if exit_status.str.startswith("exit_recorded").all():
                quality_status = "settled"
            elif exit_status.str.contains("missing|pending", case=False, regex=True).all():
                quality_status = "data_missing"
            else:
                quality_status = "partial_settlement"
        rows.append(
            {
                "trade_date": trade_date,
                "strategy_id": strategy_id,
                "candidate_id": item["candidate_id"],
                "selected_count": selected_count,
                "estimated_avg_fill_ratio": avg_fill_ratio,
                "partial_fill_count": partial_fill_count,
                "zero_fill_count": zero_fill_count,
                "limit_up_block_count": limit_up_block_count,
                "limit_down_block_count": limit_down_block_count,
                "suspend_count": suspend_count,
                "status": quality_status,
                "line_id": line_id(strategy_id),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["trade_date", "strategy_id"], kind="stable").reset_index(drop=True)


def build_settlement_summary(ledger_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, ledger in sorted(ledger_frames.items()):
        if ledger.empty:
            continue
        for _, row in ledger.iterrows():
            rows.append(
                {
                    "signal_date": str(row["signal_date"]),
                    "settlement_date": str(row["settlement_date"]),
                    "line_id": line_id(str(row["strategy_id"])),
                    "strategy_id": str(row["strategy_id"]),
                    "ledger_type": str(row["ledger_type"]),
                    "num_positions": int(row["num_positions"]),
                    "num_winners": int(row["num_winners"]),
                    "num_losers": int(row["num_losers"]),
                    "daily_return": pd.to_numeric(row["daily_return"], errors="coerce"),
                    "avg_win": pd.to_numeric(row["avg_win"], errors="coerce"),
                    "avg_loss": pd.to_numeric(row["avg_loss"], errors="coerce"),
                    "profit_factor": pd.to_numeric(row["profit_factor"], errors="coerce"),
                    "notes": "settled T+1 paper ledger" if pd.notna(row["daily_return"]) else "pending_or_data_missing",
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["signal_date", "strategy_id", "ledger_type"], kind="stable").reset_index(drop=True)


def max_drawdown_from_returns(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def build_cumulative_summary(settlement_summary: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    if settlement_summary.empty:
        return pd.DataFrame()
    tail_days = (
        status.assign(trade_date=status["trade_date"].astype(str))
        .groupby("strategy_id")["tail_down_flag"]
        .sum()
        .to_dict()
    )
    rows: list[dict[str, Any]] = []
    for (strategy_id, ledger_type), group in settlement_summary.groupby(["strategy_id", "ledger_type"], sort=False):
        group = group.sort_values("signal_date", kind="stable")
        daily_returns = pd.to_numeric(group["daily_return"], errors="coerce")
        total_trades = int(pd.to_numeric(group["num_positions"], errors="coerce").fillna(0).sum())
        total_winners = int(pd.to_numeric(group["num_winners"], errors="coerce").fillna(0).sum())
        gain = float(daily_returns[daily_returns > 0].sum()) if (daily_returns > 0).any() else 0.0
        loss = float(-daily_returns[daily_returns < 0].sum()) if (daily_returns < 0).any() else 0.0
        rows.append(
            {
                "line_id": line_id(str(strategy_id)),
                "strategy_id": str(strategy_id),
                "ledger_type": str(ledger_type),
                "active_days": int(len(group)),
                "total_trades": total_trades,
                "cumulative_return": float((1.0 + daily_returns.fillna(0.0)).prod() - 1.0),
                "PF": gain / loss if loss > 0 else np.nan,
                "max_drawdown": max_drawdown_from_returns(daily_returns),
                "daily_win_rate": float((daily_returns > 0).mean()) if len(group) else np.nan,
                "trade_win_rate": float(total_winners / total_trades) if total_trades else np.nan,
                "tail_down_trigger_days": int(tail_days.get(strategy_id, 0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["strategy_id", "ledger_type"], kind="stable").reset_index(drop=True)


def write_sha_report(output_root: Path, files: list[Path]) -> pd.DataFrame:
    sha_path = output_root / "sha256" / "forward_shadow_cumulative_sha256.csv"
    ensure_dir(sha_path.parent)
    rows = []
    for path in files:
        if not path.exists():
            continue
        rows.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    out = pd.DataFrame(rows)
    out.to_csv(sha_path, index=False)
    return out


def reconcile_outputs(output_root: Path) -> dict[str, Any]:
    status_path = output_root / "forward_shadow_candidate_status.csv"
    status = read_csv(status_path, dtype={"trade_date": str, "strategy_id": str, "candidate_id": str})
    if status.empty:
        return {"status": "skipped", "reason": f"missing_or_empty_status:{status_path}"}

    signal_frames = load_settlement_frames(output_root, "daily_signals", "_signals.csv")
    entry_frames = load_settlement_frames(output_root, "daily_entry_prices", "_entry_prices.csv")
    settlement_frames = load_exit_pair_frames(output_root, "daily_exit_settlement", "_exit_settlement.csv")
    ledger_frames = load_exit_pair_frames(output_root, "daily_ledgers", "_exit_ledgers.csv")

    trade_details = build_trade_details(status, signal_frames, entry_frames, settlement_frames)
    daily_summary = build_daily_summary(status, ledger_frames)
    execution_quality = build_execution_quality(status, settlement_frames)
    no_trade_days = status[status["selected_count"].fillna(0).astype(int).eq(0)].copy()
    no_trade_days["line_id"] = no_trade_days["strategy_id"].map(line_id)
    no_trade_days = no_trade_days.sort_values(["trade_date", "strategy_id"], kind="stable").reset_index(drop=True)
    settlement_summary = build_settlement_summary(ledger_frames)
    cumulative_summary = build_cumulative_summary(settlement_summary, status)

    out_files = {
        "trade_details": output_root / "forward_shadow_trade_details.csv",
        "daily_summary": output_root / "forward_shadow_daily_summary.csv",
        "execution_quality": output_root / "forward_shadow_execution_quality.csv",
        "no_trade_days": output_root / "forward_shadow_no_trade_days.csv",
        "settlement_summary": output_root / "forward_shadow_settlement_summary.csv",
        "cumulative_summary": output_root / "forward_shadow_cumulative_daily_summary.csv",
    }
    trade_details.to_csv(out_files["trade_details"], index=False)
    daily_summary.to_csv(out_files["daily_summary"], index=False)
    execution_quality.to_csv(out_files["execution_quality"], index=False)
    no_trade_days.to_csv(out_files["no_trade_days"], index=False)
    settlement_summary.to_csv(out_files["settlement_summary"], index=False)
    cumulative_summary.to_csv(out_files["cumulative_summary"], index=False)
    sha = write_sha_report(output_root, list(out_files.values()))

    return {
        "status": "success",
        "trade_details_rows": int(len(trade_details)),
        "daily_summary_rows": int(len(daily_summary)),
        "execution_quality_rows": int(len(execution_quality)),
        "no_trade_days_rows": int(len(no_trade_days)),
        "settlement_summary_rows": int(len(settlement_summary)),
        "cumulative_summary_rows": int(len(cumulative_summary)),
        "sha_rows": int(len(sha)),
        "output_root": str(output_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconcile forward-shadow cumulative paper outputs from frozen daily artifacts.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = reconcile_outputs(Path(args.output_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
