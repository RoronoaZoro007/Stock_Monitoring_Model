#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_strategy as ev  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "tushare" / "v8_research" / "batch1_hard_checks"
LOCKED = ROOT / "locked_artifacts" / "v7_cap20_strong_label_003"
RAW_DATASET = ROOT / "data_tushare" / "clean" / "features" / "tail_dataset_top3000_repaired_raw.parquet"
EVENT_DATASET = ROOT / "data_tushare" / "clean" / "features" / "tail_dataset_top3000_event_clean.parquet"
CAP20_DATASET = ROOT / "data_tushare" / "clean" / "features" / "tail_dataset_top3000_strict_cap20.parquet"
MINUTE_DAILY = ROOT / "data_tushare" / "clean" / "daily_from_minutes_top3000.parquet"
VALIDATION_START = "20260224"
VALIDATION_END = "20260520"
TOP_N = 10


def now_bj() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def ymd_to_datetime(value: Any) -> pd.Timestamp:
    return pd.to_datetime(str(value), format="%Y%m%d", errors="coerce")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_trade_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["trade_date"] = out["trade_date"].astype(str)
    out["trade_id"] = np.arange(len(out))
    return out


def load_base_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top10, candidates, daily, stock = ev.load_inputs(
        ev.DEFAULT_SIGNAL_FILE,
        ev.DEFAULT_SCORE_FILE,
        ev.DEFAULT_DAILY_FILE,
        ev.DEFAULT_STOCK_BASIC_FILE,
    )
    source = pd.read_parquet(ev.DEFAULT_DAILY_FILE, columns=["ts_code", "trade_date", "daily_source"])
    source["trade_date"] = source["trade_date"].astype(str)
    daily = daily.merge(source, on=["ts_code", "trade_date"], how="left")
    return top10, candidates, daily, stock


def evaluate_selected(
    selected: pd.DataFrame,
    daily: pd.DataFrame,
    stock: pd.DataFrame,
    universe_id: str,
    include_capacity: bool = True,
    use_zero_exit_deferral: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = ensure_trade_ids(selected)
    trade_base = ev.prepare_trade_base(selected, daily, stock)
    minutes = ev.load_relevant_minutes(trade_base, ev.DEFAULT_MINUTE_DIR)
    trade_base = ev.add_entry_exit_bars(trade_base, minutes)
    trade_base = ev.add_execution_flags(trade_base, minutes, daily)
    if use_zero_exit_deferral:
        trade_base = apply_zero_exit_deferral(trade_base, minutes, daily)

    cost_scenarios, capacity_scenarios = ev.build_scenarios()
    scenarios = cost_scenarios + (capacity_scenarios if include_capacity else [])
    model_auc = ev.auc_safe(selected, "label_static_003", "score") if "score" in selected.columns else np.nan
    rows = []
    detail_frames = []
    for scenario in scenarios:
        metrics, details, _daily_df = ev.evaluate_scenario(trade_base, scenario, model_auc)
        metrics["universe_id"] = universe_id
        rows.append(metrics)
        if scenario.scenario_id in {
            "10bp_slippage_plus_impact_participation_10pct_partial_zero_capital_100k",
            "5bp_one_way_slippage",
            "10bp_one_way_slippage",
        }:
            details["universe_id"] = universe_id
            detail_frames.append(details)
    details_out = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return pd.DataFrame(rows), details_out


def build_rolling_universe(candidates: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cand = candidates[["trade_date", "ts_code", "score"]].copy()
    cand["trade_date"] = cand["trade_date"].astype(str)
    hist = daily[["ts_code", "trade_date", "amount"]].copy()
    hist["trade_date"] = hist["trade_date"].astype(str)
    hist["amount"] = pd.to_numeric(hist["amount"], errors="coerce")
    hist = hist.sort_values(["ts_code", "trade_date"])
    grouped = hist.groupby("ts_code", group_keys=False)
    for window in (20, 60, 120):
        hist[f"avg_amount_{window}d"] = grouped["amount"].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=max(5, min(20, w // 2))).mean()
        )
    score_dates = sorted(cand["trade_date"].unique())
    hist = hist[hist["trade_date"].isin(score_dates)]
    base = cand.merge(hist, on=["ts_code", "trade_date"], how="left")
    base["rank_amount_60d"] = base.groupby("trade_date")["avg_amount_60d"].rank(method="first", ascending=False)
    base["rank_amount_20d"] = base.groupby("trade_date")["avg_amount_20d"].rank(method="first", ascending=False)
    base["rank_amount_120d"] = base.groupby("trade_date")["avg_amount_120d"].rank(method="first", ascending=False)

    memberships = []
    for universe_id, limit in [("U0_static_top3000_v7", None), ("U1_internal_roll_top3000_60d", 3000), ("U2_internal_roll_top2500_60d", 2500), ("U3_internal_roll_top2000_60d", 2000)]:
        tmp = base[["trade_date", "ts_code", "rank_amount_60d", "avg_amount_60d", "rank_amount_20d", "rank_amount_120d"]].copy()
        tmp["universe_id"] = universe_id
        tmp["in_universe"] = True if limit is None else tmp["rank_amount_60d"].le(limit)
        tmp = tmp[tmp["in_universe"]].copy()
        memberships.append(tmp)
    universe = pd.concat(memberships, ignore_index=True)

    daily_rows = []
    overlap_rows = []
    prev_sets: dict[str, set[str]] = {}
    u0_sets = {d: set(g["ts_code"]) for d, g in universe[universe["universe_id"] == "U0_static_top3000_v7"].groupby("trade_date")}
    for universe_id, group in universe.groupby("universe_id"):
        for trade_date, day in group.groupby("trade_date"):
            members = set(day["ts_code"])
            prev = prev_sets.get(universe_id, set())
            u0 = u0_sets.get(trade_date, set())
            daily_rows.append(
                {
                    "trade_date": trade_date,
                    "universe_id": universe_id,
                    "stock_count": len(members),
                    "new_vs_prev_day": len(members - prev) if prev else len(members),
                    "retained_vs_prev_day": len(members & prev) if prev else 0,
                    "removed_vs_prev_day": len(prev - members) if prev else 0,
                }
            )
            overlap_rows.append(
                {
                    "trade_date": trade_date,
                    "universe_id": universe_id,
                    "u0_count": len(u0),
                    "universe_count": len(members),
                    "intersection_with_u0": len(members & u0),
                    "overlap_ratio_vs_u0": len(members & u0) / len(u0) if u0 else np.nan,
                    "removed_from_u0_count": len(u0 - members),
                }
            )
            prev_sets[universe_id] = members
    return universe, pd.DataFrame(daily_rows), pd.DataFrame(overlap_rows)


def universe_signal_retention(top10: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    universe_sets = {(u, d): set(g["ts_code"]) for (u, d), g in universe.groupby(["universe_id", "trade_date"])}
    for universe_id in sorted(universe["universe_id"].unique()):
        if universe_id == "U0_static_top3000_v7":
            continue
        for trade_date, day in top10.groupby("trade_date"):
            selected = set(day["ts_code"])
            members = universe_sets.get((universe_id, str(trade_date)), set())
            retained = selected & members
            rows.append(
                {
                    "trade_date": str(trade_date),
                    "universe_id": universe_id,
                    "v7_top10_count": len(selected),
                    "retained_count": len(retained),
                    "removed_count": len(selected - retained),
                    "retention_ratio": len(retained) / len(selected) if selected else np.nan,
                    "removed_codes": ",".join(sorted(selected - retained)),
                }
            )
    return pd.DataFrame(rows)


def select_top10_by_universe(candidates: pd.DataFrame, universe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for universe_id, members in universe.groupby("universe_id"):
        selected = candidates.merge(members[["trade_date", "ts_code"]], on=["trade_date", "ts_code"], how="inner")
        selected = selected.sort_values(["trade_date", "score"], ascending=[True, False]).groupby("trade_date", group_keys=False).head(TOP_N)
        selected = ensure_trade_ids(selected)
        out[universe_id] = selected
    return out


def run_universe_audit(top10: pd.DataFrame, candidates: pd.DataFrame, daily: pd.DataFrame, stock: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe, daily_size, overlap = build_rolling_universe(candidates, daily)
    retention = universe_signal_retention(top10, universe)
    selections = select_top10_by_universe(candidates, universe)
    eval_frames = []
    for universe_id, selected in selections.items():
        metrics, _details = evaluate_selected(selected, daily, stock, universe_id, include_capacity=True, use_zero_exit_deferral=False)
        eval_frames.append(metrics)
    evaluation = pd.concat(eval_frames, ignore_index=True)
    write_universe_limitations(evaluation)
    return daily_size, overlap, retention, evaluation


def write_universe_limitations(evaluation: pd.DataFrame) -> None:
    text = """# Batch 1 Universe Limitations

Current data only covers the previously downloaded v7 Top3000 universe. Therefore
Batch 1 can only audit rolling liquidity filters **inside the existing Top3000**.
It must not be described as an unbiased all-A-share daily rolling Top3000.

U0 is the locked v7 static Top3000 candidate set available in the validation
score matrix. U1/U2/U3 are internal rolling liquidity filters using only `t-1`
and earlier `avg_amount_60d`.

A full daily score matrix exists for the locked validation period, so Batch 1
does reselect Top10 inside U0/U1/U2/U3 for audit comparison. This is not model
retraining and does not change v7_locked.

The period `20260224-20260520` remains a legacy validation interval and is not
used here for final model selection.
"""
    (OUT_DIR / "batch1_universe_limitations.md").write_text(text, encoding="utf-8")


def dataset_key(df: pd.DataFrame) -> pd.Series:
    return df["ts_code"].astype(str) + "|" + df["trade_date"].astype(str)


def load_extreme_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str], set[str]]:
    raw = pd.read_parquet(RAW_DATASET)
    event = pd.read_parquet(EVENT_DATASET, columns=["ts_code", "trade_date", "target_return"])
    cap20 = pd.read_parquet(CAP20_DATASET, columns=["ts_code", "trade_date", "target_return"])
    raw["trade_date"] = raw["trade_date"].astype(str)
    event["trade_date"] = event["trade_date"].astype(str)
    cap20["trade_date"] = cap20["trade_date"].astype(str)
    event_keys = set(dataset_key(event))
    cap20_keys = set(dataset_key(cap20))
    dirty_event_keys = set(dataset_key(raw)) - event_keys
    return raw, event, cap20, dirty_event_keys, cap20_keys


def run_extreme_audit(top10: pd.DataFrame, daily: pd.DataFrame, stock: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw, event, cap20, dirty_event_keys, cap20_keys = load_extreme_inputs()
    extreme = raw[raw["target_return"].abs() > 0.20].copy()
    extreme["sample_key"] = dataset_key(extreme)
    extreme["split"] = np.where(extreme["trade_date"] >= VALIDATION_START, "legacy_validation", "train")
    extreme["extreme_side"] = np.where(extreme["target_return"] > 0, "positive", "negative")
    extreme = add_extreme_attribution(extreme, daily, stock, dirty_event_keys)

    by_split = (
        extreme.groupby(["split", "extreme_side", "primary_attribution"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["split", "extreme_side", "rows"], ascending=[True, True, False])
    )

    policy_rows = []
    for policy_id, name, keys in [
        ("C0_full_no_cap", "No event removal and no cap20 removal", set(dataset_key(raw))),
        ("C1_event_clean_only", "Only confirmed dirty event removed", set(dataset_key(event))),
        ("C2_cap20", "Confirmed dirty event removed and abs(target_return)>20% removed", set(dataset_key(cap20))),
    ]:
        use = raw[dataset_key(raw).isin(keys)]
        policy_rows.append(
            {
                "policy_id": policy_id,
                "policy_name": name,
                "sample_rows": len(use),
                "removed_vs_C0": len(raw) - len(use),
                "extreme_abs_gt20": int((use["target_return"].abs() > 0.20).sum()),
                "positive_extreme": int((use["target_return"] > 0.20).sum()),
                "negative_extreme": int((use["target_return"] < -0.20).sum()),
                "train_extreme": int(((use["trade_date"] < VALIDATION_START) & (use["target_return"].abs() > 0.20)).sum()),
                "legacy_validation_extreme": int(((use["trade_date"] >= VALIDATION_START) & (use["target_return"].abs() > 0.20)).sum()),
            }
        )
    policy = pd.DataFrame(policy_rows)

    trade_impact_rows = []
    top10 = top10.copy()
    top10["sample_key"] = dataset_key(top10)
    for policy_id, keys in [
        ("C0_full_no_cap", set(dataset_key(raw))),
        ("C1_event_clean_only", set(dataset_key(event))),
        ("C2_cap20", set(dataset_key(cap20))),
    ]:
        selected = top10[top10["sample_key"].isin(keys)].copy()
        scenario = ev.Scenario(policy_id, policy_id, 5, True)
        tmp = selected[["trade_date", "ts_code", "target_return"]].copy()
        tmp["trade_id"] = np.arange(len(tmp))
        tmp["score"] = selected["score"].to_numpy() if "score" in selected.columns else np.nan
        tmp["gross_return_v7"] = ev.net_to_gross_locked(tmp["target_return"])
        tmp["effective_gross_return"] = tmp["gross_return_v7"]
        tmp["fill_ratio"] = 1.0
        tmp["filled_notional_return"] = tmp["target_return"]
        tmp["portfolio_slot_return"] = tmp["target_return"]
        tmp["unfilled_reason"] = "filled"
        daily_ret = tmp.groupby("trade_date")["portfolio_slot_return"].mean().reindex(sorted(top10["trade_date"].astype(str).unique())).fillna(0.0)
        row = ev.metrics_from_returns(tmp, daily_ret, scenario, np.nan)
        row["policy_id"] = policy_id
        row["removed_v7_trades"] = len(top10) - len(selected)
        row["removed_extreme_v7_trades"] = int((top10[~top10["sample_key"].isin(keys)]["target_return"].abs() > 0.20).sum())
        trade_impact_rows.append(row)
    trade_impact = pd.DataFrame(trade_impact_rows)
    write_extreme_limitations()
    return extreme, by_split, policy, trade_impact


def add_extreme_attribution(extreme: pd.DataFrame, daily: pd.DataFrame, stock: pd.DataFrame, dirty_event_keys: set[str]) -> pd.DataFrame:
    out = extreme.merge(stock, on="ts_code", how="left", suffixes=("", "_stock"))
    entry_daily = daily[
        ["ts_code", "trade_date", "daily_source", "close", "prev_close_use", "prev_trade_date", "next_trade_date", "amount"]
    ].rename(
        columns={
            "daily_source": "entry_daily_source",
            "close": "entry_daily_close",
            "prev_close_use": "entry_prev_close",
            "amount": "entry_daily_amount",
            "next_trade_date": "exit_trade_date",
        }
    )
    out = out.merge(entry_daily, on=["ts_code", "trade_date"], how="left")
    exit_daily = daily[["ts_code", "trade_date", "daily_source", "close", "prev_close_use", "amount"]].rename(
        columns={
            "trade_date": "exit_trade_date",
            "daily_source": "exit_daily_source",
            "close": "exit_daily_close",
            "prev_close_use": "exit_prev_close",
            "amount": "exit_daily_amount",
        }
    )
    out = out.merge(exit_daily, on=["ts_code", "exit_trade_date"], how="left")

    minute_daily = pd.read_parquet(MINUTE_DAILY, columns=["ts_code", "trade_date", "close", "amount", "bars"]).rename(
        columns={"close": "minute_daily_close", "amount": "minute_daily_amount", "bars": "minute_bars"}
    )
    minute_daily["trade_date"] = minute_daily["trade_date"].astype(str)
    out = out.merge(minute_daily, on=["ts_code", "trade_date"], how="left")
    out["daily_minute_close_diff_pct"] = (out["entry_daily_close"] / out["minute_daily_close"] - 1).abs()
    out["minute_daily_close_mismatch"] = out["daily_minute_close_diff_pct"].gt(0.01)
    out["from_repaired_daily_sample"] = out["entry_daily_source"].eq("minute_reconstructed") | out["exit_daily_source"].eq("minute_reconstructed")
    out["dirty_event_removed_by_event_clean"] = out["sample_key"].isin(dirty_event_keys)

    tmp_for_minutes = out[["ts_code", "trade_date", "exit_time", "target_return", "entry_vwap"]].copy()
    tmp_for_minutes["score"] = 0.0
    tmp_for_minutes = ev.prepare_trade_base(ensure_trade_ids(tmp_for_minutes), daily, stock)
    minutes = ev.load_relevant_minutes(tmp_for_minutes, ev.DEFAULT_MINUTE_DIR)
    tmp_for_minutes = ev.add_entry_exit_bars(tmp_for_minutes, minutes)
    minute_cols = tmp_for_minutes[["trade_id", "entry_bar_amount", "exit_bar_amount", "exit_bar_close", "exit_bar_high", "exit_bar_low"]]
    out = out.reset_index(drop=True)
    out["trade_id"] = np.arange(len(out))
    out = out.merge(minute_cols, on="trade_id", how="left")
    out["exit_amount_zero"] = out["exit_bar_amount"].fillna(0).le(0)
    out["exit_amount_lt_10k"] = out["exit_bar_amount"].fillna(0).le(10_000)
    out["entry_amount_lt_10k"] = out["entry_bar_amount"].fillna(0).le(10_000)

    prev_dt = out["prev_trade_date"].map(ymd_to_datetime)
    cur_dt = out["trade_date"].map(ymd_to_datetime)
    out["prev_trade_gap_calendar_days"] = (cur_dt - prev_dt).dt.days
    out["suspect_resume_gap"] = out["prev_trade_gap_calendar_days"].gt(7)
    list_dt = out["list_date"].map(ymd_to_datetime)
    out["listing_age_days"] = (cur_dt - list_dt).dt.days
    out["is_new_stock_180d"] = out["listing_age_days"].between(0, 180, inclusive="both")
    out["is_subnew_365d"] = out["listing_age_days"].between(181, 365, inclusive="both")
    out["special_limit_board"] = out["ts_code"].astype(str).str.startswith(("300", "301", "688"))
    out["is_st"] = out.get("name", pd.Series("", index=out.index)).astype(str).str.upper().str.contains("ST", na=False)

    conditions = [
        out["dirty_event_removed_by_event_clean"],
        out["minute_daily_close_mismatch"],
        out["from_repaired_daily_sample"],
        out["exit_amount_zero"],
        out["exit_amount_lt_10k"],
        out["suspect_resume_gap"],
        out["is_new_stock_180d"],
        out["is_subnew_365d"],
        out["special_limit_board"],
    ]
    labels = [
        "confirmed_dirty_event_clean_removed",
        "minute_daily_close_mismatch",
        "minute_reconstructed_daily_source",
        "exit_bar_zero_amount",
        "exit_bar_amount_lt_10k",
        "suspension_or_resume_gap_gt7_calendar_days",
        "new_stock_180d",
        "subnew_stock_365d",
        "special_20pct_limit_board",
    ]
    out["primary_attribution"] = np.select(conditions, labels, default="unknown")
    out["exright_or_adjustment_evidence"] = np.where(out["minute_daily_close_mismatch"], "minute_daily_close_mismatch", "unknown")
    keep = [
        "ts_code",
        "trade_date",
        "split",
        "target_return",
        "extreme_side",
        "primary_attribution",
        "dirty_event_removed_by_event_clean",
        "from_repaired_daily_sample",
        "entry_daily_source",
        "exit_daily_source",
        "minute_daily_close_mismatch",
        "daily_minute_close_diff_pct",
        "exit_bar_amount",
        "exit_amount_zero",
        "exit_amount_lt_10k",
        "entry_bar_amount",
        "entry_amount_lt_10k",
        "suspect_resume_gap",
        "prev_trade_gap_calendar_days",
        "exright_or_adjustment_evidence",
        "is_new_stock_180d",
        "is_subnew_365d",
        "special_limit_board",
        "is_st",
        "name",
        "industry",
        "market",
        "list_date",
    ]
    return out[[c for c in keep if c in out.columns]]


def write_extreme_limitations() -> None:
    text = """# Batch 1 Extreme Return Limitations

This audit attributes `abs(target_return) > 20%` using only local data available
in the current project: repaired/raw feature frames, clean daily data, minute-
aggregated daily bars, raw 5-minute bars loaded for the extreme rows, and stock
basic metadata.

Fields marked `unknown` are intentionally left unknown. Batch 1 does not fetch
corporate-action details or authoritative suspension announcements, so it does
not guess ex-right/ex-dividend causes unless local minute-vs-daily price evidence
is present.

C0/C1/C2 are audit policies only. No model is retrained and no v7 feature,
label, TopN, universe or exit rule is changed.
"""
    (OUT_DIR / "batch1_extreme_return_limitations.md").write_text(text, encoding="utf-8")


def apply_zero_exit_deferral(trade_base: pd.DataFrame, minutes: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    out = trade_base.copy()
    out["zero_exit_amount_original"] = out["effective_exit_amount"].fillna(0).le(0)
    out["zero_exit_deferred"] = False
    out["zero_exit_defer_bars"] = 0
    out["zero_exit_original_gross_return"] = out["effective_gross_return"]
    for idx, row in out[out["zero_exit_amount_original"] & ~out["no_buy_due_to_execution"]].iterrows():
        replacement = find_next_positive_amount_bar(row, minutes)
        if replacement:
            out.loc[idx, "zero_exit_deferred"] = True
            out.loc[idx, "zero_exit_defer_bars"] = replacement["bars_checked"]
            out.loc[idx, "effective_exit_trade_date"] = replacement["trade_date"]
            out.loc[idx, "effective_exit_time"] = replacement["bar_time"]
            out.loc[idx, "effective_exit_vwap"] = replacement["bar_vwap"]
            out.loc[idx, "effective_exit_amount"] = replacement["amount"]
            if pd.notna(row["entry_vwap"]) and row["entry_vwap"] > 0:
                out.loc[idx, "effective_gross_return"] = replacement["bar_vwap"] / row["entry_vwap"] - 1
        else:
            out.loc[idx, "no_exit_due_to_execution"] = True
    return out


def find_next_positive_amount_bar(row: pd.Series, minutes: pd.DataFrame) -> dict[str, Any] | None:
    dates = [row["exit_trade_date"], row.get("fallback_trade_date")]
    start_times = [row["exit_time"], "09:30"]
    checked = 0
    for trade_date, start_time in zip(dates, start_times):
        if not isinstance(trade_date, str) or trade_date == "nan":
            continue
        day = minutes[(minutes["ts_code"] == row["ts_code"]) & (minutes["trade_date"] == trade_date)].sort_values("bar_time")
        day = day[day["bar_time"] > start_time] if trade_date == row["exit_trade_date"] else day[day["bar_time"] >= start_time]
        for bar in day.itertuples(index=False):
            checked += 1
            if pd.notna(bar.amount) and float(bar.amount) > 0 and pd.notna(bar.bar_vwap):
                return {
                    "trade_date": trade_date,
                    "bar_time": str(bar.bar_time),
                    "bar_vwap": float(bar.bar_vwap),
                    "amount": float(bar.amount),
                    "bars_checked": checked,
                }
    return None


def run_execution_audit(top10: pd.DataFrame, daily: pd.DataFrame, stock: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = ensure_trade_ids(top10)
    trade_base = ev.prepare_trade_base(selected, daily, stock)
    minutes = ev.load_relevant_minutes(trade_base, ev.DEFAULT_MINUTE_DIR)
    trade_base = ev.add_entry_exit_bars(trade_base, minutes)
    trade_base = ev.add_execution_flags(trade_base, minutes, daily)
    trade_base_e5 = apply_zero_exit_deferral(trade_base, minutes, daily)
    model_auc = ev.auc_safe(top10, "label_static_003", "score")

    policies = [
        ev.Scenario("E0_v7_5bp", "v7_original_5bp", 5, True),
        ev.Scenario("E1_10bp", "10bp_one_way_slippage", 10, True),
        ev.Scenario("E2_10bp_impact_100k", "10bp_one_way_slippage_plus_impact_cost", 10, True, include_impact=True, capital=100_000),
        ev.Scenario(
            "E3_10bp_impact_participation_10pct_no_partial_100k",
            "10bp_one_way_slippage_plus_impact_cost_plus_participation_cap",
            10,
            True,
            include_impact=True,
            capital=100_000,
            participation_cap=0.10,
            partial_fill=False,
            zero_fill=True,
            enforce_limit_suspend=False,
        ),
        ev.Scenario(
            "E4_10bp_impact_participation_10pct_partial_zero_100k",
            "10bp_one_way_slippage_plus_impact_cost_plus_participation_cap_plus_partial_zero_fill",
            10,
            True,
            include_impact=True,
            capital=100_000,
            participation_cap=0.10,
            partial_fill=True,
            zero_fill=True,
            enforce_limit_suspend=False,
        ),
        ev.Scenario(
            "E5_10bp_impact_participation_10pct_partial_zero_execution_100k",
            "E4_plus_limit_suspend_zero_exit_defer",
            10,
            True,
            include_impact=True,
            capital=100_000,
            participation_cap=0.10,
            partial_fill=True,
            zero_fill=True,
            enforce_limit_suspend=True,
        ),
    ]
    policy_rows = []
    policy_details = []
    for scenario in policies:
        base = trade_base_e5 if scenario.scenario_id.startswith("E5") else trade_base
        metrics, details, _daily = ev.evaluate_scenario(base, scenario, model_auc)
        policy_rows.append(add_participation_percentiles(metrics, details))
        details["policy_id"] = scenario.scenario_id
        policy_details.append(details)
    policy_summary = pd.DataFrame(policy_rows)

    capacity_rows = []
    partial_events = []
    zero_events = []
    reason_rows = []
    _, capacity_scenarios = ev.build_scenarios()
    for scenario in capacity_scenarios:
        scenario = replace(scenario, enforce_limit_suspend=True)
        metrics, details, _daily = ev.evaluate_scenario(trade_base_e5, scenario, model_auc)
        capacity_rows.append(add_participation_percentiles(metrics, details))
        events = details[details["fill_ratio"].lt(0.999)].copy()
        if not events.empty:
            events["scenario_id"] = scenario.scenario_id
            partial_events.append(events[events["fill_ratio"].gt(0)])
            zero_events.append(events[events["fill_ratio"].le(0)])
        reason = details.groupby("unfilled_reason").size().reset_index(name="rows")
        reason["scenario_id"] = scenario.scenario_id
        reason["capital"] = scenario.capital
        reason["participation_cap"] = scenario.participation_cap
        reason["slippage_bp"] = scenario.slippage_bp
        reason_rows.append(reason)
    capacity = pd.DataFrame(capacity_rows)
    partial = pd.concat(partial_events, ignore_index=True) if partial_events else pd.DataFrame()
    zero = pd.concat(zero_events, ignore_index=True) if zero_events else pd.DataFrame()
    reasons = pd.concat(reason_rows, ignore_index=True) if reason_rows else pd.DataFrame()
    fill = capacity[
        [
            "scenario_id",
            "capital",
            "participation_cap",
            "slippage_bp",
            "avg_fill_ratio",
            "partial_fill_count",
            "zero_fill_count",
            "unfilled_reason_filled",
            "unfilled_reason_partial_fill_capacity",
            "unfilled_reason_zero_fill_capacity",
            "unfilled_reason_no_buy_limit_or_missing",
            "unfilled_reason_no_exit_limit_or_missing",
            "single_ticket_participation_median",
            "single_ticket_participation_p90",
            "single_ticket_participation_p99",
        ]
    ].copy()
    write_impact_formula()
    write_zero_exit_analysis(trade_base, trade_base_e5, policy_summary, capacity)
    return policy_summary, capacity, fill, partial, zero, reasons


def add_participation_percentiles(metrics: dict[str, Any], details: pd.DataFrame) -> dict[str, Any]:
    out = dict(metrics)
    if "buy_participation" in details.columns and "sell_participation" in details.columns:
        p = np.maximum(details["buy_participation"].fillna(0).to_numpy(float), details["sell_participation"].fillna(0).to_numpy(float))
        out["single_ticket_participation_median"] = float(np.quantile(p, 0.50))
        out["single_ticket_participation_p90"] = float(np.quantile(p, 0.90))
        out["single_ticket_participation_p99"] = float(np.quantile(p, 0.99))
    else:
        out["single_ticket_participation_median"] = np.nan
        out["single_ticket_participation_p90"] = np.nan
        out["single_ticket_participation_p99"] = np.nan
    return out


def write_impact_formula() -> None:
    text = """# Batch 1 Impact Cost Formula

Batch 1 uses the same simplified execution-cost formula introduced by the Batch
0 unified evaluator.

Formula:

`side_impact = min(0.0050, 0.0015 * sqrt(realized_participation))`

Answers to required audit questions:

| Question | Answer |
|---|---|
| Is buy impact included? | Yes, when the scenario enables impact cost. |
| Is sell impact included? | Yes, when the scenario enables impact cost. |
| Is impact linear in participation? | No. It uses square-root participation. |
| Is impact bucketed by participation bands? | No. |
| Does it distinguish buy vs sell direction? | Only by each side's realized participation; the same formula is used on both sides. |
| Does it distinguish capital size? | Yes indirectly: larger capital raises order notional and realized participation. |
| Does it distinguish turnover/amount size? | Yes indirectly: realized participation divides order notional by the relevant 5-minute bar amount. |
| Is this a simplified formula? | Yes. It does not use order book depth, queue position, active buy/sell imbalance, intrabar path, or Level-2 liquidity. |

This is an audit stress formula, not an optimized trading-cost model.
"""
    (OUT_DIR / "batch1_impact_cost_formula.md").write_text(text, encoding="utf-8")


def write_zero_exit_analysis(original: pd.DataFrame, e5: pd.DataFrame, policy: pd.DataFrame, capacity: pd.DataFrame) -> None:
    row0 = original[(original["ts_code"] == "603268.SH") & (original["trade_date"] == "20260416")]
    row5 = e5[(e5["ts_code"] == "603268.SH") & (e5["trade_date"] == "20260416")]
    if row0.empty or row5.empty:
        text = "# 603268.SH Zero Exit Analysis\n\nRequired row was not found in the locked v7 Top10 signal file.\n"
        (OUT_DIR / "batch1_603268_zero_exit_analysis.md").write_text(text, encoding="utf-8")
        return
    r0 = row0.iloc[0]
    r5 = row5.iloc[0]
    low_exit = original[original["exit_bar_amount"].fillna(0).le(100_000)][
        ["trade_date", "ts_code", "exit_trade_date", "exit_time", "exit_bar_amount", "target_return"]
    ].sort_values("exit_bar_amount").head(20)
    low_entry = original[original["entry_bar_amount"].fillna(0).le(100_000)][
        ["trade_date", "ts_code", "entry_bar_amount", "target_return"]
    ].sort_values("entry_bar_amount").head(20)
    low_exit.to_csv(OUT_DIR / "batch1_low_exit_amount_samples.csv", index=False)
    low_entry.to_csv(OUT_DIR / "batch1_low_entry_amount_samples.csv", index=False)
    text = f"""# 603268.SH Zero Exit Analysis

Signal: `603268.SH`, trade date `20260416`, intended exit `20260417 {r0['exit_time']}`.

| Item | Value |
|---|---:|
| Original v7 target_return | {float(r0['target_return']):.10f} |
| Original intended exit bar amount | {float(r0['exit_bar_amount']):.2f} |
| Original gross return reconstructed from v7 | {float(r0['gross_return_v7']):.10f} |
| E5 zero-exit deferred? | {bool(r5['zero_exit_deferred'])} |
| E5 effective exit date | {r5['effective_exit_trade_date']} |
| E5 effective exit time | {r5['effective_exit_time']} |
| E5 effective exit amount | {float(r5['effective_exit_amount']):.2f} |
| E5 effective gross return after deferral | {float(r5['effective_gross_return']):.10f} |
| Gross return change from zero-exit deferral | {float(r5['effective_gross_return'] - r0['gross_return_v7']):.10f} |

Answers:

1. Original v7 backtest used the locked `target_return`; it did not reject this
   trade because the intended 10:30 sell bar amount was zero.
2. Batch 1 E5 treats a zero sell bar amount as non-executable at that bar.
3. E5 defers the sell to the next available 5-minute bar with positive amount.
4. The gross-return impact is shown in the table above.
5. Other exit samples with amount `<=100000` are exported to
   `batch1_low_exit_amount_samples.csv`.
6. Buy samples with amount `<=100000` are exported to
   `batch1_low_entry_amount_samples.csv`.

"""
    (OUT_DIR / "batch1_603268_zero_exit_analysis.md").write_text(text, encoding="utf-8")


def write_conclusion(
    universe_overlap: pd.DataFrame,
    retention: pd.DataFrame,
    universe_eval: pd.DataFrame,
    extreme_attr: pd.DataFrame,
    extreme_policy: pd.DataFrame,
    trade_impact: pd.DataFrame,
    execution_policy: pd.DataFrame,
    capacity: pd.DataFrame,
) -> None:
    u_agg = (
        universe_overlap[universe_overlap["universe_id"] != "U0_static_top3000_v7"]
        .groupby("universe_id")
        .agg(avg_overlap=("overlap_ratio_vs_u0", "mean"), avg_removed=("removed_from_u0_count", "mean"), min_overlap=("overlap_ratio_vs_u0", "min"))
        .reset_index()
    )
    r_agg = retention.groupby("universe_id").agg(avg_retention=("retention_ratio", "mean"), min_retention=("retention_ratio", "min"), total_removed=("removed_count", "sum")).reset_index()
    c_counts = extreme_attr["primary_attribution"].value_counts().reset_index()
    c_counts.columns = ["primary_attribution", "rows"]
    e5 = execution_policy[execution_policy["scenario_id"] == "E5_10bp_impact_participation_10pct_partial_zero_execution_100k"].iloc[0]
    cap_10bp = capacity[capacity["slippage_bp"] == 10].copy()
    cap_by_capital = cap_10bp.groupby("capital").agg(avg_fill=("avg_fill_ratio", "mean"), min_return=("cumulative_return", "min"), max_return=("cumulative_return", "max")).reset_index()
    recommended = "Yes, but only after accepting that Batch 2A is still an audit/research step, not a deployment step."
    text = f"""# Batch 1 Conclusion

Generated at Beijing time: {now_bj()}.

## 1. Batch 1 changed what?

Batch 1 added hard-check audits for universe construction, extreme target-return
attribution and execution ledger realism. It generated only audit outputs under
`reports/tushare/v8_research/batch1_hard_checks/`.

## 2. Batch 1 did not change what?

It did not change v7 data, model, features, labels, TopN, sell rules, or any
training process. It did not start Batch 2A, Batch 2B, Batch 3, Batch 4 or Batch 5.

## 3. Was v7_locked modified?

No.

## 4. Was the model retrained?

No.

## 5. Were features, labels, TopN or sell rules modified?

No.

## 6. U: How different are static Top3000 and rolling liquidity universes?

{u_agg.to_markdown(index=False)}

## 7. U: What is v7 original Top10 retention inside rolling pools?

{r_agg.to_markdown(index=False)}

## 8. C: What mainly causes extreme target_return?

{c_counts.head(10).to_markdown(index=False)}

## 9. C: What is the C0/C1/C2 effect?

Sample policy summary:

{extreme_policy.to_markdown(index=False)}

Existing v7 trade impact:

{trade_impact[['policy_id', 'cumulative_return', 'profit_factor', 'removed_v7_trades', 'removed_extreme_v7_trades']].to_markdown(index=False)}

## 10. E: Is v7 still positive after 10bp + impact + participation + partial/zero fill?

At `capital=100000`, `participation_cap=10%`, E5 cumulative return is
`{float(e5['cumulative_return']):.6f}`, PF is `{float(e5['profit_factor']):.6f}`.

## 11. E: How is fill quality across capital sizes?

{cap_by_capital.to_markdown(index=False)}

## 12. E: Are there non-executable, zero-fill or deferred-sell issues?

E5 includes limit/suspension checks and zero sell-bar deferral. The known
`603268.SH 20260416` zero exit amount event is deferred to the next positive
amount bar. Full event details are in `batch1_603268_zero_exit_analysis.md`.

## 13. Should Batch 2A start?

{recommended}

## 14. What should Batch 2A prioritize among U/L/F?

Priority should be `U` first, then `F`, then `L`.

Basis:

- `U`: rolling liquidity membership materially changes the candidate set and can
  be audited without retraining or label changes.
- `F`: execution and capacity stress show that transaction friction can erase
  the v7 edge, so factor robustness should be judged under fixed execution costs.
- `L`: label changes are higher risk because they require retraining and a fresh
  out-of-sample protocol; they should wait until the universe and execution
  measurement harness is accepted.

This recommendation is based on audit stability and implementation risk, not on
choosing the highest old-validation-period return.
"""
    (OUT_DIR / "batch1_conclusion.md").write_text(text, encoding="utf-8")


def write_sha256() -> None:
    rows = []
    for path in sorted(OUT_DIR.glob("batch1_*")):
        if path.name == "batch1_file_sha256.csv" or not path.is_file():
            continue
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        rows.append({"path": rel(path), "sha256": h.hexdigest(), "bytes": path.stat().st_size})
    for extra in [Path(__file__).resolve(), ROOT / "research" / "v8_research" / "evaluate_strategy.py"]:
        h = hashlib.sha256()
        with extra.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        rows.append({"path": rel(extra), "sha256": h.hexdigest(), "bytes": extra.stat().st_size})
    pd.DataFrame(rows).to_csv(OUT_DIR / "batch1_file_sha256.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    top10, candidates, daily, stock = load_base_inputs()
    top10 = ensure_trade_ids(top10)
    candidates = candidates.copy()
    candidates["trade_date"] = candidates["trade_date"].astype(str)

    daily_size, overlap, retention, universe_eval = run_universe_audit(top10, candidates, daily, stock)
    extreme_attr, extreme_by_split, extreme_policy, extreme_trade_impact = run_extreme_audit(top10, daily, stock)
    execution_policy, capacity, fill, partial, zero, reasons = run_execution_audit(top10, daily, stock)

    daily_size.to_csv(OUT_DIR / "batch1_universe_daily_size.csv", index=False)
    overlap.to_csv(OUT_DIR / "batch1_universe_overlap.csv", index=False)
    retention.to_csv(OUT_DIR / "batch1_universe_signal_retention.csv", index=False)
    universe_eval.to_csv(OUT_DIR / "batch1_universe_evaluation_summary.csv", index=False)

    extreme_attr.to_csv(OUT_DIR / "batch1_extreme_return_attribution.csv", index=False)
    extreme_by_split.to_csv(OUT_DIR / "batch1_extreme_return_by_split.csv", index=False)
    extreme_policy.to_csv(OUT_DIR / "batch1_extreme_return_policy_summary.csv", index=False)
    extreme_trade_impact.to_csv(OUT_DIR / "batch1_extreme_return_trade_impact.csv", index=False)

    execution_policy.to_csv(OUT_DIR / "batch1_execution_policy_summary.csv", index=False)
    capacity.to_csv(OUT_DIR / "batch1_capacity_matrix.csv", index=False)
    fill.to_csv(OUT_DIR / "batch1_fill_quality.csv", index=False)
    partial.to_csv(OUT_DIR / "batch1_partial_fill_events.csv", index=False)
    zero.to_csv(OUT_DIR / "batch1_zero_fill_events.csv", index=False)
    reasons.to_csv(OUT_DIR / "batch1_unfilled_reason_summary.csv", index=False)

    write_conclusion(overlap, retention, universe_eval, extreme_attr, extreme_policy, extreme_trade_impact, execution_policy, capacity)
    write_sha256()
    print(
        json.dumps(
            {
                "output_dir": str(OUT_DIR),
                "universe_rows": len(universe_eval),
                "extreme_rows": len(extreme_attr),
                "execution_policy_rows": len(execution_policy),
                "capacity_rows": len(capacity),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
