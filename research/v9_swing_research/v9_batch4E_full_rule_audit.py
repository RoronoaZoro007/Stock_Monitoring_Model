#!/usr/bin/env python3
"""v9 Batch 4E: full simple-rule audit.

This batch extends Batch 4D without adding or tuning rules. It audits:
- T+5 exit delay risk when the planned exit close is suspended or limit-down.
- Monthly cash ledger stability.
- Weak-market samples under the frozen market-regime labels.

No model training, feature change, label change, parameter search, or v7_locked
change is performed.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from v9_batch4D_deep_simple_rule_ledger import (
    HOLDOUT_END,
    HOLDOUT_START,
    PRIMARY_LABEL,
    REPORT_ROOT,
    SCENARIOS,
    STAGE_STATUS_PATH,
    TARGET_RULES,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    add_position_returns,
    build_scenario_daily,
    expand_position_days,
    load_target_trades,
    max_drawdown_from_equity,
    profit_factor_from_pnl,
    read_price_panel,
    sha256_file,
    summarize_ledger,
)


ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT / "data_tushare" / "clean" / "v9"
LABEL_PATH = CLEAN_DIR / "v9_swing_labels_3_5_10.parquet"
REPORT_DIR = REPORT_ROOT / "batch4E_full_rule_audit"
GLOBAL_HANDOFF_PATH = REPORT_ROOT / "v9_current_handoff.md"
ROADMAP_PATH = REPORT_ROOT / "v9_execution_roadmap.md"

MAX_EXIT_DELAY_DAYS = 5
PLANNED_POLICY = "planned_t5_close"
DELAY_POLICY = "delay_if_exit_blocked_max5"


def finite_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def write_config() -> None:
    lines = [
        "batch: batch4E_full_rule_audit",
        "training_enabled: false",
        "rule_search_enabled: false",
        "target_rules:",
    ]
    lines += [f"  - {r}" for r in TARGET_RULES]
    lines += [
        "exit_policies:",
        f"  - {PLANNED_POLICY}",
        f"  - {DELAY_POLICY}",
        f"max_exit_delay_trading_days: {MAX_EXIT_DELAY_DAYS}",
        "exit_delay_trigger: exit_suspend_flag_5d or exit_limit_down_close_flag_5d",
        "exit_delay_search: next tradable daily close where suspend_flag=false and limit_down_close_flag=false",
        "unresolved_delay_policy: mark at last finite close inside max-delay window and flag unresolved",
        "monthly_ledger: grouped by calendar month from overlapping daily capital ledger",
        "weak_sample_policy: use frozen trend_regime=weak signal days; do not create new regime thresholds",
        "scenarios:",
    ]
    for scenario in SCENARIOS:
        lines.append(f"  - {scenario.scenario_id}:")
        for k, v in asdict(scenario).items():
            if k == "scenario_id":
                continue
            lines.append(f"      {k}: {v}")
    (REPORT_DIR / "batch4E_config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_limitations() -> None:
    text = """# Batch 4E Limitations

- This is a full audit of two pre-selected simple rules, not an optimization batch.
- It does not train models, add rules, tune thresholds, change labels, change TopN, or modify v7_locked.
- Exit-delay simulation uses daily close data, not intraday order book data.
- If a blocked exit remains unresolved after 5 trading days, the position is marked at the last finite close inside the delay window and explicitly flagged as unresolved.
- Weak-market validation uses the frozen `trend_regime=weak` labels. The latest research holdout has no weak signal days, so truly recent weak-market validation is unavailable.
- Validation weak-market samples have already been observed in earlier audits; they are stress evidence, not final unseen proof.
- Industry fields remain current-snapshot caveats inherited from previous v9 batches.
"""
    (REPORT_DIR / "batch4E_limitations.md").write_text(text, encoding="utf-8")


def load_exit_price_panel() -> tuple[pd.DataFrame, list[str], dict[str, int], dict[str, pd.DataFrame]]:
    cols = ["trade_date", "ts_code", "adj_close", "suspend_flag", "limit_down_close_flag"]
    prices = pd.read_parquet(LABEL_PATH, columns=cols)
    prices["trade_date"] = prices["trade_date"].astype(str)
    prices["adj_close"] = finite_numeric(prices["adj_close"])
    for col in ["suspend_flag", "limit_down_close_flag"]:
        prices[col] = prices[col].fillna(False).astype(bool)
    prices = prices.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    dates = sorted(prices["trade_date"].unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    prices["date_idx"] = prices["trade_date"].map(date_to_idx).astype("int32")
    by_code = {code: g.reset_index(drop=True) for code, g in prices.groupby("ts_code", sort=False)}
    return prices, dates, date_to_idx, by_code


def build_delay_trades(
    trades: pd.DataFrame,
    date_to_idx: dict[str, int],
    by_code: dict[str, pd.DataFrame],
    max_delay_days: int = MAX_EXIT_DELAY_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    delayed = trades.copy()
    delayed["original_exit_date_5d"] = delayed["exit_date_5d"]
    delayed["original_fwd_ret_5d_open"] = delayed[PRIMARY_LABEL]
    delayed["exit_delay_days"] = 0
    delayed["exit_delay_triggered"] = False
    delayed["exit_delay_resolved"] = True
    delayed["exit_delay_unresolved"] = False
    delayed["delayed_exit_date"] = delayed["exit_date_5d"]
    delayed["delayed_exit_adj_close"] = delayed["entry_adj_open"] * (1.0 + delayed[PRIMARY_LABEL])

    for idx, row in delayed.iterrows():
        blocked = bool(row.get("exit_suspend_flag_5d", False)) or bool(row.get("exit_limit_down_close_flag_5d", False))
        if not blocked:
            continue
        code = row["ts_code"]
        original_exit = str(row["exit_date_5d"])
        original_idx = date_to_idx.get(original_exit)
        code_prices = by_code.get(code)
        event = {
            "trade_id": int(row["trade_id"]),
            "rule_id": row["rule_id"],
            "split": row["split"],
            "trade_date": row["trade_date"],
            "ts_code": code,
            "industry": row.get("industry"),
            "original_exit_date": original_exit,
            "exit_suspend_flag_5d": bool(row.get("exit_suspend_flag_5d", False)),
            "exit_limit_down_close_flag_5d": bool(row.get("exit_limit_down_close_flag_5d", False)),
            "resolved": False,
            "unresolved": False,
            "delay_days": np.nan,
            "delayed_exit_date": None,
            "original_return": float(row[PRIMARY_LABEL]) if pd.notna(row[PRIMARY_LABEL]) else np.nan,
            "delayed_return": np.nan,
            "return_delta": np.nan,
        }
        if original_idx is None or code_prices is None or code_prices.empty:
            event["unresolved"] = True
            rows.append(event)
            delayed.loc[idx, "exit_delay_triggered"] = True
            delayed.loc[idx, "exit_delay_resolved"] = False
            delayed.loc[idx, "exit_delay_unresolved"] = True
            continue

        window = code_prices[
            code_prices["date_idx"].between(original_idx + 1, original_idx + max_delay_days)
            & code_prices["adj_close"].notna()
        ].copy()
        tradable = window[(~window["suspend_flag"]) & (~window["limit_down_close_flag"])]
        if not tradable.empty:
            chosen = tradable.iloc[0]
            resolved = True
            unresolved = False
        elif not window.empty:
            chosen = window.iloc[-1]
            resolved = False
            unresolved = True
        else:
            chosen = None
            resolved = False
            unresolved = True

        delayed.loc[idx, "exit_delay_triggered"] = True
        delayed.loc[idx, "exit_delay_resolved"] = resolved
        delayed.loc[idx, "exit_delay_unresolved"] = unresolved
        if chosen is not None:
            delayed_exit_date = str(chosen["trade_date"])
            delayed_adj_close = float(chosen["adj_close"])
            delayed_ret = delayed_adj_close / float(row["entry_adj_open"]) - 1.0
            delay_days = int(chosen["date_idx"]) - int(original_idx)
            delayed.loc[idx, "exit_date_5d"] = delayed_exit_date
            delayed.loc[idx, PRIMARY_LABEL] = delayed_ret
            delayed.loc[idx, "exit_delay_days"] = delay_days
            delayed.loc[idx, "delayed_exit_date"] = delayed_exit_date
            delayed.loc[idx, "delayed_exit_adj_close"] = delayed_adj_close
            delayed.loc[idx, "exit_suspend_flag_5d"] = bool(chosen["suspend_flag"])
            delayed.loc[idx, "exit_limit_down_close_flag_5d"] = bool(chosen["limit_down_close_flag"])
            event.update(
                {
                    "resolved": resolved,
                    "unresolved": unresolved,
                    "delay_days": delay_days,
                    "delayed_exit_date": delayed_exit_date,
                    "delayed_return": delayed_ret,
                    "return_delta": delayed_ret - float(row[PRIMARY_LABEL]),
                }
            )
        rows.append(event)

    events = pd.DataFrame(rows)
    if events.empty:
        events = pd.DataFrame(
            columns=[
                "trade_id",
                "rule_id",
                "split",
                "trade_date",
                "ts_code",
                "industry",
                "original_exit_date",
                "exit_suspend_flag_5d",
                "exit_limit_down_close_flag_5d",
                "resolved",
                "unresolved",
                "delay_days",
                "delayed_exit_date",
                "original_return",
                "delayed_return",
                "return_delta",
            ]
        )
    else:
        for col in ["original_exit_date", "delayed_exit_date"]:
            if col in events.columns:
                events[col] = events[col].astype("string")
    return delayed, events


def build_policy_ledger(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    dates: list[str],
    date_to_idx: dict[str, int],
    exit_policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pos = expand_position_days(trades, dates, date_to_idx)
    pos = add_position_returns(pos, prices)
    daily_frames = []
    term_frames = []
    summary_frames = []
    for scenario in SCENARIOS:
        daily, terms = build_scenario_daily(pos, trades, scenario)
        summary = summarize_ledger(daily, terms, scenario)
        daily["exit_policy"] = exit_policy
        terms["exit_policy"] = exit_policy
        summary["exit_policy"] = exit_policy
        daily_frames.append(daily)
        term_frames.append(terms)
        summary_frames.append(summary)
    return (
        pd.concat(daily_frames, ignore_index=True),
        pd.concat(term_frames, ignore_index=True),
        pd.concat(summary_frames, ignore_index=True),
    )


def build_exit_delay_impact(summary_all: pd.DataFrame) -> pd.DataFrame:
    key = ["scenario_id", "rule_id", "split"]
    planned = summary_all[summary_all["exit_policy"].eq(PLANNED_POLICY)].copy()
    delayed = summary_all[summary_all["exit_policy"].eq(DELAY_POLICY)].copy()
    cols = [
        "cumulative_return",
        "max_drawdown",
        "profit_factor",
        "daily_win_rate",
        "trade_win_rate",
        "total_net_pnl_yuan",
        "total_cost_yuan",
        "avg_trade_fill_ratio",
        "zero_fill_trades",
        "partial_fill_trades",
    ]
    merged = planned[key + cols].merge(delayed[key + cols], on=key, suffixes=("_planned", "_delayed"), how="outer")
    for col in cols:
        if f"{col}_planned" in merged.columns and f"{col}_delayed" in merged.columns:
            merged[f"{col}_delta"] = merged[f"{col}_delayed"] - merged[f"{col}_planned"]
    return merged.sort_values(key)


def build_monthly_ledger(daily_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for keys, g in daily_all.groupby(["exit_policy", "scenario_id", "rule_id", "split"], sort=False):
        exit_policy, scenario_id, rule_id, split = keys
        g = g.sort_values("trade_date").copy()
        g["month"] = g["trade_date"].astype(str).str.slice(0, 6)
        for month, m in g.groupby("month", sort=False):
            start_equity = float(m["prev_equity_yuan"].iloc[0])
            end_equity = float(m["equity_yuan"].iloc[-1])
            rows.append(
                {
                    "exit_policy": exit_policy,
                    "scenario_id": scenario_id,
                    "rule_id": rule_id,
                    "split": split,
                    "month": month,
                    "ledger_days": int(len(m)),
                    "start_equity_yuan": start_equity,
                    "end_equity_yuan": end_equity,
                    "monthly_return": end_equity / start_equity - 1.0 if start_equity else np.nan,
                    "net_pnl_yuan": float(m["net_pnl_yuan"].sum()),
                    "total_cost_yuan": float(m["total_cost_yuan"].sum()),
                    "traded_notional_yuan": float(m["traded_notional_yuan"].sum()),
                    "avg_exposure": float(m["gross_exposure_ratio"].mean()),
                    "max_exposure": float(m["gross_exposure_ratio"].max()),
                    "avg_active_positions": float(m["active_positions"].mean()),
                    "max_drawdown_in_month": max_drawdown_from_equity(m["equity_yuan"]),
                    "daily_win_rate": float(m["daily_return"].gt(0).mean()),
                }
            )
    monthly = pd.DataFrame(rows).sort_values(["exit_policy", "scenario_id", "rule_id", "split", "month"])

    stability_rows = []
    for keys, g in monthly.groupby(["exit_policy", "scenario_id", "rule_id", "split"], sort=False):
        exit_policy, scenario_id, rule_id, split = keys
        g = g.sort_values("month")
        neg = g["monthly_return"].lt(0).to_numpy()
        longest = 0
        current = 0
        for value in neg:
            current = current + 1 if value else 0
            longest = max(longest, current)
        stability_rows.append(
            {
                "exit_policy": exit_policy,
                "scenario_id": scenario_id,
                "rule_id": rule_id,
                "split": split,
                "months": int(len(g)),
                "positive_months": int(g["monthly_return"].gt(0).sum()),
                "negative_months": int(g["monthly_return"].lt(0).sum()),
                "monthly_win_rate": float(g["monthly_return"].gt(0).mean()),
                "mean_monthly_return": float(g["monthly_return"].mean()),
                "median_monthly_return": float(g["monthly_return"].median()),
                "best_month_return": float(g["monthly_return"].max()),
                "worst_month_return": float(g["monthly_return"].min()),
                "longest_negative_month_streak": int(longest),
                "avg_monthly_turnover": float((g["traded_notional_yuan"] / g["start_equity_yuan"].replace(0, np.nan)).mean()),
            }
        )
    stability = pd.DataFrame(stability_rows).sort_values(["exit_policy", "scenario_id", "rule_id", "split"])
    return monthly, stability


def build_weak_sample_validation(
    planned_trades: pd.DataFrame,
    delayed_trades: pd.DataFrame,
    prices: pd.DataFrame,
    dates: list[str],
    date_to_idx: dict[str, int],
) -> pd.DataFrame:
    samples = [
        ("train_weak_signal_days", "train", planned_trades["split"].eq("train") & planned_trades["trend_regime"].eq("weak")),
        (
            "validation_weak_signal_days",
            "validation",
            planned_trades["split"].eq("validation") & planned_trades["trend_regime"].eq("weak"),
        ),
        (
            "research_holdout_weak_signal_days",
            "research_holdout",
            planned_trades["split"].eq("research_holdout") & planned_trades["trend_regime"].eq("weak"),
        ),
        ("all_weak_signal_days", "all", planned_trades["trend_regime"].eq("weak")),
    ]
    rows = []
    for exit_policy, source in [(PLANNED_POLICY, planned_trades), (DELAY_POLICY, delayed_trades)]:
        for sample_id, sample_split, mask in samples:
            sample = source.loc[mask].copy()
            if sample.empty:
                for scenario in SCENARIOS:
                    for rule_id in TARGET_RULES:
                        rows.append(
                            {
                                "sample_id": sample_id,
                                "sample_split": sample_split,
                                "source_split": "none",
                                "exit_policy": exit_policy,
                                "scenario_id": scenario.scenario_id,
                                "rule_id": rule_id,
                                "signal_days": 0,
                                "trades": 0,
                                "cumulative_return": np.nan,
                                "max_drawdown": np.nan,
                                "profit_factor": np.nan,
                                "daily_win_rate": np.nan,
                                "trade_win_rate": np.nan,
                                "avg_trade_fill_ratio": np.nan,
                                "zero_fill_trades": np.nan,
                                "partial_fill_trades": np.nan,
                            }
                        )
                continue
            daily, _terms, summary = build_policy_ledger(sample, prices, dates, date_to_idx, exit_policy)
            summary = summary.assign(sample_id=sample_id, sample_split=sample_split)
            for _, r in summary.iterrows():
                rows.append(
                    {
                        "sample_id": sample_id,
                        "sample_split": sample_split,
                        "source_split": r["split"],
                        "exit_policy": exit_policy,
                        "scenario_id": r["scenario_id"],
                        "rule_id": r["rule_id"],
                        "signal_days": int(sample[sample["rule_id"].eq(r["rule_id"])]["trade_date"].nunique()),
                        "trades": int(r["trades"]),
                        "cumulative_return": float(r["cumulative_return"]),
                        "max_drawdown": float(r["max_drawdown"]),
                        "profit_factor": float(r["profit_factor"]),
                        "daily_win_rate": float(r["daily_win_rate"]),
                        "trade_win_rate": float(r["trade_win_rate"]),
                        "avg_trade_fill_ratio": float(r["avg_trade_fill_ratio"]),
                        "zero_fill_trades": int(r["zero_fill_trades"]),
                        "partial_fill_trades": int(r["partial_fill_trades"]),
                    }
                )
    return pd.DataFrame(rows).sort_values(["sample_id", "exit_policy", "scenario_id", "rule_id"])


def build_weak_year_summary(terms_all: pd.DataFrame) -> pd.DataFrame:
    weak = terms_all[terms_all["trend_regime"].eq("weak")].copy()
    if weak.empty:
        return pd.DataFrame()
    weak["year"] = weak["trade_date"].astype(str).str.slice(0, 4)
    rows = []
    for keys, g in weak.groupby(["exit_policy", "scenario_id", "rule_id", "split", "year"], sort=False):
        exit_policy, scenario_id, rule_id, split, year = keys
        pnl = finite_numeric(g["net_trade_pnl_yuan"])
        target = finite_numeric(g["target_order_yuan"])
        rows.append(
            {
                "exit_policy": exit_policy,
                "scenario_id": scenario_id,
                "rule_id": rule_id,
                "split": split,
                "year": year,
                "signal_days": int(g["trade_date"].nunique()),
                "trades": int(len(g)),
                "net_pnl_yuan": float(pnl.sum()),
                "avg_net_return_on_target": float((pnl / target.replace(0, np.nan)).mean()),
                "trade_win_rate": float(pnl.gt(0).mean()),
                "profit_factor": profit_factor_from_pnl(pnl),
                "avg_fill_ratio": float(g["fill_ratio"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["exit_policy", "scenario_id", "rule_id", "year"])


def write_conclusion(
    summary_all: pd.DataFrame,
    impact: pd.DataFrame,
    monthly_stability: pd.DataFrame,
    weak_validation: pd.DataFrame,
    exit_events: pd.DataFrame,
) -> str:
    base = summary_all[
        summary_all["exit_policy"].eq(DELAY_POLICY)
        & summary_all["scenario_id"].eq("base_1m_5pct_signal_40bps")
        & summary_all["split"].eq("research_holdout")
    ].copy()
    strict = summary_all[
        summary_all["exit_policy"].eq(DELAY_POLICY)
        & summary_all["scenario_id"].eq("strict_5m_1pct_signal_40bps")
        & summary_all["split"].eq("research_holdout")
    ].copy()
    best_base = base.sort_values("cumulative_return", ascending=False).head(1)
    best_strict = strict.sort_values("cumulative_return", ascending=False).head(1)
    weak_holdout = weak_validation[
        weak_validation["sample_id"].eq("research_holdout_weak_signal_days")
        & weak_validation["trades"].fillna(0).gt(0)
    ]
    validation_weak_base = weak_validation[
        weak_validation["sample_id"].eq("validation_weak_signal_days")
        & weak_validation["exit_policy"].eq(DELAY_POLICY)
        & weak_validation["scenario_id"].eq("base_1m_5pct_signal_40bps")
    ].copy()
    event_count = int(len(exit_events))
    unresolved_count = int(exit_events.get("unresolved", pd.Series(dtype=bool)).fillna(False).sum()) if event_count else 0
    monthly_base = monthly_stability[
        monthly_stability["exit_policy"].eq(DELAY_POLICY)
        & monthly_stability["scenario_id"].eq("base_1m_5pct_signal_40bps")
        & monthly_stability["split"].eq("research_holdout")
    ].copy()

    gate = "review_required_no_forward_tracking"
    if (
        not best_base.empty
        and float(best_base.iloc[0]["cumulative_return"]) > 0
        and float(best_base.iloc[0]["max_drawdown"]) > -0.25
        and not weak_holdout.empty
        and unresolved_count == 0
    ):
        gate = "candidate_for_forward_shadow_review"

    lines = [
        "# Batch 4E Full Simple Rule Audit",
        "",
        "## Scope",
        "",
        "- Extended Batch 4D with exit-delay simulation, monthly cash ledger review and weak-market sample validation.",
        "- No model training, no new rule search, no parameter tuning, no feature/label/horizon change, and no v7_locked modification.",
        "",
        "## Core Findings",
        "",
    ]
    if not best_base.empty:
        r = best_base.iloc[0]
        lines.append(
            f"- Best delayed-exit research-holdout base rule: `{r['rule_id']}`, cumulative return `{r['cumulative_return']:.4f}`, max drawdown `{r['max_drawdown']:.4f}`, PF `{r['profit_factor']:.4f}`."
        )
    if not best_strict.empty:
        r = best_strict.iloc[0]
        lines.append(
            f"- Best delayed-exit strict capacity rule: `{r['rule_id']}`, cumulative return `{r['cumulative_return']:.4f}`, avg fill `{r['avg_trade_fill_ratio']:.4f}`, max drawdown `{r['max_drawdown']:.4f}`."
        )
    lines.append(f"- Exit-delay events found: `{event_count}`; unresolved after {MAX_EXIT_DELAY_DAYS} trading days: `{unresolved_count}`.")
    if not validation_weak_base.empty:
        r = validation_weak_base.sort_values("cumulative_return", ascending=False).iloc[0]
        lines.append(
            f"- Validation weak-sample best delayed base rule: `{r['rule_id']}`, cumulative return `{r['cumulative_return']:.4f}`, PF `{r['profit_factor']:.4f}`, trades `{int(r['trades'])}`."
        )
    lines.append("- Research-holdout still has no weak trend-regime signal days; recent weak-market robustness remains unproven.")
    if not monthly_base.empty:
        r = monthly_base.sort_values("monthly_win_rate", ascending=False).iloc[0]
        lines.append(
            f"- Best research-holdout delayed base monthly profile: `{r['rule_id']}`, monthly win rate `{r['monthly_win_rate']:.4f}`, worst month `{r['worst_month_return']:.4f}`, longest negative-month streak `{int(r['longest_negative_month_streak'])}`."
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Exit-delay risk is now simulated rather than only flagged, but it is still based on daily bars and not intraday order-book liquidity.",
        "- The rules remain more defensible than the failed Batch4B models, but the evidence is not sufficient for live readiness.",
        "- Weak-market evidence is historical/validation stress evidence; it is not a newest-period unseen weak-market test.",
        "- High-cost stress and monthly drawdown behavior must remain gating evidence before any forward paper tracking.",
        "",
        "## Gate",
        "",
        f"- Gate result: `{gate}`.",
        "- Do not resume complex model training from this result.",
        "- Do not start live trading or broker integration.",
        "- If accepted, the next step is a decision review: either keep historical research paused and wait for future weak-market data, or run only a paper-only forward shadow with explicit caveats.",
        "",
    ]
    (REPORT_DIR / "batch4E_conclusion.md").write_text("\n".join(lines), encoding="utf-8")
    return gate


def update_handoff(gate: str) -> None:
    text = f"""# Handoff: Batch 4E Full Simple Rule Audit Completed

## Completed Batch

- Completed: `Batch 4E full simple-rule audit`.
- Output directory: `reports/tushare/v9_swing_research/batch4E_full_rule_audit/`.
- Scope: exit-delay simulation, monthly cash ledger, weak-market sample validation for two pre-selected simple rules.

## Gate

- Gate result: `{gate}`.

## Required Inputs For Next Step

- `batch4E_exit_delay_impact_summary.csv`
- `batch4E_monthly_cash_ledger.csv`
- `batch4E_monthly_stability_summary.csv`
- `batch4E_weak_sample_validation.csv`
- `batch4E_conclusion.md`

## Next Step

- Do not resume complex model training.
- Do not start live trading or broker integration.
- If accepted, run a decision review before any paper-only forward shadow tracking.
"""
    GLOBAL_HANDOFF_PATH.write_text(text, encoding="utf-8")


def update_stage_status(gate: str) -> None:
    columns = ["stage", "batch", "status", "output_dir", "gate_result", "next_action"]
    row = {
        "stage": "Stage 7 Full Simple Rule Audit",
        "batch": "batch4E_full_rule_audit",
        "status": "completed",
        "output_dir": "reports/tushare/v9_swing_research/batch4E_full_rule_audit",
        "gate_result": "blocked" if gate == "review_required_no_forward_tracking" else "review",
        "next_action": "Decision review required; do not start live trading, broker integration, or complex model training.",
    }
    if STAGE_STATUS_PATH.exists():
        df = pd.read_csv(STAGE_STATUS_PATH)
        for col in columns:
            if col not in df.columns:
                df[col] = ""
        df = df[columns]
        df = df[df["batch"].ne(row["batch"])]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = df[columns]
    df.to_csv(STAGE_STATUS_PATH, index=False)


def update_roadmap(gate: str) -> None:
    text = ROADMAP_PATH.read_text(encoding="utf-8") if ROADMAP_PATH.exists() else "# v9 Swing Research Execution Roadmap\n"
    marker = "\n## Batch 4E Full Simple Rule Audit\n"
    addition = f"""{marker}
- Status: completed.
- Output directory: `reports/tushare/v9_swing_research/batch4E_full_rule_audit/`.
- Scope: exit-delay simulation, monthly cash ledger and weak-market sample validation for `low_stock_ret_60d` and `combo_low_liquidity_weak_momentum`.
- Gate: `{gate}`.
- Next: decision review only; no complex model training and no live readiness claim.
"""
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n" + addition
    else:
        text = text.rstrip() + "\n" + addition
    ROADMAP_PATH.write_text(text, encoding="utf-8")


def write_hashes() -> None:
    rows = []
    for path in sorted(REPORT_DIR.glob("batch4E_*")):
        if path.name == "batch4E_file_sha256.csv" or not path.is_file():
            continue
        rows.append({"file": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    for path in [GLOBAL_HANDOFF_PATH, ROADMAP_PATH, STAGE_STATUS_PATH]:
        if path.exists():
            rows.append({"file": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    with (REPORT_DIR / "batch4E_file_sha256.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "sha256", "size_bytes"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_config()
    write_limitations()
    print({"event": "load_trades"}, flush=True)
    trades = load_target_trades()
    print({"event": "load_exit_prices"}, flush=True)
    _exit_prices, _exit_dates, exit_date_to_idx, exit_by_code = load_exit_price_panel()
    delayed_trades, exit_events = build_delay_trades(trades, exit_date_to_idx, exit_by_code)
    exit_events.to_csv(REPORT_DIR / "batch4E_exit_delay_events.csv", index=False)

    print({"event": "read_prices"}, flush=True)
    prices, dates, date_to_idx = read_price_panel()
    daily_frames = []
    term_frames = []
    summary_frames = []
    for policy, source in [(PLANNED_POLICY, trades), (DELAY_POLICY, delayed_trades)]:
        print({"event": "build_policy_ledger", "exit_policy": policy, "trades": int(len(source))}, flush=True)
        daily, terms, summary = build_policy_ledger(source, prices, dates, date_to_idx, policy)
        daily_frames.append(daily)
        term_frames.append(terms)
        summary_frames.append(summary)

    daily_all = pd.concat(daily_frames, ignore_index=True)
    terms_all = pd.concat(term_frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)
    summary_all.to_csv(REPORT_DIR / "batch4E_exit_delay_ledger_summary.csv", index=False)
    daily_all.to_csv(REPORT_DIR / "batch4E_policy_daily_ledger.csv", index=False)
    terms_all.to_parquet(REPORT_DIR / "_local_batch4E_trade_terms.parquet", index=False, compression="zstd")

    impact = build_exit_delay_impact(summary_all)
    impact.to_csv(REPORT_DIR / "batch4E_exit_delay_impact_summary.csv", index=False)
    monthly, monthly_stability = build_monthly_ledger(daily_all)
    monthly.to_csv(REPORT_DIR / "batch4E_monthly_cash_ledger.csv", index=False)
    monthly_stability.to_csv(REPORT_DIR / "batch4E_monthly_stability_summary.csv", index=False)

    print({"event": "weak_sample_validation"}, flush=True)
    weak_validation = build_weak_sample_validation(trades, delayed_trades, prices, dates, date_to_idx)
    weak_validation.to_csv(REPORT_DIR / "batch4E_weak_sample_validation.csv", index=False)
    weak_year = build_weak_year_summary(terms_all)
    weak_year.to_csv(REPORT_DIR / "batch4E_weak_year_summary.csv", index=False)

    gate = write_conclusion(summary_all, impact, monthly_stability, weak_validation, exit_events)
    update_handoff(gate)
    update_stage_status(gate)
    update_roadmap(gate)
    write_hashes()
    print({"event": "done", "out_dir": str(REPORT_DIR), "gate_result": gate}, flush=True)


if __name__ == "__main__":
    main()
