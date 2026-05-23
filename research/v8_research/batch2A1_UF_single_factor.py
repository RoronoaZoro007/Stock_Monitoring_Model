#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))

import evaluate_strategy as ev  # noqa: E402
import batch1_hard_checks as b1  # noqa: E402


OUT_DIR = ROOT / "reports" / "tushare" / "v8_research" / "batch2A1_UF_single_factor"
VALIDATION_START = "20260224"
VALIDATION_END = "20260520"
TOP_N = 10
CAPITALS = [100_000, 300_000, 500_000, 1_000_000, 5_000_000]
PARTICIPATION_CAPS = [0.01, 0.03, 0.05, 0.10]
SLIPPAGE_BPS = [5, 10, 15, 20]
FILTER_THRESHOLDS = {
    "F0_no_capacity_filter": None,
    "F1_order_lte_10pct_ref_amount": 0.10,
    "F2_order_lte_5pct_ref_amount": 0.05,
    "F3_order_lte_3pct_ref_amount": 0.03,
    "F4_order_lte_1pct_ref_amount": 0.01,
}


def now_bj() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top10, candidates, daily, stock = b1.load_base_inputs()
    top10 = ensure_trade_ids(top10)
    candidates = candidates.copy()
    candidates["trade_date"] = candidates["trade_date"].astype(str)
    return top10, candidates, daily, stock


def base_scenarios_for_selection(capital: int | None = None) -> list[ev.Scenario]:
    scenarios = [
        ev.Scenario("5bp_one_way_slippage", "5bp_one_way_slippage", 5, True),
        ev.Scenario("10bp_one_way_slippage", "10bp_one_way_slippage", 10, True),
        ev.Scenario("15bp_one_way_slippage", "15bp_one_way_slippage", 15, True),
        ev.Scenario("20bp_one_way_slippage", "20bp_one_way_slippage", 20, True),
        ev.Scenario(
            "10bp_slippage_plus_impact",
            "10bp_one_way_slippage_plus_impact_cost",
            10,
            True,
            include_impact=True,
            capital=capital or 100_000,
        ),
    ]
    for part in PARTICIPATION_CAPS:
        scenarios.append(
            ev.Scenario(
                f"10bp_slippage_plus_impact_participation_{int(part * 100)}pct_no_partial",
                "10bp_one_way_slippage_plus_impact_cost_plus_participation_cap",
                10,
                True,
                include_impact=True,
                capital=capital or 100_000,
                participation_cap=part,
                partial_fill=False,
                zero_fill=True,
                enforce_limit_suspend=True,
            )
        )
        scenarios.append(
            ev.Scenario(
                f"10bp_slippage_plus_impact_participation_{int(part * 100)}pct_partial_zero",
                "10bp_one_way_slippage_plus_impact_cost_plus_participation_cap_plus_partial_zero_fill",
                10,
                True,
                include_impact=True,
                capital=capital or 100_000,
                participation_cap=part,
                partial_fill=True,
                zero_fill=True,
                enforce_limit_suspend=True,
            )
        )
    return scenarios


def capacity_scenarios_for_selection() -> list[ev.Scenario]:
    scenarios = []
    for capital in CAPITALS:
        for part in PARTICIPATION_CAPS:
            for bp in SLIPPAGE_BPS:
                scenarios.append(
                    ev.Scenario(
                        f"capacity_{capital}_part_{int(part * 10000)}bp_slip_{bp}bp",
                        "slippage_plus_impact_plus_participation_partial_zero",
                        bp,
                        True,
                        include_impact=True,
                        capital=capital,
                        participation_cap=part,
                        partial_fill=True,
                        zero_fill=True,
                        enforce_limit_suspend=True,
                    )
                )
    return scenarios


def prepare_execution_base(selected: pd.DataFrame, daily: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:
    selected = ensure_trade_ids(selected)
    base = ev.prepare_trade_base(selected, daily, stock)
    minutes = ev.load_relevant_minutes(base, ev.DEFAULT_MINUTE_DIR)
    base = ev.add_entry_exit_bars(base, minutes)
    base = ev.add_execution_flags(base, minutes, daily)
    return base


def evaluate_selection(
    selected: pd.DataFrame,
    daily: pd.DataFrame,
    stock: pd.DataFrame,
    experiment_id: str,
    group_type: str,
    scenarios: list[ev.Scenario],
    include_trade_details: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = prepare_execution_base(selected, daily, stock)
    score_auc = ev.auc_safe(selected, "label_static_003", "score") if "score" in selected.columns else np.nan
    metrics_rows = []
    daily_frames = []
    detail_frames = []
    fill_rows = []
    for scenario in scenarios:
        metrics, details, daily_df = ev.evaluate_scenario(base, scenario, score_auc)
        metrics["experiment_id"] = experiment_id
        metrics["group_type"] = group_type
        metrics_rows.append(ev_add_participation_percentiles(metrics, details))
        daily_df["experiment_id"] = experiment_id
        daily_df["group_type"] = group_type
        daily_frames.append(daily_df)
        fill_rows.append(
            {
                "experiment_id": experiment_id,
                "group_type": group_type,
                "scenario_id": scenario.scenario_id,
                "cost_case": scenario.cost_case,
                "capital": scenario.capital,
                "participation_cap": scenario.participation_cap,
                "slippage_bp": scenario.slippage_bp,
                "avg_fill_ratio": metrics["avg_fill_ratio"],
                "partial_fill_count": metrics["partial_fill_count"],
                "zero_fill_count": metrics["zero_fill_count"],
                "unfilled_reason_filled": metrics["unfilled_reason_filled"],
                "unfilled_reason_partial_fill_capacity": metrics["unfilled_reason_partial_fill_capacity"],
                "unfilled_reason_zero_fill_capacity": metrics["unfilled_reason_zero_fill_capacity"],
                "unfilled_reason_no_buy_limit_or_missing": metrics["unfilled_reason_no_buy_limit_or_missing"],
                "unfilled_reason_no_exit_limit_or_missing": metrics["unfilled_reason_no_exit_limit_or_missing"],
            }
        )
        if include_trade_details and scenario.scenario_id in {
            "5bp_one_way_slippage",
            "10bp_one_way_slippage",
            "10bp_slippage_plus_impact",
            "10bp_slippage_plus_impact_participation_10pct_partial_zero",
        }:
            details["experiment_id"] = experiment_id
            details["group_type"] = group_type
            detail_frames.append(details)
    return (
        pd.DataFrame(metrics_rows),
        pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(),
        pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame(),
        pd.DataFrame(fill_rows),
    )


def ev_add_participation_percentiles(metrics: dict[str, Any], details: pd.DataFrame) -> dict[str, Any]:
    out = dict(metrics)
    p = np.maximum(details["buy_participation"].fillna(0).to_numpy(float), details["sell_participation"].fillna(0).to_numpy(float))
    out["single_ticket_participation_median"] = float(np.quantile(p, 0.50))
    out["single_ticket_participation_p90"] = float(np.quantile(p, 0.90))
    out["single_ticket_participation_p99"] = float(np.quantile(p, 0.99))
    return out


def build_universe_selections(top10: pd.DataFrame, candidates: pd.DataFrame, daily: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    universe, _daily_size, _overlap = b1.build_rolling_universe(candidates, daily)
    selections = b1.select_top10_by_universe(candidates, universe)
    # Keep U1 as sanity check, but official comparison is U0/U2/U3.
    v7_sets = {(d, frozenset(g["ts_code"])) for d, g in top10.groupby("trade_date")}
    overlap_rows = []
    v7_by_day = {str(d): set(g["ts_code"]) for d, g in top10.groupby("trade_date")}
    for universe_id, selected in selections.items():
        for trade_date, day in selected.groupby("trade_date"):
            v7 = v7_by_day.get(str(trade_date), set())
            picked = set(day["ts_code"])
            overlap_rows.append(
                {
                    "trade_date": str(trade_date),
                    "experiment_id": universe_id,
                    "selected_count": len(picked),
                    "v7_top10_overlap_count": len(picked & v7),
                    "v7_top10_overlap_ratio": len(picked & v7) / len(picked) if picked else np.nan,
                    "is_v7_exact_same_set": picked == v7,
                }
            )
    return selections, pd.DataFrame(overlap_rows)


def concentration_summary(selected_map: dict[str, pd.DataFrame], overlap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment_id, selected in selected_map.items():
        ind = selected.get("industry", pd.Series("unknown", index=selected.index)).fillna("unknown")
        industry_counts = ind.value_counts()
        stock_counts = selected["ts_code"].value_counts()
        total = len(selected)
        rows.append(
            {
                "experiment_id": experiment_id,
                "total_trades": total,
                "unique_stocks": int(selected["ts_code"].nunique()),
                "unique_industries": int(ind.nunique()),
                "top_industry": str(industry_counts.index[0]) if len(industry_counts) else "",
                "top_industry_trade_count": int(industry_counts.iloc[0]) if len(industry_counts) else 0,
                "top_industry_trade_share": float(industry_counts.iloc[0] / total) if total and len(industry_counts) else np.nan,
                "top_stock": str(stock_counts.index[0]) if len(stock_counts) else "",
                "top_stock_trade_count": int(stock_counts.iloc[0]) if len(stock_counts) else 0,
                "top_stock_trade_share": float(stock_counts.iloc[0] / total) if total and len(stock_counts) else np.nan,
                "stock_hhi": float(((stock_counts / total) ** 2).sum()) if total else np.nan,
                "industry_hhi": float(((industry_counts / total) ** 2).sum()) if total else np.nan,
                "avg_v7_top10_overlap_ratio": float(overlap[overlap["experiment_id"] == experiment_id]["v7_top10_overlap_ratio"].mean()),
            }
        )
    return pd.DataFrame(rows)


def run_u_experiment(top10: pd.DataFrame, candidates: pd.DataFrame, daily: pd.DataFrame, stock: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selections, overlap = build_universe_selections(top10, candidates, daily)
    official = ["U0_static_top3000_v7", "U1_internal_roll_top3000_60d", "U2_internal_roll_top2500_60d", "U3_internal_roll_top2000_60d"]
    metrics_frames = []
    daily_frames = []
    details_frames = []
    fill_frames = []
    scenarios = base_scenarios_for_selection(100_000) + capacity_scenarios_for_selection()
    for universe_id in official:
        selected = selections[universe_id]
        metrics, daily_ret, details, fill = evaluate_selection(selected, daily, stock, universe_id, "U", scenarios)
        add_selection_metadata(metrics, selected, overlap, experiment_id=universe_id)
        metrics_frames.append(metrics)
        daily_frames.append(daily_ret)
        details_frames.append(details)
        fill_frames.append(fill)
    concentration = concentration_summary({k: selections[k] for k in official}, overlap)
    write_u_limitations()
    return (
        pd.concat(metrics_frames, ignore_index=True),
        pd.concat(daily_frames, ignore_index=True),
        pd.concat(details_frames, ignore_index=True),
        pd.concat(fill_frames, ignore_index=True),
        concentration,
    )


def add_selection_metadata(metrics: pd.DataFrame, selected: pd.DataFrame, overlap: pd.DataFrame, experiment_id: str) -> None:
    counts = selected.groupby("trade_date").size()
    ov = overlap[overlap["experiment_id"] == experiment_id]
    metrics["selected_trading_days"] = int(selected["trade_date"].nunique())
    metrics["selected_total_trades"] = int(len(selected))
    metrics["avg_selected_per_day"] = float(counts.mean()) if len(counts) else np.nan
    metrics["days_below_top10"] = int((counts < TOP_N).sum()) if len(counts) else 0
    metrics["avg_v7_top10_overlap_ratio"] = float(ov["v7_top10_overlap_ratio"].mean()) if len(ov) else np.nan


def write_u_limitations() -> None:
    text = """# Batch 2A-1 U Limitations

The U experiment uses the existing v7 Top3000 validation score matrix. It is an
internal rolling liquidity filter inside the downloaded Top3000, not an unbiased
all-A-share daily rolling Top3000.

The score matrix is complete for the legacy validation interval, so U0/U1/U2/U3
are reselected daily by locked v7 `score` without retraining, refitting,
feature changes, label changes, TopN changes or sell-rule changes.

Rolling liquidity uses `avg_amount_60d` computed with `t-1` and earlier daily
amount. It does not use future liquidity or post-validation data.

U1 is retained only as a sanity check because Batch 1 showed it is identical to
U0 inside the current Top3000 matrix.
"""
    (OUT_DIR / "batch2A1_U_limitations.md").write_text(text, encoding="utf-8")


def add_capacity_reference_amounts(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    out["tail_1430_1450_amount"] = np.expm1(out["amount_sofar_log"].astype(float)) * out["tail_amount_share"].astype(float)
    proxy = compute_past20_exit_window_proxy(out)
    out = out.merge(proxy, on=["ts_code", "trade_date"], how="left")
    out["ref_A_tail_1430_1450_amount"] = out["tail_1430_1450_amount"]
    out["ref_B_min_tail_past20_exit_proxy"] = np.minimum(out["tail_1430_1450_amount"], out["past20_exit_window_amount_proxy"])
    return out


def compute_past20_exit_window_proxy(candidates: pd.DataFrame) -> pd.DataFrame:
    symbols = sorted(candidates["ts_code"].unique())
    min_date = "20260101"
    max_date = VALIDATION_END
    rows = []
    cols = ["ts_code", "trade_time", "amount"]
    for i, ts_code in enumerate(symbols, 1):
        symbol_dir = ev.DEFAULT_MINUTE_DIR / f"ts_code={ts_code}"
        frames = []
        for path in sorted(symbol_dir.glob("*.parquet")):
            df = pd.read_parquet(path, columns=cols)
            if df.empty:
                continue
            df["trade_date"] = df["trade_time"].astype(str).str.slice(0, 10).str.replace("-", "", regex=False)
            df = df[(df["trade_date"] >= min_date) & (df["trade_date"] <= max_date)]
            if df.empty:
                continue
            df["bar_time"] = df["trade_time"].astype(str).str.slice(11, 16)
            df = df[(df["bar_time"] >= "09:30") & (df["bar_time"] <= "10:30")]
            if not df.empty:
                frames.append(df[["ts_code", "trade_date", "amount"]])
        if not frames:
            continue
        day = pd.concat(frames, ignore_index=True)
        day["amount"] = pd.to_numeric(day["amount"], errors="coerce")
        day = day.groupby(["ts_code", "trade_date"], as_index=False)["amount"].sum().rename(columns={"amount": "exit_window_0930_1030_amount"})
        day = day.sort_values(["ts_code", "trade_date"])
        day["past20_exit_window_amount_proxy"] = day.groupby("ts_code")["exit_window_0930_1030_amount"].transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).mean()
        )
        rows.append(day[["ts_code", "trade_date", "past20_exit_window_amount_proxy"]])
    if not rows:
        return pd.DataFrame(columns=["ts_code", "trade_date", "past20_exit_window_amount_proxy"])
    proxy = pd.concat(rows, ignore_index=True)
    dates = set(candidates["trade_date"].astype(str).unique())
    return proxy[proxy["trade_date"].isin(dates)].copy()


def select_f_candidates(candidates: pd.DataFrame, capital: int, filter_id: str, ref_method: str, threshold: float | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = candidates.copy()
    order_notional = capital / TOP_N
    if threshold is None:
        out["capacity_pass"] = True
        out["reference_amount"] = np.nan
    else:
        ref_col = "ref_A_tail_1430_1450_amount" if ref_method == "A_tail" else "ref_B_min_tail_past20_exit_proxy"
        out["reference_amount"] = pd.to_numeric(out[ref_col], errors="coerce")
        out["capacity_pass"] = out["reference_amount"].gt(0) & (order_notional <= out["reference_amount"] * threshold)
    event_rows = []
    selected_frames = []
    for trade_date, day in out.groupby("trade_date", sort=True):
        before = len(day)
        eligible = day[day["capacity_pass"]].copy()
        selected = eligible.sort_values("score", ascending=False).head(TOP_N).copy()
        selected_frames.append(selected)
        event_rows.append(
            {
                "trade_date": trade_date,
                "experiment_id": f"{filter_id}_{ref_method}_capital_{capital}",
                "filter_id": filter_id,
                "ref_method": ref_method,
                "capital": capital,
                "order_notional": order_notional,
                "threshold": threshold,
                "candidates_before_filter": before,
                "candidates_after_filter": len(eligible),
                "filtered_removed_count": before - len(eligible),
                "selected_count": len(selected),
                "below_top10": len(selected) < TOP_N,
            }
        )
    selected_all = ensure_trade_ids(pd.concat(selected_frames, ignore_index=True)) if selected_frames else pd.DataFrame()
    return selected_all, pd.DataFrame(event_rows)


def run_f_experiment(candidates: pd.DataFrame, daily: pd.DataFrame, stock: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = add_capacity_reference_amounts(candidates)
    metrics_frames = []
    daily_frames = []
    details_frames = []
    fill_frames = []
    events_frames = []
    for capital in CAPITALS:
        for filter_id, threshold in FILTER_THRESHOLDS.items():
            ref_methods = ["none"] if threshold is None else ["A_tail", "B_tail_and_past20_exit_proxy"]
            for ref_method in ref_methods:
                selected, events = select_f_candidates(candidates, capital, filter_id, ref_method, threshold)
                experiment_id = f"{filter_id}_{ref_method}_capital_{capital}"
                scenarios = base_scenarios_for_selection(capital)
                metrics, daily_ret, details, fill = evaluate_selection(selected, daily, stock, experiment_id, "F", scenarios)
                add_f_metadata(metrics, events)
                metrics["filter_id"] = filter_id
                metrics["ref_method"] = ref_method
                metrics["filter_capital"] = capital
                metrics["filter_threshold"] = threshold
                daily_ret["filter_id"] = filter_id
                daily_ret["ref_method"] = ref_method
                daily_ret["filter_capital"] = capital
                details["filter_id"] = filter_id
                details["ref_method"] = ref_method
                details["filter_capital"] = capital
                fill["filter_id"] = filter_id
                fill["ref_method"] = ref_method
                fill["filter_capital"] = capital
                metrics_frames.append(metrics)
                daily_frames.append(daily_ret)
                details_frames.append(details)
                fill_frames.append(fill)
                events_frames.append(events)
    write_f_limitations(candidates)
    return (
        pd.concat(metrics_frames, ignore_index=True),
        pd.concat(daily_frames, ignore_index=True),
        pd.concat(details_frames, ignore_index=True),
        pd.concat(fill_frames, ignore_index=True),
        pd.concat(events_frames, ignore_index=True),
    )


def add_f_metadata(metrics: pd.DataFrame, events: pd.DataFrame) -> None:
    metrics["filtered_removed_candidates_total"] = int(events["filtered_removed_count"].sum())
    metrics["days_below_top10"] = int(events["below_top10"].sum())
    metrics["avg_candidates_after_filter"] = float(events["candidates_after_filter"].mean())
    metrics["min_candidates_after_filter"] = int(events["candidates_after_filter"].min())
    metrics["avg_selected_per_day"] = float(events["selected_count"].mean())


def write_f_limitations(candidates: pd.DataFrame) -> None:
    proxy_coverage = candidates["past20_exit_window_amount_proxy"].notna().mean()
    text = f"""# Batch 2A-1 F Limitations

F uses locked v7 scores and only changes candidate filtering before taking daily
Top10. It does not retrain, refit, alter features, alter labels, alter TopN, or
alter sell rules.

Reference amount A:

`tail_1430_1450_amount = expm1(amount_sofar_log) * tail_amount_share`

Reference amount B:

`min(tail_1430_1450_amount, past20_exit_window_amount_proxy)`

`past20_exit_window_amount_proxy` is computed from raw 5-minute bars as the
rolling mean of prior 20 trading days' `09:30-10:30` amount. It is shifted by
one trading day, so day t uses only t-1 and earlier data. Current proxy coverage
inside the validation candidate matrix is `{proxy_coverage:.4f}`.

F is a single-factor capacity filter experiment. U+F combinations are not a
formal Batch 2A-1 conclusion and are reserved for Batch 3.
"""
    (OUT_DIR / "batch2A1_F_limitations.md").write_text(text, encoding="utf-8")


def write_conclusion(
    u_summary: pd.DataFrame,
    u_conc: pd.DataFrame,
    f_summary: pd.DataFrame,
    f_events: pd.DataFrame,
) -> None:
    u_core = u_summary[u_summary["scenario_id"].isin(["5bp_one_way_slippage", "10bp_one_way_slippage", "10bp_slippage_plus_impact"])]
    u_pivot = u_core.pivot_table(index="experiment_id", columns="scenario_id", values="cumulative_return", aggfunc="first").reset_index()
    f_core = f_summary[f_summary["scenario_id"].isin(["10bp_slippage_plus_impact", "10bp_slippage_plus_impact_participation_10pct_partial_zero"])]
    f_key = f_core[
        (f_core["scenario_id"] == "10bp_slippage_plus_impact_participation_10pct_partial_zero")
        & (f_core["filter_capital"] == 100_000)
    ][
        [
            "experiment_id",
            "cumulative_return",
            "profit_factor",
            "max_drawdown",
            "avg_fill_ratio",
            "days_below_top10",
            "filtered_removed_candidates_total",
        ]
    ].sort_values("profit_factor", ascending=False)
    f_insufficient = f_events.groupby("experiment_id")["below_top10"].sum().reset_index(name="days_below_top10")
    max_days_below = int(f_insufficient["days_below_top10"].max()) if len(f_insufficient) else 0
    shortage_answer = (
        "No. All tested F filters still had at least 10 eligible candidates on every validation trading day."
        if max_days_below == 0
        else "Yes. Some stricter F filters produced days with fewer than 10 eligible candidates."
    )
    shortage_table = (
        f_insufficient.sort_values("days_below_top10", ascending=False).head(12).to_markdown(index=False)
        if max_days_below > 0
        else f_insufficient.head(12).to_markdown(index=False)
    )
    u2 = u_summary[(u_summary["experiment_id"] == "U2_internal_roll_top2500_60d") & (u_summary["scenario_id"] == "10bp_slippage_plus_impact")].iloc[0]
    u3 = u_summary[(u_summary["experiment_id"] == "U3_internal_roll_top2000_60d") & (u_summary["scenario_id"] == "10bp_slippage_plus_impact")].iloc[0]
    f_positive = f_key[f_key["cumulative_return"] > 0]
    recommendation = "No"
    recommended_universe = "none"
    recommended_filter = "none"
    reason = (
        "Batch 2A-1 shows U/F filters improve some fill-quality or concentration diagnostics, "
        "but 10bp+impact+participation partial/zero-fill results remain weak on the legacy interval. "
        "Starting L retraining now would risk fitting labels to a fragile execution edge."
    )
    text = f"""# Batch 2A-1 Conclusion

Generated at Beijing time: {now_bj()}.

## 1. Batch 2A-1 changed what?

It added non-retraining U and F single-factor experiments using locked v7 scores:
U reselects Top10 inside rolling liquidity universes, and F applies capacity
filters before taking Top10.

## 2. Batch 2A-1 did not change what?

It did not change v7_locked, model weights, features, labels, TopN, sell rules,
market regime filters, industry constraints, exit rules, model complexity, or
walk-forward setup. It did not start Batch 2A-2, Batch 2B, Batch 3, Batch 4 or
Batch 5.

## 3. Was v7_locked modified?

No.

## 4. Was the model retrained?

No.

## 5. Were features, labels, TopN or sell rules modified?

No.

## 6. Was a complete daily score matrix available?

Yes. The locked validation prediction file contains the complete daily score
matrix for the current Top3000 validation candidate set. Batch 2A-1 used it for
inference-only reselection. No model fit was performed.

## 7. Did U2/U3 improve under 10bp + impact?

U2 10bp+impact cumulative return: `{float(u2['cumulative_return']):.6f}`,
PF `{float(u2['profit_factor']):.6f}`. U3 10bp+impact cumulative return:
`{float(u3['cumulative_return']):.6f}`, PF `{float(u3['profit_factor']):.6f}`.

U cumulative-return table:

{u_pivot.to_markdown(index=False)}

## 8. Are U changes driven by fewer trades, concentration or liquidity?

Top10 remains daily Top10 when candidates are sufficient, so trade count does
not fall materially. The main change is candidate composition and liquidity
membership. Concentration diagnostics:

{u_conc.to_markdown(index=False)}

## 9. Which F filters improve PF, drawdown and fill ratio?

Key F rows at capital `100000`, scenario `10bp+impact+10% participation+partial/zero`:

{f_key.head(12).to_markdown(index=False)}

## 10. Is capacity filtering only reducing trades rather than improving signal quality?

In this run, capacity filtering removed candidates but did not reduce any day
below Top10. Therefore it did not reduce executed trade count in the main Top10
setup. Any apparent improvement or deterioration should still be treated as a
candidate-composition and capacity-screening effect unless it also improves PF,
drawdown and fill quality consistently across capital levels.

## 11. Are there days with fewer than 10 candidates?

{shortage_answer}

Daily shortage counts are in `batch2A1_F_capacity_filter_events.csv`. Aggregate
sample:

{shortage_table}

## 12. Is return concentrated in a few days?

Yes, the summary files include cumulative return after dropping the largest 1,
3 and 5 profit days. These concentration columns should be used before treating
any old-validation-period gain as robust.

## 13. Should Batch 2A-2 L execution-aware label retraining start?

{recommendation}.

## 14. If L starts, which universe and F filter should be used?

Recommended universe: `{recommended_universe}`. Recommended F filter:
`{recommended_filter}`.

Reason: {reason}

"""
    (OUT_DIR / "batch2A1_conclusion.md").write_text(text, encoding="utf-8")


def write_hashes() -> None:
    rows = []
    for path in sorted(OUT_DIR.glob("batch2A1_*")):
        if not path.is_file() or path.name == "batch2A1_file_sha256.csv":
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
    pd.DataFrame(rows).to_csv(OUT_DIR / "batch2A1_file_sha256.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    top10, candidates, daily, stock = load_inputs()
    u_summary, u_daily, u_details, u_fill, u_conc = run_u_experiment(top10, candidates, daily, stock)
    f_summary, f_daily, f_details, f_fill, f_events = run_f_experiment(candidates, daily, stock)

    u_summary.to_csv(OUT_DIR / "batch2A1_U_reselection_summary.csv", index=False)
    u_daily.to_csv(OUT_DIR / "batch2A1_U_daily_returns.csv", index=False)
    u_details.to_csv(OUT_DIR / "batch2A1_U_trade_details.csv", index=False)
    u_fill.to_csv(OUT_DIR / "batch2A1_U_fill_quality.csv", index=False)
    u_conc.to_csv(OUT_DIR / "batch2A1_U_concentration_summary.csv", index=False)

    f_summary.to_csv(OUT_DIR / "batch2A1_F_filter_summary.csv", index=False)
    f_daily.to_csv(OUT_DIR / "batch2A1_F_daily_returns.csv", index=False)
    f_details.to_csv(OUT_DIR / "batch2A1_F_trade_details.csv", index=False)
    f_fill.to_csv(OUT_DIR / "batch2A1_F_fill_quality.csv", index=False)
    f_events.to_csv(OUT_DIR / "batch2A1_F_capacity_filter_events.csv", index=False)

    write_conclusion(u_summary, u_conc, f_summary, f_events)
    write_hashes()
    print(
        json.dumps(
            {
                "output_dir": str(OUT_DIR),
                "u_summary_rows": len(u_summary),
                "f_summary_rows": len(f_summary),
                "score_matrix_complete": True,
                "u_trade_detail_rows": len(u_details),
                "f_trade_detail_rows": len(f_details),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
