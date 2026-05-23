#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))

import evaluate_strategy as ev  # noqa: E402
import batch2A15_alpha_attribution as a15  # noqa: E402


OUT_DIR = ROOT / "reports" / "tushare" / "v8_research" / "batch2B_R0_market_regime_diagnostic"
RANDOM_RUNS = 1000
RANDOM_SEED = 20260523
PRIMARY_SCENARIOS = a15.PRIMARY_SCENARIOS
PRIMARY_SCENARIO_IDS = a15.PRIMARY_SCENARIO_IDS
MAIN_SCENARIO_ID = "10bp_slippage_plus_impact"

REGIME_SPECS = [
    {
        "regime_id": "R1_tail_down",
        "regime_side": "positive_diagnostic",
        "regime_field": "tail_direction",
        "regime_value": "tail_down",
        "description": "market_tail_ret_median < 0",
    },
    {
        "regime_id": "R2_market_down",
        "regime_side": "positive_diagnostic",
        "regime_field": "market_direction",
        "regime_value": "market_down",
        "description": "market_ret_median_1450 < 0",
    },
    {
        "regime_id": "R3_low_tail_vol",
        "regime_side": "positive_diagnostic",
        "regime_field": "tail_vol_regime",
        "regime_value": "low_tail_vol",
        "description": "market_tail_vol_proxy below validation median",
    },
    {
        "regime_id": "R4_tail_breadth_Q2",
        "regime_side": "positive_diagnostic",
        "regime_field": "market_tail_breadth_positive_qbucket",
        "regime_value": "Q2",
        "description": "market_tail_breadth_positive second quintile",
    },
    {
        "regime_id": "R5_tail_ret_median_Q2",
        "regime_side": "positive_diagnostic",
        "regime_field": "market_tail_ret_median_qbucket",
        "regime_value": "Q2",
        "description": "market_tail_ret_median second quintile",
    },
    {
        "regime_id": "R6_market_dispersion_Q1_low",
        "regime_side": "positive_diagnostic",
        "regime_field": "market_dispersion_proxy_qbucket",
        "regime_value": "Q1_low",
        "description": "market_dispersion_proxy lowest quintile",
    },
    {
        "regime_id": "C1_tail_up",
        "regime_side": "negative_control",
        "regime_field": "tail_direction",
        "regime_value": "tail_up",
        "description": "market_tail_ret_median >= 0",
    },
    {
        "regime_id": "C2_market_up",
        "regime_side": "negative_control",
        "regime_field": "market_direction",
        "regime_value": "market_up",
        "description": "market_ret_median_1450 >= 0",
    },
    {
        "regime_id": "C3_tail_breadth_Q5_high",
        "regime_side": "negative_control",
        "regime_field": "market_tail_breadth_positive_qbucket",
        "regime_value": "Q5_high",
        "description": "market_tail_breadth_positive highest quintile",
    },
    {
        "regime_id": "C4_tail_ret_median_Q5_high",
        "regime_side": "negative_control",
        "regime_field": "market_tail_ret_median_qbucket",
        "regime_value": "Q5_high",
        "description": "market_tail_ret_median highest quintile",
    },
]

STRATEGY_SPECS = [
    {"strategy_id": "S0_v7_original_top10", "strategy_name": "S0: v7 original Top10", "mask_col": None},
    {"strategy_id": "S1_U2_filter_only_no_refill", "strategy_name": "S1: U2 filter-only, no refill", "mask_col": "in_u2"},
    {"strategy_id": "S2_U3_filter_only_no_refill", "strategy_name": "S2: U3 filter-only, no refill", "mask_col": "in_u3"},
]


def now_bj() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_prepared() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    top10_raw, candidates_raw, daily, stock, all_dates = a15.load_inputs()
    candidates, top10_liq, universe = a15.add_liquidity_fields(candidates_raw, top10_raw, daily)
    top10 = a15.add_universe_flags(top10_liq, universe)
    top10, market_day = a15.add_market_state_fields(candidates, top10)
    top10_base = a15.prepare_execution_base(top10, daily, stock)
    return top10, top10_base, daily, stock, all_dates, market_day


def strategy_mask(top10: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    if spec["mask_col"] is None:
        return pd.Series(True, index=top10.index)
    return top10[spec["mask_col"]].astype(bool)


def regime_dates(market_day: pd.DataFrame, spec: dict[str, Any]) -> list[str]:
    field = spec["regime_field"]
    value = spec["regime_value"]
    hit = market_day[market_day[field].astype(str) == str(value)]
    return sorted(hit["trade_date"].astype(str).unique())


def concentration(selected: pd.DataFrame, daily_impact: pd.DataFrame) -> dict[str, Any]:
    total = len(selected)
    industries = selected.get("industry", pd.Series("unknown", index=selected.index)).fillna("unknown")
    industry_counts = industries.value_counts()
    stock_counts = selected["ts_code"].value_counts() if total else pd.Series(dtype=int)
    daily = daily_impact.sort_values("daily_return", ascending=False)
    top_profit = daily[daily["daily_return"] > 0]["daily_return"]
    total_pos = float(top_profit.sum())
    return {
        "total_trades": int(total),
        "unique_stocks": int(selected["ts_code"].nunique()) if total else 0,
        "top_stock": str(stock_counts.index[0]) if len(stock_counts) else "",
        "top_stock_count": int(stock_counts.iloc[0]) if len(stock_counts) else 0,
        "top_stock_share": float(stock_counts.iloc[0] / total) if total and len(stock_counts) else np.nan,
        "stock_hhi": float(((stock_counts / total) ** 2).sum()) if total else np.nan,
        "unique_industries": int(industries.nunique()) if total else 0,
        "top_industry": str(industry_counts.index[0]) if len(industry_counts) else "",
        "top_industry_count": int(industry_counts.iloc[0]) if len(industry_counts) else 0,
        "top_industry_share": float(industry_counts.iloc[0] / total) if total and len(industry_counts) else np.nan,
        "industry_hhi": float(((industry_counts / total) ** 2).sum()) if total else np.nan,
        "top1_profit_day_return": float(top_profit.head(1).sum()) if len(top_profit) else 0.0,
        "top3_profit_day_return_sum": float(top_profit.head(3).sum()) if len(top_profit) else 0.0,
        "top5_profit_day_return_sum": float(top_profit.head(5).sum()) if len(top_profit) else 0.0,
        "top3_profit_day_positive_contribution_share": float(top_profit.head(3).sum() / total_pos) if total_pos else np.nan,
    }


def evaluate_combo(
    top10: pd.DataFrame,
    top10_base: pd.DataFrame,
    all_dates: list[str],
    strategy: dict[str, Any],
    regime: dict[str, Any],
    gate_dates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s_mask = strategy_mask(top10, strategy)
    date_mask = top10["trade_date"].astype(str).isin(gate_dates)
    selected = top10[s_mask & date_mask].copy()
    base = top10_base[top10_base["trade_id"].isin(set(selected["trade_id"].astype(int)))].copy()
    group_id = f"{strategy['strategy_id']}__{regime['regime_id']}"
    metrics, details, _daily_gate = a15.evaluate_trade_base(
        base,
        selected,
        group_id,
        "R0_market_regime_gate",
        PRIMARY_SCENARIOS,
        gate_dates,
    )

    daily_frames = []
    counts_by_day = selected.groupby("trade_date").size().reindex(all_dates).fillna(0).astype(int)
    for scenario_id in PRIMARY_SCENARIO_IDS:
        scenario_details = details[details["scenario_id"] == scenario_id]
        daily_series = scenario_details.groupby("trade_date")["portfolio_slot_return"].mean()
        daily_series = daily_series.reindex(all_dates).fillna(0.0)
        daily_frames.append(
            pd.DataFrame(
                {
                    "trade_date": all_dates,
                    "scenario_id": scenario_id,
                    "daily_return": daily_series.to_numpy(float),
                    "gate_active": [d in set(gate_dates) for d in all_dates],
                    "selected_trade_count": counts_by_day.to_numpy(int),
                    "group_id": group_id,
                    "strategy_id": strategy["strategy_id"],
                    "regime_id": regime["regime_id"],
                }
            )
        )
    daily_all = pd.concat(daily_frames, ignore_index=True)
    impact_daily_gate = daily_all[
        (daily_all["scenario_id"] == MAIN_SCENARIO_ID) & (daily_all["gate_active"])
    ][["trade_date", "daily_return"]]
    conc = pd.DataFrame([{**concentration(selected, impact_daily_gate), "group_id": group_id}])

    gate_count = len(gate_dates)
    counts_gate = selected.groupby("trade_date").size().reindex(gate_dates).fillna(0).astype(int)
    metadata = {
        "strategy_id": strategy["strategy_id"],
        "strategy_name": strategy["strategy_name"],
        "regime_id": regime["regime_id"],
        "regime_side": regime["regime_side"],
        "regime_field": regime["regime_field"],
        "regime_value": regime["regime_value"],
        "regime_description": regime["description"],
        "gate_day_count": gate_count,
        "no_trade_days": len(all_dates) - gate_count,
        "active_trade_days": int((counts_gate > 0).sum()),
        "zero_holding_gate_days": int((counts_gate == 0).sum()),
        "avg_daily_holdings": float(counts_gate.mean()) if gate_count else 0.0,
        "avg_daily_holdings_all_legacy_days": float(counts_by_day.mean()),
    }
    for key, value in metadata.items():
        metrics[key] = value
        details[key] = value
        conc[key] = value
    return metrics, daily_all, details, conc


def evaluate_base_strategies(
    top10: pd.DataFrame,
    top10_base: pd.DataFrame,
    all_dates: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    daily_by_strategy: dict[str, pd.DataFrame] = {}
    for spec in STRATEGY_SPECS:
        mask = strategy_mask(top10, spec)
        selected = top10[mask].copy()
        base = top10_base[top10_base["trade_id"].isin(set(selected["trade_id"].astype(int)))].copy()
        metrics, details, _daily = a15.evaluate_trade_base(base, selected, spec["strategy_id"], "R0_base_strategy", PRIMARY_SCENARIOS, all_dates)
        counts = selected.groupby("trade_date").size().reindex(all_dates).fillna(0).astype(int)
        daily_frames = []
        for scenario_id in PRIMARY_SCENARIO_IDS:
            s = details[details["scenario_id"] == scenario_id].groupby("trade_date")["portfolio_slot_return"].mean()
            s = s.reindex(all_dates).fillna(0.0)
            daily_frames.append(
                pd.DataFrame(
                    {
                        "trade_date": all_dates,
                        "scenario_id": scenario_id,
                        "daily_return": s.to_numpy(float),
                        "selected_trade_count": counts.to_numpy(int),
                    }
                )
            )
        metrics["strategy_id"] = spec["strategy_id"]
        metrics["strategy_name"] = spec["strategy_name"]
        metrics["avg_daily_holdings"] = float(counts.mean())
        rows.append(metrics)
        daily_by_strategy[spec["strategy_id"]] = pd.concat(daily_frames, ignore_index=True)
    return pd.concat(rows, ignore_index=True), daily_by_strategy


def max_drawdown_from_values(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    equity = np.cumprod(1 + np.nan_to_num(values, nan=0.0))
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1))


def matched_random_summary(
    regime_summary: pd.DataFrame,
    base_daily: dict[str, pd.DataFrame],
    all_dates: list[str],
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    scenario_lookup = {
        (row["group_id"], row["scenario_id"]): row
        for _, row in regime_summary.iterrows()
    }
    for group_id, group in regime_summary.groupby("group_id"):
        first = group.iloc[0]
        n = int(first["gate_day_count"])
        strategy_id = first["strategy_id"]
        if n <= 0:
            continue
        for scenario_id in PRIMARY_SCENARIO_IDS:
            actual = scenario_lookup[(group_id, scenario_id)]
            series = (
                base_daily[strategy_id][base_daily[strategy_id]["scenario_id"] == scenario_id]
                .set_index("trade_date")["daily_return"]
                .reindex(all_dates)
                .fillna(0.0)
            )
            arr = series.to_numpy(float)
            outcomes = np.empty(RANDOM_RUNS)
            drawdowns = np.empty(RANDOM_RUNS)
            for i in range(RANDOM_RUNS):
                idx = rng.choice(len(arr), size=n, replace=False)
                picked = arr[idx]
                outcomes[i] = np.prod(1 + picked) - 1
                drawdowns[i] = max_drawdown_from_values(picked)
            actual_return = float(actual["cumulative_return"])
            rows.append(
                {
                    "group_id": group_id,
                    "strategy_id": first["strategy_id"],
                    "regime_id": first["regime_id"],
                    "scenario_id": scenario_id,
                    "matched_day_count": n,
                    "random_runs": RANDOM_RUNS,
                    "actual_cumulative_return": actual_return,
                    "random_cumulative_return_mean": float(outcomes.mean()),
                    "random_cumulative_return_median": float(np.median(outcomes)),
                    "random_cumulative_return_p05": float(np.quantile(outcomes, 0.05)),
                    "random_cumulative_return_p95": float(np.quantile(outcomes, 0.95)),
                    "random_cumulative_return_min": float(outcomes.min()),
                    "random_cumulative_return_max": float(outcomes.max()),
                    "actual_percentile_vs_random": float((outcomes <= actual_return).mean()),
                    "random_max_drawdown_mean": float(drawdowns.mean()),
                    "random_max_drawdown_p05": float(np.quantile(drawdowns, 0.05)),
                    "random_max_drawdown_p95": float(np.quantile(drawdowns, 0.95)),
                }
            )
    return pd.DataFrame(rows)


def candidate_screening(
    regime_summary: pd.DataFrame,
    random_summary: pd.DataFrame,
    base_summary: pd.DataFrame,
) -> pd.DataFrame:
    main = regime_summary[regime_summary["scenario_id"] == MAIN_SCENARIO_ID].copy()
    rand = random_summary[random_summary["scenario_id"] == MAIN_SCENARIO_ID][
        ["group_id", "actual_percentile_vs_random", "random_cumulative_return_p95", "random_cumulative_return_mean"]
    ]
    base = base_summary[base_summary["scenario_id"] == MAIN_SCENARIO_ID][["strategy_id", "max_drawdown", "cumulative_return", "profit_factor"]]
    base = base.rename(
        columns={
            "max_drawdown": "base_strategy_max_drawdown",
            "cumulative_return": "base_strategy_cumulative_return",
            "profit_factor": "base_strategy_profit_factor",
        }
    )
    out = main.merge(rand, on="group_id", how="left").merge(base, on="strategy_id", how="left")
    out["condition_positive_10bp_impact"] = out["cumulative_return"] > 0
    out["condition_pf_gt_105"] = out["profit_factor"] > 1.05
    out["condition_trade_days_ge_20"] = out["gate_day_count"] >= 20
    out["condition_drop_top3_not_materially_negative"] = out["cum_return_drop_top3_profit_days"] >= -0.02
    out["condition_random_percentile_gt_90"] = out["actual_percentile_vs_random"] > 0.90
    out["condition_random_percentile_gt_95"] = out["actual_percentile_vs_random"] > 0.95
    out["drawdown_delta_vs_base"] = out["max_drawdown"] - out["base_strategy_max_drawdown"]
    out["condition_drawdown_not_materially_worse_than_base"] = out["drawdown_delta_vs_base"] >= -0.02
    out["condition_avg_daily_holdings_ge_2"] = out["avg_daily_holdings"] >= 2.0
    conditions = [
        "condition_positive_10bp_impact",
        "condition_pf_gt_105",
        "condition_trade_days_ge_20",
        "condition_drop_top3_not_materially_negative",
        "condition_random_percentile_gt_90",
        "condition_drawdown_not_materially_worse_than_base",
        "condition_avg_daily_holdings_ge_2",
    ]
    out["screen_pass_count"] = out[conditions].sum(axis=1).astype(int)
    out["candidate_status"] = np.where(
        out[conditions].all(axis=1),
        "forward_shadow_candidate",
        np.where(out["gate_day_count"] < 20, "observation_only_n_lt_20", "diagnostic_only"),
    )
    return out.sort_values(["candidate_status", "actual_percentile_vs_random", "cumulative_return"], ascending=[True, False, False])


def write_limitations() -> None:
    text = """# Batch 2B-R0 Limitations

Batch 2B-R0 is a market-regime gate diagnostic only. It does not retrain the
model, alter features, alter labels, alter the sell rule, alter TopN, or move
`v7_locked`.

Only three base strategies are tested: S0 original v7 Top10, S1 U2 filter-only
without refill, and S2 U3 filter-only without refill.

Only the pre-registered single market-state gates R1-R6 and C1-C4 are tested.
No multi-regime stacking, industry constraint, exit-rule optimization, dynamic
TopN, model retraining, or execution-aware label training is performed.

Matched random baselines use the same base strategy and the same number of
selected days as each gate, with 1000 random draws from the legacy validation
trading dates.

`condition_drop_top3_not_materially_negative` is operationalized as cumulative
return after dropping the largest 3 profit days being greater than or equal to
-2%. `condition_drawdown_not_materially_worse_than_base` is operationalized as
max drawdown no more than 2 percentage points worse than the corresponding base
strategy. These thresholds are diagnostic flags, not final parameters.

The legacy validation period must not be used as a final untouched test set.
No result in this folder is a live-trading recommendation.
"""
    (OUT_DIR / "batch2B_R0_limitations.md").write_text(text, encoding="utf-8")


def write_conclusion(
    regime_summary: pd.DataFrame,
    random_summary: pd.DataFrame,
    screening: pd.DataFrame,
) -> None:
    main = screening.copy()
    positive = main[main["cumulative_return"] > 0].sort_values("cumulative_return", ascending=False)
    stable = screening[
        (screening["cumulative_return"] > 0)
        & (screening["cum_return_drop_top3_profit_days"] >= -0.02)
    ].sort_values("cumulative_return", ascending=False)
    small = main[main["gate_day_count"] < 20].sort_values("cumulative_return", ascending=False)
    random_good = screening[screening["actual_percentile_vs_random"] > 0.90].sort_values("actual_percentile_vs_random", ascending=False)
    candidates = screening[screening["candidate_status"] == "forward_shadow_candidate"].copy()

    cols = [
        "strategy_id",
        "regime_id",
        "gate_day_count",
        "cumulative_return",
        "profit_factor",
        "max_drawdown",
        "cum_return_drop_top3_profit_days",
        "actual_percentile_vs_random",
        "avg_daily_holdings",
    ]
    text = f"""# Batch 2B-R0 Market Regime Gate Diagnostic Conclusion

Generated at Beijing time: {now_bj()}.

## 1. Batch 2B-R0 changed what?

It added a pre-registered market-regime gate diagnostic for S0 original v7
Top10, S1 U2 filter-only without refill, and S2 U3 filter-only without refill.
It also added matched random baselines for every S + R/C combination.

## 2. Batch 2B-R0 did not change what?

It did not change v7_locked, model weights, features, labels, TopN main rule,
sell rules, execution-aware labels, Batch 3, Batch 4 or Batch 5.

## 3. Was v7_locked modified?

No.

## 4. Was the model retrained?

No.

## 5. Were features, labels or sell rules modified?

No.

## 6. Which market-state combinations are positive after 10bp+impact?

{positive[cols].to_markdown(index=False) if len(positive) else 'No positive 10bp+impact combinations were found.'}

## 7. Which combinations remain stable after dropping the largest 3 profit days?

{stable[cols].to_markdown(index=False) if len(stable) else 'No combination remained non-materially-negative after dropping the largest 3 profit days.'}

## 8. Which combinations are small-sample observations?

{small[cols].to_markdown(index=False) if len(small) else 'No tested combination had fewer than 20 selected days.'}

## 9. Which combinations are significantly better than matched random baseline?

{random_good[cols].to_markdown(index=False) if len(random_good) else 'No combination exceeded the 90th percentile of its matched random baseline.'}

## 10. Should a forward shadow candidate list be built?

{"Yes, but only for paper tracking. The qualifying rows are listed in `batch2B_R0_candidate_screening.csv`." if len(candidates) else "No formal forward-shadow list passed all diagnostic gates. Partial positives should remain watchlist-only."}

Forward-shadow candidates:

{candidates[cols + ['candidate_status']].to_markdown(index=False) if len(candidates) else 'None.'}

## 11. Is L label retraining still not recommended?

Yes. This batch is diagnostic and does not provide enough evidence to start
execution-aware label retraining.

## 12. Should further historical optimization pause and move to forward paper tracking?

Yes. Historical diagnostics can continue only as narrow audits. Strategy
selection should move to forward paper tracking because the legacy validation
period is already heavily inspected.

## 13. Live-trading status

No direct live-trading recommendation is made.
"""
    (OUT_DIR / "batch2B_R0_conclusion.md").write_text(text, encoding="utf-8")


def write_hashes() -> None:
    rows = []
    for path in sorted(OUT_DIR.glob("batch2B_R0_*")):
        if not path.is_file() or path.name == "batch2B_R0_file_sha256.csv":
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
    pd.DataFrame(rows).to_csv(OUT_DIR / "batch2B_R0_file_sha256.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    top10, top10_base, _daily, _stock, all_dates, market_day = load_prepared()
    base_summary, base_daily = evaluate_base_strategies(top10, top10_base, all_dates)

    summary_frames = []
    daily_frames = []
    trade_frames = []
    concentration_frames = []
    for strategy in STRATEGY_SPECS:
        for regime in REGIME_SPECS:
            dates = regime_dates(market_day, regime)
            metrics, daily_all, details, conc = evaluate_combo(top10, top10_base, all_dates, strategy, regime, dates)
            summary_frames.append(metrics)
            daily_frames.append(daily_all)
            trade_frames.append(details)
            concentration_frames.append(conc)

    regime_summary = pd.concat(summary_frames, ignore_index=True)
    regime_daily = pd.concat(daily_frames, ignore_index=True)
    regime_trades = pd.concat(trade_frames, ignore_index=True)
    regime_concentration = pd.concat(concentration_frames, ignore_index=True)
    random_summary = matched_random_summary(regime_summary, base_daily, all_dates)
    screening = candidate_screening(regime_summary, random_summary, base_summary)

    regime_summary.to_csv(OUT_DIR / "batch2B_R0_regime_summary.csv", index=False)
    regime_daily.to_csv(OUT_DIR / "batch2B_R0_regime_daily_returns.csv", index=False)
    regime_trades.to_csv(OUT_DIR / "batch2B_R0_regime_trades.csv", index=False)
    random_summary.to_csv(OUT_DIR / "batch2B_R0_matched_random_summary.csv", index=False)
    regime_concentration.to_csv(OUT_DIR / "batch2B_R0_regime_concentration.csv", index=False)
    screening.to_csv(OUT_DIR / "batch2B_R0_candidate_screening.csv", index=False)
    base_summary.to_csv(OUT_DIR / "batch2B_R0_base_strategy_summary.csv", index=False)
    write_limitations()
    write_conclusion(regime_summary, random_summary, screening)
    write_hashes()
    print(
        json.dumps(
            {
                "output_dir": str(OUT_DIR),
                "regime_summary_rows": len(regime_summary),
                "matched_random_rows": len(random_summary),
                "forward_shadow_candidates": int((screening["candidate_status"] == "forward_shadow_candidate").sum()),
                "v7_locked_modified": False,
                "model_retrained": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
