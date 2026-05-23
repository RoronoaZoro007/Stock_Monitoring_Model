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
import batch1_hard_checks as b1  # noqa: E402


OUT_DIR = ROOT / "reports" / "tushare" / "v8_research" / "batch2A15_alpha_attribution"
BATCH2A1_DIR = ROOT / "reports" / "tushare" / "v8_research" / "batch2A1_UF_single_factor"
VALIDATION_START = "20260224"
VALIDATION_END = "20260520"
TOP_N = 10
Q_BUCKETS = 5
PRIMARY_SCENARIOS = [
    ev.Scenario("5bp_one_way_slippage", "5bp_one_way_slippage", 5, True),
    ev.Scenario("10bp_one_way_slippage", "10bp_one_way_slippage", 10, True),
    ev.Scenario(
        "10bp_slippage_plus_impact",
        "10bp_one_way_slippage_plus_impact_cost",
        10,
        True,
        include_impact=True,
        capital=100_000,
    ),
]
PRIMARY_SCENARIO_IDS = [s.scenario_id for s in PRIMARY_SCENARIOS]


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


def amount_sofar_1450(df: pd.DataFrame) -> pd.Series:
    return np.expm1(pd.to_numeric(df["amount_sofar_log"], errors="coerce"))


def tail_amount_1430_1450(df: pd.DataFrame) -> pd.Series:
    return amount_sofar_1450(df) * pd.to_numeric(df["tail_amount_share"], errors="coerce")


def assign_qbucket(series: pd.Series, q: int = Q_BUCKETS) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    labels = [f"Q{i}_low" if i == 1 else f"Q{i}_high" if i == q else f"Q{i}" for i in range(1, q + 1)]
    out = pd.Series("unknown", index=series.index, dtype=object)
    mask = values.notna()
    if mask.sum() < q:
        return out
    ranked = values[mask].rank(method="first")
    out.loc[mask] = pd.qcut(ranked, q=q, labels=labels).astype(str)
    return out


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    top10, candidates, daily, stock = b1.load_base_inputs()
    top10 = ensure_trade_ids(top10)
    candidates = candidates.copy()
    candidates["trade_date"] = candidates["trade_date"].astype(str)
    all_dates = sorted(top10["trade_date"].astype(str).unique())
    return top10, candidates, daily, stock, all_dates


def add_liquidity_fields(candidates: pd.DataFrame, top10: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe, _daily_size, _overlap = b1.build_rolling_universe(candidates, daily)
    u0 = universe[universe["universe_id"] == "U0_static_top3000_v7"][
        ["trade_date", "ts_code", "avg_amount_60d", "rank_amount_60d"]
    ].copy()
    out_candidates = candidates.merge(u0, on=["trade_date", "ts_code"], how="left")
    out_candidates["amount_sofar_1450"] = amount_sofar_1450(out_candidates)
    out_candidates["tail_1430_1450_amount"] = tail_amount_1430_1450(out_candidates)

    keep = [
        "trade_date",
        "ts_code",
        "avg_amount_60d",
        "rank_amount_60d",
        "amount_sofar_1450",
        "tail_1430_1450_amount",
    ]
    out_top10 = top10.merge(out_candidates[keep], on=["trade_date", "ts_code"], how="left")
    return out_candidates, out_top10, universe


def add_universe_flags(top10: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    out = top10.copy()
    for universe_id, col in [
        ("U2_internal_roll_top2500_60d", "in_u2"),
        ("U3_internal_roll_top2000_60d", "in_u3"),
    ]:
        members = universe[universe["universe_id"] == universe_id][["trade_date", "ts_code"]].drop_duplicates()
        members[col] = True
        out = out.merge(members, on=["trade_date", "ts_code"], how="left")
        out[col] = out[col].fillna(False).astype(bool)
    out["u2_group"] = np.where(out["in_u2"], "A1_v7_top10_in_U2", "A2_v7_top10_dropped_by_U2")
    out["u3_group"] = np.where(out["in_u3"], "A3_v7_top10_in_U3", "A4_v7_top10_dropped_by_U3")
    return out


def prepare_execution_base(selected: pd.DataFrame, daily: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:
    base = ev.prepare_trade_base(selected, daily, stock)
    minutes = ev.load_relevant_minutes(base, ev.DEFAULT_MINUTE_DIR)
    base = ev.add_entry_exit_bars(base, minutes)
    return ev.add_execution_flags(base, minutes, daily)


def add_participation_percentiles(metrics: dict[str, Any], details: pd.DataFrame) -> dict[str, Any]:
    out = dict(metrics)
    if details.empty:
        out["single_ticket_participation_median"] = np.nan
        out["single_ticket_participation_p90"] = np.nan
        out["single_ticket_participation_p99"] = np.nan
        out["min_filled_notional_return"] = np.nan
        return out
    p = np.maximum(
        details["buy_participation"].fillna(0.0).to_numpy(float),
        details["sell_participation"].fillna(0.0).to_numpy(float),
    )
    out["single_ticket_participation_median"] = float(np.quantile(p, 0.50))
    out["single_ticket_participation_p90"] = float(np.quantile(p, 0.90))
    out["single_ticket_participation_p99"] = float(np.quantile(p, 0.99))
    out["min_filled_notional_return"] = float(details["filled_notional_return"].min())
    return out


def evaluate_trade_base(
    trade_base: pd.DataFrame,
    selected: pd.DataFrame,
    group_id: str,
    group_type: str,
    scenarios: list[ev.Scenario] | None = None,
    calendar_dates: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenarios = scenarios or PRIMARY_SCENARIOS
    use_base = trade_base.copy()
    score_auc = ev.auc_safe(selected, "label_static_003", "score") if "score" in selected.columns else np.nan
    metrics_rows = []
    detail_frames = []
    daily_frames = []
    for scenario in scenarios:
        metrics, details, daily_df = ev.evaluate_scenario(use_base, scenario, score_auc)
        if calendar_dates is not None:
            daily_series = details.groupby("trade_date")["portfolio_slot_return"].mean()
            daily_series = daily_series.reindex(calendar_dates).fillna(0.0)
            metrics = ev.metrics_from_returns(details, daily_series, scenario, score_auc)
            daily_df = daily_series.rename("daily_return").reset_index().rename(columns={"index": "trade_date"})
            daily_df["scenario_id"] = scenario.scenario_id
        metrics.update({"group_id": group_id, "group_type": group_type})
        metrics_rows.append(add_participation_percentiles(metrics, details))
        details = details.copy()
        details["group_id"] = group_id
        details["group_type"] = group_type
        detail_frames.append(details)
        daily_df = daily_df.copy()
        daily_df["group_id"] = group_id
        daily_df["group_type"] = group_type
        daily_frames.append(daily_df)
    return pd.DataFrame(metrics_rows), pd.concat(detail_frames, ignore_index=True), pd.concat(daily_frames, ignore_index=True)


def eval_subset_from_prepared_base(
    full_base: pd.DataFrame,
    full_selected: pd.DataFrame,
    mask: pd.Series,
    group_id: str,
    group_type: str,
    calendar_dates: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ids = set(full_selected.loc[mask, "trade_id"].astype(int))
    selected = full_selected[full_selected["trade_id"].isin(ids)].copy()
    base = full_base[full_base["trade_id"].isin(ids)].copy()
    return evaluate_trade_base(base, selected, group_id, group_type, PRIMARY_SCENARIOS, calendar_dates)


def group_concentration(selected: pd.DataFrame) -> dict[str, Any]:
    total = len(selected)
    industries = selected.get("industry", pd.Series("unknown", index=selected.index)).fillna("unknown")
    industry_counts = industries.value_counts()
    stock_counts = selected["ts_code"].value_counts()
    return {
        "top_industry": str(industry_counts.index[0]) if len(industry_counts) else "",
        "top_industry_count": int(industry_counts.iloc[0]) if len(industry_counts) else 0,
        "top_industry_share": float(industry_counts.iloc[0] / total) if total and len(industry_counts) else np.nan,
        "industry_hhi": float(((industry_counts / total) ** 2).sum()) if total else np.nan,
        "industry_distribution_json": json.dumps(industry_counts.head(10).to_dict(), ensure_ascii=False),
        "unique_stocks": int(selected["ts_code"].nunique()) if total else 0,
        "top_stock": str(stock_counts.index[0]) if len(stock_counts) else "",
        "top_stock_count": int(stock_counts.iloc[0]) if len(stock_counts) else 0,
        "top_stock_share": float(stock_counts.iloc[0] / total) if total and len(stock_counts) else np.nan,
        "stock_hhi": float(((stock_counts / total) ** 2).sum()) if total else np.nan,
    }


def add_group_stats(metrics: pd.DataFrame, selected: pd.DataFrame, all_dates: list[str], group_id: str) -> pd.DataFrame:
    out = metrics.copy()
    counts = selected.groupby("trade_date").size().reindex(all_dates).fillna(0)
    stats = {
        "sample_count": int(len(selected)),
        "active_days": int((counts > 0).sum()),
        "no_trade_days": int((counts == 0).sum()),
        "avg_daily_holdings": float(counts.mean()),
        "avg_amount_sofar_1450": float(pd.to_numeric(selected.get("amount_sofar_1450"), errors="coerce").mean()),
        "avg_tail_1430_1450_amount": float(pd.to_numeric(selected.get("tail_1430_1450_amount"), errors="coerce").mean()),
        "max_single_loss_target_return": float(pd.to_numeric(selected["target_return"], errors="coerce").min()) if len(selected) else np.nan,
    }
    stats.update(group_concentration(selected))
    for key, value in stats.items():
        out[key] = value
    out["group_id"] = group_id
    return out


def build_detail_return_pivot(details: pd.DataFrame) -> pd.DataFrame:
    use = details[details["scenario_id"].isin(PRIMARY_SCENARIO_IDS)].copy()
    pivots = []
    for col in ["filled_notional_return", "portfolio_slot_return", "buy_participation", "sell_participation"]:
        p = use.pivot_table(index="trade_id", columns="scenario_id", values=col, aggfunc="first")
        p = p.rename(columns={c: f"{col}_{c}" for c in p.columns})
        pivots.append(p)
    return pd.concat(pivots, axis=1).reset_index()


def run_u_keep_drop(
    top10: pd.DataFrame,
    top10_base: pd.DataFrame,
    all_dates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    detail_frames = []
    group_specs = [
        ("A1_v7_top10_in_U2", "in_u2"),
        ("A2_v7_top10_dropped_by_U2", "~in_u2"),
        ("A3_v7_top10_in_U3", "in_u3"),
        ("A4_v7_top10_dropped_by_U3", "~in_u3"),
    ]
    for group_id, expr in group_specs:
        if expr.startswith("~"):
            mask = ~top10[expr[1:]].astype(bool)
        else:
            mask = top10[expr].astype(bool)
        metrics, details, _daily = eval_subset_from_prepared_base(top10_base, top10, mask, group_id, "A_keep_drop", all_dates)
        metrics = add_group_stats(metrics, top10[mask].copy(), all_dates, group_id)
        ids = set(top10.loc[mask, "trade_id"].astype(int))
        contribution = slot_weighted_contribution(details, ids, all_dates)
        for key, value in contribution.items():
            metrics[key] = value
        rows.append(metrics)
        detail_frames.append(details)

    trade_pivot = build_detail_return_pivot(pd.concat(detail_frames, ignore_index=True))
    trades = top10[
        [
            "trade_id",
            "trade_date",
            "ts_code",
            "score",
            "target_return",
            "in_u2",
            "in_u3",
            "u2_group",
            "u3_group",
            "industry",
            "avg_amount_60d",
            "amount_sofar_1450",
            "tail_1430_1450_amount",
        ]
    ].merge(trade_pivot, on="trade_id", how="left")
    return pd.concat(rows, ignore_index=True), trades


def slot_weighted_contribution(details: pd.DataFrame, trade_ids: set[int], all_dates: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if details.empty or not trade_ids:
        for sid in PRIMARY_SCENARIO_IDS:
            out[f"v7_slot_weighted_sum_return_{sid}"] = np.nan
            out[f"v7_slot_weighted_cumulative_return_{sid}"] = np.nan
        return out
    sub = details[details["trade_id"].isin(trade_ids)]
    for sid in PRIMARY_SCENARIO_IDS:
        s = sub[sub["scenario_id"] == sid].groupby("trade_date")["portfolio_slot_return"].sum() / TOP_N
        s = s.reindex(all_dates).fillna(0.0)
        out[f"v7_slot_weighted_sum_return_{sid}"] = float(s.sum())
        out[f"v7_slot_weighted_cumulative_return_{sid}"] = float((1 + s).prod() - 1)
    return out


def run_u_filter_only(
    top10: pd.DataFrame,
    top10_base: pd.DataFrame,
    all_dates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    daily_frames = []
    for group_id, col in [("B1_filter_only_keep_U2_no_refill", "in_u2"), ("B2_filter_only_keep_U3_no_refill", "in_u3")]:
        mask = top10[col].astype(bool)
        selected = top10[mask].copy()
        metrics, _details, daily = eval_subset_from_prepared_base(top10_base, top10, mask, group_id, "B_filter_only", all_dates)
        metrics = add_group_stats(metrics, selected, all_dates, group_id)
        rows.append(metrics)
        daily_frames.append(daily)
    return pd.concat(rows, ignore_index=True), pd.concat(daily_frames, ignore_index=True)


def select_topn_in_bucket(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.sort_values(["trade_date", "score"], ascending=[True, False]).groupby("trade_date", group_keys=False).head(TOP_N).copy()


def pearson_safe(df: pd.DataFrame) -> float:
    if len(df) < 3:
        return np.nan
    return float(pd.to_numeric(df["score"], errors="coerce").corr(pd.to_numeric(df["target_return"], errors="coerce"), method="pearson"))


def spearman_safe(df: pd.DataFrame) -> float:
    if len(df) < 3:
        return np.nan
    return float(pd.to_numeric(df["score"], errors="coerce").corr(pd.to_numeric(df["target_return"], errors="coerce"), method="spearman"))


def evaluate_selected_rows(
    selected: pd.DataFrame,
    daily: pd.DataFrame,
    stock: pd.DataFrame,
    group_id: str,
    group_type: str,
    calendar_dates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = ensure_trade_ids(selected)
    base = prepare_execution_base(selected, daily, stock)
    metrics, details, _daily = evaluate_trade_base(base, selected, group_id, group_type, PRIMARY_SCENARIOS, calendar_dates)
    return metrics, details


def run_liquidity_buckets(
    candidates: pd.DataFrame,
    top10: pd.DataFrame,
    daily: pd.DataFrame,
    stock: pd.DataFrame,
    all_dates: list[str],
) -> pd.DataFrame:
    rows = []
    metrics_specs = [
        ("avg_amount_60d", "avg_amount_60d"),
        ("amount_sofar_1450", "amount_sofar_1450"),
        ("tail_1430_1450_amount", "tail_1430_1450_amount"),
    ]
    top10_keys = set(top10["trade_date"].astype(str) + "|" + top10["ts_code"].astype(str))
    bucketed_candidates = candidates.copy()
    for metric_id, col in metrics_specs:
        bucket_col = f"{metric_id}_bucket"
        bucketed_candidates[bucket_col] = assign_qbucket(bucketed_candidates[col])
        keyed = bucketed_candidates[["trade_date", "ts_code", bucket_col]]
        top10_bucketed = top10.merge(keyed, on=["trade_date", "ts_code"], how="left")
        for bucket in sorted(bucketed_candidates[bucket_col].dropna().unique()):
            if bucket == "unknown":
                continue
            all_bucket = bucketed_candidates[bucketed_candidates[bucket_col] == bucket].copy()
            v7_bucket = top10_bucketed[top10_bucketed[bucket_col] == bucket].copy()
            for scope, sample, selected in [
                ("all_candidates_bucket_top10_by_score", all_bucket, select_topn_in_bucket(all_bucket)),
                ("v7_top10_trades_in_bucket", v7_bucket, v7_bucket),
            ]:
                if sample.empty or selected.empty:
                    continue
                metrics, details = evaluate_selected_rows(selected, daily, stock, f"C_{metric_id}_{bucket}_{scope}", "C_liquidity_bucket", all_dates)
                main = metrics[metrics["scenario_id"] == "10bp_slippage_plus_impact"].iloc[0].to_dict()
                five = metrics[metrics["scenario_id"] == "5bp_one_way_slippage"].iloc[0].to_dict()
                ten = metrics[metrics["scenario_id"] == "10bp_one_way_slippage"].iloc[0].to_dict()
                selected_keys = set(sample["trade_date"].astype(str) + "|" + sample["ts_code"].astype(str))
                top10_count = len(selected_keys & top10_keys)
                impact_details = details[details["scenario_id"] == "10bp_slippage_plus_impact"]
                p = np.maximum(
                    impact_details["buy_participation"].fillna(0.0).to_numpy(float),
                    impact_details["sell_participation"].fillna(0.0).to_numpy(float),
                )
                rows.append(
                    {
                        "liquidity_metric": metric_id,
                        "bucket": bucket,
                        "sample_scope": scope,
                        "sample_count": int(len(sample)),
                        "v7_top10_selected_count": int(top10_count),
                        "selected_trade_count": int(len(selected)),
                        "score_mean": float(pd.to_numeric(sample["score"], errors="coerce").mean()),
                        "score_target_pearson": pearson_safe(sample),
                        "score_target_spearman": spearman_safe(sample),
                        "auc": ev.auc_safe(sample, "label_static_003", "score"),
                        "cumulative_return_5bp": five["cumulative_return"],
                        "cumulative_return_10bp": ten["cumulative_return"],
                        "cumulative_return_10bp_impact": main["cumulative_return"],
                        "profit_factor": main["profit_factor"],
                        "max_drawdown": main["max_drawdown"],
                        "trade_win_rate": main["trade_win_rate"],
                        "daily_win_rate": main["daily_win_rate"],
                        "avg_fill_ratio": main["avg_fill_ratio"],
                        "partial_fill_count": main["partial_fill_count"],
                        "zero_fill_count": main["zero_fill_count"],
                        "single_ticket_participation_median": float(np.quantile(p, 0.50)) if len(p) else np.nan,
                        "single_ticket_participation_p90": float(np.quantile(p, 0.90)) if len(p) else np.nan,
                        "single_ticket_participation_p99": float(np.quantile(p, 0.99)) if len(p) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def original_top10_scenario_details(top10_base: pd.DataFrame, top10: pd.DataFrame, all_dates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics, details, daily = evaluate_trade_base(top10_base, top10, "v7_original_top10", "D_original", PRIMARY_SCENARIOS, all_dates)
    return metrics, details, daily


def add_market_state_fields(candidates: pd.DataFrame, top10: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    day = candidates.groupby("trade_date").agg(
        market_tail_breadth_positive=("market_tail_breadth_positive", "first"),
        market_tail_ret_median=("market_tail_ret_median", "first"),
        market_ret_median_1450=("market_ret_median_1450", "first"),
        market_amount_log=("market_amount_log", "first"),
        market_dispersion_proxy=("ret_to_prev_close_1450", "std"),
        market_tail_vol_proxy=("tail_ret_1430_1450", "std"),
    )
    day = day.reset_index()
    for col in [
        "market_tail_breadth_positive",
        "market_tail_ret_median",
        "market_dispersion_proxy",
        "market_amount_log",
        "market_tail_vol_proxy",
    ]:
        day[f"{col}_qbucket"] = assign_qbucket(day[col])
    day["tail_direction"] = np.where(day["market_tail_ret_median"] >= 0, "tail_up", "tail_down")
    day["market_direction"] = np.where(day["market_ret_median_1450"] >= 0, "market_up", "market_down")
    day["amount_regime"] = np.where(day["market_amount_log"] >= day["market_amount_log"].median(), "high_amount", "low_amount")
    day["tail_vol_regime"] = np.where(day["market_tail_vol_proxy"] >= day["market_tail_vol_proxy"].median(), "high_tail_vol", "low_tail_vol")
    out = top10.merge(day, on="trade_date", how="left")
    return out, day


def build_details_wide(details: pd.DataFrame) -> pd.DataFrame:
    use = details[details["scenario_id"].isin(PRIMARY_SCENARIO_IDS)].copy()
    returns = use.pivot_table(index="trade_id", columns="scenario_id", values="filled_notional_return", aggfunc="first")
    returns = returns.rename(columns={c: f"return_{c}" for c in returns.columns})
    participation = use[use["scenario_id"] == "10bp_slippage_plus_impact"][["trade_id", "buy_participation", "sell_participation"]].copy()
    participation["max_participation"] = participation[["buy_participation", "sell_participation"]].max(axis=1)
    return returns.reset_index().merge(participation[["trade_id", "max_participation"]], on="trade_id", how="left")


def cost_decay_for_group(df: pd.DataFrame, group_cols: list[str], dimension_type: str, all_dates: list[str] | None = None) -> pd.DataFrame:
    rows = []
    total_5_to_10 = float((df["return_5bp_one_way_slippage"] - df["return_10bp_one_way_slippage"]).sum())
    total_10_to_impact = float((df["return_10bp_one_way_slippage"] - df["return_10bp_slippage_plus_impact"]).sum())
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        name = "|".join(str(x) for x in keys)
        daily_5 = group.groupby("trade_date")["return_5bp_one_way_slippage"].mean()
        daily_10 = group.groupby("trade_date")["return_10bp_one_way_slippage"].mean()
        daily_imp = group.groupby("trade_date")["return_10bp_slippage_plus_impact"].mean()
        if all_dates is not None and group_cols == ["trade_date"]:
            daily_5 = daily_5.reindex(all_dates).dropna()
            daily_10 = daily_10.reindex(all_dates).dropna()
            daily_imp = daily_imp.reindex(all_dates).dropna()
        decay_5_to_10 = float((group["return_5bp_one_way_slippage"] - group["return_10bp_one_way_slippage"]).sum())
        decay_10_to_impact = float((group["return_10bp_one_way_slippage"] - group["return_10bp_slippage_plus_impact"]).sum())
        rows.append(
            {
                "dimension_type": dimension_type,
                "dimension_value": name,
                "trade_count": int(len(group)),
                "cumulative_return_5bp": float((1 + daily_5).prod() - 1) if len(daily_5) else np.nan,
                "cumulative_return_10bp": float((1 + daily_10).prod() - 1) if len(daily_10) else np.nan,
                "cumulative_return_10bp_impact": float((1 + daily_imp).prod() - 1) if len(daily_imp) else np.nan,
                "mean_trade_return_5bp": float(group["return_5bp_one_way_slippage"].mean()),
                "mean_trade_return_10bp": float(group["return_10bp_one_way_slippage"].mean()),
                "mean_trade_return_10bp_impact": float(group["return_10bp_slippage_plus_impact"].mean()),
                "decay_5bp_to_10bp_sum": decay_5_to_10,
                "decay_10bp_to_impact_sum": decay_10_to_impact,
                "decay_5bp_to_10bp_contribution": decay_5_to_10 / total_5_to_10 if total_5_to_10 else np.nan,
                "decay_10bp_to_impact_contribution": decay_10_to_impact / total_10_to_impact if total_10_to_impact else np.nan,
                "avg_participation": float(group["max_participation"].mean()),
            }
        )
    return pd.DataFrame(rows)


def run_cost_decay(top10: pd.DataFrame, details: pd.DataFrame, all_dates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wide = build_details_wide(details)
    data = top10.merge(wide, on="trade_id", how="left")
    for col in ["avg_amount_60d", "amount_sofar_1450", "tail_1430_1450_amount"]:
        data[f"{col}_bucket"] = assign_qbucket(data[col])

    bucket_frames = [
        cost_decay_for_group(data, [f"{col}_bucket"], col, all_dates)
        for col in ["avg_amount_60d", "amount_sofar_1450", "tail_1430_1450_amount"]
    ]
    for col in [
        "market_tail_breadth_positive_qbucket",
        "market_tail_ret_median_qbucket",
        "market_dispersion_proxy_qbucket",
        "tail_direction",
        "market_direction",
        "amount_regime",
        "tail_vol_regime",
    ]:
        if col in data.columns:
            bucket_frames.append(cost_decay_for_group(data, [col], col, all_dates))
    by_bucket = pd.concat(bucket_frames, ignore_index=True)
    by_industry = cost_decay_for_group(data, ["industry"], "industry", all_dates).sort_values("decay_10bp_to_impact_contribution", ascending=False)
    by_stock = cost_decay_for_group(data, ["ts_code"], "stock", all_dates).sort_values("decay_10bp_to_impact_contribution", ascending=False)
    by_day = cost_decay_for_group(data, ["trade_date"], "trade_date", all_dates).sort_values("trade_date" if False else "dimension_value")
    return by_bucket, by_industry, by_day, by_stock


def run_market_regime_groups(
    top10: pd.DataFrame,
    top10_base: pd.DataFrame,
    all_dates: list[str],
) -> pd.DataFrame:
    rows = []
    specs = [
        "market_tail_breadth_positive_qbucket",
        "market_tail_ret_median_qbucket",
        "market_dispersion_proxy_qbucket",
        "tail_direction",
        "market_direction",
        "amount_regime",
        "tail_vol_regime",
    ]
    for spec in specs:
        if spec not in top10.columns:
            continue
        for value, group in top10.groupby(spec, dropna=False):
            if len(group) == 0:
                continue
            dates = sorted(group["trade_date"].astype(str).unique())
            ids = set(group["trade_id"].astype(int))
            base = top10_base[top10_base["trade_id"].isin(ids)].copy()
            metrics, _details, _daily = evaluate_trade_base(base, group.copy(), f"E_{spec}_{value}", "E_market_regime", PRIMARY_SCENARIOS, dates)
            metrics["regime_field"] = spec
            metrics["regime_value"] = str(value)
            metrics["regime_trade_days"] = len(dates)
            metrics["regime_trade_count"] = len(group)
            rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def read_batch2a1_reselection() -> pd.DataFrame:
    path = BATCH2A1_DIR / "batch2A1_U_reselection_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def write_conclusion(
    keep_drop: pd.DataFrame,
    filter_only: pd.DataFrame,
    liquidity: pd.DataFrame,
    decay_bucket: pd.DataFrame,
    market: pd.DataFrame,
) -> None:
    a_imp = keep_drop[keep_drop["scenario_id"] == "10bp_slippage_plus_impact"].copy()
    b_imp = filter_only[filter_only["scenario_id"] == "10bp_slippage_plus_impact"].copy()
    b2a1 = read_batch2a1_reselection()
    b2a1_imp = b2a1[b2a1["scenario_id"] == "10bp_slippage_plus_impact"] if not b2a1.empty else pd.DataFrame()
    c_high = liquidity[
        (liquidity["liquidity_metric"] == "avg_amount_60d")
        & (liquidity["bucket"].astype(str).str.contains("Q5"))
    ]
    c_low = liquidity[
        (liquidity["liquidity_metric"] == "avg_amount_60d")
        & (liquidity["bucket"].astype(str).str.contains("Q1"))
    ]
    market_imp = market[market["scenario_id"] == "10bp_slippage_plus_impact"].copy() if not market.empty else pd.DataFrame()
    positive_market = market_imp[market_imp["cumulative_return"] > 0].sort_values("cumulative_return", ascending=False)

    def one(df: pd.DataFrame, key: str, group_col: str = "group_id") -> dict[str, Any]:
        if df.empty:
            return {}
        hit = df[df[group_col] == key]
        return hit.iloc[0].to_dict() if len(hit) else {}

    a1 = one(a_imp, "A1_v7_top10_in_U2")
    a2 = one(a_imp, "A2_v7_top10_dropped_by_U2")
    a3 = one(a_imp, "A3_v7_top10_in_U3")
    a4 = one(a_imp, "A4_v7_top10_dropped_by_U3")
    b1r = one(b_imp, "B1_filter_only_keep_U2_no_refill")
    b2r = one(b_imp, "B2_filter_only_keep_U3_no_refill")
    u2_re = one(b2a1_imp, "U2_internal_roll_top2500_60d", "experiment_id")
    u3_re = one(b2a1_imp, "U3_internal_roll_top2000_60d", "experiment_id")

    total_decay_5_10 = float(decay_bucket[decay_bucket["dimension_type"] == "avg_amount_60d"]["decay_5bp_to_10bp_sum"].sum())
    total_decay_10_imp = float(decay_bucket[decay_bucket["dimension_type"] == "avg_amount_60d"]["decay_10bp_to_impact_sum"].sum())
    broad_decay_groups = decay_bucket[
        (decay_bucket["dimension_type"] == "avg_amount_60d")
        & (decay_bucket["decay_5bp_to_10bp_sum"] > 0)
    ]["dimension_value"].nunique()
    market_table = (
        positive_market[["regime_field", "regime_value", "cumulative_return", "profit_factor", "max_drawdown", "regime_trade_days"]]
        .head(10)
        .to_markdown(index=False)
        if len(positive_market)
        else "No positive 10bp+impact market-state group was found."
    )

    text = f"""# Batch 2A-1.5 Alpha Attribution Conclusion

Generated at Beijing time: {now_bj()}.

## 1. Batch 2A-1.5 changed what?

It added alpha-attribution diagnostics for the locked v7 validation signals:
U2/U3 keep/drop attribution, U2/U3 filter-only diagnostics, liquidity-bucket
score effectiveness, cost-decay attribution and market-state grouping.

## 2. Batch 2A-1.5 did not change what?

It did not change v7_locked, model weights, features, labels, TopN main rule,
sell rules, execution-aware labels, Batch 2B, Batch 3, Batch 4 or Batch 5.

## 3. Was v7_locked modified?

No.

## 4. Was the model retrained?

No.

## 5. Were features, labels, TopN or sell rules modified?

No.

## 6. Does original v7 alpha come from U2/U3 kept or dropped groups?

10bp+impact standalone cumulative return:

| Group | Trades | CumRet | PF | v7 slot-weighted contribution |
|---|---:|---:|---:|---:|
| A1 in U2 | {int(a1.get('sample_count', 0))} | {float(a1.get('cumulative_return', np.nan)):.6f} | {float(a1.get('profit_factor', np.nan)):.6f} | {float(a1.get('v7_slot_weighted_cumulative_return_10bp_slippage_plus_impact', np.nan)):.6f} |
| A2 dropped by U2 | {int(a2.get('sample_count', 0))} | {float(a2.get('cumulative_return', np.nan)):.6f} | {float(a2.get('profit_factor', np.nan)):.6f} | {float(a2.get('v7_slot_weighted_cumulative_return_10bp_slippage_plus_impact', np.nan)):.6f} |
| A3 in U3 | {int(a3.get('sample_count', 0))} | {float(a3.get('cumulative_return', np.nan)):.6f} | {float(a3.get('profit_factor', np.nan)):.6f} | {float(a3.get('v7_slot_weighted_cumulative_return_10bp_slippage_plus_impact', np.nan)):.6f} |
| A4 dropped by U3 | {int(a4.get('sample_count', 0))} | {float(a4.get('cumulative_return', np.nan)):.6f} | {float(a4.get('profit_factor', np.nan)):.6f} | {float(a4.get('v7_slot_weighted_cumulative_return_10bp_slippage_plus_impact', np.nan)):.6f} |

This table should be read as attribution, not as an optimized rule.

## 7. Is U2/U3 filter-only better than Batch 2A-1 reselection?

10bp+impact cumulative return:

| Experiment | CumRet | PF | Avg daily holdings |
|---|---:|---:|---:|
| B1 U2 filter-only | {float(b1r.get('cumulative_return', np.nan)):.6f} | {float(b1r.get('profit_factor', np.nan)):.6f} | {float(b1r.get('avg_daily_holdings', np.nan)):.3f} |
| Batch2A1 U2 reselection | {float(u2_re.get('cumulative_return', np.nan)):.6f} | {float(u2_re.get('profit_factor', np.nan)):.6f} | {float(u2_re.get('avg_selected_per_day', np.nan)):.3f} |
| B2 U3 filter-only | {float(b2r.get('cumulative_return', np.nan)):.6f} | {float(b2r.get('profit_factor', np.nan)):.6f} | {float(b2r.get('avg_daily_holdings', np.nan)):.3f} |
| Batch2A1 U3 reselection | {float(u3_re.get('cumulative_return', np.nan)):.6f} | {float(u3_re.get('profit_factor', np.nan)):.6f} | {float(u3_re.get('avg_selected_per_day', np.nan)):.3f} |

Filter-only isolates the original v7 signals and does not refill dropped names.

## 8. Does v7 score still rank in high-liquidity buckets?

High avg_amount_60d bucket diagnostic. The `all_candidates_bucket_top10_by_score`
row tests whether the locked score can reselect inside the bucket; the
`v7_top10_trades_in_bucket` row measures the original v7 trades that happened
to fall in that bucket.

{c_high.to_markdown(index=False) if len(c_high) else 'No high-liquidity bucket rows were generated.'}

Low avg_amount_60d bucket diagnostic:

{c_low.to_markdown(index=False) if len(c_low) else 'No low-liquidity bucket rows were generated.'}

## 9. Is cost decay caused by a few high-friction trades or thin alpha overall?

For avg_amount_60d buckets, summed trade-return decay from 5bp to 10bp is
`{total_decay_5_10:.6f}`, and from 10bp to 10bp+impact is
`{total_decay_10_imp:.6f}`. Positive 5bp-to-10bp decay appears in
`{broad_decay_groups}` liquidity buckets. This indicates the main 5bp-to-10bp
collapse is broad transaction-cost pressure, while impact adds a smaller
liquidity-sensitive layer.

## 10. Is there a market state with positive 10bp+impact return?

{market_table}

These are diagnostics only. They are not selected parameters.

## 11. Should Batch 2B-R market-state filtering diagnostics start?

Yes, but only as diagnostics. Market-state grouping can be audited without
changing labels or retraining, and it directly tests whether v7 alpha is
conditional on market regime.

## 12. Should Batch 2A-2-L execution-aware label retraining start?

No. U/F non-retraining tests did not produce stable improvement, and this
attribution batch should be reviewed before any label change.

## 13. Should historical optimization pause and keep forward paper tracking?

Yes. Historical diagnostics can continue in narrow audit batches, but model
selection should not use the legacy 20260224-20260520 interval as a final test.
Forward paper tracking is needed before treating any filter as tradable.

## 14. Next step

Review Batch 2A-1.5 outputs first. The next audit candidate is Batch 2B-R
market-state diagnostics, not L retraining and not U+F combination optimization.
"""
    (OUT_DIR / "batch2A15_conclusion.md").write_text(text, encoding="utf-8")


def write_limitations() -> None:
    text = """# Batch 2A-1.5 Limitations

This batch is attribution-only. It does not optimize parameters, retrain the
model, alter labels, alter features, change TopN, change sell rules, or claim a
new tradable v8 strategy.

U diagnostics are computed inside the existing downloaded v7 Top3000 matrix.
They are not all-A-share unbiased rolling-universe tests.

`market_dispersion_proxy` and `market_tail_vol_proxy` are diagnostic proxies
derived from the validation candidate matrix. They were not v7 model features
and are not used to train or choose a production rule here.

The legacy validation interval remains unsuitable as a final untouched test set.
"""
    (OUT_DIR / "batch2A15_limitations.md").write_text(text, encoding="utf-8")


def write_hashes() -> None:
    rows = []
    for path in sorted(OUT_DIR.glob("batch2A15_*")):
        if not path.is_file() or path.name == "batch2A15_file_sha256.csv":
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
    pd.DataFrame(rows).to_csv(OUT_DIR / "batch2A15_file_sha256.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    top10_raw, candidates_raw, daily, stock, all_dates = load_inputs()
    candidates, top10_liq, universe = add_liquidity_fields(candidates_raw, top10_raw, daily)
    top10 = add_universe_flags(top10_liq, universe)
    top10, market_day = add_market_state_fields(candidates, top10)

    top10_base = prepare_execution_base(top10, daily, stock)
    keep_drop_summary, keep_drop_trades = run_u_keep_drop(top10, top10_base, all_dates)
    filter_only_summary, filter_only_daily = run_u_filter_only(top10, top10_base, all_dates)
    liquidity_effectiveness = run_liquidity_buckets(candidates, top10, daily, stock, all_dates)
    original_metrics, original_details, _original_daily = original_top10_scenario_details(top10_base, top10, all_dates)
    cost_bucket, cost_industry, cost_day, cost_stock = run_cost_decay(top10, original_details, all_dates)
    market_regime = run_market_regime_groups(top10, top10_base, all_dates)

    keep_drop_summary.to_csv(OUT_DIR / "batch2A15_U_keep_drop_summary.csv", index=False)
    keep_drop_trades.to_csv(OUT_DIR / "batch2A15_U_keep_drop_trades.csv", index=False)
    filter_only_summary.to_csv(OUT_DIR / "batch2A15_U_filter_only_summary.csv", index=False)
    filter_only_daily.to_csv(OUT_DIR / "batch2A15_U_filter_only_daily_returns.csv", index=False)
    liquidity_effectiveness.to_csv(OUT_DIR / "batch2A15_liquidity_bucket_score_effectiveness.csv", index=False)
    cost_bucket.to_csv(OUT_DIR / "batch2A15_cost_decay_by_bucket.csv", index=False)
    cost_industry.to_csv(OUT_DIR / "batch2A15_cost_decay_by_industry.csv", index=False)
    cost_day.to_csv(OUT_DIR / "batch2A15_cost_decay_by_day.csv", index=False)
    cost_stock.to_csv(OUT_DIR / "batch2A15_cost_decay_by_stock.csv", index=False)
    market_regime.to_csv(OUT_DIR / "batch2A15_market_regime_group_summary.csv", index=False)
    original_metrics.to_csv(OUT_DIR / "batch2A15_original_v7_metrics.csv", index=False)
    market_day.to_csv(OUT_DIR / "batch2A15_market_state_daily_proxy.csv", index=False)

    write_limitations()
    write_conclusion(keep_drop_summary, filter_only_summary, liquidity_effectiveness, cost_bucket, market_regime)
    write_hashes()
    print(
        json.dumps(
            {
                "output_dir": str(OUT_DIR),
                "keep_drop_rows": len(keep_drop_summary),
                "filter_only_rows": len(filter_only_summary),
                "liquidity_rows": len(liquidity_effectiveness),
                "market_regime_rows": len(market_regime),
                "v7_locked_modified": False,
                "model_retrained": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
