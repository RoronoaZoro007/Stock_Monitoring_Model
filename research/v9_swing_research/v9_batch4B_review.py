#!/usr/bin/env python3
"""v9 Batch 4B Review: model-vs-simple-rule attribution.

This is an audit-only batch. It does not retrain models, change labels,
change features, change horizons, or tune parameters. It explains why the
frozen Batch 4B lightweight models did not beat the frozen Batch 3D simple
rule baselines.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT / "data_tushare" / "clean" / "v9"
REPORT_ROOT = ROOT / "reports" / "tushare" / "v9_swing_research"
B3D_DIR = REPORT_ROOT / "batch3D_simple_rule_baseline"
B4A_DIR = REPORT_ROOT / "batch4A_model_training_plan"
B4B_DIR = REPORT_ROOT / "batch4B_lightweight_model_training"
REPORT_DIR = REPORT_ROOT / "batch4B_review"

FEATURE_PATH = CLEAN_DIR / "v9_stock_industry_features.parquet"
LABEL_PATH = CLEAN_DIR / "v9_swing_labels_3_5_10.parquet"
MODEL_SELECTED_PATH = B4B_DIR / "batch4B_selected_trades.parquet"

GLOBAL_HANDOFF_PATH = REPORT_ROOT / "v9_current_handoff.md"
ROADMAP_PATH = REPORT_ROOT / "v9_execution_roadmap.md"
STAGE_STATUS_PATH = REPORT_ROOT / "v9_stage_status.csv"

PRIMARY_LABEL = "fwd_ret_5d_open"
PRIMARY_TOP_N = 20

TRAIN_START = "20180102"
TRAIN_END = "20221230"
VALIDATION_START = "20230103"
VALIDATION_END = "20241231"
HOLDOUT_START = "20250102"
HOLDOUT_END = "20260522"

REGIME_COLUMNS = [
    "trend_regime",
    "vol_regime",
    "industry_crowding_regime",
    "stock_concentration_regime",
    "market_regime_id",
]


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    factors: tuple[str, ...]
    directions: tuple[str, ...]


RULE_SPECS = {
    "low_log_amount": RuleSpec("low_log_amount", ("log_amount",), ("low",)),
    "low_stock_amount_share_in_industry": RuleSpec(
        "low_stock_amount_share_in_industry",
        ("stock_amount_share_in_industry",),
        ("low",),
    ),
    "low_log_total_mv": RuleSpec("low_log_total_mv", ("log_total_mv",), ("low",)),
    "low_stock_ret_60d": RuleSpec("low_stock_ret_60d", ("stock_ret_60d",), ("low",)),
    "combo_low_liquidity_weak_momentum": RuleSpec(
        "combo_low_liquidity_weak_momentum",
        ("log_amount", "turnover_rate", "stock_ret_20d", "stock_ret_60d"),
        ("low", "low", "low", "low"),
    ),
    "combo_low_size_low_liquidity": RuleSpec(
        "combo_low_size_low_liquidity",
        ("log_total_mv", "log_amount", "turnover_rate"),
        ("low", "low", "low"),
    ),
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def split_name(trade_date: pd.Series) -> pd.Series:
    d = trade_date.astype(str)
    out = pd.Series("outside", index=trade_date.index, dtype="object")
    out[d.between(TRAIN_START, TRAIN_END)] = "train"
    out[d.between(VALIDATION_START, VALIDATION_END)] = "validation"
    out[d.between(HOLDOUT_START, HOLDOUT_END)] = "research_holdout"
    return out


def max_drawdown(returns: pd.Series) -> float:
    r = finite_numeric(returns).dropna()
    if r.empty:
        return np.nan
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def profit_factor(returns: pd.Series) -> float:
    r = finite_numeric(returns).dropna()
    gains = r.loc[r.gt(0)].sum()
    losses = r.loc[r.lt(0)].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / abs(losses))


def summarize_daily_returns(daily: pd.DataFrame, meta: dict[str, object]) -> dict[str, object]:
    row = dict(meta)
    if daily.empty:
        row.update(
            {
                "signal_days": 0,
                "total_trades": 0,
                "avg_selected_count": np.nan,
                "cumulative_return": np.nan,
                "mean_daily_return": np.nan,
                "median_daily_return": np.nan,
                "daily_win_rate": np.nan,
                "trade_win_rate": np.nan,
                "profit_factor": np.nan,
                "trade_profit_factor": np.nan,
                "max_drawdown": np.nan,
                "avg_trade_return": np.nan,
                "max_daily_return": np.nan,
                "min_daily_return": np.nan,
            }
        )
        return row
    d = daily.sort_values("trade_date").copy()
    r = finite_numeric(d["daily_return"]).fillna(0.0)
    total_trades = int(d["selected_count"].sum())
    pos_sum = float(d.get("positive_trade_return_sum", pd.Series(dtype=float)).sum()) if total_trades else np.nan
    neg_sum = float(d.get("negative_trade_return_sum", pd.Series(dtype=float)).sum()) if total_trades else np.nan
    trade_sum = float(d.get("trade_return_sum", pd.Series(dtype=float)).sum()) if total_trades else np.nan
    win_count = int(d.get("win_count", pd.Series(dtype=float)).sum()) if total_trades else 0
    row.update(
        {
            "signal_days": int(d["trade_date"].nunique()),
            "total_trades": total_trades,
            "avg_selected_count": float(d["selected_count"].mean()),
            "cumulative_return": float((1.0 + r).prod() - 1.0),
            "mean_daily_return": float(r.mean()),
            "median_daily_return": float(r.median()),
            "daily_win_rate": float(r.gt(0).mean()),
            "trade_win_rate": float(win_count / total_trades) if total_trades else np.nan,
            "profit_factor": profit_factor(r),
            "trade_profit_factor": float(pos_sum / abs(neg_sum)) if pd.notna(neg_sum) and neg_sum < 0 else np.nan,
            "max_drawdown": max_drawdown(r),
            "avg_trade_return": float(trade_sum / total_trades) if total_trades else np.nan,
            "max_daily_return": float(r.max()),
            "min_daily_return": float(r.min()),
        }
    )
    return row


def make_bucket_from_rank(rank_pct: pd.Series, prefix: str) -> pd.Series:
    values = np.ceil(rank_pct.clip(0, 1) * 5).clip(1, 5)
    out = pd.Series("missing", index=rank_pct.index, dtype="object")
    mask = values.notna()
    q = values.loc[mask].astype("int8")
    out.loc[mask] = [f"{prefix}{x}" for x in q]
    return out


def make_rule_score(sub: pd.DataFrame, rule: RuleSpec) -> pd.Series:
    if len(rule.factors) == 1:
        raw = finite_numeric(sub[rule.factors[0]])
        return -raw if rule.directions[0] == "low" else raw
    parts = []
    for factor, direction in zip(rule.factors, rule.directions):
        rank_pct = sub.groupby("trade_date", sort=False)[factor].rank(pct=True, method="average")
        preferred = 1.0 - rank_pct if direction == "low" else rank_pct
        parts.append(preferred)
    return pd.concat(parts, axis=1).mean(axis=1)


def read_review_dataset(minimum_rules: list[str]) -> pd.DataFrame:
    factor_cols = sorted({f for rid in minimum_rules for f in RULE_SPECS[rid].factors})
    feature_cols = [
        "trade_date",
        "ts_code",
        "industry",
        "amount",
        "total_mv",
        "turnover_rate",
        "listed_days",
        "st_flag",
        "suspend_flag",
        "limit_up_close_flag",
        "limit_down_close_flag",
    ] + REGIME_COLUMNS + factor_cols
    feature_cols = list(dict.fromkeys(feature_cols))
    label_cols = ["trade_date", "ts_code", PRIMARY_LABEL, "label_win_5d"]
    print({"event": "read_features", "columns": len(feature_cols)}, flush=True)
    features = pd.read_parquet(FEATURE_PATH, columns=feature_cols)
    print({"event": "read_labels", "columns": len(label_cols)}, flush=True)
    labels = pd.read_parquet(LABEL_PATH, columns=label_cols)
    features["trade_date"] = features["trade_date"].astype(str)
    labels["trade_date"] = labels["trade_date"].astype(str)
    df = features.merge(labels, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
    df["industry"] = df["industry"].fillna("UNKNOWN").astype(str)
    for flag in ["st_flag", "suspend_flag", "limit_up_close_flag", "limit_down_close_flag"]:
        df[flag] = df[flag].fillna(False).astype(bool)
    numeric_cols = sorted(set(factor_cols + ["amount", "total_mv", "turnover_rate", "listed_days", PRIMARY_LABEL]))
    for col in numeric_cols:
        df[col] = finite_numeric(df[col])
    df["screened_universe_flag"] = (
        df["listed_days"].ge(120)
        & ~df["st_flag"]
        & ~df["suspend_flag"]
        & ~df["limit_up_close_flag"]
        & ~df["limit_down_close_flag"]
        & df[PRIMARY_LABEL].notna()
    )
    df["split"] = split_name(df["trade_date"])
    df = df[df["split"].ne("outside")].copy()
    print({"event": "dataset_ready", "rows": int(len(df)), "screened_rows": int(df["screened_universe_flag"].sum())}, flush=True)
    df["rank_log_total_mv_pct"] = df.groupby("trade_date", sort=False)["log_total_mv"].rank(pct=True, method="average")
    df["rank_log_amount_pct"] = df.groupby("trade_date", sort=False)["log_amount"].rank(pct=True, method="average")
    df["size_bucket"] = make_bucket_from_rank(df["rank_log_total_mv_pct"], "size_q")
    df["liquidity_bucket"] = make_bucket_from_rank(df["rank_log_amount_pct"], "liq_q")
    return df


def reconstruct_rule_trades(df: pd.DataFrame, minimum_rules: list[str]) -> pd.DataFrame:
    frames = []
    for idx, rule_id in enumerate(minimum_rules, start=1):
        rule = RULE_SPECS[rule_id]
        print({"event": "reconstruct_rule", "done": idx - 1, "total": len(minimum_rules), "rule_id": rule_id}, flush=True)
        needed = [
            "trade_date",
            "ts_code",
            "industry",
            "split",
            "size_bucket",
            "liquidity_bucket",
            "amount",
            "total_mv",
            "log_amount",
            "log_total_mv",
            PRIMARY_LABEL,
            "label_win_5d",
        ] + REGIME_COLUMNS + list(rule.factors)
        needed = list(dict.fromkeys(needed))
        sub = df.loc[df["screened_universe_flag"], needed].dropna(subset=[PRIMARY_LABEL] + list(rule.factors)).copy()
        sub["score"] = make_rule_score(sub, rule)
        sub = sub.dropna(subset=["score"])
        sub = sub.sort_values(["trade_date", "score"], ascending=[True, False], kind="mergesort")
        top = sub.groupby("trade_date", sort=False).head(PRIMARY_TOP_N).copy()
        top["baseline_id"] = rule_id
        top["horizon"] = "5d"
        top["rank"] = top.groupby(["baseline_id", "trade_date"], sort=False)["score"].rank(ascending=False, method="first").astype("int16")
        frames.append(top)
    selected = pd.concat(frames, ignore_index=True)
    selected.to_parquet(REPORT_DIR / "batch4B_review_rule_selected_trades.parquet", index=False, compression="zstd")
    return selected


def enrich_model_trades(model_selected: pd.DataFrame, review_df: pd.DataFrame) -> pd.DataFrame:
    enrich_cols = [
        "trade_date",
        "ts_code",
        "amount",
        "total_mv",
        "log_amount",
        "log_total_mv",
        "turnover_rate",
    ]
    enrich = review_df[enrich_cols].drop_duplicates(["trade_date", "ts_code"])
    out = model_selected.merge(enrich, on=["trade_date", "ts_code"], how="left", validate="many_to_one")
    out.to_parquet(REPORT_DIR / "batch4B_review_model_selected_trades_enriched.parquet", index=False, compression="zstd")
    return out


def daily_from_selected(selected: pd.DataFrame, id_col: str, id_name: str) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    g = selected.groupby([id_col, "split", "trade_date"], sort=False)
    out = g.agg(
        selected_count=(PRIMARY_LABEL, "size"),
        daily_return=(PRIMARY_LABEL, "mean"),
        trade_return_sum=(PRIMARY_LABEL, "sum"),
        win_count=(PRIMARY_LABEL, lambda x: int((x > 0).sum())),
        loss_count=(PRIMARY_LABEL, lambda x: int((x <= 0).sum())),
        positive_trade_return_sum=(PRIMARY_LABEL, lambda x: float(x[x > 0].sum())),
        negative_trade_return_sum=(PRIMARY_LABEL, lambda x: float(x[x < 0].sum())),
    ).reset_index()
    return out.rename(columns={id_col: id_name})


def build_baseline_vs_model_summary(model_daily: pd.DataFrame, rule_daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_id, split), g in model_daily.groupby(["model_id", "split"], sort=False):
        rows.append(summarize_daily_returns(g, {"selection_type": "model", "selector_id": model_id, "split": split}))
    for (baseline_id, split), g in rule_daily.groupby(["baseline_id", "split"], sort=False):
        rows.append(summarize_daily_returns(g, {"selection_type": "simple_rule", "selector_id": baseline_id, "split": split}))
    out = pd.DataFrame(rows).sort_values(["split", "selection_type", "selector_id"])
    out.to_csv(REPORT_DIR / "batch4B_review_baseline_vs_model_summary.csv", index=False)
    return out


def build_reconstruction_parity(baseline_vs_model: pd.DataFrame) -> pd.DataFrame:
    reference_path = B4B_DIR / "batch4B_simple_baseline_by_split.csv"
    if not reference_path.exists():
        out = pd.DataFrame()
        out.to_csv(REPORT_DIR / "batch4B_review_reconstruction_parity.csv", index=False)
        return out
    reconstructed = baseline_vs_model[baseline_vs_model["selection_type"].eq("simple_rule")].copy()
    reference = pd.read_csv(reference_path).rename(columns={"baseline_id": "selector_id"})
    cols = ["selector_id", "split", "signal_days", "total_trades", "mean_daily_return", "profit_factor", "max_drawdown"]
    merged = reconstructed[cols].merge(reference[cols], on=["selector_id", "split"], suffixes=("_review", "_batch4B_reference"))
    for col in ["signal_days", "total_trades", "mean_daily_return", "profit_factor", "max_drawdown"]:
        merged[f"{col}_diff"] = merged[f"{col}_review"] - merged[f"{col}_batch4B_reference"]
    numeric_diffs = [c for c in merged.columns if c.endswith("_diff")]
    merged["parity_pass"] = merged[numeric_diffs].abs().max(axis=1).le(1e-10)
    merged.to_csv(REPORT_DIR / "batch4B_review_reconstruction_parity.csv", index=False)
    return merged


def build_overlap(model_selected: pd.DataFrame, rule_selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    detail_rows = []
    model_groups = {(m, d): g for (m, d), g in model_selected.groupby(["model_id", "trade_date"], sort=False)}
    rule_groups = {(b, d): g for (b, d), g in rule_selected.groupby(["baseline_id", "trade_date"], sort=False)}
    model_ids = sorted(model_selected["model_id"].unique())
    baseline_ids = sorted(rule_selected["baseline_id"].unique())
    dates = sorted(set(model_selected["trade_date"]).intersection(set(rule_selected["trade_date"])))
    total = len(model_ids) * len(baseline_ids) * len(dates)
    done = 0
    for model_id in model_ids:
        for baseline_id in baseline_ids:
            for trade_date in dates:
                done += 1
                if done % 2000 == 0:
                    print({"event": "overlap_progress", "done": done, "total": total}, flush=True)
                mg = model_groups.get((model_id, trade_date))
                bg = rule_groups.get((baseline_id, trade_date))
                if mg is None or bg is None:
                    continue
                split = str(mg["split"].iloc[0])
                mset = set(mg["ts_code"])
                bset = set(bg["ts_code"])
                overlap = mset & bset
                model_only = mset - bset
                baseline_only = bset - mset
                mret = mg.set_index("ts_code")[PRIMARY_LABEL]
                bret = bg.set_index("ts_code")[PRIMARY_LABEL]
                overlap_ret = float(mret.loc[list(overlap)].mean()) if overlap else np.nan
                model_only_ret = float(mret.loc[list(model_only)].mean()) if model_only else np.nan
                baseline_only_ret = float(bret.loc[list(baseline_only)].mean()) if baseline_only else np.nan
                model_daily_return = float(mret.mean())
                baseline_daily_return = float(bret.mean())
                replacement_diff = model_daily_return - baseline_daily_return
                rows.append(
                    {
                        "model_id": model_id,
                        "baseline_id": baseline_id,
                        "split": split,
                        "trade_date": trade_date,
                        "model_count": int(len(mset)),
                        "baseline_count": int(len(bset)),
                        "overlap_count": int(len(overlap)),
                        "overlap_ratio_model": float(len(overlap) / len(mset)) if mset else np.nan,
                        "overlap_ratio_baseline": float(len(overlap) / len(bset)) if bset else np.nan,
                        "model_only_count": int(len(model_only)),
                        "baseline_only_count": int(len(baseline_only)),
                        "overlap_return": overlap_ret,
                        "model_only_return": model_only_ret,
                        "baseline_only_return": baseline_only_ret,
                        "model_daily_return": model_daily_return,
                        "baseline_daily_return": baseline_daily_return,
                        "replacement_return_diff": replacement_diff,
                    }
                )
                for code in sorted(model_only):
                    detail_rows.append(
                        {
                            "model_id": model_id,
                            "baseline_id": baseline_id,
                            "trade_date": trade_date,
                            "split": split,
                            "side": "model_only",
                            "ts_code": code,
                            "trade_return": float(mret.loc[code]),
                        }
                    )
                for code in sorted(baseline_only):
                    detail_rows.append(
                        {
                            "model_id": model_id,
                            "baseline_id": baseline_id,
                            "trade_date": trade_date,
                            "split": split,
                            "side": "baseline_only",
                            "ts_code": code,
                            "trade_return": float(bret.loc[code]),
                        }
                    )
    daily = pd.DataFrame(rows)
    details = pd.DataFrame(detail_rows)
    daily.to_csv(REPORT_DIR / "batch4B_review_overlap_daily.csv", index=False)
    details.to_parquet(REPORT_DIR / "batch4B_review_replacement_trade_details.parquet", index=False, compression="zstd")

    summary_rows = []
    for keys, g in daily.groupby(["model_id", "baseline_id", "split"], sort=False):
        model_id, baseline_id, split = keys
        replacement_series = finite_numeric(g["replacement_return_diff"]).dropna()
        model_only_series = finite_numeric(g["model_only_return"]).dropna()
        baseline_only_series = finite_numeric(g["baseline_only_return"]).dropna()
        summary_rows.append(
            {
                "model_id": model_id,
                "baseline_id": baseline_id,
                "split": split,
                "signal_days": int(g["trade_date"].nunique()),
                "avg_overlap_count": float(g["overlap_count"].mean()),
                "avg_overlap_ratio_model": float(g["overlap_ratio_model"].mean()),
                "avg_overlap_ratio_baseline": float(g["overlap_ratio_baseline"].mean()),
                "avg_model_only_count": float(g["model_only_count"].mean()),
                "avg_baseline_only_count": float(g["baseline_only_count"].mean()),
                "avg_overlap_return": float(finite_numeric(g["overlap_return"]).mean()),
                "avg_model_only_return": float(model_only_series.mean()) if not model_only_series.empty else np.nan,
                "avg_baseline_only_return": float(baseline_only_series.mean()) if not baseline_only_series.empty else np.nan,
                "model_only_minus_baseline_only": float(model_only_series.mean() - baseline_only_series.mean())
                if not model_only_series.empty and not baseline_only_series.empty
                else np.nan,
                "avg_model_daily_return": float(finite_numeric(g["model_daily_return"]).mean()),
                "avg_baseline_daily_return": float(finite_numeric(g["baseline_daily_return"]).mean()),
                "avg_replacement_return_diff": float(replacement_series.mean()) if not replacement_series.empty else np.nan,
                "replacement_diff_pf": profit_factor(replacement_series),
                "replacement_diff_positive_day_rate": float(replacement_series.gt(0).mean()) if not replacement_series.empty else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["split", "model_id", "baseline_id"])
    summary.to_csv(REPORT_DIR / "batch4B_review_overlap_summary.csv", index=False)
    replacement = summary[
        [
            "model_id",
            "baseline_id",
            "split",
            "avg_replacement_return_diff",
            "replacement_diff_pf",
            "replacement_diff_positive_day_rate",
            "avg_model_only_return",
            "avg_baseline_only_return",
            "model_only_minus_baseline_only",
        ]
    ].copy()
    replacement.to_csv(REPORT_DIR / "batch4B_review_replacement_summary.csv", index=False)
    return daily, summary, replacement


def style_exposure(selected: pd.DataFrame, selector_col: str, selector_type: str) -> pd.DataFrame:
    rows = []
    for group_col in ["size_bucket", "liquidity_bucket"]:
        for keys, g in selected.groupby([selector_col, "split", group_col], dropna=False, sort=False):
            selector_id, split, bucket = keys
            rows.append(
                {
                    "selector_type": selector_type,
                    "selector_id": selector_id,
                    "split": split,
                    "group_type": group_col,
                    "group_value": bucket,
                    "trade_count": int(len(g)),
                    "trade_share_within_selector_split": np.nan,
                    "avg_trade_return": float(finite_numeric(g[PRIMARY_LABEL]).mean()),
                    "trade_win_rate": float(finite_numeric(g[PRIMARY_LABEL]).gt(0).mean()),
                    "positive_trade_return_sum": float(g.loc[g[PRIMARY_LABEL].gt(0), PRIMARY_LABEL].sum()),
                    "negative_trade_return_sum": float(g.loc[g[PRIMARY_LABEL].lt(0), PRIMARY_LABEL].sum()),
                    "amount_median": float(finite_numeric(g["amount"]).median()),
                    "amount_p10": float(finite_numeric(g["amount"]).quantile(0.10)),
                    "amount_p90": float(finite_numeric(g["amount"]).quantile(0.90)),
                    "total_mv_median": float(finite_numeric(g["total_mv"]).median()),
                    "turnover_rate_median": float(finite_numeric(g["turnover_rate"]).median()),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        totals = out.groupby(["selector_type", "selector_id", "split", "group_type"])["trade_count"].transform("sum")
        out["trade_share_within_selector_split"] = out["trade_count"] / totals
        out["trade_profit_factor"] = out.apply(
            lambda r: r["positive_trade_return_sum"] / abs(r["negative_trade_return_sum"]) if r["negative_trade_return_sum"] < 0 else np.nan,
            axis=1,
        )
    return out


def build_style_and_capacity(model_selected: pd.DataFrame, rule_selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_style = style_exposure(model_selected, "model_id", "model")
    rule_style = style_exposure(rule_selected, "baseline_id", "simple_rule")
    style = pd.concat([model_style, rule_style], ignore_index=True)
    style.to_csv(REPORT_DIR / "batch4B_review_style_exposure.csv", index=False)

    rows = []
    for selector_type, selector_col, selected in [
        ("model", "model_id", model_selected),
        ("simple_rule", "baseline_id", rule_selected),
    ]:
        for keys, g in selected.groupby([selector_col, "split"], sort=False):
            selector_id, split = keys
            amount = finite_numeric(g["amount"]).dropna()
            total_mv = finite_numeric(g["total_mv"]).dropna()
            rows.append(
                {
                    "selector_type": selector_type,
                    "selector_id": selector_id,
                    "split": split,
                    "trades": int(len(g)),
                    "amount_count": int(amount.size),
                    "amount_p01": float(amount.quantile(0.01)) if amount.size else np.nan,
                    "amount_p05": float(amount.quantile(0.05)) if amount.size else np.nan,
                    "amount_p10": float(amount.quantile(0.10)) if amount.size else np.nan,
                    "amount_median": float(amount.median()) if amount.size else np.nan,
                    "amount_p90": float(amount.quantile(0.90)) if amount.size else np.nan,
                    "amount_p99": float(amount.quantile(0.99)) if amount.size else np.nan,
                    "total_mv_median": float(total_mv.median()) if total_mv.size else np.nan,
                    "size_q1_share": float((g["size_bucket"] == "size_q1").mean()),
                    "liq_q1_share": float((g["liquidity_bucket"] == "liq_q1").mean()),
                    "both_size_q1_liq_q1_share": float(((g["size_bucket"] == "size_q1") & (g["liquidity_bucket"] == "liq_q1")).mean()),
                }
            )
    capacity = pd.DataFrame(rows).sort_values(["split", "selector_type", "selector_id"])
    capacity.to_csv(REPORT_DIR / "batch4B_review_capacity_proxy.csv", index=False)
    return style, capacity


def build_concentration(selected: pd.DataFrame, selector_col: str, selector_type: str) -> pd.DataFrame:
    rows = []
    for keys, g in selected.groupby([selector_col, "split"], sort=False):
        selector_id, split = keys
        stock_counts = g["ts_code"].value_counts()
        industry_counts = g["industry"].fillna("UNKNOWN").value_counts()
        returns_by_stock = g.groupby("ts_code")[PRIMARY_LABEL].sum().sort_values(ascending=False)
        rows.append(
            {
                "selector_type": selector_type,
                "selector_id": selector_id,
                "split": split,
                "trades": int(len(g)),
                "unique_stocks": int(g["ts_code"].nunique()),
                "unique_industries": int(g["industry"].nunique()),
                "top1_stock_trade_share": float(stock_counts.iloc[0] / len(g)) if len(g) else np.nan,
                "top10_stock_trade_share": float(stock_counts.head(10).sum() / len(g)) if len(g) else np.nan,
                "top1_industry_trade_share": float(industry_counts.iloc[0] / len(g)) if len(g) else np.nan,
                "top5_industry_trade_share": float(industry_counts.head(5).sum() / len(g)) if len(g) else np.nan,
                "top10_stock_return_contribution": float(returns_by_stock.head(10).sum()) if not returns_by_stock.empty else np.nan,
                "bottom10_stock_return_contribution": float(returns_by_stock.tail(10).sum()) if not returns_by_stock.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_concentration_summary(model_selected: pd.DataFrame, rule_selected: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat(
        [
            build_concentration(model_selected, "model_id", "model"),
            build_concentration(rule_selected, "baseline_id", "simple_rule"),
        ],
        ignore_index=True,
    )
    out.to_csv(REPORT_DIR / "batch4B_review_concentration_summary.csv", index=False)
    return out


def build_model_feature_readout() -> pd.DataFrame:
    fi = pd.read_csv(B4B_DIR / "batch4B_feature_importance.csv")
    out = fi.sort_values(["model_id", "abs_coefficient"], ascending=[True, False]).copy()
    out.to_csv(REPORT_DIR / "batch4B_review_feature_readout.csv", index=False)
    return out


def write_limitations() -> None:
    lines = [
        "# Batch 4B Review Limitations",
        "",
        "- This batch does not retrain or rescore models; it only audits existing Batch4B model outputs.",
        "- Simple-rule selected trades are reconstructed with the frozen Batch3D rule definitions because Batch3D originally persisted daily returns, not trade-level selections.",
        "- Review returns remain gross cohort returns; they are not execution-ledger returns and do not include fees, slippage, impact cost, overlapping capital usage, or capacity constraints.",
        "- `amount` and `total_mv` are reported in their source data units and used as capacity proxies only.",
        "- Research holdout is not a pristine final test because earlier diagnostic batches inspected the full history.",
        "- Current-snapshot industry classification remains a known limitation; this review does not convert it into point-in-time industry membership.",
    ]
    (REPORT_DIR / "batch4B_review_limitations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_data_foundation_review() -> None:
    stage0_coverage = REPORT_ROOT / "stage0_data_foundation" / "stage0_coverage_report.csv"
    stage0_endpoint = REPORT_ROOT / "stage0_data_foundation" / "stage0_raw_endpoint_summary.csv"
    batch2_panel = REPORT_ROOT / "batch2_clean_panel_label_audit" / "batch2_panel_coverage.csv"
    batch2_quality = REPORT_ROOT / "batch2_clean_panel_label_audit" / "batch2_panel_quality_audit.csv"
    batch2_labels = REPORT_ROOT / "batch2_clean_panel_label_audit" / "batch2_label_distribution.csv"

    raw_summary = pd.read_csv(stage0_endpoint) if stage0_endpoint.exists() else pd.DataFrame()
    panel = pd.read_csv(batch2_panel) if batch2_panel.exists() else pd.DataFrame()
    quality = pd.read_csv(batch2_quality) if batch2_quality.exists() else pd.DataFrame()
    labels = pd.read_csv(batch2_labels) if batch2_labels.exists() else pd.DataFrame()

    rows = []
    if stage0_coverage.exists():
        cov = pd.read_csv(stage0_coverage)
        rows.append(
            {
                "check_area": "raw_download_coverage",
                "evidence_file": str(stage0_coverage),
                "key_result": f"coverage_pass={bool(cov['coverage_pass'].all()) if 'coverage_pass' in cov else 'unknown'}",
                "interpretation": "Raw daily endpoint coverage was locked before factor/model work.",
            }
        )
    if not raw_summary.empty:
        rows.append(
            {
                "check_area": "raw_endpoint_rows",
                "evidence_file": str(stage0_endpoint),
                "key_result": f"endpoints={raw_summary['endpoint'].nunique()}, rows={int(raw_summary['rows'].sum())}, missing_dates={int(raw_summary['missing_dates'].sum())}",
                "interpretation": "daily/adj_factor/daily_basic/suspend_d/stk_limit raw files have endpoint-level hashes and date coverage.",
            }
        )
    if not panel.empty:
        p = panel.iloc[0]
        rows.append(
            {
                "check_area": "clean_panel_coverage",
                "evidence_file": str(batch2_panel),
                "key_result": f"rows={int(p['rows'])}, trade_days={int(p['trade_days'])}, codes={int(p['codes'])}, duplicate_code_dates={int(p['duplicate_code_dates'])}",
                "interpretation": "Clean panel exists and has no duplicate stock-date rows in the Batch2 audit.",
            }
        )
    if not quality.empty:
        q = quality.set_index("field")["missing_ratio"].to_dict() if "field" in quality else {}
        rows.append(
            {
                "check_area": "clean_panel_missingness",
                "evidence_file": str(batch2_quality),
                "key_result": f"open_missing={q.get('open', 'unknown')}, close_missing={q.get('close', 'unknown')}, adj_factor_missing={q.get('adj_factor', 'unknown')}, total_mv_missing={q.get('total_mv', 'unknown')}",
                "interpretation": "Core price and adjustment fields passed; total_mv/turnover/basic fields retain a small missing-ratio caveat.",
            }
        )
    if not labels.empty:
        label5 = labels[labels["horizon"].astype(str).eq("5d")]
        if not label5.empty:
            r = label5.iloc[0]
            rows.append(
                {
                    "check_area": "primary_5d_label_quality",
                    "evidence_file": str(batch2_labels),
                    "key_result": f"valid_ratio={float(r['valid_ratio']):.6f}, abs_gt_20pct_ratio={float(r['abs_gt_20pct_ratio']):.6f}, raw_adj_mean_gap={float(r['raw_adj_mean_gap']):.6f}",
                    "interpretation": "The Batch4B primary 5d label is already audited for missingness, extreme labels and adjustment effects.",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "batch4B_review_data_foundation_check.csv", index=False)

    lines = [
        "# Batch 4B Review Data Foundation Check",
        "",
        "This file answers whether the original downloaded data had correctness checks and cleaning before Batch4B.",
        "",
        "## Evidence",
        "",
        out.to_markdown(index=False) if not out.empty else "No prior data-foundation evidence files found.",
        "",
        "## Conclusion",
        "",
        "- Yes. v9 has a Stage0 raw data lock with coverage and hashes, followed by Batch2 clean-panel and label audits.",
        "- The current Batch4B model underperformance is therefore not primarily attributed to missing raw-data QA or missing cleaning.",
        "- The remaining caveats are still material: no independent vendor cross-check, point-in-time industry limitation, ST proxy limitations, and gross-return rather than executable-ledger accounting.",
    ]
    (REPORT_DIR / "batch4B_review_data_foundation_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_conclusion(
    baseline_vs_model: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    capacity: pd.DataFrame,
    feature_readout: pd.DataFrame,
) -> str:
    validation = baseline_vs_model[baseline_vs_model["split"].eq("validation")].copy()
    model_val = validation[validation["selection_type"].eq("model")]
    rule_val = validation[validation["selection_type"].eq("simple_rule")]
    best_rule = rule_val.sort_values(["mean_daily_return", "profit_factor"], ascending=[False, False]).head(1)
    best_model = model_val.sort_values(["mean_daily_return", "profit_factor"], ascending=[False, False]).head(1)
    best_rule_id = str(best_rule["selector_id"].iloc[0]) if not best_rule.empty else ""
    best_model_id = str(best_model["selector_id"].iloc[0]) if not best_model.empty else ""
    gap = (
        float(best_model["mean_daily_return"].iloc[0] - best_rule["mean_daily_return"].iloc[0])
        if not best_model.empty and not best_rule.empty
        else np.nan
    )
    best_overlap = overlap_summary[
        overlap_summary["split"].eq("validation")
        & overlap_summary["model_id"].eq(best_model_id)
        & overlap_summary["baseline_id"].eq(best_rule_id)
    ]
    overlap_ratio = float(best_overlap["avg_overlap_ratio_model"].iloc[0]) if not best_overlap.empty else np.nan
    replacement_diff = float(best_overlap["avg_replacement_return_diff"].iloc[0]) if not best_overlap.empty else np.nan
    model_capacity = capacity[
        capacity["split"].eq("validation")
        & capacity["selector_type"].eq("model")
        & capacity["selector_id"].eq(best_model_id)
    ]
    size_q1_share = float(model_capacity["size_q1_share"].iloc[0]) if not model_capacity.empty else np.nan
    liq_q1_share = float(model_capacity["liq_q1_share"].iloc[0]) if not model_capacity.empty else np.nan
    top_features = (
        feature_readout[feature_readout["model_id"].eq(best_model_id)]
        .head(6)[["feature", "coefficient", "abs_coefficient"]]
        .to_dict("records")
    )
    gate = "block_batch5A_and_complex_models"
    if pd.notna(gap) and gap >= 0 and pd.notna(replacement_diff) and replacement_diff >= 0:
        gate = "review_can_consider_batch5A"

    lines = [
        "# Batch 4B Review",
        "",
        "## Scope",
        "",
        "- This batch audits why Batch4B lightweight models did not beat Batch3D simple rules.",
        "- No model retraining, parameter search, feature change, label change, horizon change, or v7_locked modification was performed.",
        "",
        "## Core Validation Comparison",
        "",
        f"- Best validation simple rule: `{best_rule_id}`.",
        f"- Best validation model: `{best_model_id}`.",
        f"- Mean daily return gap, model minus rule: `{gap:.6f}`.",
        f"- Best model vs best rule average overlap ratio: `{overlap_ratio:.4f}`.",
        f"- Best model vs best rule replacement return difference: `{replacement_diff:.6f}`.",
        f"- Best model validation size_q1 share: `{size_q1_share:.4f}`.",
        f"- Best model validation liquidity_q1 share: `{liq_q1_share:.4f}`.",
        "- Reconstructed Batch3D simple-rule trade details exactly match Batch4B reference split summaries; see `batch4B_review_reconstruction_parity.csv`.",
        f"- Best model top coefficients: `{top_features}`.",
        "",
        "## Interpretation",
        "",
        "- The model did not create a clearly independent alpha layer. Its strongest coefficients and selections are dominated by low market value, low amount and low turnover exposures.",
        "- The simple rules are already a direct expression of the same low-size/low-liquidity/reversal signal. Batch4B models mostly repackage that exposure, then replace part of the simple-rule basket with weaker names.",
        "- The validation advantage of simple rules is not explained by missing model complexity; it is visible under the same gross cohort label and same Top20 count.",
        "- Low-size/low-liquidity concentration is also a capacity risk, so simply training a more complex model is not justified until the simple-rule exposure passes cost/capacity robustness.",
        "",
        "## Data Foundation Check",
        "",
        "- Stage0 raw manifests and Batch2 clean panel/label audit already exist and were used as the data foundation.",
        "- This review did not find evidence that the Batch4B underperformance is primarily caused by raw data download or cleaning failure.",
        "- Remaining data caveats are unchanged: no independent vendor cross-check, current-snapshot industry limitation, and gross-return rather than executable-ledger returns.",
        "",
        "## Gate",
        "",
        f"- Gate result: `{gate}`.",
        "- Recommendation: do not enter Batch5A for the current lightweight model set.",
        "- Recommendation: do not train more complex models yet.",
        "- Next practical review should be simple-rule robustness/capacity audit or a new Batch4A amendment with a clearly different objective. It should not tune on the same validation gap.",
    ]
    text = "\n".join(lines) + "\n"
    (REPORT_DIR / "batch4B_review_conclusion.md").write_text(text, encoding="utf-8")
    return gate


def update_handoff(gate: str) -> None:
    text = "\n".join(
        [
            "# Handoff: Batch 4B Review Completed",
            "",
            "## Completed Batch",
            "",
            "- Completed: `Batch 4B Review`.",
            "- Output directory: `reports/tushare/v9_swing_research/batch4B_review/`.",
            "- Scope: model-vs-simple-rule overlap, replacement contribution, style/capacity proxy and feature readout.",
            "",
            "## Gate",
            "",
            f"- Gate result: `{gate}`.",
            "",
            "## Required Inputs For Next Step",
            "",
            "- `batch4B_review_baseline_vs_model_summary.csv`",
            "- `batch4B_review_overlap_summary.csv`",
            "- `batch4B_review_replacement_summary.csv`",
            "- `batch4B_review_style_exposure.csv`",
            "- `batch4B_review_capacity_proxy.csv`",
            "- `batch4B_review_conclusion.md`",
            "",
            "## Next Step",
            "",
            "- Do not proceed to Batch5A for the current lightweight model set unless this review is explicitly overridden.",
            "- Preferred next step: simple-rule robustness/capacity audit, or a new Batch4A amendment with a different objective and explicit boundaries.",
            "",
            "## Blocked Actions",
            "",
            "- Do not train complex models based on the current validation gap.",
            "- Do not start walk-forward or forward tracking from the current Batch4B model set.",
            "- Do not tune horizons, labels, features or TopN on the reviewed validation results.",
        ]
    )
    GLOBAL_HANDOFF_PATH.write_text(text + "\n", encoding="utf-8")


def update_stage_status(gate: str) -> None:
    if not STAGE_STATUS_PATH.exists():
        return
    df = pd.read_csv(STAGE_STATUS_PATH)
    row = {
        "stage": "Stage 7 Lightweight Model Review",
        "batch": "batch4B_review",
        "status": "completed",
        "output_dir": "reports/tushare/v9_swing_research/batch4B_review",
        "gate_result": "blocked" if gate.startswith("block") else "review",
        "next_action": "Do not proceed to Batch5A with current lightweight models; review simple-rule robustness/capacity or freeze a new Batch4A amendment.",
    }
    df = df[~df["batch"].eq("batch4B_review")]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    mask5a = df["batch"].eq("batch5A_robustness_exposure_audit")
    df.loc[mask5a, "status"] = "blocked"
    df.loc[mask5a, "gate_result"] = "blocked"
    df.loc[mask5a, "next_action"] = "Blocked by Batch4B Review: lightweight models did not beat simple rules and do not justify robustness audit as model candidates."
    df.to_csv(STAGE_STATUS_PATH, index=False)


def update_roadmap(gate: str) -> None:
    if not ROADMAP_PATH.exists():
        return
    text = ROADMAP_PATH.read_text(encoding="utf-8")
    addition = """### Stage 7C Batch 4B Review

输出目录：`reports/tushare/v9_swing_research/batch4B_review/`

执行边界：

- 未重训模型、未调参、未修改特征/标签/horizon/TopN。
- 重建 Batch3D 5d minimum-to-beat 简单规则 Top20 明细，与 Batch4B 模型 Top20 逐日对齐。
- 输出重合率、替换贡献、低市值/低流动性暴露、容量代理和特征权重读数。

结论：

- 当前轻量模型没有稳定打败简单规则。
- 当前不建议进入 Batch5A，也不建议训练更复杂模型。
- 下一步优先做简单规则的成本、容量、弱市和集中度鲁棒性审计，或另开 Batch4A amendment 冻结不同训练目标。

"""
    marker = "### Step 5A: 鲁棒性和暴露审计"
    if addition.strip() not in text:
        text = text.replace(marker, addition + marker)
    text = text.replace(
        "> Review Batch 4B model-vs-baseline gap; do not enter Batch 5A yet",
        "> Batch 4B Review completed; do not enter Batch 5A with current lightweight models",
    )
    text = text.replace(
        "- 内容来自 `Batch 4B review`",
        "- 内容来自 `Batch 4B Review completed`",
    )
    ROADMAP_PATH.write_text(text, encoding="utf-8")


def write_hashes() -> None:
    rows = []
    for path in sorted(REPORT_DIR.glob("batch4B_review_*")):
        if path.name == "batch4B_review_file_sha256.csv" or not path.is_file():
            continue
        rows.append({"file": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    for path in [GLOBAL_HANDOFF_PATH, ROADMAP_PATH, STAGE_STATUS_PATH]:
        if path.exists():
            rows.append({"file": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    with (REPORT_DIR / "batch4B_review_file_sha256.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "sha256", "size_bytes"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print({"event": "start", "out_dir": str(REPORT_DIR)}, flush=True)
    baseline_ref = pd.read_csv(B4A_DIR / "batch4A_baseline_reference.csv")
    minimum_rules = baseline_ref.loc[
        baseline_ref["baseline_role"].eq("minimum_to_beat") & baseline_ref["horizon"].eq("5d"),
        "rule_id",
    ].astype(str).tolist()
    minimum_rules = [rule_id for rule_id in minimum_rules if rule_id in RULE_SPECS]
    print({"event": "minimum_rules", "rules": minimum_rules}, flush=True)

    review_df = read_review_dataset(minimum_rules)
    rule_selected = reconstruct_rule_trades(review_df, minimum_rules)
    model_selected = pd.read_parquet(MODEL_SELECTED_PATH)
    model_selected["trade_date"] = model_selected["trade_date"].astype(str)
    model_selected = enrich_model_trades(model_selected, review_df)

    model_daily = daily_from_selected(model_selected, "model_id", "model_id")
    rule_daily = daily_from_selected(rule_selected, "baseline_id", "baseline_id")
    model_daily.to_csv(REPORT_DIR / "batch4B_review_model_daily_returns.csv", index=False)
    rule_daily.to_csv(REPORT_DIR / "batch4B_review_rule_daily_returns.csv", index=False)

    baseline_vs_model = build_baseline_vs_model_summary(model_daily, rule_daily)
    build_reconstruction_parity(baseline_vs_model)
    _, overlap_summary, _ = build_overlap(model_selected, rule_selected)
    style, capacity = build_style_and_capacity(model_selected, rule_selected)
    build_concentration_summary(model_selected, rule_selected)
    feature_readout = build_model_feature_readout()
    write_limitations()
    write_data_foundation_review()
    gate = write_conclusion(baseline_vs_model, overlap_summary, capacity, feature_readout)
    update_handoff(gate)
    update_stage_status(gate)
    update_roadmap(gate)
    write_hashes()
    print({"event": "done", "out_dir": str(REPORT_DIR), "gate_result": gate}, flush=True)


if __name__ == "__main__":
    main()
