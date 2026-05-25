#!/usr/bin/env python3
"""v9 Batch 3D: simple-rule baselines.

This batch converts Batch 3C diagnostic IC candidates into simple, deterministic
ranking rules and compares them with matched random baselines. It does not train
models, tune parameters, or enable concept/theme factors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT / "data_tushare" / "clean" / "v9"
REPORT_DIR = ROOT / "reports" / "tushare" / "v9_swing_research" / "batch3D_simple_rule_baseline"
CHECKPOINT_DIR = REPORT_DIR / "_checkpoints"
GLOBAL_HANDOFF_PATH = ROOT / "reports" / "tushare" / "v9_swing_research" / "v9_current_handoff.md"
ROADMAP_PATH = ROOT / "reports" / "tushare" / "v9_swing_research" / "v9_execution_roadmap.md"
STAGE_STATUS_PATH = ROOT / "reports" / "tushare" / "v9_swing_research" / "v9_stage_status.csv"

LABEL_PATH = CLEAN_DIR / "v9_swing_labels_3_5_10.parquet"
FEATURE_PATH = CLEAN_DIR / "v9_stock_industry_features.parquet"
REGIME_PATH = CLEAN_DIR / "v9_market_regime_formal.parquet"
BATCH3C_CANDIDATE_PATH = ROOT / "reports" / "tushare" / "v9_swing_research" / "batch3C_factor_ic_decile" / "batch3C_candidate_factor_screen.csv"

HORIZONS = [3, 5, 10]
PRIMARY_TOP_N = 20
RANDOM_SIMS = 500
RANDOM_SEED = 20260525


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    rule_group: str
    factors: tuple[str, ...]
    directions: tuple[str, ...]
    description: str
    uses_industry_snapshot: bool = False
    primary_rule: bool = True


RULE_SPECS = [
    RuleSpec("low_log_amount", "liquidity_inverse", ("log_amount",), ("low",), "Select the lowest daily amount names."),
    RuleSpec("low_turnover_rate", "liquidity_inverse", ("turnover_rate",), ("low",), "Select the lowest turnover names."),
    RuleSpec(
        "low_stock_amount_share_in_industry",
        "liquidity_crowding_inverse",
        ("stock_amount_share_in_industry",),
        ("low",),
        "Select stocks with low amount share inside their current industry.",
    ),
    RuleSpec("low_log_total_mv", "size_inverse", ("log_total_mv",), ("low",), "Select smaller total market value names."),
    RuleSpec("low_stock_ret_5d", "short_reversal", ("stock_ret_5d",), ("low",), "Classic short-term reversal proxy: weak 5d return."),
    RuleSpec("low_stock_ret_20d", "medium_reversal", ("stock_ret_20d",), ("low",), "Inverse 20d momentum based on Batch 3C negative IC."),
    RuleSpec("low_stock_ret_60d", "medium_reversal", ("stock_ret_60d",), ("low",), "Inverse 60d momentum based on Batch 3C negative IC."),
    RuleSpec("high_stock_ret_20d", "momentum_control", ("stock_ret_20d",), ("high",), "Naive 20d momentum control."),
    RuleSpec("high_stock_ret_60d", "momentum_control", ("stock_ret_60d",), ("high",), "Naive 60d momentum control."),
    RuleSpec(
        "low_stock_rel_industry_ret_20d",
        "relative_strength_inverse",
        ("stock_rel_industry_ret_20d",),
        ("low",),
        "Select names weak versus their current industry over 20d.",
        uses_industry_snapshot=True,
    ),
    RuleSpec(
        "low_stock_rel_industry_ret_60d",
        "relative_strength_inverse",
        ("stock_rel_industry_ret_60d",),
        ("low",),
        "Select names weak versus their current industry over 60d.",
        uses_industry_snapshot=True,
    ),
    RuleSpec(
        "low_industry_crowding_score",
        "industry_crowding_inverse",
        ("industry_crowding_score",),
        ("low",),
        "Select names in low-crowding current-snapshot industries.",
        uses_industry_snapshot=True,
    ),
    RuleSpec(
        "high_industry_ret_20d",
        "industry_momentum_control",
        ("industry_ret_20d",),
        ("high",),
        "Naive current-snapshot industry momentum control.",
        uses_industry_snapshot=True,
    ),
    RuleSpec(
        "low_industry_ret_60d",
        "industry_reversal",
        ("industry_ret_60d",),
        ("low",),
        "Current-snapshot industry reversal diagnostic.",
        uses_industry_snapshot=True,
    ),
    RuleSpec(
        "combo_low_liquidity",
        "combo",
        ("log_amount", "turnover_rate", "stock_amount_share_in_industry"),
        ("low", "low", "low"),
        "Average preferred-rank of low amount, low turnover and low stock industry amount share.",
    ),
    RuleSpec(
        "combo_weak_momentum",
        "combo",
        ("stock_ret_20d", "stock_ret_60d", "stock_rel_industry_ret_20d", "stock_rel_industry_ret_60d"),
        ("low", "low", "low", "low"),
        "Average preferred-rank of weak medium-term absolute and relative momentum.",
        uses_industry_snapshot=True,
    ),
    RuleSpec(
        "combo_low_liquidity_weak_momentum",
        "combo",
        ("log_amount", "turnover_rate", "stock_ret_20d", "stock_ret_60d"),
        ("low", "low", "low", "low"),
        "Average preferred-rank of low liquidity and weak medium-term momentum.",
    ),
    RuleSpec(
        "combo_low_size_low_liquidity",
        "combo",
        ("log_total_mv", "log_amount", "turnover_rate"),
        ("low", "low", "low"),
        "Average preferred-rank of smaller size and lower trading activity.",
    ),
]


REGIME_COLUMNS = ["trend_regime", "vol_regime", "industry_crowding_regime", "stock_concentration_regime", "extreme_selloff_flag"]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))


def checkpoint_path(*parts: str) -> Path:
    return CHECKPOINT_DIR / ("__".join(safe_name(part) for part in parts) + ".csv")


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def read_checkpoint(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def finite_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


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


def summarize_daily(daily: pd.DataFrame, meta: dict) -> dict:
    if daily.empty:
        row = dict(meta)
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
    ordered = daily.sort_values("trade_date")
    returns = finite_numeric(ordered["daily_return"]).fillna(0.0)
    total_trades = int(ordered["selected_count"].sum())
    trade_sum = float(ordered["trade_return_sum"].sum()) if total_trades else np.nan
    win_count = int(ordered["win_count"].sum()) if "win_count" in ordered else 0
    positive_trade_sum = float(ordered["positive_trade_return_sum"].sum()) if "positive_trade_return_sum" in ordered else np.nan
    negative_trade_sum = float(ordered["negative_trade_return_sum"].sum()) if "negative_trade_return_sum" in ordered else np.nan
    row = dict(meta)
    row.update(
        {
            "signal_days": int(ordered["trade_date"].nunique()),
            "total_trades": total_trades,
            "avg_selected_count": float(ordered["selected_count"].mean()),
            "cumulative_return": float((1.0 + returns).prod() - 1.0),
            "mean_daily_return": float(returns.mean()),
            "median_daily_return": float(returns.median()),
            "daily_win_rate": float(returns.gt(0).mean()),
            "trade_win_rate": float(win_count / total_trades) if total_trades else np.nan,
            "profit_factor": profit_factor(returns),
            "trade_profit_factor": float(positive_trade_sum / abs(negative_trade_sum)) if negative_trade_sum and negative_trade_sum < 0 else np.nan,
            "max_drawdown": max_drawdown(returns),
            "avg_trade_return": float(trade_sum / total_trades) if total_trades else np.nan,
            "max_daily_return": float(returns.max()),
            "min_daily_return": float(returns.min()),
        }
    )
    return row


def read_dataset() -> pd.DataFrame:
    factor_cols = sorted({factor for rule in RULE_SPECS for factor in rule.factors})
    feature_cols = [
        "trade_date",
        "ts_code",
        "industry",
        "listed_days",
        "st_flag",
        "suspend_flag",
        "limit_up_close_flag",
        "limit_down_close_flag",
    ] + factor_cols + REGIME_COLUMNS
    label_cols = ["trade_date", "ts_code"] + [f"fwd_ret_{h}d_open" for h in HORIZONS]
    features = pd.read_parquet(FEATURE_PATH, columns=feature_cols)
    labels = pd.read_parquet(LABEL_PATH, columns=label_cols)
    features["trade_date"] = features["trade_date"].astype(str)
    labels["trade_date"] = labels["trade_date"].astype(str)
    df = features.merge(labels, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
    df["industry"] = df["industry"].fillna("UNKNOWN").astype(str)
    for col in factor_cols + [f"fwd_ret_{h}d_open" for h in HORIZONS] + ["listed_days"]:
        df[col] = finite_numeric(df[col])
    for flag in ["st_flag", "suspend_flag", "limit_up_close_flag", "limit_down_close_flag", "extreme_selloff_flag"]:
        df[flag] = df[flag].fillna(False).astype(bool)
    df["screened_universe_flag"] = (
        df["listed_days"].ge(120)
        & ~df["st_flag"]
        & ~df["suspend_flag"]
        & ~df["limit_up_close_flag"]
        & ~df["limit_down_close_flag"]
    )
    return df


def make_rule_score(sub: pd.DataFrame, rule: RuleSpec) -> pd.Series:
    if len(rule.factors) == 1:
        factor = rule.factors[0]
        direction = rule.directions[0]
        raw = finite_numeric(sub[factor])
        return -raw if direction == "low" else raw
    parts = []
    for factor, direction in zip(rule.factors, rule.directions):
        rank_pct = sub.groupby("trade_date", sort=False)[factor].rank(pct=True, method="average")
        preferred = 1.0 - rank_pct if direction == "low" else rank_pct
        parts.append(preferred)
    return pd.concat(parts, axis=1).mean(axis=1)


def evaluate_rule(df: pd.DataFrame, rule: RuleSpec, horizon: int, top_n: int, refresh: bool) -> pd.DataFrame:
    ckpt = checkpoint_path("rule_daily", rule.rule_id, f"{horizon}d", f"top{top_n}")
    if ckpt.exists() and not refresh:
        cached = read_checkpoint(ckpt)
        print({"event": "checkpoint_skip", "step": "rule_daily", "rule_id": rule.rule_id, "horizon": f"{horizon}d", "rows": len(cached)}, flush=True)
        return cached

    started = time.perf_counter()
    target = f"fwd_ret_{horizon}d_open"
    needed = ["trade_date", "ts_code", "industry", target] + list(rule.factors)
    sub = df.loc[df["screened_universe_flag"] & df[target].notna(), needed].dropna(subset=[target] + list(rule.factors)).copy()
    if sub.empty:
        out = pd.DataFrame()
        write_csv_atomic(out, ckpt)
        return out
    sub["_score"] = make_rule_score(sub, rule)
    sub = sub.dropna(subset=["_score"])
    sub["_rank"] = sub.groupby("trade_date", sort=False)["_score"].rank(method="first", ascending=False)
    selected = sub.loc[sub["_rank"].le(top_n), ["trade_date", "ts_code", "industry", target, "_score"]].copy()
    if selected.empty:
        out = pd.DataFrame()
    else:
        selected["_positive"] = selected[target].gt(0).astype(int)
        selected["_negative"] = selected[target].lt(0).astype(int)
        selected["_pos_ret"] = selected[target].where(selected[target].gt(0), 0.0)
        selected["_neg_ret"] = selected[target].where(selected[target].lt(0), 0.0)
        out = (
            selected.groupby("trade_date", sort=False)
            .agg(
                selected_count=("ts_code", "size"),
                daily_return=(target, "mean"),
                trade_return_sum=(target, "sum"),
                win_count=("_positive", "sum"),
                loss_count=("_negative", "sum"),
                positive_trade_return_sum=("_pos_ret", "sum"),
                negative_trade_return_sum=("_neg_ret", "sum"),
                avg_rule_score=("_score", "mean"),
                unique_industries=("industry", "nunique"),
            )
            .reset_index()
        )
    if not out.empty:
        out.insert(0, "top_n", top_n)
        out.insert(0, "horizon", f"{horizon}d")
        out.insert(0, "rule_id", rule.rule_id)
    write_csv_atomic(out, ckpt)
    print(
        {
            "event": "checkpoint_write",
            "step": "rule_daily",
            "rule_id": rule.rule_id,
            "horizon": f"{horizon}d",
            "rows": len(out),
            "elapsed_sec": round(time.perf_counter() - started, 1),
        },
        flush=True,
    )
    return out


def build_random_baseline(
    df: pd.DataFrame,
    rule: RuleSpec,
    horizon: int,
    rule_daily: pd.DataFrame,
    top_n: int,
    sims: int,
    seed: int,
    refresh: bool,
) -> pd.DataFrame:
    ckpt = checkpoint_path("random_matched", rule.rule_id, f"{horizon}d", f"top{top_n}", f"sims{sims}")
    if ckpt.exists() and not refresh:
        cached = read_checkpoint(ckpt)
        print({"event": "checkpoint_skip", "step": "random_matched", "rule_id": rule.rule_id, "horizon": f"{horizon}d", "rows": len(cached)}, flush=True)
        return cached
    if rule_daily.empty:
        out = pd.DataFrame()
        write_csv_atomic(out, ckpt)
        return out
    started = time.perf_counter()
    target = f"fwd_ret_{horizon}d_open"
    needed = ["trade_date", target] + list(rule.factors)
    pool = df.loc[df["screened_universe_flag"] & df[target].notna(), needed].dropna(subset=[target] + list(rule.factors)).copy()
    counts = rule_daily.set_index("trade_date")["selected_count"].astype(int).to_dict()
    pool = pool.loc[pool["trade_date"].isin(counts)]
    arrays = {date: g[target].to_numpy(dtype=float) for date, g in pool.groupby("trade_date", sort=False)}
    dates = [date for date in sorted(counts) if date in arrays and len(arrays[date]) > 0]
    daily_matrix = np.full((sims, len(dates)), np.nan, dtype=float)
    rng = np.random.default_rng(seed + horizon * 1000 + abs(hash(rule.rule_id)) % 100000)
    for col_idx, date in enumerate(dates):
        arr = arrays[date]
        n = min(int(counts[date]), len(arr))
        if n <= 0:
            continue
        idx = rng.integers(0, len(arr), size=(sims, n))
        daily_matrix[:, col_idx] = arr[idx].mean(axis=1)
        if (col_idx + 1) % 500 == 0 or col_idx + 1 == len(dates):
            print(
                {
                    "event": "progress",
                    "step": "random_matched",
                    "rule_id": rule.rule_id,
                    "horizon": f"{horizon}d",
                    "dates_done": col_idx + 1,
                    "dates_total": len(dates),
                    "elapsed_sec": round(time.perf_counter() - started, 1),
                },
                flush=True,
            )
    rows = []
    for sim_id in range(sims):
        returns = pd.Series(daily_matrix[sim_id, :]).dropna()
        if returns.empty:
            continue
        rows.append(
            {
                "rule_id": rule.rule_id,
                "horizon": f"{horizon}d",
                "top_n": top_n,
                "sim_id": sim_id,
                "signal_days": int(returns.size),
                "cumulative_return": float((1.0 + returns).prod() - 1.0),
                "mean_daily_return": float(returns.mean()),
                "daily_win_rate": float(returns.gt(0).mean()),
                "profit_factor": profit_factor(returns),
                "max_drawdown": max_drawdown(returns),
            }
        )
    out = pd.DataFrame(rows)
    write_csv_atomic(out, ckpt)
    print(
        {
            "event": "checkpoint_write",
            "step": "random_matched",
            "rule_id": rule.rule_id,
            "horizon": f"{horizon}d",
            "rows": len(out),
            "elapsed_sec": round(time.perf_counter() - started, 1),
        },
        flush=True,
    )
    return out


def summarize_random(random_df: pd.DataFrame, rule_row: dict) -> dict:
    if random_df.empty:
        return {
            "rule_id": rule_row["rule_id"],
            "horizon": rule_row["horizon"],
            "top_n": rule_row["top_n"],
            "random_sims": 0,
            "random_cumret_mean": np.nan,
            "random_cumret_median": np.nan,
            "random_cumret_p05": np.nan,
            "random_cumret_p95": np.nan,
            "random_cumret_min": np.nan,
            "random_cumret_max": np.nan,
            "rule_cumret_percentile": np.nan,
        }
    r = finite_numeric(random_df["cumulative_return"]).dropna()
    rule_ret = rule_row["cumulative_return"]
    return {
        "rule_id": rule_row["rule_id"],
        "horizon": rule_row["horizon"],
        "top_n": rule_row["top_n"],
        "random_sims": int(r.size),
        "random_cumret_mean": float(r.mean()),
        "random_cumret_median": float(r.median()),
        "random_cumret_p05": float(r.quantile(0.05)),
        "random_cumret_p95": float(r.quantile(0.95)),
        "random_cumret_min": float(r.min()),
        "random_cumret_max": float(r.max()),
        "rule_cumret_percentile": float(r.le(rule_ret).mean()) if pd.notna(rule_ret) else np.nan,
    }


def rule_daily_by_regime(all_daily: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    if all_daily.empty:
        return pd.DataFrame()
    merged = all_daily.merge(regime[["trade_date"] + REGIME_COLUMNS], on="trade_date", how="left", validate="many_to_one")
    rows = []
    for col in REGIME_COLUMNS:
        for keys, g in merged.groupby(["rule_id", "horizon", "top_n", col], dropna=False, sort=False):
            rule_id, horizon, top_n, value = keys
            rows.append(summarize_daily(g, {"rule_id": rule_id, "horizon": horizon, "top_n": top_n, "regime_type": col, "regime_value": value}))
    return pd.DataFrame(rows)


def write_rule_config() -> None:
    lines = [
        "batch: batch3D_simple_rule_baseline",
        "top_n: 20",
        f"random_simulations: {RANDOM_SIMS}",
        f"random_seed: {RANDOM_SEED}",
        "label: fwd_ret_{h}d_open",
        "label_definition: entry_adj_open(T+1) to exit_adj_close(T+h)",
        "costs: none_gross_diagnostic",
        "position_weighting: equal_weight_selected_names_per_signal_date",
        "overlap_accounting: cohort_return_diagnostic_not_full_capital_ledger",
        "concept_theme_features_enabled: false",
        "model_training_enabled: false",
        "resume_policy: default_resume_from_checkpoints; use --refresh to recompute from scratch",
        "rules:",
    ]
    for rule in RULE_SPECS:
        lines.extend(
            [
                f"  - rule_id: {rule.rule_id}",
                f"    rule_group: {rule.rule_group}",
                f"    factors: [{', '.join(rule.factors)}]",
                f"    directions: [{', '.join(rule.directions)}]",
                f"    uses_industry_snapshot: {str(rule.uses_industry_snapshot).lower()}",
                f"    primary_rule: {str(rule.primary_rule).lower()}",
                f"    description: {json.dumps(rule.description, ensure_ascii=False)}",
            ]
        )
    (REPORT_DIR / "batch3D_rule_config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_limitations() -> None:
    lines = [
        "# Batch 3D Limitations",
        "",
        "- This batch is a simple-rule baseline, not a model and not a trading strategy.",
        "- No model training, hyperparameter optimization, feature selection by returns, concept/theme data or live trading is performed.",
        "- Primary return is gross cohort return: each signal date selects Top20 equal-weight names and uses `T+1 open -> T+h close` labels.",
        "- Cohort compounding is diagnostic and does not model overlapping capital usage across 3/5/10 trading-day holding periods.",
        "- Transaction costs, impact cost, partial fills, stop rules and capacity are not included in this batch.",
        "- Industry-derived rules use current-snapshot industry classification and are diagnostic-only until point-in-time industry data is locked.",
        "- Random baselines are matched to each rule's eligible pool, active dates and selected count.",
        "- Long-running work is checkpointed under `_checkpoints/`; rerun without `--refresh` resumes completed checkpoints.",
    ]
    (REPORT_DIR / "batch3D_limitations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_handoff(summary: pd.DataFrame, random_summary: pd.DataFrame, gate_result: str) -> None:
    top = summary.sort_values(["eligible_for_training_plan", "rule_cumret_percentile", "cumulative_return"], ascending=[False, False, False]).head(15)
    payload = top[
        [
            "rule_id",
            "horizon",
            "cumulative_return",
            "mean_daily_return",
            "daily_win_rate",
            "profit_factor",
            "max_drawdown",
            "rule_cumret_percentile",
            "uses_industry_snapshot",
            "eligible_for_training_plan",
        ]
    ].to_dict("records")
    lines = [
        "# Handoff: Batch 3D to Batch 4A",
        "",
        "## Completed Batch",
        "",
        "- Completed: `Batch 3D simple rule baseline`",
        "- Output directory: `reports/tushare/v9_swing_research/batch3D_simple_rule_baseline/`",
        "- No model training, hyperparameter search, concept/theme feature enablement, or live/sim trading was performed.",
        "- Long-running computations support checkpoint/resume by default.",
        "",
        "## Core Conclusion",
        "",
        f"- Gate result: `{gate_result}`",
        f"- Deterministic rule rows: `{len(summary)}`",
        f"- Matched random rows: `{len(random_summary)}`",
        f"- Top rule diagnostics: `{json.dumps(payload, ensure_ascii=False, default=str)}`",
        "- Main caveat: results are gross cohort diagnostics and do not include overlapping capital ledger, costs, impact or capacity.",
        "- Industry snapshot rules remain diagnostic-only.",
        "",
        "## Required Inputs For Next Step",
        "",
        "- `reports/tushare/v9_swing_research/batch3D_simple_rule_baseline/batch3D_rule_backtest_summary.csv`",
        "- `reports/tushare/v9_swing_research/batch3D_simple_rule_baseline/batch3D_random_baseline_summary.csv`",
        "- `reports/tushare/v9_swing_research/batch3D_simple_rule_baseline/batch3D_rule_config.yaml`",
        "- `reports/tushare/v9_swing_research/batch3D_simple_rule_baseline/batch3D_limitations.md`",
        "",
        "## Next Step",
        "",
        "If accepted, proceed only to `Batch 4A model training plan freeze`, not direct training.",
        "Batch 4A must freeze horizon, features, label, split, baseline references and allowed training fields before any model is trained.",
        "",
        "## Blocked Actions",
        "",
        "- Do not train models before Batch 4A is accepted.",
        "- Do not treat current-snapshot industry rules as clean point-in-time training features.",
        "- Do not use these gross cohort returns as executable strategy returns.",
    ]
    text = "\n".join(lines) + "\n"
    (REPORT_DIR / "batch3D_handoff_to_batch4A.md").write_text(text, encoding="utf-8")
    GLOBAL_HANDOFF_PATH.write_text(text, encoding="utf-8")


def write_conclusion(summary: pd.DataFrame, coverage: pd.DataFrame, gate_result: str) -> None:
    top = summary.sort_values(["eligible_for_training_plan", "rule_cumret_percentile", "cumulative_return"], ascending=[False, False, False]).head(15)
    eligible_count = int(summary["eligible_for_training_plan"].sum()) if "eligible_for_training_plan" in summary else 0
    lines = [
        "# Batch 3D Simple Rule Baseline",
        "",
        "## Scope",
        "",
        "- This batch evaluates deterministic simple rules and matched random baselines.",
        "- It does not train models, tune parameters, use concept/theme features, or modify v7_locked.",
        "- All long-running steps are checkpointed and resumable; `--refresh` is required for full recompute.",
        "",
        "## Method",
        "",
        "- Primary TopN: `20` names per signal date.",
        "- Return label: `fwd_ret_{h}d_open = exit_adj_close(T+h) / entry_adj_open(T+1) - 1`.",
        "- Daily rule return: equal-weight mean return of selected names.",
        "- Cumulative return: compounded daily cohort returns; this is not a full overlapping capital ledger.",
        "- Costs, impact, capacity and partial fills are not included in this batch.",
        "",
        "## Coverage",
        "",
        coverage.to_markdown(index=False),
        "",
        "## Gate",
        "",
        f"- Gate result: `{gate_result}`",
        f"- Eligible non-industry-snapshot rule rows passing pre-registered screen: `{eligible_count}`",
        "- Eligibility screen: cumulative_return > 0, daily_win_rate > 50%, signal_days >= 1000, matched random percentile >= 95%, and no current-snapshot industry dependency.",
        "",
        "## Top Rule Diagnostics",
        "",
        top[
            [
                "rule_id",
                "horizon",
                "cumulative_return",
                "mean_daily_return",
                "daily_win_rate",
                "profit_factor",
                "max_drawdown",
                "rule_cumret_percentile",
                "uses_industry_snapshot",
                "eligible_for_training_plan",
            ]
        ].to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- If eligible rules exist, Batch 4A may freeze a model training plan, but no training should start yet.",
        "- The strongest rules must be checked for costs, capacity, crowding and weak-market robustness before any forward tracking.",
        "- Industry-snapshot rules remain diagnostic-only.",
        "",
        "## Next Step",
        "",
        "- Proceed to `Batch 4A model training plan freeze` only after reviewing this report.",
        "- Do not start Batch 4B training directly.",
    ]
    (REPORT_DIR / "batch3D_conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hashes() -> None:
    rows = []
    for path in sorted(REPORT_DIR.glob("batch3D_*")):
        if path.name == "batch3D_file_sha256.csv" or not path.is_file():
            continue
        rows.append({"file": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    for path in [GLOBAL_HANDOFF_PATH, ROADMAP_PATH, STAGE_STATUS_PATH]:
        if path.exists():
            rows.append({"file": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    pd.DataFrame(rows).to_csv(REPORT_DIR / "batch3D_file_sha256.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Clear Batch 3D checkpoints and recompute from scratch.")
    parser.add_argument("--top-n", type=int, default=PRIMARY_TOP_N)
    parser.add_argument("--random-sims", type=int, default=RANDOM_SIMS)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.refresh and CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        {
            "event": "run_mode",
            "refresh": bool(args.refresh),
            "resume": not bool(args.refresh),
            "top_n": args.top_n,
            "random_sims": args.random_sims,
            "checkpoint_dir": str(CHECKPOINT_DIR),
        },
        flush=True,
    )

    print({"event": "read_dataset"}, flush=True)
    df = read_dataset()
    coverage_rows = []
    for horizon in HORIZONS:
        target = f"fwd_ret_{horizon}d_open"
        coverage_rows.append(
            {
                "horizon": f"{horizon}d",
                "rows": len(df),
                "screened_rows": int(df["screened_universe_flag"].sum()),
                "screened_valid_target_rows": int((df["screened_universe_flag"] & df[target].notna()).sum()),
                "screened_valid_target_ratio": float((df["screened_universe_flag"] & df[target].notna()).mean()),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(REPORT_DIR / "batch3D_coverage_summary.csv", index=False)
    write_rule_config()
    write_limitations()

    all_daily_frames = []
    summary_rows = []
    random_summary_rows = []
    random_full_rows = []
    rule_meta = {rule.rule_id: rule for rule in RULE_SPECS}

    for horizon in HORIZONS:
        for rule_idx, rule in enumerate(RULE_SPECS, start=1):
            print(
                {
                    "event": "rule_start",
                    "rule_done": rule_idx - 1,
                    "rules_total": len(RULE_SPECS),
                    "rule_id": rule.rule_id,
                    "horizon": f"{horizon}d",
                },
                flush=True,
            )
            daily = evaluate_rule(df, rule, horizon, args.top_n, refresh=args.refresh)
            if not daily.empty:
                all_daily_frames.append(daily)
            row = summarize_daily(
                daily,
                {
                    "rule_id": rule.rule_id,
                    "rule_group": rule.rule_group,
                    "horizon": f"{horizon}d",
                    "top_n": args.top_n,
                    "factors": "|".join(rule.factors),
                    "directions": "|".join(rule.directions),
                    "uses_industry_snapshot": rule.uses_industry_snapshot,
                    "primary_rule": rule.primary_rule,
                },
            )
            random_df = build_random_baseline(df, rule, horizon, daily, args.top_n, args.random_sims, RANDOM_SEED, refresh=args.refresh)
            if not random_df.empty:
                random_full_rows.append(random_df)
            random_summary = summarize_random(random_df, row)
            row.update(random_summary)
            summary_rows.append(row)
            random_summary_rows.append(random_summary)

    summary = pd.DataFrame(summary_rows)
    summary["eligible_for_training_plan"] = (
        summary["primary_rule"].astype(bool)
        & ~summary["uses_industry_snapshot"].astype(bool)
        & summary["cumulative_return"].gt(0)
        & summary["daily_win_rate"].gt(0.50)
        & summary["signal_days"].ge(1000)
        & summary["rule_cumret_percentile"].ge(0.95)
    )
    summary = summary.sort_values(["eligible_for_training_plan", "rule_cumret_percentile", "cumulative_return"], ascending=[False, False, False])
    summary.to_csv(REPORT_DIR / "batch3D_rule_backtest_summary.csv", index=False)
    summary.to_csv(REPORT_DIR / "batch3D_rule_by_horizon.csv", index=False)

    random_summary = pd.DataFrame(random_summary_rows)
    random_summary.to_csv(REPORT_DIR / "batch3D_random_baseline_summary.csv", index=False)
    if random_full_rows:
        pd.concat(random_full_rows, ignore_index=True).to_csv(REPORT_DIR / "batch3D_random_simulation_metrics.csv", index=False)

    all_daily = pd.concat(all_daily_frames, ignore_index=True) if all_daily_frames else pd.DataFrame()
    all_daily.to_csv(REPORT_DIR / "batch3D_rule_daily_returns.csv", index=False)
    regime = pd.read_parquet(REGIME_PATH, columns=["trade_date"] + REGIME_COLUMNS)
    regime["trade_date"] = regime["trade_date"].astype(str)
    by_regime = rule_daily_by_regime(all_daily, regime)
    by_regime.to_csv(REPORT_DIR / "batch3D_rule_by_regime.csv", index=False)

    gate_result = "pass_to_batch4A_plan_freeze" if int(summary["eligible_for_training_plan"].sum()) > 0 else "review_required_before_batch4A"
    write_handoff(summary, random_summary, gate_result)
    write_conclusion(summary, coverage, gate_result)
    write_hashes()
    print(
        {
            "out_dir": str(REPORT_DIR),
            "rule_rows": int(len(summary)),
            "eligible_rows": int(summary["eligible_for_training_plan"].sum()),
            "random_summary_rows": int(len(random_summary)),
            "gate_result": gate_result,
            "next_step": "Batch 4A model training plan freeze" if gate_result.startswith("pass") else "review Batch 3D limitations",
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
