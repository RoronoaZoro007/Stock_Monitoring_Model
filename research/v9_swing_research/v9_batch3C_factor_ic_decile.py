#!/usr/bin/env python3
"""v9 Batch 3C: factor IC and decile diagnostics.

This batch tests whether pre-registered daily/industry factors have sorting
power for 3/5/10 trading-day forward labels. It does not train models, run
simple-rule backtests, or enable concept/theme factors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT / "data_tushare" / "clean" / "v9"
REPORT_DIR = ROOT / "reports" / "tushare" / "v9_swing_research" / "batch3C_factor_ic_decile"
CHECKPOINT_DIR = REPORT_DIR / "_checkpoints"
GLOBAL_HANDOFF_PATH = ROOT / "reports" / "tushare" / "v9_swing_research" / "v9_current_handoff.md"
ROADMAP_PATH = ROOT / "reports" / "tushare" / "v9_swing_research" / "v9_execution_roadmap.md"
STAGE_STATUS_PATH = ROOT / "reports" / "tushare" / "v9_swing_research" / "v9_stage_status.csv"

LABEL_PATH = CLEAN_DIR / "v9_swing_labels_3_5_10.parquet"
FEATURE_PATH = CLEAN_DIR / "v9_stock_industry_features.parquet"
REGIME_PATH = CLEAN_DIR / "v9_market_regime_formal.parquet"

HORIZONS = [3, 5, 10]
MIN_GROUP_N = 30


@dataclass(frozen=True)
class FactorSpec:
    factor: str
    group: str
    description: str
    expected_direction: str
    point_in_time_status: str


FACTOR_SPECS = [
    FactorSpec("stock_ret_1d", "price_momentum_reversal", "Stock 1-day adjusted return at signal date.", "unknown", "usable_after_t_close"),
    FactorSpec("stock_ret_5d", "price_momentum_reversal", "Stock 5-day adjusted return at signal date.", "unknown", "usable_after_t_close"),
    FactorSpec("stock_ret_20d", "price_momentum", "Stock 20-day adjusted return at signal date.", "unknown", "usable_after_t_close"),
    FactorSpec("stock_ret_60d", "price_momentum", "Stock 60-day adjusted return at signal date.", "unknown", "usable_after_t_close"),
    FactorSpec("stock_rel_industry_ret_1d", "relative_strength", "Stock 1-day return minus industry 1-day return.", "unknown", "diagnostic_industry_snapshot"),
    FactorSpec("stock_rel_industry_ret_5d", "relative_strength", "Stock 5-day return minus industry 5-day return.", "unknown", "diagnostic_industry_snapshot"),
    FactorSpec("stock_rel_industry_ret_20d", "relative_strength", "Stock 20-day return minus industry 20-day return.", "unknown", "diagnostic_industry_snapshot"),
    FactorSpec("stock_rel_industry_ret_60d", "relative_strength", "Stock 60-day return minus industry 60-day return.", "unknown", "diagnostic_industry_snapshot"),
    FactorSpec("industry_ret_5d", "industry_strength", "Industry 5-day equal-weight return.", "unknown", "diagnostic_industry_snapshot"),
    FactorSpec("industry_ret_20d", "industry_strength", "Industry 20-day equal-weight return.", "unknown", "diagnostic_industry_snapshot"),
    FactorSpec("industry_ret_60d", "industry_strength", "Industry 60-day equal-weight return.", "unknown", "diagnostic_industry_snapshot"),
    FactorSpec("industry_breadth_20d", "industry_breadth", "Industry 20-day rolling positive-return breadth.", "positive", "diagnostic_industry_snapshot"),
    FactorSpec("industry_breadth_60d", "industry_breadth", "Industry 60-day rolling positive-return breadth.", "positive", "diagnostic_industry_snapshot"),
    FactorSpec("industry_amount_share", "industry_crowding", "Industry amount share of full-market amount.", "unknown", "diagnostic_industry_snapshot"),
    FactorSpec("industry_amount_share_roll20d", "industry_crowding", "Industry amount share 20-day rolling mean.", "unknown", "diagnostic_industry_snapshot"),
    FactorSpec("industry_amount_share_roll60d", "industry_crowding", "Industry amount share 60-day rolling mean.", "unknown", "diagnostic_industry_snapshot"),
    FactorSpec("industry_crowding_score", "industry_crowding", "Industry amount-share percentile versus own prior 252 trading days.", "unknown", "diagnostic_industry_snapshot"),
    FactorSpec("stock_amount_share_in_industry", "liquidity_crowding", "Stock amount as share of its industry's amount.", "unknown", "usable_after_t_close"),
    FactorSpec("turnover_rate", "liquidity", "Daily turnover rate.", "unknown", "usable_after_t_close"),
    FactorSpec("log_amount", "liquidity", "Log daily amount.", "unknown", "usable_after_t_close"),
    FactorSpec("log_total_mv", "size", "Log total market value.", "unknown", "usable_after_t_close"),
]

FACTOR_COLUMNS = [spec.factor for spec in FACTOR_SPECS]


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


def read_dataset() -> pd.DataFrame:
    feature_cols = [
        "trade_date",
        "ts_code",
        "industry",
        "trend_regime",
        "vol_regime",
        "industry_crowding_regime",
        "stock_concentration_regime",
        "market_regime_id",
        "listed_days",
        "st_flag",
        "suspend_flag",
        "limit_up_close_flag",
        "limit_down_close_flag",
    ] + FACTOR_COLUMNS
    features = pd.read_parquet(FEATURE_PATH, columns=feature_cols)
    label_cols = ["trade_date", "ts_code"] + [f"fwd_ret_{h}d_open" for h in HORIZONS]
    labels = pd.read_parquet(LABEL_PATH, columns=label_cols)
    features["trade_date"] = features["trade_date"].astype(str)
    labels["trade_date"] = labels["trade_date"].astype(str)
    df = features.merge(labels, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
    df["year"] = df["trade_date"].str[:4]
    df["industry"] = df["industry"].fillna("UNKNOWN").astype(str)
    for flag in ["st_flag", "suspend_flag", "limit_up_close_flag", "limit_down_close_flag"]:
        df[flag] = df[flag].fillna(False).astype(bool)
    df["listed_days"] = finite_numeric(df["listed_days"])
    for col in FACTOR_COLUMNS + [f"fwd_ret_{h}d_open" for h in HORIZONS]:
        df[col] = finite_numeric(df[col])
    df["screened_universe_flag"] = (
        df["listed_days"].ge(120)
        & ~df["st_flag"]
        & ~df["suspend_flag"]
        & ~df["limit_up_close_flag"]
        & ~df["limit_down_close_flag"]
    )
    df["size_bucket"] = make_daily_bucket(df, "log_total_mv", 5, "size_q")
    df["liquidity_bucket"] = make_daily_bucket(df, "log_amount", 5, "liq_q")
    return df


def make_daily_bucket(df: pd.DataFrame, col: str, buckets: int, prefix: str) -> pd.Series:
    rank_pct = df.groupby("trade_date", sort=False)[col].rank(pct=True, method="average")
    q = np.ceil(rank_pct * buckets)
    q = q.clip(lower=1, upper=buckets)
    return q.astype("Int64").astype(str).replace("<NA>", "missing").map(lambda x: f"{prefix}{x}" if x != "missing" else "missing")


def group_corr_from_ranks(ranked: pd.DataFrame, group_cols: list[str], x_col: str, y_col: str, min_n: int = MIN_GROUP_N) -> pd.DataFrame:
    tmp = ranked[group_cols + [x_col, y_col]].dropna()
    if tmp.empty:
        return pd.DataFrame(columns=group_cols + ["ic", "n"])
    tmp["_x2"] = tmp[x_col] * tmp[x_col]
    tmp["_y2"] = tmp[y_col] * tmp[y_col]
    tmp["_xy"] = tmp[x_col] * tmp[y_col]
    agg = (
        tmp.groupby(group_cols, dropna=False, sort=False)
        .agg(n=(x_col, "size"), sx=(x_col, "sum"), sy=(y_col, "sum"), sxx=("_x2", "sum"), syy=("_y2", "sum"), sxy=("_xy", "sum"))
        .reset_index()
    )
    n = agg["n"].astype(float)
    cov = agg["sxy"] - agg["sx"] * agg["sy"] / n
    varx = agg["sxx"] - agg["sx"] * agg["sx"] / n
    vary = agg["syy"] - agg["sy"] * agg["sy"] / n
    denom = np.sqrt(varx * vary)
    agg["ic"] = cov / denom.replace(0, np.nan)
    agg.loc[agg["n"].lt(min_n), "ic"] = np.nan
    return agg[group_cols + ["ic", "n"]]


def ic_all_factors_for_groups(
    df: pd.DataFrame,
    target: str,
    universe_mask: pd.Series,
    group_cols: list[str],
    factors: list[str] = FACTOR_COLUMNS,
    min_n: int = MIN_GROUP_N,
    progress_label: str | None = None,
    progress_every: int = 5000,
) -> pd.DataFrame:
    """Compute rank IC for all factors in one pass over group_cols."""
    cols = group_cols + factors + [target]
    sub = df.loc[universe_mask, cols].dropna(subset=[target]).copy()
    if sub.empty:
        return pd.DataFrame(columns=group_cols + ["factor", "ic", "n"])
    rows = []
    grouped = sub.groupby(group_cols, dropna=False, sort=False)
    total_groups = int(grouped.ngroups)
    started = time.perf_counter()
    if progress_label:
        print(
            {
                "event": "progress_start",
                "step": progress_label,
                "groups_total": total_groups,
                "rows_total": int(len(sub)),
            },
            flush=True,
        )
    for group_idx, (keys, g) in enumerate(grouped, start=1):
        if not isinstance(keys, tuple):
            keys = (keys,)
        y = finite_numeric(g[target])
        if y.notna().sum() < min_n:
            if progress_label and (group_idx % progress_every == 0 or group_idx == total_groups):
                print(
                    {
                        "event": "progress",
                        "step": progress_label,
                        "groups_done": group_idx,
                        "groups_total": total_groups,
                        "pct": round(group_idx / total_groups * 100, 2) if total_groups else 100.0,
                        "elapsed_sec": round(time.perf_counter() - started, 1),
                        "ic_rows": len(rows),
                    },
                    flush=True,
                )
            continue
        y_rank = y.rank(pct=True, method="average")
        x_rank = g[factors].rank(pct=True, method="average")
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = x_rank.corrwith(y_rank)
        n = x_rank.notna().mul(y_rank.notna(), axis=0).sum()
        key_payload = {field: value for field, value in zip(group_cols, keys)}
        for factor in factors:
            factor_n = int(n.get(factor, 0))
            if factor_n < min_n:
                continue
            rows.append({**key_payload, "factor": factor, "ic": float(corr.get(factor, np.nan)), "n": factor_n})
        if progress_label and (group_idx % progress_every == 0 or group_idx == total_groups):
            print(
                {
                    "event": "progress",
                    "step": progress_label,
                    "groups_done": group_idx,
                    "groups_total": total_groups,
                    "pct": round(group_idx / total_groups * 100, 2) if total_groups else 100.0,
                    "elapsed_sec": round(time.perf_counter() - started, 1),
                    "ic_rows": len(rows),
                },
                flush=True,
            )
    if progress_label:
        print(
            {
                "event": "progress_done",
                "step": progress_label,
                "groups_done": total_groups,
                "groups_total": total_groups,
                "elapsed_sec": round(time.perf_counter() - started, 1),
                "ic_rows": len(rows),
            },
            flush=True,
        )
    return pd.DataFrame(rows)


def ic_all_factors_checkpointed_by_date(
    df: pd.DataFrame,
    target: str,
    universe_mask: pd.Series,
    group_cols: list[str],
    checkpoint_label: str,
    refresh: bool,
    chunk_size: int,
    progress_every: int,
    factors: list[str] = FACTOR_COLUMNS,
) -> pd.DataFrame:
    valid_dates = sorted(df.loc[universe_mask & df[target].notna(), "trade_date"].dropna().astype(str).unique())
    chunks = [valid_dates[i : i + chunk_size] for i in range(0, len(valid_dates), chunk_size)]
    frames = []
    print(
        {
            "event": "checkpoint_plan",
            "step": checkpoint_label,
            "date_chunks_total": len(chunks),
            "chunk_size": chunk_size,
            "resume_enabled": not refresh,
        },
        flush=True,
    )
    for chunk_idx, dates in enumerate(chunks, start=1):
        ckpt = checkpoint_path("ic", checkpoint_label, f"chunk_{chunk_idx:04d}_of_{len(chunks):04d}")
        if ckpt.exists() and not refresh:
            cached = read_checkpoint(ckpt)
            frames.append(cached)
            print(
                {
                    "event": "checkpoint_skip",
                    "step": checkpoint_label,
                    "chunk_done": chunk_idx,
                    "chunks_total": len(chunks),
                    "rows": int(len(cached)),
                    "file": str(ckpt),
                },
                flush=True,
            )
            continue
        chunk_mask = universe_mask & df["trade_date"].astype(str).isin(dates)
        ic = ic_all_factors_for_groups(
            df,
            target,
            chunk_mask,
            group_cols,
            factors=factors,
            progress_label=f"{checkpoint_label}:chunk_{chunk_idx}/{len(chunks)}",
            progress_every=progress_every,
        )
        write_csv_atomic(ic, ckpt)
        frames.append(ic)
        print(
            {
                "event": "checkpoint_write",
                "step": checkpoint_label,
                "chunk_done": chunk_idx,
                "chunks_total": len(chunks),
                "rows": int(len(ic)),
                "file": str(ckpt),
            },
            flush=True,
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=group_cols + ["factor", "ic", "n"])


def summarize_ic(ic_df: pd.DataFrame, group_fields: list[str], meta: dict) -> pd.DataFrame:
    rows = []
    if ic_df.empty:
        return pd.DataFrame()
    for keys, g in ic_df.groupby(group_fields, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        valid = finite_numeric(g["ic"]).dropna()
        row = dict(meta)
        row.update({field: value for field, value in zip(group_fields, keys)})
        row["ic_days"] = int(valid.size)
        row["mean_ic"] = float(valid.mean()) if valid.size else np.nan
        row["median_ic"] = float(valid.median()) if valid.size else np.nan
        row["std_ic"] = float(valid.std(ddof=1)) if valid.size > 1 else np.nan
        row["icir"] = float(valid.mean() / valid.std(ddof=1) * np.sqrt(252)) if valid.size > 1 and valid.std(ddof=1) != 0 else np.nan
        row["positive_ic_rate"] = float(valid.gt(0).mean()) if valid.size else np.nan
        row["mean_group_n"] = float(g.loc[g["ic"].notna(), "n"].mean()) if valid.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def rank_ic_for(df: pd.DataFrame, factor: str, target: str, universe_mask: pd.Series, group_cols: list[str]) -> pd.DataFrame:
    cols = ["trade_date", factor, target] + [c for c in group_cols if c != "trade_date"]
    sub = df.loc[universe_mask, cols].dropna(subset=[factor, target]).copy()
    if sub.empty:
        return pd.DataFrame(columns=group_cols + ["ic", "n"])
    sub["_factor_rank"] = sub.groupby(group_cols, dropna=False, sort=False)[factor].rank(pct=True, method="average")
    sub["_target_rank"] = sub.groupby(group_cols, dropna=False, sort=False)[target].rank(pct=True, method="average")
    return group_corr_from_ranks(sub, group_cols, "_factor_rank", "_target_rank")


def build_daily_ic_and_summaries(df: pd.DataFrame, refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_frames = []
    summary_frames = []
    for universe_name, base_mask in [
        ("all_valid", pd.Series(True, index=df.index)),
        ("screened", df["screened_universe_flag"]),
    ]:
        for horizon in HORIZONS:
            target = f"fwd_ret_{horizon}d_open"
            mask = base_mask & df[target].notna()
            ic = ic_all_factors_checkpointed_by_date(
                df,
                target,
                mask,
                ["trade_date"],
                checkpoint_label=f"daily_ic_{universe_name}_{horizon}d",
                refresh=refresh,
                chunk_size=250,
                progress_every=250,
            )
            if ic.empty:
                continue
            ic.insert(0, "universe", universe_name)
            ic.insert(1, "horizon", f"{horizon}d")
            daily_frames.append(ic)
            summary = summarize_ic(ic, ["universe", "horizon", "factor"], {})
            if not summary.empty:
                meta = build_factor_dictionary()[["factor", "factor_group", "point_in_time_status"]]
                summary = summary.merge(meta, on="factor", how="left")
                summary_frames.append(summary)
    daily_ic = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    return daily_ic, summary


def build_split_ic(df: pd.DataFrame, split_col: str, split_name: str, refresh: bool) -> pd.DataFrame:
    frames = []
    base_mask = df["screened_universe_flag"]
    for horizon in HORIZONS:
        target = f"fwd_ret_{horizon}d_open"
        mask = base_mask & df[target].notna() & df[split_col].notna()
        progress_every = 5000 if split_name == "industry" else 500
        chunk_size = 100 if split_name == "industry" else 250
        ic = ic_all_factors_checkpointed_by_date(
            df,
            target,
            mask,
            ["trade_date", split_col],
            checkpoint_label=f"split_ic_{split_name}_{horizon}d",
            refresh=refresh,
            chunk_size=chunk_size,
            progress_every=progress_every,
        )
        if ic.empty:
            continue
        summary = summarize_ic(ic, ["factor", split_col], {"split_type": split_name, "horizon": f"{horizon}d"})
        if summary.empty:
            continue
        summary = summary.rename(columns={split_col: "split_value"})
        meta = build_factor_dictionary()[["factor", "factor_group"]]
        summary = summary.merge(meta, on="factor", how="left")
        frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_year_ic(daily_ic: pd.DataFrame) -> pd.DataFrame:
    if daily_ic.empty:
        return pd.DataFrame()
    tmp = daily_ic.loc[daily_ic["universe"].eq("screened")].copy()
    tmp["year"] = tmp["trade_date"].astype(str).str[:4]
    return summarize_ic(tmp, ["horizon", "factor", "year"], {"universe": "screened"})


def build_regime_ic(daily_ic: pd.DataFrame) -> pd.DataFrame:
    if daily_ic.empty:
        return pd.DataFrame()
    regime = pd.read_parquet(
        REGIME_PATH,
        columns=["trade_date", "trend_regime", "vol_regime", "industry_crowding_regime", "stock_concentration_regime", "extreme_selloff_flag"],
    )
    regime["trade_date"] = regime["trade_date"].astype(str)
    tmp = daily_ic.loc[daily_ic["universe"].eq("screened")].merge(regime, on="trade_date", how="left", validate="many_to_one")
    frames = []
    for col in ["trend_regime", "vol_regime", "industry_crowding_regime", "stock_concentration_regime", "extreme_selloff_flag"]:
        s = summarize_ic(tmp, ["horizon", "factor", col], {"universe": "screened", "split_type": col})
        if not s.empty:
            s = s.rename(columns={col: "split_value"})
            frames.append(s)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_decile_and_spread(df: pd.DataFrame, refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    decile_rows = []
    spread_rows = []
    base_mask = df["screened_universe_flag"]
    for horizon in HORIZONS:
        target = f"fwd_ret_{horizon}d_open"
        for factor_idx, spec in enumerate(FACTOR_SPECS, start=1):
            decile_ckpt = checkpoint_path("decile", f"{horizon}d", spec.factor)
            spread_ckpt = checkpoint_path("spread", f"{horizon}d", spec.factor)
            if decile_ckpt.exists() and spread_ckpt.exists() and not refresh:
                cached_decile = read_checkpoint(decile_ckpt)
                cached_spread = read_checkpoint(spread_ckpt)
                if not cached_decile.empty:
                    decile_rows.append(cached_decile)
                if not cached_spread.empty:
                    spread_rows.append(cached_spread)
                print(
                    {
                        "event": "checkpoint_skip",
                        "step": f"decile:{horizon}d",
                        "factors_done": factor_idx,
                        "factors_total": len(FACTOR_SPECS),
                        "current_factor": spec.factor,
                        "decile_rows": int(len(cached_decile)),
                        "spread_rows": int(len(cached_spread)),
                    },
                    flush=True,
                )
                continue
            print(
                {
                    "event": "progress",
                    "step": f"decile:{horizon}d",
                    "factors_done": factor_idx - 1,
                    "factors_total": len(FACTOR_SPECS),
                    "current_factor": spec.factor,
                },
                flush=True,
            )
            cols = ["trade_date", spec.factor, target]
            sub = df.loc[base_mask & df[target].notna() & df[spec.factor].notna(), cols].copy()
            if sub.empty:
                write_csv_atomic(pd.DataFrame(), decile_ckpt)
                write_csv_atomic(pd.DataFrame(), spread_ckpt)
                continue
            sub["_rank_pct"] = sub.groupby("trade_date", sort=False)[spec.factor].rank(pct=True, method="average")
            sub["decile"] = np.ceil(sub["_rank_pct"] * 10).clip(1, 10).astype(int)
            daily_decile = sub.groupby(["trade_date", "decile"], sort=False).agg(daily_mean_return=(target, "mean"), obs=(target, "size")).reset_index()
            by_decile = daily_decile.groupby("decile", sort=True).agg(
                days=("trade_date", "nunique"),
                observations=("obs", "sum"),
                mean_daily_return=("daily_mean_return", "mean"),
                median_daily_return=("daily_mean_return", "median"),
                positive_daily_rate=("daily_mean_return", lambda x: float(pd.to_numeric(x, errors="coerce").gt(0).mean())),
            ).reset_index()
            by_decile.insert(0, "factor", spec.factor)
            by_decile.insert(0, "horizon", f"{horizon}d")
            by_decile.insert(0, "universe", "screened")
            write_csv_atomic(by_decile, decile_ckpt)
            decile_rows.append(by_decile)
            pivot = daily_decile.pivot(index="trade_date", columns="decile", values="daily_mean_return")
            spread_frame = pd.DataFrame()
            if 1 in pivot and 10 in pivot:
                spread = (pivot[10] - pivot[1]).dropna()
                mean_spread = float(spread.mean()) if len(spread) else np.nan
                std_spread = float(spread.std(ddof=1)) if len(spread) > 1 else np.nan
                preferred_direction = "high_factor" if mean_spread >= 0 else "low_factor"
                spread_frame = pd.DataFrame(
                    [
                        {
                        "universe": "screened",
                        "horizon": f"{horizon}d",
                        "factor": spec.factor,
                        "spread_days": int(len(spread)),
                        "high_minus_low_mean": mean_spread,
                        "high_minus_low_median": float(spread.median()) if len(spread) else np.nan,
                        "high_minus_low_tstat": float(mean_spread / std_spread * np.sqrt(len(spread))) if len(spread) > 1 and std_spread != 0 else np.nan,
                        "positive_spread_rate": float(spread.gt(0).mean()) if len(spread) else np.nan,
                        "direction_adjusted_mean_spread": abs(mean_spread) if pd.notna(mean_spread) else np.nan,
                        "preferred_direction": preferred_direction,
                        }
                    ]
                )
                spread_rows.append(spread_frame)
            write_csv_atomic(spread_frame, spread_ckpt)
        print(
            {
                "event": "progress_done",
                "step": f"decile:{horizon}d",
                "factors_done": len(FACTOR_SPECS),
                "factors_total": len(FACTOR_SPECS),
            },
            flush=True,
        )
    decile = pd.concat(decile_rows, ignore_index=True) if decile_rows else pd.DataFrame()
    spread = pd.concat(spread_rows, ignore_index=True) if spread_rows else pd.DataFrame()
    return decile, spread


def build_candidate_screen(rankic: pd.DataFrame, spread: pd.DataFrame) -> pd.DataFrame:
    primary = rankic.loc[rankic["universe"].eq("screened")].copy()
    merged = primary.merge(spread[["horizon", "factor", "direction_adjusted_mean_spread"]], on=["horizon", "factor"], how="left")
    merged["abs_mean_ic"] = merged["mean_ic"].abs()
    merged["candidate_flag"] = (
        merged["ic_days"].ge(500)
        & merged["abs_mean_ic"].ge(0.01)
        & merged["positive_ic_rate"].sub(0.5).abs().ge(0.02)
        & merged["direction_adjusted_mean_spread"].gt(0)
    )
    return merged.sort_values(["candidate_flag", "abs_mean_ic", "direction_adjusted_mean_spread"], ascending=[False, False, False])


def build_factor_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "factor": spec.factor,
                "factor_group": spec.group,
                "description": spec.description,
                "expected_direction": spec.expected_direction,
                "point_in_time_status": spec.point_in_time_status,
                "enabled_for_ic": True,
                "enabled_for_training": False,
                "notes": "Training enablement is blocked until Batch 4A feature freeze.",
            }
            for spec in FACTOR_SPECS
        ]
    )


def build_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    screened = int(df["screened_universe_flag"].sum())
    for horizon in HORIZONS:
        target = f"fwd_ret_{horizon}d_open"
        rows.append(
            {
                "horizon": f"{horizon}d",
                "rows": total,
                "valid_target_rows": int(df[target].notna().sum()),
                "valid_target_ratio": float(df[target].notna().mean()),
                "screened_rows": screened,
                "screened_valid_target_rows": int((df["screened_universe_flag"] & df[target].notna()).sum()),
                "screened_valid_target_ratio_of_total": float((df["screened_universe_flag"] & df[target].notna()).mean()),
            }
        )
    return pd.DataFrame(rows)


def write_limitations() -> None:
    lines = [
        "# Batch 3C Factor IC Limitations",
        "",
        "## Scope",
        "",
        "- This batch computes factor RankIC, ICIR, decile returns and split diagnostics only.",
        "- It does not train models, run simple-rule backtests, choose trading parameters, or enable concept/theme factors.",
        "",
        "## Universe",
        "",
        "- `all_valid` keeps all rows with valid target and factor values.",
        "- Primary `screened` universe additionally excludes ST rows, suspended rows, stocks listed for less than 120 days, and current-day close limit-up/limit-down rows.",
        "- This is a diagnostic tradability screen, not a final trading universe.",
        "",
        "## Point-In-Time",
        "",
        "- Factors use T-day close/day-level data and are suitable for post-close swing research, not intraday decisions.",
        "- Industry factors remain diagnostic because `stock_basic.industry` is a current snapshot field.",
        "- Concept/theme factors are disabled because no point-in-time concept membership data is locked.",
        "",
        "## Interpretation",
        "",
        "- Positive IC means higher factor values rank with higher future returns.",
        "- Negative IC can still be economically useful as an inverse rank, but must be validated against simple-rule baselines in Batch 3D.",
        "- IC significance alone is not a trading result; costs, turnover and execution remain untested here.",
    ]
    (REPORT_DIR / "batch3C_factor_limitations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_handoff(candidate: pd.DataFrame) -> None:
    candidates = candidate.loc[candidate["candidate_flag"]].head(20)
    payload = candidates[["horizon", "factor", "mean_ic", "icir", "positive_ic_rate", "direction_adjusted_mean_spread"]].to_dict("records")
    lines = [
        "# Handoff: Batch 3C to Batch 3D",
        "",
        "## Completed Batch",
        "",
        "- Completed: `Batch 3C factor IC and decile diagnostics`",
        "- Output directory: `reports/tushare/v9_swing_research/batch3C_factor_ic_decile/`",
        "- No simple-rule backtest, model training, or parameter optimization was performed.",
        "- Long-running steps are checkpointed under `_checkpoints/`; rerun without `--refresh` resumes completed chunks, while `--refresh` clears checkpoints and recomputes from scratch.",
        "",
        "## Core Conclusion",
        "",
        f"- Candidate factor rows by pre-registered screen: `{int(candidate['candidate_flag'].sum())}`",
        f"- Top candidate diagnostics: `{json.dumps(payload, ensure_ascii=False, default=str)}`",
        "- Concept/theme factors remain disabled.",
        "- Industry snapshot factors remain diagnostic-only unless point-in-time industry classification is added or leakage risk is explicitly accepted.",
        "- The strongest diagnostics are mostly negative IC signals: low liquidity, low turnover, low stock amount share inside industry, and weak medium-term momentum rank higher for future returns.",
        "",
        "## Required Inputs For Next Step",
        "",
        "- `reports/tushare/v9_swing_research/batch3C_factor_ic_decile/batch3C_candidate_factor_screen.csv`",
        "- `reports/tushare/v9_swing_research/batch3C_factor_ic_decile/batch3C_rankic_summary.csv`",
        "- `reports/tushare/v9_swing_research/batch3C_factor_ic_decile/batch3C_decile_return_summary.csv`",
        "- `reports/tushare/v9_swing_research/batch3C_factor_ic_decile/batch3C_top_bottom_spread.csv`",
        "- `data_tushare/clean/v9/v9_swing_labels_3_5_10.parquet`",
        "- `data_tushare/clean/v9/v9_stock_industry_features.parquet`",
        "",
        "## Next Step",
        "",
        "Run `Batch 3D simple rule baseline` to test whether simple rules built from the strongest IC factors can beat random baselines.",
        "Batch 3D must also be implemented with checkpoint/resume support before any long-running computation.",
        "",
        "## Blocked Actions",
        "",
        "- Do not train models.",
        "- Do not freeze model features until Batch 3D passes.",
        "- Do not enable concept/theme factors without a point-in-time data foundation.",
    ]
    text = "\n".join(lines) + "\n"
    (REPORT_DIR / "batch3C_handoff_to_batch3D.md").write_text(text, encoding="utf-8")
    GLOBAL_HANDOFF_PATH.write_text(text, encoding="utf-8")


def write_conclusion(rankic: pd.DataFrame, spread: pd.DataFrame, candidate: pd.DataFrame, coverage: pd.DataFrame) -> None:
    top = candidate.head(15)
    pass_count = int(candidate["candidate_flag"].sum())
    lines = [
        "# Batch 3C Factor IC and Decile Diagnostics",
        "",
        "## Scope",
        "",
        "- This batch computes RankIC, ICIR, decile returns and split diagnostics for 3d/5d/10d labels.",
        "- It does not run simple-rule backtests, train models, tune parameters, or enable concept/theme factors.",
        "- Long-running computations are checkpointed and resumable by default; use `--refresh` only when a full recompute is intended.",
        "",
        "## Coverage",
        "",
        coverage.to_markdown(index=False),
        "",
        "## Candidate Screen",
        "",
        "- Candidate screen is diagnostic: `ic_days >= 500`, `abs(mean_ic) >= 0.01`, `abs(positive_ic_rate - 0.5) >= 0.02`, and direction-adjusted Top-Bottom spread > 0.",
        f"- Candidate rows passing screen: `{pass_count}`",
        "",
        "## Top Diagnostics",
        "",
        top[["horizon", "factor", "factor_group", "mean_ic", "icir", "positive_ic_rate", "direction_adjusted_mean_spread", "candidate_flag"]].to_markdown(index=False),
        "",
        "## Gate Result",
        "",
        "- Gate result: `pass_to_batch3D`." if pass_count > 0 else "- Gate result: `review_required_before_batch3D`.",
        "- Reason: factor diagnostics are complete. Batch 3D is still required before any model training.",
        "- Main interpretation: the strongest rows are negative IC diagnostics, so Batch 3D should test inverse-ranked simple rules rather than assuming momentum continuation.",
        "",
        "## Next Step",
        "",
        "- Execute `Batch 3D simple rule baseline` using the candidate factor screen and random baselines.",
        "- Do not train models yet.",
    ]
    (REPORT_DIR / "batch3C_conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hashes() -> None:
    rows = []
    for path in sorted(REPORT_DIR.glob("batch3C_*")):
        if path.name == "batch3C_file_sha256.csv" or not path.is_file():
            continue
        rows.append({"file": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    for path in [GLOBAL_HANDOFF_PATH, ROADMAP_PATH, STAGE_STATUS_PATH]:
        if path.exists():
            rows.append({"file": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    pd.DataFrame(rows).to_csv(REPORT_DIR / "batch3C_file_sha256.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Recompute from scratch and clear Batch 3C checkpoints.")
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.refresh and CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print({"event": "run_mode", "refresh": bool(args.refresh), "resume": not bool(args.refresh), "checkpoint_dir": str(CHECKPOINT_DIR)}, flush=True)

    print({"event": "read_dataset"}, flush=True)
    df = read_dataset()
    coverage = build_coverage(df)
    coverage.to_csv(REPORT_DIR / "batch3C_coverage_summary.csv", index=False)
    build_factor_dictionary().to_csv(REPORT_DIR / "batch3C_factor_dictionary.csv", index=False)

    print({"event": "daily_ic"}, flush=True)
    daily_ic, rankic_summary = build_daily_ic_and_summaries(df, refresh=args.refresh)
    daily_ic.to_csv(REPORT_DIR / "batch3C_daily_ic.csv", index=False)
    rankic_summary.to_csv(REPORT_DIR / "batch3C_rankic_summary.csv", index=False)

    print({"event": "split_ic", "split": "year"}, flush=True)
    build_year_ic(daily_ic).to_csv(REPORT_DIR / "batch3C_ic_by_year.csv", index=False)
    print({"event": "split_ic", "split": "regime"}, flush=True)
    build_regime_ic(daily_ic).to_csv(REPORT_DIR / "batch3C_ic_by_regime.csv", index=False)
    print({"event": "split_ic", "split": "industry"}, flush=True)
    build_split_ic(df, "industry", "industry", refresh=args.refresh).to_csv(REPORT_DIR / "batch3C_ic_by_industry.csv", index=False)
    print({"event": "split_ic", "split": "size_bucket"}, flush=True)
    size = build_split_ic(df, "size_bucket", "size_bucket", refresh=args.refresh)
    print({"event": "split_ic", "split": "liquidity_bucket"}, flush=True)
    liquidity = build_split_ic(df, "liquidity_bucket", "liquidity_bucket", refresh=args.refresh)
    pd.concat([size, liquidity], ignore_index=True).to_csv(REPORT_DIR / "batch3C_ic_by_size_liquidity.csv", index=False)

    print({"event": "decile"}, flush=True)
    decile, spread = build_decile_and_spread(df, refresh=args.refresh)
    decile.to_csv(REPORT_DIR / "batch3C_decile_return_summary.csv", index=False)
    spread.to_csv(REPORT_DIR / "batch3C_top_bottom_spread.csv", index=False)
    candidate = build_candidate_screen(rankic_summary, spread)
    candidate.to_csv(REPORT_DIR / "batch3C_candidate_factor_screen.csv", index=False)

    write_limitations()
    write_handoff(candidate)
    write_conclusion(rankic_summary, spread, candidate, coverage)
    write_hashes()
    print(
        {
            "out_dir": str(REPORT_DIR),
            "rows": int(len(df)),
            "screened_rows": int(df["screened_universe_flag"].sum()),
            "factors": len(FACTOR_SPECS),
            "candidate_rows": int(candidate["candidate_flag"].sum()),
            "next_step": "Batch 3D simple rule baseline",
        }
    )


if __name__ == "__main__":
    main()
