#!/usr/bin/env python3
"""v9 Batch 3B: industry/theme feature engineering.

This batch builds industry-level and stock-relative industry diagnostics for
future factor IC work. It does not run factor IC, train models, tune rules, or
enable concept/theme features without point-in-time membership data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT / "data_tushare" / "clean" / "v9"
REPORT_DIR = ROOT / "reports" / "tushare" / "v9_swing_research" / "batch3B_industry_theme_features"
GLOBAL_HANDOFF_PATH = ROOT / "reports" / "tushare" / "v9_swing_research" / "v9_current_handoff.md"

PANEL_PATH = CLEAN_DIR / "v9_daily_panel.parquet"
REGIME_PATH = CLEAN_DIR / "v9_market_regime_formal.parquet"
INDUSTRY_FEATURE_PATH = CLEAN_DIR / "v9_industry_daily_features.parquet"
STOCK_INDUSTRY_FEATURE_PATH = CLEAN_DIR / "v9_stock_industry_features.parquet"

ROLLING_CROWDING_WINDOW = 252
ROLLING_CROWDING_MIN_PERIODS = 60


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def rolling_compound_return(ret: pd.Series, window: int, min_periods: int) -> pd.Series:
    return (1.0 + ret).rolling(window, min_periods=min_periods).apply(np.prod, raw=True) - 1.0


def prior_rolling_percentile(series: pd.Series, window: int = ROLLING_CROWDING_WINDOW, min_periods: int = ROLLING_CROWDING_MIN_PERIODS) -> pd.Series:
    """Current value percentile versus prior rolling window only."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for idx, current in enumerate(values):
        if np.isnan(current):
            continue
        start = max(0, idx - window)
        hist = values[start:idx]
        hist = hist[~np.isnan(hist)]
        if len(hist) >= min_periods:
            out[idx] = float((hist <= current).mean())
    return pd.Series(out, index=series.index)


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "ts_code",
        "trade_date",
        "industry",
        "adj_close",
        "amount",
        "total_mv",
        "circ_mv",
        "turnover_rate",
        "listed_days",
        "st_flag",
        "suspend_flag",
        "limit_up_close_flag",
        "limit_down_close_flag",
    ]
    panel = pd.read_parquet(PANEL_PATH, columns=cols)
    regime = pd.read_parquet(
        REGIME_PATH,
        columns=[
            "trade_date",
            "trend_regime",
            "vol_regime",
            "industry_crowding_regime",
            "stock_concentration_regime",
            "market_regime_id",
            "extreme_selloff_flag",
        ],
    )
    panel["trade_date"] = panel["trade_date"].astype(str)
    regime["trade_date"] = regime["trade_date"].astype(str)
    panel["industry"] = panel["industry"].fillna("UNKNOWN").astype(str)
    panel.loc[panel["industry"].str.strip().eq("") | panel["industry"].eq("None"), "industry"] = "UNKNOWN"
    for col in ["adj_close", "amount", "total_mv", "circ_mv", "turnover_rate", "listed_days"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel["amount"] = panel["amount"].fillna(0.0).clip(lower=0.0)
    for flag in ["st_flag", "suspend_flag", "limit_up_close_flag", "limit_down_close_flag"]:
        panel[flag] = panel[flag].fillna(False).astype(bool)
    return panel, regime


def build_features(refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if INDUSTRY_FEATURE_PATH.exists() and STOCK_INDUSTRY_FEATURE_PATH.exists() and not refresh:
        industry = pd.read_parquet(INDUSTRY_FEATURE_PATH)
        stock = pd.read_parquet(STOCK_INDUSTRY_FEATURE_PATH)
        return industry, stock

    panel, regime = read_inputs()
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    grp_code = panel.groupby("ts_code", sort=False)["adj_close"]
    panel["stock_ret_1d"] = grp_code.pct_change()
    panel["stock_ret_5d"] = grp_code.pct_change(5)
    panel["stock_ret_20d"] = grp_code.pct_change(20)
    panel["stock_ret_60d"] = grp_code.pct_change(60)

    market_amount = panel.groupby("trade_date", sort=True)["amount"].sum().reset_index(name="market_amount")
    industry = (
        panel.groupby(["trade_date", "industry"], sort=True)
        .agg(
            industry_stock_count=("ts_code", "nunique"),
            industry_amount=("amount", "sum"),
            industry_ew_ret_1d=("stock_ret_1d", "mean"),
            industry_breadth_positive=("stock_ret_1d", lambda x: float(pd.to_numeric(x, errors="coerce").gt(0).mean())),
            industry_total_mv=("total_mv", "sum"),
            industry_circ_mv=("circ_mv", "sum"),
            industry_turnover_median=("turnover_rate", "median"),
            industry_st_stock_count=("st_flag", "sum"),
            industry_suspend_stock_count=("suspend_flag", "sum"),
            industry_limit_up_close_count=("limit_up_close_flag", "sum"),
            industry_limit_down_close_count=("limit_down_close_flag", "sum"),
        )
        .reset_index()
    )
    industry = industry.merge(market_amount, on="trade_date", how="left", validate="many_to_one")
    industry["industry_amount_share"] = industry["industry_amount"] / industry["market_amount"].replace(0, np.nan)
    industry["industry_mv_share"] = industry["industry_total_mv"] / industry.groupby("trade_date")["industry_total_mv"].transform("sum").replace(0, np.nan)
    industry = industry.merge(regime, on="trade_date", how="left", validate="many_to_one")
    industry = industry.sort_values(["industry", "trade_date"]).reset_index(drop=True)

    by_industry = industry.groupby("industry", sort=False)
    for window, min_periods in [(5, 3), (20, 10), (60, 30)]:
        industry[f"industry_ret_{window}d"] = by_industry["industry_ew_ret_1d"].transform(
            lambda s, w=window, m=min_periods: rolling_compound_return(pd.to_numeric(s, errors="coerce"), w, m)
        )
    for window, min_periods in [(20, 10), (60, 30)]:
        industry[f"industry_breadth_{window}d"] = by_industry["industry_breadth_positive"].transform(
            lambda s, w=window, m=min_periods: pd.to_numeric(s, errors="coerce").rolling(w, min_periods=m).mean()
        )
        industry[f"industry_amount_share_roll{window}d"] = by_industry["industry_amount_share"].transform(
            lambda s, w=window, m=min_periods: pd.to_numeric(s, errors="coerce").rolling(w, min_periods=m).mean()
        )

    industry["industry_amount_share_pct_252_tminus1"] = by_industry["industry_amount_share"].transform(prior_rolling_percentile)
    industry["industry_crowding_score"] = industry["industry_amount_share_pct_252_tminus1"]
    industry["industry_crowding_state"] = "crowding_normal"
    industry.loc[industry["industry_crowding_score"].isna(), "industry_crowding_state"] = "insufficient_history"
    industry.loc[industry["industry_crowding_score"].ge(0.80), "industry_crowding_state"] = "crowding_high"
    industry.loc[industry["industry_crowding_score"].le(0.20), "industry_crowding_state"] = "crowding_low"

    for col in ["industry_ret_5d", "industry_ret_20d", "industry_ret_60d", "industry_amount_share"]:
        industry[f"{col}_rank_pct_cs"] = industry.groupby("trade_date")[col].rank(pct=True, method="average")

    merge_cols = [
        "trade_date",
        "industry",
        "industry_stock_count",
        "industry_amount",
        "industry_amount_share",
        "industry_mv_share",
        "industry_ew_ret_1d",
        "industry_ret_5d",
        "industry_ret_20d",
        "industry_ret_60d",
        "industry_breadth_positive",
        "industry_breadth_20d",
        "industry_breadth_60d",
        "industry_amount_share_roll20d",
        "industry_amount_share_roll60d",
        "industry_crowding_score",
        "industry_crowding_state",
        "industry_ret_20d_rank_pct_cs",
        "industry_amount_share_rank_pct_cs",
        "trend_regime",
        "vol_regime",
        "industry_crowding_regime",
        "stock_concentration_regime",
        "market_regime_id",
        "extreme_selloff_flag",
    ]
    stock_cols = [
        "trade_date",
        "ts_code",
        "industry",
        "stock_ret_1d",
        "stock_ret_5d",
        "stock_ret_20d",
        "stock_ret_60d",
        "amount",
        "total_mv",
        "circ_mv",
        "turnover_rate",
        "listed_days",
        "st_flag",
        "suspend_flag",
        "limit_up_close_flag",
        "limit_down_close_flag",
    ]
    stock = panel[stock_cols].merge(industry[merge_cols], on=["trade_date", "industry"], how="left", validate="many_to_one")
    for window in [1, 5, 20, 60]:
        stock[f"stock_rel_industry_ret_{window}d"] = stock[f"stock_ret_{window}d"] - stock[f"industry_ret_{window}d" if window != 1 else "industry_ew_ret_1d"]
    stock["stock_amount_share_in_industry"] = stock["amount"] / stock["industry_amount"].replace(0, np.nan)
    stock["log_total_mv"] = np.log(pd.to_numeric(stock["total_mv"], errors="coerce").where(lambda s: s > 0))
    stock["log_amount"] = np.log(pd.to_numeric(stock["amount"], errors="coerce").where(lambda s: s > 0))

    industry.to_parquet(INDUSTRY_FEATURE_PATH, index=False, compression="zstd")
    stock.to_parquet(STOCK_INDUSTRY_FEATURE_PATH, index=False, compression="zstd")
    return industry, stock


def build_feature_dictionary() -> pd.DataFrame:
    rows = [
        ("industry", "industry", "Current snapshot industry from stock_basic; diagnostic only unless point-in-time source is added.", "stock_basic.industry", "diagnostic_only_current_snapshot"),
        ("industry_ew_ret_1d", "industry_daily", "Equal-weight average stock_ret_1d within industry on trade_date.", "adj_close", "usable_after_t_close"),
        ("industry_ret_5d", "industry_daily", "Rolling compounded industry_ew_ret_1d over 5 trading days.", "industry_ew_ret_1d", "usable_after_t_close"),
        ("industry_ret_20d", "industry_daily", "Rolling compounded industry_ew_ret_1d over 20 trading days.", "industry_ew_ret_1d", "usable_after_t_close"),
        ("industry_ret_60d", "industry_daily", "Rolling compounded industry_ew_ret_1d over 60 trading days.", "industry_ew_ret_1d", "usable_after_t_close"),
        ("industry_breadth_positive", "industry_daily", "Share of industry stocks with positive stock_ret_1d.", "adj_close", "usable_after_t_close"),
        ("industry_amount_share", "industry_daily", "Industry amount divided by full-market amount on trade_date.", "amount", "usable_after_t_close"),
        ("industry_crowding_score", "industry_daily", "Current industry_amount_share percentile versus the same industry's prior 252 trading days, min 60 days.", "industry_amount_share", "prior_window_no_future_distribution"),
        ("industry_crowding_state", "industry_daily", "crowding_high if score >= 0.8, low if <= 0.2, normal otherwise.", "industry_crowding_score", "prior_window_no_future_distribution"),
        ("stock_rel_industry_ret_1d", "stock_daily", "Stock 1d return minus same-day industry_ew_ret_1d.", "stock_ret_1d, industry_ew_ret_1d", "usable_after_t_close"),
        ("stock_rel_industry_ret_5d", "stock_daily", "Stock 5d return minus industry_ret_5d.", "stock_ret_5d, industry_ret_5d", "usable_after_t_close"),
        ("stock_rel_industry_ret_20d", "stock_daily", "Stock 20d return minus industry_ret_20d.", "stock_ret_20d, industry_ret_20d", "usable_after_t_close"),
        ("stock_rel_industry_ret_60d", "stock_daily", "Stock 60d return minus industry_ret_60d.", "stock_ret_60d, industry_ret_60d", "usable_after_t_close"),
        ("stock_amount_share_in_industry", "stock_daily", "Stock amount divided by same-day industry amount.", "amount, industry_amount", "usable_after_t_close"),
    ]
    return pd.DataFrame(rows, columns=["field", "level", "definition", "source_fields", "point_in_time_status"])


def build_crowding_summary(industry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = industry.groupby("industry", dropna=False)
    for name, g in grouped:
        rows.append(
            {
                "industry": name,
                "days": int(len(g)),
                "first_date": g["trade_date"].min(),
                "last_date": g["trade_date"].max(),
                "avg_stock_count": float(g["industry_stock_count"].mean()),
                "avg_amount_share": float(g["industry_amount_share"].mean()),
                "p95_amount_share": float(g["industry_amount_share"].quantile(0.95)),
                "crowding_high_days": int(g["industry_crowding_state"].eq("crowding_high").sum()),
                "crowding_low_days": int(g["industry_crowding_state"].eq("crowding_low").sum()),
                "avg_ret_20d": float(g["industry_ret_20d"].mean()),
                "avg_ret_60d": float(g["industry_ret_60d"].mean()),
                "latest_crowding_state": g.sort_values("trade_date")["industry_crowding_state"].iloc[-1],
            }
        )
    return pd.DataFrame(rows).sort_values(["crowding_high_days", "avg_amount_share"], ascending=[False, False])


def build_theme_status() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_group": "concept_theme_membership",
                "status": "disabled",
                "reason": "No point-in-time concept/theme membership dataset has been locked in Stage 0.",
                "allowed_use": "Do not use for IC or training until a separate data-foundation batch locks source, coverage, fields and point-in-time rules.",
            },
            {
                "feature_group": "industry_snapshot",
                "status": "diagnostic_only",
                "reason": "stock_basic.industry is a current snapshot field and may contain classification look-ahead risk.",
                "allowed_use": "Allowed for diagnostics and grouping; training use requires explicit acceptance or point-in-time replacement.",
            },
        ]
    )


def write_limitations() -> None:
    lines = [
        "# Batch 3B Point-In-Time Limitations",
        "",
        "## Scope",
        "",
        "- This batch builds industry-level and stock-relative industry diagnostics.",
        "- It does not compute factor IC, run simple-rule backtests, train models, or choose parameters.",
        "",
        "## Industry Field Risk",
        "",
        "- `stock_basic.industry` is a current snapshot field according to Stage 0.",
        "- Therefore industry features are allowed for diagnostics and grouping, but are not automatically approved as training features.",
        "- If training needs industry classification, a point-in-time industry source should be locked or the leakage risk must be explicitly accepted.",
        "",
        "## Concept / Theme Features",
        "",
        "- Concept/theme features are disabled in this batch.",
        "- Reason: no point-in-time historical concept membership or concept index dataset has been locked.",
        "- Any future concept feature work must start with a separate data-foundation batch before IC or training.",
        "",
        "## Timing",
        "",
        "- Return, breadth, amount-share and crowding features use same-day close/amount information; they are suitable for post-close swing research, not intraday pre-close signal generation.",
        "- Industry crowding score uses current amount share compared with the same industry's prior 252 trading days, with a 60-day minimum history. The percentile window excludes the current day and future days.",
    ]
    (REPORT_DIR / "batch3B_point_in_time_limitations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_handoff(industry: pd.DataFrame, stock: pd.DataFrame, summary: pd.DataFrame) -> None:
    top_industries = summary.head(10)[["industry", "crowding_high_days", "avg_amount_share"]].to_dict("records")
    lines = [
        "# Handoff: Batch 3B to Batch 3C",
        "",
        "## Completed Batch",
        "",
        "- Completed: `Batch 3B industry/theme features`",
        "- Output directory: `reports/tushare/v9_swing_research/batch3B_industry_theme_features/`",
        "- Industry feature file: `data_tushare/clean/v9/v9_industry_daily_features.parquet`",
        "- Stock-industry feature file: `data_tushare/clean/v9/v9_stock_industry_features.parquet`",
        "- No factor IC, simple-rule backtest, model training, or parameter optimization was performed.",
        "",
        "## Core Conclusion",
        "",
        f"- Industry daily feature rows: `{len(industry)}`",
        f"- Stock-industry feature rows: `{len(stock)}`",
        f"- Industry count: `{industry['industry'].nunique()}`",
        f"- Top crowding industries by high-crowding days: `{json.dumps(top_industries, ensure_ascii=False)}`",
        "- Concept/theme features remain disabled because no point-in-time source is locked.",
        "- Industry features are diagnostic-only unless point-in-time industry classification is added or leakage risk is explicitly accepted.",
        "",
        "## Required Inputs For Next Step",
        "",
        "- `data_tushare/clean/v9/v9_swing_labels_3_5_10.parquet`",
        "- `data_tushare/clean/v9/v9_market_regime_formal.parquet`",
        "- `data_tushare/clean/v9/v9_industry_daily_features.parquet`",
        "- `data_tushare/clean/v9/v9_stock_industry_features.parquet`",
        "- `reports/tushare/v9_swing_research/batch3B_industry_theme_features/batch3B_industry_feature_dictionary.csv`",
        "- `reports/tushare/v9_swing_research/batch3B_industry_theme_features/batch3B_point_in_time_limitations.md`",
        "",
        "## Next Step",
        "",
        "Run `Batch 3C factor IC and decile diagnostics` only after accepting that industry snapshot fields are diagnostic-only.",
        "",
        "Batch 3C should:",
        "",
        "- Compute RankIC, ICIR, decile returns and Top-Bottom spreads for 3d/5d/10d labels.",
        "- Split IC by market regime, year, industry, size and liquidity.",
        "- Keep concept/theme factors disabled unless their data foundation is locked first.",
        "",
        "## Blocked Actions",
        "",
        "- Do not train models.",
        "- Do not run simple-rule baseline before Batch 3C unless explicitly requested.",
        "- Do not treat current-snapshot industry as a clean point-in-time training feature.",
    ]
    text = "\n".join(lines) + "\n"
    (REPORT_DIR / "batch3B_handoff_to_batch3C.md").write_text(text, encoding="utf-8")
    GLOBAL_HANDOFF_PATH.write_text(text, encoding="utf-8")


def write_conclusion(industry: pd.DataFrame, stock: pd.DataFrame, summary: pd.DataFrame) -> None:
    crowd_counts = industry["industry_crowding_state"].value_counts(dropna=False).reset_index()
    crowd_counts.columns = ["industry_crowding_state", "rows"]
    theme_status = build_theme_status()
    lines = [
        "# Batch 3B Industry / Theme Feature Engineering",
        "",
        "## Scope",
        "",
        "- This batch builds industry-level strength, breadth, amount-share and crowding diagnostics.",
        "- It builds stock-relative industry return and stock share-in-industry diagnostics.",
        "- It does not run factor IC, simple-rule backtests, model training, or parameter selection.",
        "",
        "## Output Coverage",
        "",
        f"- Industry daily feature rows: `{len(industry)}`",
        f"- Stock-industry feature rows: `{len(stock)}`",
        f"- Trade dates: `{industry['trade_date'].nunique()}`",
        f"- Industries: `{industry['industry'].nunique()}`",
        "",
        "## Industry Crowding State Distribution",
        "",
        crowd_counts.to_markdown(index=False),
        "",
        "## Top Crowding Industries",
        "",
        summary.head(15)[["industry", "days", "avg_amount_share", "p95_amount_share", "crowding_high_days", "latest_crowding_state"]].to_markdown(index=False),
        "",
        "## Theme / Concept Status",
        "",
        theme_status.to_markdown(index=False),
        "",
        "## Gate Result",
        "",
        "- Gate result: `pass`.",
        "- Reason: industry diagnostics are built, concept/theme features are explicitly disabled, and point-in-time limitations are documented.",
        "",
        "## Next Step",
        "",
        "- Execute `Batch 3C factor IC and decile diagnostics` after accepting current industry limitations.",
        "- Do not train models or run simple-rule baseline yet.",
    ]
    (REPORT_DIR / "batch3B_conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hashes() -> None:
    rows = []
    for path in sorted(REPORT_DIR.glob("batch3B_*")):
        if path.name == "batch3B_file_sha256.csv" or not path.is_file():
            continue
        rows.append({"file": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    for path in [INDUSTRY_FEATURE_PATH, STOCK_INDUSTRY_FEATURE_PATH, GLOBAL_HANDOFF_PATH]:
        rows.append({"file": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    pd.DataFrame(rows).to_csv(REPORT_DIR / "batch3B_file_sha256.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    industry, stock = build_features(refresh=args.refresh)
    feature_dict = build_feature_dictionary()
    crowding_summary = build_crowding_summary(industry)
    theme_status = build_theme_status()

    feature_dict.to_csv(REPORT_DIR / "batch3B_industry_feature_dictionary.csv", index=False)
    crowding_summary.to_csv(REPORT_DIR / "batch3B_industry_crowding_summary.csv", index=False)
    theme_status.to_csv(REPORT_DIR / "batch3B_theme_feature_status.csv", index=False)
    write_limitations()
    write_handoff(industry, stock, crowding_summary)
    write_conclusion(industry, stock, crowding_summary)
    write_hashes()

    print(
        {
            "out_dir": str(REPORT_DIR),
            "industry_feature_path": str(INDUSTRY_FEATURE_PATH),
            "stock_industry_feature_path": str(STOCK_INDUSTRY_FEATURE_PATH),
            "industry_rows": int(len(industry)),
            "stock_rows": int(len(stock)),
            "industries": int(industry["industry"].nunique()),
            "next_step": "Batch 3C factor IC and decile diagnostics",
        }
    )


if __name__ == "__main__":
    main()
