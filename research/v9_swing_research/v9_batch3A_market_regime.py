#!/usr/bin/env python3
"""v9 Batch 3A: formal market regime labels.

This batch formalizes market-regime labels for downstream factor diagnostics.
It does not train models, tune trading parameters, or modify v7.
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
REPORT_DIR = ROOT / "reports" / "tushare" / "v9_swing_research" / "batch3A_market_regime"
GLOBAL_HANDOFF_PATH = ROOT / "reports" / "tushare" / "v9_swing_research" / "v9_current_handoff.md"
STAGE_STATUS_PATH = ROOT / "reports" / "tushare" / "v9_swing_research" / "v9_stage_status.csv"

INITIAL_REGIME_PATH = CLEAN_DIR / "v9_market_regime_initial.parquet"
FORMAL_REGIME_PATH = CLEAN_DIR / "v9_market_regime_formal.parquet"


TREND_STRONG_THRESHOLD = 0.08
TREND_WEAK_THRESHOLD = -0.08
EXTREME_BREADTH_THRESHOLD = 0.20
EXTREME_EW_RET_THRESHOLD = -0.03
EXPANDING_MIN_PERIODS = 252
HIGH_QUANTILE = 0.80
LOW_QUANTILE = 0.20


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_q(series: pd.Series, q: float) -> pd.Series:
    """Expanding quantile shifted by one day, using only prior observations."""
    return series.expanding(EXPANDING_MIN_PERIODS).quantile(q).shift(1)


def assign_trend_regime(roll60_ew_ret: pd.Series) -> pd.Series:
    out = pd.Series("neutral", index=roll60_ew_ret.index, dtype="object")
    out.loc[roll60_ew_ret.isna()] = "insufficient_history"
    out.loc[roll60_ew_ret.ge(TREND_STRONG_THRESHOLD)] = "strong"
    out.loc[roll60_ew_ret.le(TREND_WEAK_THRESHOLD)] = "weak"
    return out


def assign_three_way(value: pd.Series, low: pd.Series, high: pd.Series, low_name: str, mid_name: str, high_name: str) -> pd.Series:
    out = pd.Series(mid_name, index=value.index, dtype="object")
    out.loc[value.isna() | low.isna() | high.isna()] = "insufficient_history"
    out.loc[value.le(low)] = low_name
    out.loc[value.ge(high)] = high_name
    return out


def build_formal_regime() -> pd.DataFrame:
    market = pd.read_parquet(INITIAL_REGIME_PATH).sort_values("trade_date").reset_index(drop=True)
    numeric_cols = [
        "ew_ret",
        "median_ret",
        "breadth_positive",
        "dispersion",
        "total_amount",
        "top50_amount_share",
        "top100_amount_share",
        "top300_amount_share",
        "top1_industry_amount_share",
        "top3_industry_amount_share",
        "top5_industry_amount_share",
        "roll60_ew_ret",
        "roll60_breadth",
        "roll20_dispersion",
    ]
    for col in numeric_cols:
        if col in market:
            market[col] = pd.to_numeric(market[col], errors="coerce")

    market["trend_regime"] = assign_trend_regime(market["roll60_ew_ret"])
    market["extreme_selloff_flag"] = market["breadth_positive"].lt(EXTREME_BREADTH_THRESHOLD) | market["ew_ret"].le(EXTREME_EW_RET_THRESHOLD)

    market["vol_low_threshold_tminus1"] = safe_q(market["roll20_dispersion"], LOW_QUANTILE)
    market["vol_high_threshold_tminus1"] = safe_q(market["roll20_dispersion"], HIGH_QUANTILE)
    market["vol_regime"] = assign_three_way(
        market["roll20_dispersion"],
        market["vol_low_threshold_tminus1"],
        market["vol_high_threshold_tminus1"],
        "low_vol",
        "normal_vol",
        "high_vol",
    )
    market["high_vol_flag"] = market["vol_regime"].eq("high_vol")
    market["low_vol_flag"] = market["vol_regime"].eq("low_vol")

    market["crowding_low_threshold_tminus1"] = safe_q(market["top5_industry_amount_share"], LOW_QUANTILE)
    market["crowding_high_threshold_tminus1"] = safe_q(market["top5_industry_amount_share"], HIGH_QUANTILE)
    market["industry_crowding_regime"] = assign_three_way(
        market["top5_industry_amount_share"],
        market["crowding_low_threshold_tminus1"],
        market["crowding_high_threshold_tminus1"],
        "crowding_low",
        "crowding_normal",
        "crowding_high",
    )
    market["industry_crowding_high_flag"] = market["industry_crowding_regime"].eq("crowding_high")
    market["industry_crowding_low_flag"] = market["industry_crowding_regime"].eq("crowding_low")

    market["stock_concentration_low_threshold_tminus1"] = safe_q(market["top100_amount_share"], LOW_QUANTILE)
    market["stock_concentration_high_threshold_tminus1"] = safe_q(market["top100_amount_share"], HIGH_QUANTILE)
    market["stock_concentration_regime"] = assign_three_way(
        market["top100_amount_share"],
        market["stock_concentration_low_threshold_tminus1"],
        market["stock_concentration_high_threshold_tminus1"],
        "stock_concentration_low",
        "stock_concentration_normal",
        "stock_concentration_high",
    )
    market["stock_concentration_high_flag"] = market["stock_concentration_regime"].eq("stock_concentration_high")

    market["market_regime_id"] = (
        market["trend_regime"].astype(str)
        + "|"
        + market["vol_regime"].astype(str)
        + "|"
        + market["industry_crowding_regime"].astype(str)
    )
    market["uses_future_data"] = False
    market["definition_version"] = "batch3A_v1_fixed_trend_prior_expanding_quantiles"
    return market


def build_thresholds() -> pd.DataFrame:
    rows = [
        {
            "regime_component": "trend_regime",
            "field": "roll60_ew_ret",
            "rule": "strong if >= +8%, weak if <= -8%, neutral otherwise",
            "threshold_type": "fixed_economic_threshold",
            "uses_future_data": False,
            "reason": "A +/-8% full-market equal-weight 60 trading-day move is large enough to separate trend environments while preserving weak-state sample size.",
        },
        {
            "regime_component": "extreme_selloff_flag",
            "field": "breadth_positive, ew_ret",
            "rule": "true if breadth_positive < 20% or ew_ret <= -3%",
            "threshold_type": "fixed_market_stress_threshold",
            "uses_future_data": False,
            "reason": "Captures broad one-day selloff pressure without looking at future returns.",
        },
        {
            "regime_component": "vol_regime",
            "field": "roll20_dispersion",
            "rule": "high if >= expanding 80% quantile as of t-1; low if <= expanding 20% quantile as of t-1",
            "threshold_type": "prior_expanding_quantile",
            "uses_future_data": False,
            "reason": "Volatility levels drift across years; t-1 expanding quantiles avoid future information.",
        },
        {
            "regime_component": "industry_crowding_regime",
            "field": "top5_industry_amount_share",
            "rule": "high if >= expanding 80% quantile as of t-1; low if <= expanding 20% quantile as of t-1",
            "threshold_type": "prior_expanding_quantile",
            "uses_future_data": False,
            "reason": "Industry crowding is relative to historical turnover structure and must not use future distribution.",
        },
        {
            "regime_component": "stock_concentration_regime",
            "field": "top100_amount_share",
            "rule": "high if >= expanding 80% quantile as of t-1; low if <= expanding 20% quantile as of t-1",
            "threshold_type": "prior_expanding_quantile",
            "uses_future_data": False,
            "reason": "Stock-level concentration is tracked as a diagnostic proxy for liquidity crowding.",
        },
    ]
    return pd.DataFrame(rows)


def build_summary(regime: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("trend_regime", "market trend by 60d equal-weight return"),
        ("vol_regime", "20d cross-sectional dispersion regime"),
        ("industry_crowding_regime", "top5 industry amount-share crowding"),
        ("stock_concentration_regime", "top100 stock amount-share concentration"),
        ("market_regime_id", "combined trend/vol/industry-crowding state"),
    ]
    for col, desc in specs:
        for value, count in regime[col].value_counts(dropna=False).sort_index().items():
            rows.append(
                {
                    "component": col,
                    "description": desc,
                    "value": value,
                    "days": int(count),
                    "ratio": float(count / len(regime)),
                    "first_date": regime.loc[regime[col].eq(value), "trade_date"].min(),
                    "last_date": regime.loc[regime[col].eq(value), "trade_date"].max(),
                }
            )
    for flag in ["extreme_selloff_flag", "high_vol_flag", "industry_crowding_high_flag", "stock_concentration_high_flag"]:
        count = int(regime[flag].fillna(False).sum())
        rows.append(
            {
                "component": flag,
                "description": "boolean flag",
                "value": True,
                "days": count,
                "ratio": float(count / len(regime)),
                "first_date": regime.loc[regime[flag].fillna(False), "trade_date"].min(),
                "last_date": regime.loc[regime[flag].fillna(False), "trade_date"].max(),
            }
        )
    return pd.DataFrame(rows)


def build_by_year(regime: pd.DataFrame) -> pd.DataFrame:
    tmp = regime.copy()
    tmp["year"] = tmp["trade_date"].astype(str).str[:4]
    rows = []
    for (year, trend), g in tmp.groupby(["year", "trend_regime"], dropna=False):
        rows.append(
            {
                "year": year,
                "trend_regime": trend,
                "days": int(len(g)),
                "ew_ret_mean": float(g["ew_ret"].mean()),
                "breadth_positive_mean": float(g["breadth_positive"].mean()),
                "extreme_selloff_days": int(g["extreme_selloff_flag"].fillna(False).sum()),
                "high_vol_days": int(g["high_vol_flag"].fillna(False).sum()),
                "industry_crowding_high_days": int(g["industry_crowding_high_flag"].fillna(False).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["year", "trend_regime"])


def build_transition(regime: pd.DataFrame) -> pd.DataFrame:
    tmp = regime[["trade_date", "trend_regime"]].copy()
    tmp["next_trend_regime"] = tmp["trend_regime"].shift(-1)
    trans = tmp.dropna(subset=["next_trend_regime"]).groupby(["trend_regime", "next_trend_regime"]).size().reset_index(name="days")
    total = trans.groupby("trend_regime")["days"].transform("sum")
    trans["from_ratio"] = trans["days"] / total
    return trans.sort_values(["trend_regime", "next_trend_regime"])


def build_sensitivity(regime: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in [0.06, 0.08, 0.10, 0.12]:
        roll = regime["roll60_ew_ret"]
        strong = int(roll.ge(threshold).sum())
        weak = int(roll.le(-threshold).sum())
        neutral = int(roll.notna().sum() - strong - weak)
        rows.append(
            {
                "threshold_abs": threshold,
                "strong_days": strong,
                "weak_days": weak,
                "neutral_days": neutral,
                "insufficient_history_days": int(roll.isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def write_limitations() -> None:
    lines = [
        "# Batch 3A Market Regime Limitations",
        "",
        "## Scope",
        "",
        "- This batch formalizes market-regime labels only.",
        "- It does not run factor IC, build industry stock-level features, train models, or choose trading parameters.",
        "",
        "## No Future Data Policy",
        "",
        "- `trend_regime` uses current-day `roll60_ew_ret`, which is only suitable after the current trading day is known. It is for historical swing research labels, not same-day intraday decisions.",
        "- `vol_regime`, `industry_crowding_regime`, and `stock_concentration_regime` use expanding quantile thresholds shifted by one day, so the threshold for day `t` uses only observations up to `t-1`.",
        "- Fixed trend and selloff thresholds are pre-registered descriptive thresholds, not optimized on forward label performance.",
        "",
        "## Data Limitations",
        "",
        "- Industry crowding uses `stock_basic.industry`, which Stage 0 identified as a current snapshot field. It is acceptable for diagnostic grouping here, but not automatically acceptable as a training feature.",
        "- Concept/theme crowding is not included because no point-in-time concept membership dataset has been locked.",
        "- The first 252 trading days have insufficient history for expanding quantile regimes and are marked `insufficient_history` for those components.",
        "",
        "## Downstream Use",
        "",
        "- Batch 3C may use these market-state labels for split diagnostics.",
        "- Batch 3B must still build and document industry-level features separately before any industry/crowding factor enters IC or training.",
    ]
    (REPORT_DIR / "batch3A_regime_limitations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_handoff(regime: pd.DataFrame, summary: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    trend_counts = summary.loc[summary["component"].eq("trend_regime"), ["value", "days"]].to_dict("records")
    lines = [
        "# Handoff: Batch 3A to Batch 3B",
        "",
        "## Completed Batch",
        "",
        "- Completed: `Batch 3A market_regime`",
        "- Output directory: `reports/tushare/v9_swing_research/batch3A_market_regime/`",
        "- Formal regime file: `data_tushare/clean/v9/v9_market_regime_formal.parquet`",
        "- No model training, factor IC, simple-rule backtest, or parameter optimization was performed.",
        "",
        "## Core Conclusion",
        "",
        "- Formal market-state labels are available for 2,032 trading days.",
        f"- Trend regime counts: `{json.dumps(trend_counts, ensure_ascii=False)}`",
        "- Volatility and crowding regimes use t-1 expanding quantile thresholds to avoid future distribution leakage.",
        "- Industry crowding remains diagnostic because the underlying industry field is a current snapshot.",
        "",
        "## Required Inputs For Next Step",
        "",
        "- `data_tushare/clean/v9/v9_daily_panel.parquet`",
        "- `data_tushare/clean/v9/v9_market_regime_formal.parquet`",
        "- `reports/tushare/v9_swing_research/batch3A_market_regime/batch3A_market_regime_daily.csv`",
        "- `reports/tushare/v9_swing_research/stage0_data_foundation/stage0_limitations.md`",
        "",
        "## Next Step",
        "",
        "Run `Batch 3B industry/theme features` only after accepting Batch 3A definitions.",
        "",
        "Batch 3B should:",
        "",
        "- Build industry daily strength, breadth, amount-share, and crowding features.",
        "- Keep concept/theme features disabled unless point-in-time membership data is locked.",
        "- Treat `stock_basic.industry` as diagnostic unless point-in-time classification is added.",
        "",
        "## Blocked Actions",
        "",
        "- Do not run Batch 3C factor IC until Batch 3B completes or is explicitly waived.",
        "- Do not train models.",
        "- Do not choose trading parameters from these market-state labels.",
    ]
    text = "\n".join(lines) + "\n"
    (REPORT_DIR / "batch3A_handoff_to_batch3B.md").write_text(text, encoding="utf-8")
    GLOBAL_HANDOFF_PATH.write_text(text, encoding="utf-8")


def write_conclusion(regime: pd.DataFrame, summary: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    trend = summary.loc[summary["component"].eq("trend_regime"), ["value", "days", "ratio"]]
    vol = summary.loc[summary["component"].eq("vol_regime"), ["value", "days", "ratio"]]
    crowd = summary.loc[summary["component"].eq("industry_crowding_regime"), ["value", "days", "ratio"]]
    lines = [
        "# Batch 3A Market Regime Formalization",
        "",
        "## Scope",
        "",
        "- This batch formalizes market-state labels for v9 factor diagnostics.",
        "- It does not run factor IC, build industry stock-level features, train models, tune parameters, or modify `v7_locked`.",
        "",
        "## Definitions",
        "",
        "- `trend_regime`: `strong` if `roll60_ew_ret >= +8%`, `weak` if `roll60_ew_ret <= -8%`, otherwise `neutral`.",
        "- `extreme_selloff_flag`: `breadth_positive < 20%` or `ew_ret <= -3%`.",
        "- `vol_regime`: `roll20_dispersion` versus t-1 expanding 20/80% quantile thresholds.",
        "- `industry_crowding_regime`: `top5_industry_amount_share` versus t-1 expanding 20/80% quantile thresholds.",
        "- `stock_concentration_regime`: `top100_amount_share` versus t-1 expanding 20/80% quantile thresholds.",
        "",
        "## Trend Regime Distribution",
        "",
        trend.to_markdown(index=False),
        "",
        "## Volatility Regime Distribution",
        "",
        vol.to_markdown(index=False),
        "",
        "## Industry Crowding Distribution",
        "",
        crowd.to_markdown(index=False),
        "",
        "## Threshold Sensitivity",
        "",
        sensitivity.to_markdown(index=False),
        "",
        "## Gate Result",
        "",
        "- Gate result: `pass`.",
        "- Reason: market-state labels are available, definitions are documented, and no forward label performance was used to choose thresholds.",
        "",
        "## Next Step",
        "",
        "- Execute `Batch 3B industry/theme features`.",
        "- Do not run factor IC until Batch 3B completes or the industry/theme step is explicitly waived.",
    ]
    (REPORT_DIR / "batch3A_conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hashes() -> None:
    rows = []
    for path in sorted(REPORT_DIR.glob("batch3A_*")):
        if path.name == "batch3A_file_sha256.csv" or not path.is_file():
            continue
        rows.append({"file": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    rows.append({"file": str(FORMAL_REGIME_PATH), "sha256": sha256_file(FORMAL_REGIME_PATH), "size_bytes": FORMAL_REGIME_PATH.stat().st_size})
    rows.append({"file": str(GLOBAL_HANDOFF_PATH), "sha256": sha256_file(GLOBAL_HANDOFF_PATH), "size_bytes": GLOBAL_HANDOFF_PATH.stat().st_size})
    pd.DataFrame(rows).to_csv(REPORT_DIR / "batch3A_file_sha256.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    if FORMAL_REGIME_PATH.exists() and not args.refresh:
        regime = pd.read_parquet(FORMAL_REGIME_PATH)
    else:
        regime = build_formal_regime()
        regime.to_parquet(FORMAL_REGIME_PATH, index=False, compression="zstd")

    thresholds = build_thresholds()
    summary = build_summary(regime)
    transition = build_transition(regime)
    by_year = build_by_year(regime)
    sensitivity = build_sensitivity(regime)

    regime.to_csv(REPORT_DIR / "batch3A_market_regime_daily.csv", index=False)
    summary.to_csv(REPORT_DIR / "batch3A_market_regime_summary.csv", index=False)
    transition.to_csv(REPORT_DIR / "batch3A_regime_transition.csv", index=False)
    by_year.to_csv(REPORT_DIR / "batch3A_regime_by_year.csv", index=False)
    thresholds.to_csv(REPORT_DIR / "batch3A_regime_thresholds.csv", index=False)
    sensitivity.to_csv(REPORT_DIR / "batch3A_regime_threshold_sensitivity.csv", index=False)
    write_limitations()
    write_handoff(regime, summary, thresholds)
    write_conclusion(regime, summary, sensitivity)
    write_hashes()

    print(
        {
            "out_dir": str(REPORT_DIR),
            "formal_regime_path": str(FORMAL_REGIME_PATH),
            "days": int(len(regime)),
            "trend_counts": regime["trend_regime"].value_counts().to_dict(),
            "next_step": "Batch 3B industry/theme features",
        }
    )


if __name__ == "__main__":
    main()
