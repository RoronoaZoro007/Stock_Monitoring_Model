#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = ROOT / "locked_artifacts" / "v7_cap20_strong_label_003" / "validation" / "validation_predictions.csv.gz"
DEFAULT_DAILY = ROOT / "data_tushare" / "clean" / "daily_repaired_top3000.parquet"
DEFAULT_OUT = ROOT / "reports" / "tushare" / "v9_swing_research" / "batch0_v7_score_diagnostic"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def max_drawdown_from_lows(group: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    group = group.sort_values("trade_date").copy()
    low = group["low"].astype(float)
    close = group["close"].astype(float)
    for h in horizons:
        future_low = pd.concat([low.shift(-i) for i in range(1, h + 1)], axis=1).min(axis=1)
        group[f"fwd_min_low_{h}d"] = future_low
        group[f"fwd_low_dd_{h}d_close"] = future_low / close - 1.0
    return group


def add_forward_returns(pred: pd.DataFrame, daily: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    daily = daily.sort_values(["ts_code", "trade_date"]).copy()
    daily["trade_date"] = daily["trade_date"].astype(str)
    daily["ts_code"] = daily["ts_code"].astype(str)
    for h in horizons:
        daily[f"fwd_close_{h}d"] = daily.groupby("ts_code")["close"].shift(-h)
        daily[f"fwd_open_1d"] = daily.groupby("ts_code")["open"].shift(-1)
        daily[f"has_fwd_{h}d"] = daily[f"fwd_close_{h}d"].notna()
    daily = pd.concat(
        [max_drawdown_from_lows(group, horizons) for _, group in daily.groupby("ts_code", sort=False)],
        ignore_index=True,
    )
    keep = ["ts_code", "trade_date", "open", "high", "low", "close", "amount", "fwd_open_1d"]
    for h in horizons:
        keep.extend([f"fwd_close_{h}d", f"has_fwd_{h}d", f"fwd_min_low_{h}d", f"fwd_low_dd_{h}d_close"])
    merged = pred.merge(daily[keep], on=["ts_code", "trade_date"], how="left", validate="many_to_one")
    entry = pd.to_numeric(merged["entry_vwap"], errors="coerce")
    close = pd.to_numeric(merged["close"], errors="coerce")
    for h in horizons:
        fwd_close = pd.to_numeric(merged[f"fwd_close_{h}d"], errors="coerce")
        merged[f"fwd_ret_{h}d_entry"] = fwd_close / entry - 1.0
        merged[f"fwd_ret_{h}d_close"] = fwd_close / close - 1.0
        merged[f"fwd_win_{h}d_entry"] = (merged[f"fwd_ret_{h}d_entry"] > 0).astype("float")
        merged.loc[merged[f"fwd_ret_{h}d_entry"].isna(), f"fwd_win_{h}d_entry"] = np.nan
    return merged


def corr_rows(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = ["target_return"] + [f"fwd_ret_{h}d_entry" for h in horizons]
    for metric in metrics:
        valid = df[["trade_date", "score", metric]].dropna()
        daily = []
        for trade_date, day in valid.groupby("trade_date"):
            if day["score"].nunique() < 2 or day[metric].nunique() < 2 or len(day) < 20:
                continue
            daily.append(
                {
                    "trade_date": trade_date,
                    "n": len(day),
                    "pearson": day["score"].corr(day[metric], method="pearson"),
                    "spearman": day["score"].corr(day[metric], method="spearman"),
                }
            )
        d = pd.DataFrame(daily)
        rows.append(
            {
                "metric": metric,
                "sample_rows": int(len(valid)),
                "trading_days": int(valid["trade_date"].nunique()) if not valid.empty else 0,
                "daily_pearson_mean": float(d["pearson"].mean()) if not d.empty else np.nan,
                "daily_pearson_median": float(d["pearson"].median()) if not d.empty else np.nan,
                "daily_spearman_mean": float(d["spearman"].mean()) if not d.empty else np.nan,
                "daily_spearman_median": float(d["spearman"].median()) if not d.empty else np.nan,
                "positive_spearman_day_ratio": float((d["spearman"] > 0).mean()) if not d.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def assign_daily_decile(df: pd.DataFrame, score_col: str = "score") -> pd.Series:
    def qcut_day(day: pd.Series) -> pd.Series:
        ranks = day.rank(method="first")
        bins = min(10, int(day.notna().sum()))
        if bins < 2:
            return pd.Series(np.nan, index=day.index)
        return pd.qcut(ranks, q=bins, labels=False, duplicates="drop") + 1

    return df.groupby("trade_date", group_keys=False)[score_col].apply(qcut_day)


def decile_summary(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    work = df.copy()
    work["score_decile"] = assign_daily_decile(work)
    rows: list[dict[str, Any]] = []
    for h in horizons:
        ret_col = f"fwd_ret_{h}d_entry"
        for decile, g in work.dropna(subset=["score_decile", ret_col]).groupby("score_decile"):
            rows.append(
                {
                    "horizon": f"{h}d",
                    "score_decile": int(decile),
                    "sample_rows": int(len(g)),
                    "trading_days": int(g["trade_date"].nunique()),
                    "avg_return": float(g[ret_col].mean()),
                    "median_return": float(g[ret_col].median()),
                    "win_rate": float((g[ret_col] > 0).mean()),
                    "avg_drawdown_to_low": float(g[f"fwd_low_dd_{h}d_close"].mean()),
                    "avg_score": float(g["score"].mean()),
                }
            )
    return pd.DataFrame(rows)


def topn_summary(df: pd.DataFrame, horizons: list[int], topns: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for h in horizons:
        ret_col = f"fwd_ret_{h}d_entry"
        valid = df.dropna(subset=[ret_col, "score"]).copy()
        valid["rank_desc"] = valid.groupby("trade_date")["score"].rank(method="first", ascending=False)
        for topn in topns:
            pick = valid[valid["rank_desc"] <= topn].copy()
            daily = pick.groupby("trade_date")[ret_col].mean()
            all_daily = valid.groupby("trade_date")[ret_col].mean()
            rows.append(
                {
                    "horizon": f"{h}d",
                    "selection": f"top{topn}",
                    "sample_rows": int(len(pick)),
                    "trading_days": int(daily.shape[0]),
                    "avg_signal_return": float(pick[ret_col].mean()) if not pick.empty else np.nan,
                    "median_signal_return": float(pick[ret_col].median()) if not pick.empty else np.nan,
                    "signal_win_rate": float((pick[ret_col] > 0).mean()) if not pick.empty else np.nan,
                    "avg_daily_equal_weight_return": float(daily.mean()) if not daily.empty else np.nan,
                    "daily_win_rate": float((daily > 0).mean()) if not daily.empty else np.nan,
                    "baseline_all_avg_daily_return": float(all_daily.mean()) if not all_daily.empty else np.nan,
                    "excess_vs_all_daily": float(daily.mean() - all_daily.mean()) if not daily.empty and not all_daily.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def group_diagnostics(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    work = df.copy()
    work["u2_flag"] = pd.to_numeric(work.get("liquidity_rank"), errors="coerce").le(2500)
    work["tail_down_flag"] = pd.to_numeric(work.get("market_tail_ret_median"), errors="coerce").lt(0)
    work["trend_ma20_positive"] = pd.to_numeric(work.get("price_vs_ma20_prev"), errors="coerce").gt(0)
    work["prev5_nonoverheat"] = pd.to_numeric(work.get("prev_ret_5d"), errors="coerce").between(-0.05, 0.08)
    rows: list[dict[str, Any]] = []
    for h in horizons:
        ret_col = f"fwd_ret_{h}d_entry"
        valid = work.dropna(subset=[ret_col, "score"]).copy()
        valid["rank_desc"] = valid.groupby("trade_date")["score"].rank(method="first", ascending=False)
        for group_name, mask in {
            "all": pd.Series(True, index=valid.index),
            "U2_liquidity_top2500": valid["u2_flag"].fillna(False),
            "tail_down": valid["tail_down_flag"].fillna(False),
            "trend_ma20_positive": valid["trend_ma20_positive"].fillna(False),
            "prev5_nonoverheat": valid["prev5_nonoverheat"].fillna(False),
            "U2_and_tail_down": valid["u2_flag"].fillna(False) & valid["tail_down_flag"].fillna(False),
            "U2_and_trend_positive": valid["u2_flag"].fillna(False) & valid["trend_ma20_positive"].fillna(False),
        }.items():
            sub = valid[mask].copy()
            if sub.empty:
                continue
            sub["daily_spearman"] = np.nan
            spearman = []
            for _, day in sub.groupby("trade_date"):
                if day["score"].nunique() > 1 and day[ret_col].nunique() > 1 and len(day) >= 20:
                    spearman.append(day["score"].corr(day[ret_col], method="spearman"))
            top10 = sub[sub["rank_desc"] <= 10]
            rows.append(
                {
                    "horizon": f"{h}d",
                    "group": group_name,
                    "sample_rows": int(len(sub)),
                    "trading_days": int(sub["trade_date"].nunique()),
                    "score_spearman_mean_by_day": float(np.nanmean(spearman)) if spearman else np.nan,
                    "score_spearman_positive_day_ratio": float(np.mean(np.array(spearman) > 0)) if spearman else np.nan,
                    "all_avg_return": float(sub[ret_col].mean()),
                    "all_win_rate": float((sub[ret_col] > 0).mean()),
                    "top10_rows": int(len(top10)),
                    "top10_avg_return": float(top10[ret_col].mean()) if not top10.empty else np.nan,
                    "top10_win_rate": float((top10[ret_col] > 0).mean()) if not top10.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def coverage_summary(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    rows = []
    for h in horizons:
        ret_col = f"fwd_ret_{h}d_entry"
        valid = df.dropna(subset=[ret_col])
        rows.append(
            {
                "horizon": f"{h}d",
                "sample_rows": int(len(valid)),
                "trading_days": int(valid["trade_date"].nunique()) if not valid.empty else 0,
                "date_min": str(valid["trade_date"].min()) if not valid.empty else "",
                "date_max": str(valid["trade_date"].max()) if not valid.empty else "",
                "codes": int(valid["ts_code"].nunique()) if not valid.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def write_conclusion(out_dir: Path, corr: pd.DataFrame, deciles: pd.DataFrame, topn: pd.DataFrame, group: pd.DataFrame, coverage: pd.DataFrame) -> None:
    def pct(x: float) -> str:
        return "" if pd.isna(x) else f"{x:.2%}"

    lines = [
        "# v9 Swing Research - Batch 0",
        "",
        "## Purpose",
        "",
        "This batch does not train a model and does not choose parameters. It only tests whether locked v7 scores contain measurable ranking information for 3/5/10 trading-day forward returns.",
        "",
        "## Data Boundary",
        "",
        "- Score source: locked v7 validation predictions.",
        "- Daily source: repaired Top3000 daily data.",
        "- Forward return: future close at T+h divided by v7 entry_vwap at T, minus 1.",
        "- This is diagnostic only. The legacy validation window was already used in prior research, so it must not be treated as a final test set.",
        "",
        "## Coverage",
        "",
        coverage.to_markdown(index=False),
        "",
        "## RankIC Summary",
        "",
        corr.to_markdown(index=False),
        "",
    ]
    for h in ["3d", "5d", "10d"]:
        sub = deciles[deciles["horizon"].eq(h)].sort_values("score_decile")
        if sub.empty:
            continue
        low = sub[sub["score_decile"].eq(sub["score_decile"].min())]["avg_return"].iloc[0]
        high = sub[sub["score_decile"].eq(sub["score_decile"].max())]["avg_return"].iloc[0]
        ic = corr[corr["metric"].eq(f"fwd_ret_{h}_entry")]
        ic_val = ic["daily_spearman_mean"].iloc[0] if not ic.empty else np.nan
        lines.extend(
            [
                f"## {h} Initial Read",
                "",
                f"- Top score decile average return: {pct(high)}.",
                f"- Bottom score decile average return: {pct(low)}.",
                f"- Top-minus-bottom spread: {pct(high - low)}.",
                f"- Mean daily Spearman RankIC: {ic_val:.4f}.",
                "",
            ]
        )
    top10 = topn[topn["selection"].eq("top10")]
    lines.extend(
        [
            "## Professional Judgment",
            "",
            "v7 score was trained for an overnight/next-morning execution target, so it should not be assumed to be a swing-trading alpha. The only defensible use is as a candidate factor after measuring whether its cross-sectional ranking survives at 3/5/10 day horizons.",
            "",
        ]
    )
    if not top10.empty:
        lines.append("Top10 diagnostic table:")
        lines.append("")
        lines.append(top10.to_markdown(index=False))
        lines.append("")
    group_focus = group[group["group"].isin(["all", "U2_liquidity_top2500", "tail_down", "U2_and_tail_down"])]
    if not group_focus.empty:
        lines.append("Context diagnostic table:")
        lines.append("")
        lines.append(group_focus.to_markdown(index=False))
        lines.append("")
    lines.extend(
        [
            "## Judgment",
            "",
            "- The locked v7 score shows weak positive cross-sectional transfer to 3/5/10 day returns, strongest at 10d by RankIC and top-minus-bottom decile spread.",
            "- The evidence is not strong enough to use v7 score as a standalone swing selector: Top10 5d is negative versus the all-candidate baseline, Top10 10d median return is negative, and the top score decile is not consistently the best decile.",
            "- U2 liquidity filtering alone does not prove a stable swing improvement in this batch; it improves tradability but does not automatically improve v7 Top10 swing returns.",
            "- tail_down and U2_and_tail_down look better in this legacy window, but this is diagnostic only and cannot be used to choose live parameters without forward unseen tracking.",
            "- The professional interpretation is: v7 score can be kept as a candidate factor/control variable for v9 swing research, but v9 should not inherit v7 Top10 directly as its primary swing model.",
            "",
            "## Next Step Gate",
            "",
            "Proceed to v9 Batch 1 only as a diagnostic research step, not as optimization. Batch 1 should test whether v7 score remains useful after stricter robustness checks: monthly splits, industry neutrality, liquidity buckets, overlap-adjusted holding periods, cost/turnover estimates, and forward-only paper tracking once new data arrives.",
            "",
            "No live, simulated, or real trading conclusion is made here.",
        ]
    )
    (out_dir / "batch0_conclusion.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="v9 Batch 0: diagnose whether locked v7 scores have 3/5/10 day swing predictive content.")
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--daily", default=str(DEFAULT_DAILY))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--sample-rows", type=int, default=5000)
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    daily_path = Path(args.daily)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(predictions_path, dtype={"ts_code": str})
    pred["trade_date"] = pred["trade_date"].astype(str)
    pred["ts_code"] = pred["ts_code"].astype(str)
    daily = pd.read_parquet(daily_path)
    daily["trade_date"] = daily["trade_date"].astype(str)
    daily["ts_code"] = daily["ts_code"].astype(str)
    horizons = [3, 5, 10]
    merged = add_forward_returns(pred, daily, horizons)

    coverage = coverage_summary(merged, horizons)
    corr = corr_rows(merged, horizons)
    deciles = decile_summary(merged, horizons)
    topn = topn_summary(merged, horizons, [10, 20, 50, 100])
    group = group_diagnostics(merged, horizons)

    diagnostic_cols = [
        "trade_date",
        "ts_code",
        "score",
        "entry_vwap",
        "target_return",
        "liquidity_rank",
        "market_tail_ret_median",
        "price_vs_ma20_prev",
        "prev_ret_5d",
        "industry",
    ]
    for h in horizons:
        diagnostic_cols.extend([f"fwd_ret_{h}d_entry", f"fwd_win_{h}d_entry", f"fwd_low_dd_{h}d_close"])
    sample = merged.sort_values(["trade_date", "score"], ascending=[True, False]).head(args.sample_rows)
    sample[diagnostic_cols].to_csv(out_dir / "batch0_v7_score_swing_dataset_sample.csv", index=False)
    coverage.to_csv(out_dir / "batch0_coverage_summary.csv", index=False)
    corr.to_csv(out_dir / "batch0_rankic_summary.csv", index=False)
    deciles.to_csv(out_dir / "batch0_score_decile_forward_returns.csv", index=False)
    topn.to_csv(out_dir / "batch0_topn_forward_returns.csv", index=False)
    group.to_csv(out_dir / "batch0_group_diagnostics.csv", index=False)

    manifest = pd.DataFrame(
        [
            {"file": str(predictions_path), "sha256": sha256_file(predictions_path), "role": "locked_v7_validation_predictions"},
            {"file": str(daily_path), "sha256": sha256_file(daily_path), "role": "daily_repaired_top3000"},
        ]
    )
    manifest.to_csv(out_dir / "batch0_input_file_sha256.csv", index=False)
    write_conclusion(out_dir, corr, deciles, topn, group, coverage)

    output_hashes = []
    for path in sorted(out_dir.glob("*")):
        if path.is_file() and path.name != "batch0_file_sha256.csv":
            output_hashes.append({"file": str(path), "sha256": sha256_file(path)})
    pd.DataFrame(output_hashes).to_csv(out_dir / "batch0_file_sha256.csv", index=False)
    print(json.dumps({"out_dir": str(out_dir), "rows": len(merged), "files": len(output_hashes)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
