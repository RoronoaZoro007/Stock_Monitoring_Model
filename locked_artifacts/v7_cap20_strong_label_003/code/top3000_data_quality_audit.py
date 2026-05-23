#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data_tushare" / "manifests" / "download_tasks.sqlite3"
RANK_PATH = ROOT / "data_tushare" / "manifests" / "liquid_top3000_20251120_20260213.csv"
MINUTE_ROOT = ROOT / "data_tushare" / "raw" / "stk_mins" / "freq=5min"
DAILY_ROOT = ROOT / "data_tushare" / "raw" / "daily"
OUT_DIR = ROOT / "reports" / "tushare" / "full_top3000_data_quality"


EXPECTED_TIMES = (
    [f"09:{m:02d}" for m in range(30, 60, 5)]
    + [f"10:{m:02d}" for m in range(0, 60, 5)]
    + ["11:00", "11:05", "11:10", "11:15", "11:20", "11:25", "11:30"]
    + [f"13:{m:02d}" for m in range(5, 60, 5)]
    + [f"14:{m:02d}" for m in range(0, 60, 5)]
    + ["15:00"]
)
EXPECTED_TIME_SET = set(EXPECTED_TIMES)
MODEL_REQUIRED_TIMES = {"09:35", "13:05", "14:20", "14:30", "14:45", "14:50", "14:55"}
NEXT_EXIT_TIMES = {"09:35", "09:40", "09:45", "09:50", "10:00", "10:05", "10:30"}


@dataclass
class SymbolAudit:
    ts_code: str
    liquidity_rank: int
    files: int
    file_rows: int
    raw_rows_read: int
    dedup_rows: int
    duplicate_rows: int
    duplicate_conflict_keys: int
    invalid_datetime_rows: int
    invalid_ohlc_rows: int
    nonpositive_price_rows: int
    negative_volume_amount_rows: int
    outside_session_rows: int
    minute_date_min: str | None
    minute_date_max: str | None
    daily_expected_days: int
    minute_days: int
    full_49bar_days: int
    partial_days: int
    days_missing_model_bars: int
    days_missing_exit_bars: int
    missing_daily_days: int
    extra_minute_days_without_daily: int
    close_mismatch_gt_1bp: int
    high_mismatch_gt_5bp: int
    low_mismatch_gt_5bp: int
    vol_scaled_mismatch_gt_5pct: int
    amount_scaled_mismatch_gt_5pct: int
    median_vol_scale_ratio: float | None
    median_amount_scale_ratio: float | None


def read_rank(path: Path) -> pd.DataFrame:
    rank = pd.read_csv(path, dtype={"ts_code": str})
    rank["liquidity_rank"] = pd.to_numeric(rank["liquidity_rank"], errors="coerce").astype(int)
    return rank.sort_values("liquidity_rank").drop_duplicates("ts_code", keep="first")


def read_daily() -> pd.DataFrame:
    paths = sorted(DAILY_ROOT.glob("*.parquet"))
    if not paths:
        raise SystemExit(f"missing daily parquet files under {DAILY_ROOT}")
    daily = pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)
    for col in ["open", "high", "low", "close", "vol", "amount"]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    daily["trade_date"] = daily["trade_date"].astype(str)
    return daily.sort_values(["ts_code", "trade_date"]).drop_duplicates(["ts_code", "trade_date"], keep="last")


def load_task_summary() -> dict[str, Any]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        select group_name, status, count(*) tasks, count(distinct ts_code) symbols, coalesce(sum(row_count),0) rows
        from tasks
        where api_name='stk_mins'
        group by group_name, status
        order by group_name, status
        """
    ).fetchall()
    final_rows = con.execute(
        """
        select status, count(*) tasks, count(distinct ts_code) symbols, coalesce(sum(row_count),0) rows
        from tasks
        where group_name='minutes_replanned_162' and api_name='stk_mins'
        group by status
        order by status
        """
    ).fetchall()
    done_symbols = con.execute(
        """
        select count(distinct ts_code)
        from tasks
        where group_name='minutes_replanned_162' and api_name='stk_mins' and status='done'
        """
    ).fetchone()[0]
    con.close()
    return {
        "all_stk_mins_task_summary": [dict(row) for row in rows],
        "minutes_replanned_162_summary": [dict(row) for row in final_rows],
        "minutes_replanned_162_done_symbols": int(done_symbols),
    }


def parquet_metadata(path: Path) -> dict[str, Any]:
    meta = pq.ParquetFile(path).metadata
    return {
        "path": str(path),
        "rows": int(meta.num_rows),
        "columns": list(pq.ParquetFile(path).schema.names),
        "size_bytes": int(path.stat().st_size),
    }


def symbol_paths(ts_code: str) -> list[Path]:
    return sorted((MINUTE_ROOT / f"ts_code={ts_code}").glob("*.parquet"))


def pct_mismatch(a: pd.Series, b: pd.Series) -> pd.Series:
    base = b.abs().replace(0, np.nan)
    return (a - b).abs() / base


def audit_symbol(ts_code: str, liquidity_rank: int, paths: list[Path], daily_by_symbol: dict[str, pd.DataFrame]) -> tuple[SymbolAudit, list[dict[str, Any]], list[dict[str, Any]]]:
    file_rows = 0
    schemas: set[tuple[str, ...]] = set()
    for path in paths:
        meta = parquet_metadata(path)
        file_rows += meta["rows"]
        schemas.add(tuple(meta["columns"]))

    if not paths:
        audit = SymbolAudit(
            ts_code=ts_code,
            liquidity_rank=liquidity_rank,
            files=0,
            file_rows=0,
            raw_rows_read=0,
            dedup_rows=0,
            duplicate_rows=0,
            duplicate_conflict_keys=0,
            invalid_datetime_rows=0,
            invalid_ohlc_rows=0,
            nonpositive_price_rows=0,
            negative_volume_amount_rows=0,
            outside_session_rows=0,
            minute_date_min=None,
            minute_date_max=None,
            daily_expected_days=0,
            minute_days=0,
            full_49bar_days=0,
            partial_days=0,
            days_missing_model_bars=0,
            days_missing_exit_bars=0,
            missing_daily_days=0,
            extra_minute_days_without_daily=0,
            close_mismatch_gt_1bp=0,
            high_mismatch_gt_5bp=0,
            low_mismatch_gt_5bp=0,
            vol_scaled_mismatch_gt_5pct=0,
            amount_scaled_mismatch_gt_5pct=0,
            median_vol_scale_ratio=None,
            median_amount_scale_ratio=None,
        )
        return audit, [], []

    df = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    raw_rows = len(df)
    for col in ["open", "high", "low", "close", "vol", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["dt"] = pd.to_datetime(df["trade_time"], errors="coerce")
    invalid_dt = int(df["dt"].isna().sum())
    df_valid_dt = df[df["dt"].notna()].copy()
    df_valid_dt["trade_date"] = df_valid_dt["dt"].dt.strftime("%Y%m%d")
    df_valid_dt["bar_time"] = df_valid_dt["dt"].dt.strftime("%H:%M")

    duplicate_rows = int(df_valid_dt.duplicated(["trade_time"], keep=False).sum())
    duplicate_conflict_keys = 0
    if duplicate_rows:
        dup = df_valid_dt[df_valid_dt.duplicated(["trade_time"], keep=False)].copy()
        nunique = dup.groupby("trade_time")[["open", "high", "low", "close", "vol", "amount"]].nunique(dropna=False)
        duplicate_conflict_keys = int((nunique.max(axis=1) > 1).sum())

    nonpositive_price = int((df_valid_dt[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    invalid_ohlc = int(
        (
            (df_valid_dt["high"] < df_valid_dt[["open", "close", "low"]].max(axis=1))
            | (df_valid_dt["low"] > df_valid_dt[["open", "close", "high"]].min(axis=1))
        ).sum()
    )
    neg_vol_amt = int(((df_valid_dt["vol"] < 0) | (df_valid_dt["amount"] < 0)).sum())
    outside_session = int((~df_valid_dt["bar_time"].isin(EXPECTED_TIME_SET)).sum())

    dedup = df_valid_dt.sort_values("dt").drop_duplicates(["trade_date", "bar_time"], keep="last").copy()
    date_min = str(dedup["trade_date"].min()) if not dedup.empty else None
    date_max = str(dedup["trade_date"].max()) if not dedup.empty else None
    day_counts = dedup.groupby("trade_date")["bar_time"].nunique()
    full_days = int(day_counts.eq(len(EXPECTED_TIMES)).sum())
    partial_days = int(day_counts.lt(len(EXPECTED_TIMES)).sum())

    missing_model = 0
    missing_exit = 0
    for _, group in dedup.groupby("trade_date"):
        bars = set(group["bar_time"])
        if not MODEL_REQUIRED_TIMES.issubset(bars):
            missing_model += 1
        if not NEXT_EXIT_TIMES.issubset(bars):
            missing_exit += 1

    daily = daily_by_symbol.get(ts_code, pd.DataFrame()).copy()
    mismatch_examples: list[dict[str, Any]] = []
    if not daily.empty and date_min and date_max:
        daily_window = daily[(daily["trade_date"] >= date_min) & (daily["trade_date"] <= date_max)].copy()
    else:
        daily_window = pd.DataFrame(columns=daily.columns)

    minute_dates = set(dedup["trade_date"].unique())
    daily_dates = set(daily_window["trade_date"].unique())
    missing_daily_days = len(daily_dates - minute_dates)
    extra_minute_days = len(minute_dates - daily_dates)

    close_bad = high_bad = low_bad = vol_bad = amount_bad = 0
    vol_scale_median = amount_scale_median = None
    if not daily_window.empty and not dedup.empty:
        per_day = dedup.groupby("trade_date").agg(
            minute_close=("close", "last"),
            minute_high=("high", "max"),
            minute_low=("low", "min"),
            minute_vol=("vol", "sum"),
            minute_amount=("amount", "sum"),
            bars=("bar_time", "nunique"),
        )
        merged = per_day.reset_index().merge(
            daily_window[["trade_date", "close", "high", "low", "vol", "amount"]],
            on="trade_date",
            how="inner",
        )
        if not merged.empty:
            close_m = pct_mismatch(merged["minute_close"], merged["close"])
            high_m = pct_mismatch(merged["minute_high"], merged["high"])
            low_m = pct_mismatch(merged["minute_low"], merged["low"])
            full = merged[merged["bars"].eq(len(EXPECTED_TIMES))].copy()
            if not full.empty:
                full["vol_scale_ratio"] = full["minute_vol"] / (full["vol"] * 100.0)
                full["amount_scale_ratio"] = full["minute_amount"] / (full["amount"] * 1000.0)
                vol_scale_median = float(full["vol_scale_ratio"].median())
                amount_scale_median = float(full["amount_scale_ratio"].median())
                vol_bad = int((full["vol_scale_ratio"] - 1.0).abs().gt(0.05).sum())
                amount_bad = int((full["amount_scale_ratio"] - 1.0).abs().gt(0.05).sum())
            close_bad = int(close_m.gt(0.0001).sum())
            high_bad = int(high_m.gt(0.0005).sum())
            low_bad = int(low_m.gt(0.0005).sum())
            bad = merged[close_m.gt(0.0001) | high_m.gt(0.0005) | low_m.gt(0.0005)].head(5)
            for _, row in bad.iterrows():
                mismatch_examples.append(
                    {
                        "ts_code": ts_code,
                        "trade_date": row["trade_date"],
                        "minute_close": row["minute_close"],
                        "daily_close": row["close"],
                        "minute_high": row["minute_high"],
                        "daily_high": row["high"],
                        "minute_low": row["minute_low"],
                        "daily_low": row["low"],
                        "bars": int(row["bars"]),
                    }
                )

    issue_examples: list[dict[str, Any]] = []
    if invalid_ohlc or nonpositive_price or neg_vol_amt or outside_session or duplicate_conflict_keys:
        sample = df_valid_dt[
            ((df_valid_dt[["open", "high", "low", "close"]] <= 0).any(axis=1))
            | (
                (df_valid_dt["high"] < df_valid_dt[["open", "close", "low"]].max(axis=1))
                | (df_valid_dt["low"] > df_valid_dt[["open", "close", "high"]].min(axis=1))
            )
            | (df_valid_dt["vol"] < 0)
            | (df_valid_dt["amount"] < 0)
            | (~df_valid_dt["bar_time"].isin(EXPECTED_TIME_SET))
        ].head(5)
        for _, row in sample.iterrows():
            issue_examples.append(
                {
                    "ts_code": ts_code,
                    "trade_time": row.get("trade_time"),
                    "open": row.get("open"),
                    "high": row.get("high"),
                    "low": row.get("low"),
                    "close": row.get("close"),
                    "vol": row.get("vol"),
                    "amount": row.get("amount"),
                    "bar_time": row.get("bar_time"),
                }
            )

    audit = SymbolAudit(
        ts_code=ts_code,
        liquidity_rank=liquidity_rank,
        files=len(paths),
        file_rows=int(file_rows),
        raw_rows_read=int(raw_rows),
        dedup_rows=int(len(dedup)),
        duplicate_rows=duplicate_rows,
        duplicate_conflict_keys=duplicate_conflict_keys,
        invalid_datetime_rows=invalid_dt,
        invalid_ohlc_rows=invalid_ohlc,
        nonpositive_price_rows=nonpositive_price,
        negative_volume_amount_rows=neg_vol_amt,
        outside_session_rows=outside_session,
        minute_date_min=date_min,
        minute_date_max=date_max,
        daily_expected_days=int(len(daily_window)),
        minute_days=int(len(minute_dates)),
        full_49bar_days=full_days,
        partial_days=partial_days,
        days_missing_model_bars=missing_model,
        days_missing_exit_bars=missing_exit,
        missing_daily_days=int(missing_daily_days),
        extra_minute_days_without_daily=int(extra_minute_days),
        close_mismatch_gt_1bp=close_bad,
        high_mismatch_gt_5bp=high_bad,
        low_mismatch_gt_5bp=low_bad,
        vol_scaled_mismatch_gt_5pct=vol_bad,
        amount_scaled_mismatch_gt_5pct=amount_bad,
        median_vol_scale_ratio=vol_scale_median,
        median_amount_scale_ratio=amount_scale_median,
    )
    return audit, issue_examples, mismatch_examples


def _sum(df: pd.DataFrame, col: str) -> int:
    return int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum()) if col in df else 0


def write_markdown(summary: dict[str, Any], symbol_df: pd.DataFrame, out_dir: Path) -> None:
    top_issues = symbol_df[
        [
            "ts_code",
            "liquidity_rank",
            "files",
            "dedup_rows",
            "duplicate_rows",
            "duplicate_conflict_keys",
            "invalid_ohlc_rows",
            "partial_days",
            "missing_daily_days",
            "close_mismatch_gt_1bp",
        ]
    ].copy()
    top_issues["issue_score"] = (
        top_issues["duplicate_conflict_keys"]
        + top_issues["invalid_ohlc_rows"]
        + top_issues["missing_daily_days"]
        + top_issues["close_mismatch_gt_1bp"]
    )
    top_issues = top_issues.sort_values(["issue_score", "partial_days"], ascending=False).head(20)
    text = f"""# Top3000 原始数据质量审计

## 审计范围

- 股票池：流动性 Top3000，来自 `{RANK_PATH.name}`。
- 分钟线目录：`data_tushare/raw/stk_mins/freq=5min`。
- 审计口径：以 Top3000 股票目录中的 parquet 为落地样本；任务库用于校验下载状态和历史批次来源。
- 原始数据未修改。

## 关键结论

- Top3000 股票目录覆盖：{summary['coverage']['top3000_symbols_with_minute_dir']} / {summary['coverage']['top3000_symbols']}。
- Top3000 parquet 文件数：{summary['files']['files']}，文件元数据行数：{summary['files']['rows_from_parquet_metadata']:,}。
- 去重后分钟行数：{summary['content']['dedup_rows']:,}。
- 重复行：{summary['content']['duplicate_rows']:,}，其中价格/量额冲突时间点：{summary['content']['duplicate_conflict_keys']:,}。
- 非法 OHLC 行：{summary['content']['invalid_ohlc_rows']:,}；非正价格行：{summary['content']['nonpositive_price_rows']:,}；负成交量/额行：{summary['content']['negative_volume_amount_rows']:,}。
- 非标准 5 分钟交易时点行：{summary['content']['outside_session_rows']:,}。
- 与日线收盘价偏离超过 1bp 的交易日：{summary['daily_cross_check']['close_mismatch_gt_1bp']:,}。
- 完整 49 根 5 分钟 bar 的股票-交易日：{summary['calendar']['full_49bar_days']:,}；部分 bar 交易日：{summary['calendar']['partial_days']:,}。

## 下载任务依据

`minutes_replanned_162` 是最后一次补齐批次，它本身覆盖 {summary['task_db']['minutes_replanned_162_done_symbols']} 只股票；Top3000 中缺失于该批次的高流动性股票已由早期 `minutes` 批次覆盖，所以不能只按最终批次判断股票覆盖。

## 需要重点处理的问题

"""
    if top_issues.empty:
        text += "未发现需要按股票聚焦处理的硬错误。\n"
    else:
        text += top_issues.to_markdown(index=False)
        text += "\n"
    text += """
## 清洗建议

1. 原始 parquet 保留不动，后续建模读取时先按 `ts_code + trade_time` 去重，并保留最后落地版本。
2. 非法 OHLC、非正价格、负成交量/额、非标准交易时点行应在建模样本生成层剔除；如果计数为 0，则不需要生成重复清洗副本。
3. 对缺少关键尾盘/次日上午 bar 的股票-交易日，不做补值，直接不生成该日训练样本；这是停牌、临停、上市初期或数据缺口场景下更稳健的处理。
4. 与日线价格明显不一致的日期进入排查清单；若集中在除权除息/复权断层，应在特征样本层剔除对应隔夜标签，避免模型学习到不可交易收益。
"""
    (out_dir / "raw_audit.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Top3000 local Tushare minute data without mutating raw files")
    parser.add_argument("--rank-file", default=str(RANK_PATH))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--limit-symbols", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rank = read_rank(Path(args.rank_file))
    if args.limit_symbols:
        rank = rank.head(args.limit_symbols).copy()
    daily = read_daily()
    daily_by_symbol = {code: group for code, group in daily.groupby("ts_code")}

    task_summary = load_task_summary()
    symbol_rows: list[dict[str, Any]] = []
    issue_examples: list[dict[str, Any]] = []
    mismatch_examples: list[dict[str, Any]] = []
    schema_counter: dict[str, int] = {}

    for idx, row in enumerate(rank.itertuples(index=False), start=1):
        ts_code = str(row.ts_code)
        paths = symbol_paths(ts_code)
        for path in paths:
            cols = tuple(parquet_metadata(path)["columns"])
            schema_counter[json.dumps(cols, ensure_ascii=False)] = schema_counter.get(json.dumps(cols, ensure_ascii=False), 0) + 1
        audit, issues, mismatches = audit_symbol(ts_code, int(row.liquidity_rank), paths, daily_by_symbol)
        symbol_rows.append(asdict(audit))
        issue_examples.extend(issues)
        mismatch_examples.extend(mismatches)
        if idx % 100 == 0:
            print(f"audited_symbols={idx} rows={sum(x['dedup_rows'] for x in symbol_rows)}", flush=True)

    symbol_df = pd.DataFrame(symbol_rows)
    symbol_df.to_csv(out_dir / "symbol_quality.csv", index=False)
    pd.DataFrame(issue_examples).to_csv(out_dir / "raw_issue_examples.csv", index=False)
    pd.DataFrame(mismatch_examples).to_csv(out_dir / "daily_mismatch_examples.csv", index=False)

    top3000_dirs = int((symbol_df["files"] > 0).sum())
    summary = {
        "rank_file": str(Path(args.rank_file).resolve()),
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "coverage": {
            "top3000_symbols": int(len(rank)),
            "top3000_symbols_with_minute_dir": top3000_dirs,
            "top3000_symbols_without_minute_dir": int((symbol_df["files"] == 0).sum()),
        },
        "files": {
            "files": int(symbol_df["files"].sum()),
            "rows_from_parquet_metadata": int(symbol_df["file_rows"].sum()),
            "schemas": {key: int(value) for key, value in schema_counter.items()},
        },
        "content": {
            "raw_rows_read": _sum(symbol_df, "raw_rows_read"),
            "dedup_rows": _sum(symbol_df, "dedup_rows"),
            "duplicate_rows": _sum(symbol_df, "duplicate_rows"),
            "duplicate_conflict_keys": _sum(symbol_df, "duplicate_conflict_keys"),
            "invalid_datetime_rows": _sum(symbol_df, "invalid_datetime_rows"),
            "invalid_ohlc_rows": _sum(symbol_df, "invalid_ohlc_rows"),
            "nonpositive_price_rows": _sum(symbol_df, "nonpositive_price_rows"),
            "negative_volume_amount_rows": _sum(symbol_df, "negative_volume_amount_rows"),
            "outside_session_rows": _sum(symbol_df, "outside_session_rows"),
        },
        "calendar": {
            "daily_expected_days": _sum(symbol_df, "daily_expected_days"),
            "minute_days": _sum(symbol_df, "minute_days"),
            "full_49bar_days": _sum(symbol_df, "full_49bar_days"),
            "partial_days": _sum(symbol_df, "partial_days"),
            "days_missing_model_bars": _sum(symbol_df, "days_missing_model_bars"),
            "days_missing_exit_bars": _sum(symbol_df, "days_missing_exit_bars"),
            "missing_daily_days": _sum(symbol_df, "missing_daily_days"),
            "extra_minute_days_without_daily": _sum(symbol_df, "extra_minute_days_without_daily"),
        },
        "daily_cross_check": {
            "close_mismatch_gt_1bp": _sum(symbol_df, "close_mismatch_gt_1bp"),
            "high_mismatch_gt_5bp": _sum(symbol_df, "high_mismatch_gt_5bp"),
            "low_mismatch_gt_5bp": _sum(symbol_df, "low_mismatch_gt_5bp"),
            "vol_scaled_mismatch_gt_5pct": _sum(symbol_df, "vol_scaled_mismatch_gt_5pct"),
            "amount_scaled_mismatch_gt_5pct": _sum(symbol_df, "amount_scaled_mismatch_gt_5pct"),
            "median_symbol_vol_scale_ratio_median": float(symbol_df["median_vol_scale_ratio"].dropna().median())
            if symbol_df["median_vol_scale_ratio"].notna().any()
            else None,
            "median_symbol_amount_scale_ratio_median": float(symbol_df["median_amount_scale_ratio"].dropna().median())
            if symbol_df["median_amount_scale_ratio"].notna().any()
            else None,
        },
        "task_db": task_summary,
    }
    (out_dir / "raw_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_markdown(summary, symbol_df, out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
