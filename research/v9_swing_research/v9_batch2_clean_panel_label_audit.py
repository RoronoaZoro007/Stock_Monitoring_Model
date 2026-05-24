#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data_tushare" / "raw"
CLEAN_DIR = ROOT / "data_tushare" / "clean" / "v9"
REPORT_DIR = ROOT / "reports" / "tushare" / "v9_swing_research" / "batch2_clean_panel_label_audit"
TRADE_CAL_PATH = RAW_DIR / "bootstrap" / "trade_cal_20180101_20260522.parquet"
STOCK_BASIC_PATH = RAW_DIR / "bootstrap" / "stock_basic_all_status.parquet"
NAMECHANGE_PATH = RAW_DIR / "bootstrap" / "namechange_20180101_20260522.parquet"

PANEL_PATH = CLEAN_DIR / "v9_daily_panel.parquet"
LABEL_PATH = CLEAN_DIR / "v9_swing_labels_3_5_10.parquet"
MARKET_REGIME_PATH = CLEAN_DIR / "v9_market_regime_initial.parquet"

HORIZONS = [3, 5, 10]
STRING_COLS = [
    "ts_code",
    "trade_date",
    "name_current",
    "area",
    "industry",
    "market",
    "exchange",
    "list_status",
    "list_date",
    "delist_date",
    "suspend_timing",
    "suspend_type",
    "st_name",
    "st_change_reason",
]
BOOL_COLS = [
    "suspend_flag",
    "st_flag",
    "delist_name_flag",
    "limit_up_close_flag",
    "limit_down_close_flag",
    "limit_up_open_flag",
    "limit_down_open_flag",
]
FLOAT_COLS = [
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
    "adj_factor",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "basic_close",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "pe",
    "pe_ttm",
    "pb",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv",
    "up_limit",
    "down_limit",
    "listed_days",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_open_dates(start: str, end: str) -> list[str]:
    cal = pd.read_parquet(TRADE_CAL_PATH)
    cal["cal_date"] = cal["cal_date"].astype(str)
    cal["is_open"] = pd.to_numeric(cal["is_open"], errors="coerce").fillna(0).astype(int)
    dates = cal[cal["is_open"].eq(1) & cal["cal_date"].between(start, end)]["cal_date"].tolist()
    return sorted(set(dates))


def normalize_dates(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("-", "")[:8]


def load_stock_basic() -> pd.DataFrame:
    basic = pd.read_parquet(STOCK_BASIC_PATH)
    basic = basic.rename(columns={"name": "name_current"})
    for col in ["ts_code", "name_current", "area", "industry", "market", "exchange", "list_status", "list_date", "delist_date"]:
        if col not in basic.columns:
            basic[col] = ""
        basic[col] = basic[col].map(normalize_dates) if col in {"list_date", "delist_date"} else basic[col].fillna("").astype(str)
    keep = ["ts_code", "name_current", "area", "industry", "market", "exchange", "list_status", "list_date", "delist_date"]
    return basic[keep].drop_duplicates("ts_code", keep="first")


def load_namechange() -> pd.DataFrame:
    df = pd.read_parquet(NAMECHANGE_PATH)
    for col in ["ts_code", "name", "start_date", "end_date", "ann_date", "change_reason"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    df["start_date"] = df["start_date"].map(normalize_dates)
    df["end_date"] = df["end_date"].map(normalize_dates)
    df["end_date_filled"] = df["end_date"].where(df["end_date"].ne(""), "99991231")
    df["st_interval_flag"] = df["name"].str.contains("ST", case=False, regex=False)
    df["delist_name_flag"] = df["name"].str.contains("退", regex=False)
    return df


def read_raw(api_name: str, trade_date: str) -> pd.DataFrame:
    path = RAW_DIR / api_name / f"trade_date={trade_date}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def active_name_flags(namechange: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    active = namechange[(namechange["start_date"].le(trade_date)) & (namechange["end_date_filled"].ge(trade_date))]
    if active.empty:
        return pd.DataFrame(columns=["ts_code", "st_flag", "delist_name_flag", "st_name", "st_change_reason"])
    active = active.sort_values(["ts_code", "start_date", "ann_date"]).drop_duplicates("ts_code", keep="last")
    out = active[["ts_code", "st_interval_flag", "delist_name_flag", "name", "change_reason"]].copy()
    out = out.rename(columns={"st_interval_flag": "st_flag", "name": "st_name", "change_reason": "st_change_reason"})
    return out


def prepare_chunk(trade_date: str, stock_basic: pd.DataFrame, namechange: pd.DataFrame) -> pd.DataFrame:
    daily = read_raw("daily", trade_date)
    if daily.empty:
        return pd.DataFrame(columns=STRING_COLS + FLOAT_COLS + BOOL_COLS)
    daily["ts_code"] = daily["ts_code"].astype(str)
    daily["trade_date"] = daily["trade_date"].astype(str)

    adj = read_raw("adj_factor", trade_date)
    if not adj.empty:
        adj = adj[["ts_code", "trade_date", "adj_factor"]].copy()
        adj["ts_code"] = adj["ts_code"].astype(str)
        adj["trade_date"] = adj["trade_date"].astype(str)
        daily = daily.merge(adj, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    else:
        daily["adj_factor"] = np.nan

    basic = read_raw("daily_basic", trade_date)
    if not basic.empty:
        basic = basic.rename(columns={"close": "basic_close"})
        basic["ts_code"] = basic["ts_code"].astype(str)
        basic["trade_date"] = basic["trade_date"].astype(str)
        daily = daily.merge(basic, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    else:
        for col in [
            "basic_close",
            "turnover_rate",
            "turnover_rate_f",
            "volume_ratio",
            "pe",
            "pe_ttm",
            "pb",
            "total_share",
            "float_share",
            "free_share",
            "total_mv",
            "circ_mv",
        ]:
            daily[col] = np.nan

    limit = read_raw("stk_limit", trade_date)
    if not limit.empty:
        limit["ts_code"] = limit["ts_code"].astype(str)
        limit["trade_date"] = limit["trade_date"].astype(str)
        daily = daily.merge(limit[["ts_code", "trade_date", "up_limit", "down_limit"]], on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    else:
        daily["up_limit"] = np.nan
        daily["down_limit"] = np.nan

    suspend = read_raw("suspend_d", trade_date)
    if not suspend.empty:
        suspend = suspend.drop_duplicates(["ts_code", "trade_date"], keep="last").copy()
        suspend["ts_code"] = suspend["ts_code"].astype(str)
        suspend["trade_date"] = suspend["trade_date"].astype(str)
        suspend["suspend_flag"] = True
        daily = daily.merge(
            suspend[["ts_code", "trade_date", "suspend_flag", "suspend_timing", "suspend_type"]],
            on=["ts_code", "trade_date"],
            how="left",
            validate="one_to_one",
        )
    else:
        daily["suspend_flag"] = False
        daily["suspend_timing"] = ""
        daily["suspend_type"] = ""

    daily["suspend_flag"] = daily["suspend_flag"].fillna(False).astype(bool)
    daily = daily.merge(stock_basic, on="ts_code", how="left", validate="many_to_one")
    flags = active_name_flags(namechange, trade_date)
    daily = daily.merge(flags, on="ts_code", how="left", validate="many_to_one")

    daily["st_flag"] = daily["st_flag"].fillna(False).astype(bool)
    daily["delist_name_flag"] = daily["delist_name_flag"].fillna(False).astype(bool)
    daily["st_name"] = daily["st_name"].fillna("")
    daily["st_change_reason"] = daily["st_change_reason"].fillna("")

    for col in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount", "adj_factor", "up_limit", "down_limit"]:
        daily[col] = pd.to_numeric(daily.get(col), errors="coerce")
    for col in FLOAT_COLS:
        if col not in daily.columns:
            daily[col] = np.nan
        daily[col] = pd.to_numeric(daily[col], errors="coerce")

    daily["adj_open"] = daily["open"] * daily["adj_factor"]
    daily["adj_high"] = daily["high"] * daily["adj_factor"]
    daily["adj_low"] = daily["low"] * daily["adj_factor"]
    daily["adj_close"] = daily["close"] * daily["adj_factor"]

    with np.errstate(invalid="ignore"):
        daily["limit_up_close_flag"] = daily["close"].ge(daily["up_limit"] * 0.999)
        daily["limit_down_close_flag"] = daily["close"].le(daily["down_limit"] * 1.001)
        daily["limit_up_open_flag"] = daily["open"].ge(daily["up_limit"] * 0.999)
        daily["limit_down_open_flag"] = daily["open"].le(daily["down_limit"] * 1.001)

    list_date = pd.to_datetime(daily["list_date"].replace("", pd.NA), format="%Y%m%d", errors="coerce")
    current_date = pd.to_datetime(trade_date, format="%Y%m%d", errors="coerce")
    daily["listed_days"] = (current_date - list_date).dt.days.astype("float")

    for col in STRING_COLS:
        if col not in daily.columns:
            daily[col] = ""
        daily[col] = daily[col].fillna("").astype("string")
    for col in BOOL_COLS:
        if col not in daily.columns:
            daily[col] = False
        daily[col] = daily[col].fillna(False).astype(bool)
    for col in FLOAT_COLS:
        daily[col] = pd.to_numeric(daily[col], errors="coerce").astype("float64")

    ordered = ["ts_code", "trade_date"] + [c for c in STRING_COLS if c not in {"ts_code", "trade_date"}] + FLOAT_COLS + BOOL_COLS
    return daily[ordered].sort_values("ts_code").reset_index(drop=True)


def write_panel(open_dates: list[str], refresh: bool) -> pd.DataFrame:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if PANEL_PATH.exists() and not refresh:
        meta = pd.read_parquet(PANEL_PATH, columns=["trade_date", "ts_code"])
        return panel_coverage(meta)

    stock_basic = load_stock_basic()
    namechange = load_namechange()
    writer: pq.ParquetWriter | None = None
    progress_rows = []
    if PANEL_PATH.exists():
        PANEL_PATH.unlink()
    started = time.time()
    for idx, trade_date in enumerate(open_dates, start=1):
        chunk = prepare_chunk(trade_date, stock_basic, namechange)
        if chunk.empty:
            progress_rows.append({"trade_date": trade_date, "rows": 0, "status": "missing_daily"})
            continue
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(PANEL_PATH, table.schema, compression="zstd")
        else:
            table = table.cast(writer.schema)
        writer.write_table(table)
        progress_rows.append({"trade_date": trade_date, "rows": int(len(chunk)), "status": "ok"})
        if idx % 250 == 0:
            print(json.dumps({"event": "panel_progress", "done_dates": idx, "rows": int(sum(x["rows"] for x in progress_rows))}, ensure_ascii=False), flush=True)
    if writer is not None:
        writer.close()
    progress = pd.DataFrame(progress_rows)
    progress.to_csv(REPORT_DIR / "batch2_panel_build_progress.csv", index=False)
    meta = pd.read_parquet(PANEL_PATH, columns=["trade_date", "ts_code"])
    coverage = panel_coverage(meta)
    coverage["build_seconds"] = round(time.time() - started, 3)
    return coverage


def panel_coverage(meta: pd.DataFrame) -> pd.DataFrame:
    meta["trade_date"] = meta["trade_date"].astype(str)
    meta["ts_code"] = meta["ts_code"].astype(str)
    by_day = meta.groupby("trade_date")["ts_code"].nunique()
    return pd.DataFrame(
        [
            {
                "rows": int(len(meta)),
                "trade_days": int(meta["trade_date"].nunique()),
                "codes": int(meta["ts_code"].nunique()),
                "date_min": str(meta["trade_date"].min()),
                "date_max": str(meta["trade_date"].max()),
                "min_codes_per_day": int(by_day.min()),
                "median_codes_per_day": float(by_day.median()),
                "max_codes_per_day": int(by_day.max()),
                "duplicate_code_dates": int(meta.duplicated(["ts_code", "trade_date"]).sum()),
            }
        ]
    )


def date_shift_map(open_dates: list[str]) -> pd.DataFrame:
    rows = []
    for idx, trade_date in enumerate(open_dates):
        row = {"trade_date": trade_date, "entry_date": open_dates[idx + 1] if idx + 1 < len(open_dates) else ""}
        for h in HORIZONS:
            row[f"exit_date_{h}d"] = open_dates[idx + h] if idx + h < len(open_dates) else ""
        rows.append(row)
    return pd.DataFrame(rows)


def build_labels(open_dates: list[str], refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if LABEL_PATH.exists() and MARKET_REGIME_PATH.exists() and not refresh:
        label_columns = ["trade_date", "ts_code"]
        for h in HORIZONS:
            label_columns.extend([f"fwd_ret_{h}d_open", f"raw_fwd_ret_{h}d_open"])
        labels = pd.read_parquet(LABEL_PATH, columns=label_columns)
        return label_distribution(labels), pd.read_parquet(MARKET_REGIME_PATH)

    cols = [
        "ts_code",
        "trade_date",
        "name_current",
        "industry",
        "market",
        "open",
        "close",
        "low",
        "amount",
        "adj_factor",
        "adj_open",
        "adj_close",
        "adj_low",
        "turnover_rate",
        "turnover_rate_f",
        "total_mv",
        "circ_mv",
        "listed_days",
        "st_flag",
        "suspend_flag",
        "limit_up_close_flag",
        "limit_down_close_flag",
        "limit_up_open_flag",
        "limit_down_open_flag",
        "up_limit",
        "down_limit",
    ]
    panel = pd.read_parquet(PANEL_PATH, columns=cols)
    panel["trade_date"] = panel["trade_date"].astype(str)
    panel["ts_code"] = panel["ts_code"].astype(str)
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    panel["adj_ret_1d"] = panel.groupby("ts_code", sort=False)["adj_close"].pct_change()

    date_map = date_shift_map(open_dates)
    base = panel[
        [
            "ts_code",
            "trade_date",
            "name_current",
            "industry",
            "market",
            "close",
            "adj_close",
            "adj_ret_1d",
            "amount",
            "turnover_rate",
            "turnover_rate_f",
            "total_mv",
            "circ_mv",
            "listed_days",
            "st_flag",
            "suspend_flag",
            "limit_up_close_flag",
            "limit_down_close_flag",
            "limit_up_open_flag",
            "limit_down_open_flag",
            "adj_factor",
        ]
    ].copy()
    base = base.merge(date_map, on="trade_date", how="left", validate="many_to_one")

    entry = panel[
        [
            "ts_code",
            "trade_date",
            "open",
            "adj_open",
            "adj_factor",
            "up_limit",
            "down_limit",
            "st_flag",
            "suspend_flag",
            "limit_up_open_flag",
            "limit_down_open_flag",
        ]
    ].rename(
        columns={
            "trade_date": "entry_date",
            "open": "entry_open",
            "adj_open": "entry_adj_open",
            "adj_factor": "entry_adj_factor",
            "up_limit": "entry_up_limit",
            "down_limit": "entry_down_limit",
            "st_flag": "entry_st_flag",
            "suspend_flag": "entry_suspend_flag",
            "limit_up_open_flag": "entry_limit_up_open_flag",
            "limit_down_open_flag": "entry_limit_down_open_flag",
        }
    )
    labels = base.merge(entry, on=["ts_code", "entry_date"], how="left", validate="many_to_one")

    for h in HORIZONS:
        exit_df = panel[
            [
                "ts_code",
                "trade_date",
                "close",
                "adj_close",
                "adj_factor",
                "down_limit",
                "st_flag",
                "suspend_flag",
                "limit_down_close_flag",
                "limit_up_close_flag",
            ]
        ].rename(
            columns={
                "trade_date": f"exit_date_{h}d",
                "close": f"exit_close_{h}d",
                "adj_close": f"exit_adj_close_{h}d",
                "adj_factor": f"exit_adj_factor_{h}d",
                "down_limit": f"exit_down_limit_{h}d",
                "st_flag": f"exit_st_flag_{h}d",
                "suspend_flag": f"exit_suspend_flag_{h}d",
                "limit_down_close_flag": f"exit_limit_down_close_flag_{h}d",
                "limit_up_close_flag": f"exit_limit_up_close_flag_{h}d",
            }
        )
        labels = labels.merge(exit_df, on=["ts_code", f"exit_date_{h}d"], how="left", validate="many_to_one")
        labels[f"fwd_ret_{h}d_open"] = labels[f"exit_adj_close_{h}d"] / labels["entry_adj_open"] - 1.0
        labels[f"fwd_ret_{h}d_close"] = labels[f"exit_adj_close_{h}d"] / labels["adj_close"] - 1.0
        labels[f"raw_fwd_ret_{h}d_open"] = labels[f"exit_close_{h}d"] / labels["entry_open"] - 1.0
        labels[f"adj_factor_change_{h}d"] = labels[f"exit_adj_factor_{h}d"] / labels["adj_factor"] - 1.0
        labels[f"label_win_{h}d"] = labels[f"fwd_ret_{h}d_open"].gt(0).astype("boolean")
        labels.loc[labels[f"fwd_ret_{h}d_open"].isna(), f"label_win_{h}d"] = pd.NA

    labels.to_parquet(LABEL_PATH, index=False, compression="zstd")
    regime = build_market_regime(panel)
    regime.to_parquet(MARKET_REGIME_PATH, index=False, compression="zstd")
    return label_distribution(labels), regime


def label_distribution(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(labels)
    for h in HORIZONS:
        ret = pd.to_numeric(labels[f"fwd_ret_{h}d_open"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        raw_source = labels.get(f"raw_fwd_ret_{h}d_open")
        if raw_source is None:
            raw = pd.Series(np.nan, index=labels.index)
        else:
            raw = pd.to_numeric(raw_source, errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid_ret = ret.dropna()
        finite_gap = (ret - raw).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "horizon": f"{h}d",
                "rows": int(total),
                "valid_labels": int(ret.notna().sum()),
                "missing_labels": int(ret.isna().sum()),
                "valid_ratio": float(ret.notna().mean()),
                "mean": float(ret.mean()),
                "median": float(ret.median()),
                "std": float(ret.std()),
                "p01": float(ret.quantile(0.01)),
                "p05": float(ret.quantile(0.05)),
                "p25": float(ret.quantile(0.25)),
                "p75": float(ret.quantile(0.75)),
                "p95": float(ret.quantile(0.95)),
                "p99": float(ret.quantile(0.99)),
                "win_rate": float(valid_ret.gt(0).mean()) if len(valid_ret) else np.nan,
                "abs_gt_20pct": int(ret.abs().gt(0.20).sum()),
                "abs_gt_20pct_ratio": float(ret.abs().gt(0.20).mean()),
                "raw_adj_mean_gap": float(finite_gap.mean()) if len(finite_gap) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_market_regime(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["amount"] = pd.to_numeric(panel["amount"], errors="coerce").fillna(0.0)
    rows = []
    for trade_date, day in panel.groupby("trade_date", sort=True):
        ret = pd.to_numeric(day["adj_ret_1d"], errors="coerce")
        amount = pd.to_numeric(day["amount"], errors="coerce").fillna(0.0).clip(lower=0.0)
        total_amount = float(amount.sum())
        amount_sorted = amount.sort_values(ascending=False)
        industry_amount = day.assign(_amount=amount).groupby("industry")["_amount"].sum().sort_values(ascending=False)
        rows.append(
            {
                "trade_date": trade_date,
                "codes": int(day["ts_code"].nunique()),
                "ew_ret": float(ret.mean()),
                "median_ret": float(ret.median()),
                "breadth_positive": float(ret.gt(0).mean()),
                "dispersion": float(ret.std()),
                "total_amount": total_amount,
                "top50_amount_share": float(amount_sorted.head(50).sum() / total_amount) if total_amount else np.nan,
                "top100_amount_share": float(amount_sorted.head(100).sum() / total_amount) if total_amount else np.nan,
                "top300_amount_share": float(amount_sorted.head(300).sum() / total_amount) if total_amount else np.nan,
                "top1_industry_amount_share": float(industry_amount.head(1).sum() / total_amount) if total_amount else np.nan,
                "top3_industry_amount_share": float(industry_amount.head(3).sum() / total_amount) if total_amount else np.nan,
                "top5_industry_amount_share": float(industry_amount.head(5).sum() / total_amount) if total_amount else np.nan,
            }
        )
    market = pd.DataFrame(rows).sort_values("trade_date")
    market["roll60_ew_ret"] = (1.0 + market["ew_ret"]).rolling(60).apply(np.prod, raw=True) - 1.0
    market["roll60_breadth"] = market["breadth_positive"].rolling(60).mean()
    market["roll20_dispersion"] = market["dispersion"].rolling(20).mean()
    for col in ["roll60_ew_ret", "roll60_breadth", "top5_industry_amount_share", "roll20_dispersion"]:
        valid = market[col].dropna()
        if valid.nunique() >= 5:
            market[f"{col}_q"] = pd.qcut(market[col], 5, labels=False, duplicates="drop") + 1
        else:
            market[f"{col}_q"] = np.nan
    q30 = market["roll60_ew_ret"].quantile(0.30)
    q70 = market["roll60_ew_ret"].quantile(0.70)
    market["regime_initial"] = "neutral"
    market.loc[market["roll60_ew_ret"].le(q30), "regime_initial"] = "weak"
    market.loc[market["roll60_ew_ret"].ge(q70), "regime_initial"] = "strong"
    selloff_cut = market["ew_ret"].quantile(0.05)
    market["extreme_selloff_flag"] = market["breadth_positive"].le(0.20) | market["ew_ret"].le(selloff_cut)
    return market


def missing_label_audit(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        rows.append(
            {
                "horizon": f"{h}d",
                "rows": int(len(labels)),
                "missing_entry": int(labels["entry_adj_open"].isna().sum()),
                "missing_exit": int(labels[f"exit_adj_close_{h}d"].isna().sum()),
                "missing_label": int(labels[f"fwd_ret_{h}d_open"].isna().sum()),
                "current_st_rows": int(labels["st_flag"].fillna(False).sum()),
                "entry_st_rows": int(labels["entry_st_flag"].fillna(False).sum()),
                "exit_st_rows": int(labels[f"exit_st_flag_{h}d"].fillna(False).sum()),
                "entry_limit_up_open_rows": int(labels["entry_limit_up_open_flag"].fillna(False).sum()),
                "exit_limit_down_rows": int(labels[f"exit_limit_down_close_flag_{h}d"].fillna(False).sum()),
                "current_limit_up_close_rows": int(labels["limit_up_close_flag"].fillna(False).sum()),
                "current_limit_down_close_rows": int(labels["limit_down_close_flag"].fillna(False).sum()),
                "new_stock_lt120_rows": int(pd.to_numeric(labels["listed_days"], errors="coerce").lt(120).sum()),
            }
        )
    return pd.DataFrame(rows)


def extreme_label_attribution(labels: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for h in HORIZONS:
        ret = pd.to_numeric(labels[f"fwd_ret_{h}d_open"], errors="coerce")
        sub = labels[ret.abs().gt(0.20)].copy()
        if sub.empty:
            continue
        sub["horizon"] = f"{h}d"
        sub["fwd_ret"] = ret.loc[sub.index]
        sub["raw_fwd_ret"] = pd.to_numeric(sub[f"raw_fwd_ret_{h}d_open"], errors="coerce")
        sub["adj_factor_change"] = pd.to_numeric(sub[f"adj_factor_change_{h}d"], errors="coerce")
        sub["new_stock_lt120"] = pd.to_numeric(sub["listed_days"], errors="coerce").lt(120)
        sub["adj_effect_gt5pct"] = (sub["fwd_ret"] - sub["raw_fwd_ret"]).abs().gt(0.05) | sub["adj_factor_change"].abs().gt(0.05)
        sub["any_st"] = sub["st_flag"].fillna(False) | sub["entry_st_flag"].fillna(False) | sub[f"exit_st_flag_{h}d"].fillna(False)
        sub["entry_or_exit_limit"] = sub["entry_limit_up_open_flag"].fillna(False) | sub[f"exit_limit_down_close_flag_{h}d"].fillna(False)
        keep = [
            "horizon",
            "trade_date",
            "ts_code",
            "name_current",
            "industry",
            "entry_date",
            f"exit_date_{h}d",
            "fwd_ret",
            "raw_fwd_ret",
            "adj_factor_change",
            "listed_days",
            "new_stock_lt120",
            "adj_effect_gt5pct",
            "any_st",
            "entry_or_exit_limit",
            "entry_limit_up_open_flag",
            f"exit_limit_down_close_flag_{h}d",
            "st_flag",
            "entry_st_flag",
            f"exit_st_flag_{h}d",
        ]
        frames.append(sub[keep].rename(columns={f"exit_date_{h}d": "exit_date", f"exit_limit_down_close_flag_{h}d": "exit_limit_down_close_flag", f"exit_st_flag_{h}d": "exit_st_flag"}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def by_year_distribution(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    years = labels["trade_date"].astype(str).str[:4]
    for h in HORIZONS:
        ret = pd.to_numeric(labels[f"fwd_ret_{h}d_open"], errors="coerce")
        tmp = pd.DataFrame({"year": years, "ret": ret}).dropna()
        for year, g in tmp.groupby("year"):
            rows.append(
                {
                    "horizon": f"{h}d",
                    "year": year,
                    "rows": int(len(g)),
                    "mean": float(g["ret"].mean()),
                    "median": float(g["ret"].median()),
                    "win_rate": float(g["ret"].gt(0).mean()),
                    "p05": float(g["ret"].quantile(0.05)),
                    "p95": float(g["ret"].quantile(0.95)),
                    "abs_gt_20pct": int(g["ret"].abs().gt(0.20).sum()),
                }
            )
    return pd.DataFrame(rows)


def panel_quality_audit() -> pd.DataFrame:
    columns = ["trade_date", "ts_code", "open", "close", "adj_factor", "basic_close", "up_limit", "down_limit", "total_mv", "turnover_rate"]
    panel = pd.read_parquet(PANEL_PATH, columns=columns)
    rows = []
    for col in columns:
        if col in {"trade_date", "ts_code"}:
            continue
        rows.append({"field": col, "missing_rows": int(panel[col].isna().sum()), "missing_ratio": float(panel[col].isna().mean())})
    rows.append({"field": "duplicate_code_date", "missing_rows": int(panel.duplicated(["ts_code", "trade_date"]).sum()), "missing_ratio": 0.0})
    rows.append({"field": "non_positive_price_rows", "missing_rows": int((pd.to_numeric(panel["close"], errors="coerce") <= 0).sum()), "missing_ratio": float((pd.to_numeric(panel["close"], errors="coerce") <= 0).mean())})
    return pd.DataFrame(rows)


def write_conclusion(
    panel_cov: pd.DataFrame,
    panel_quality: pd.DataFrame,
    label_dist: pd.DataFrame,
    missing_audit: pd.DataFrame,
    extreme_summary: pd.DataFrame,
    market_regime: pd.DataFrame,
) -> None:
    regime_dist = market_regime["regime_initial"].value_counts(dropna=False).rename_axis("regime").reset_index(name="days")
    lines = [
        "# v9 Swing Research - Batch 2 Clean Panel + Label Audit",
        "",
        "## Scope",
        "",
        "- This batch merges daily, adj_factor, daily_basic, suspend_d, stk_limit, stock_basic and namechange into a clean daily panel.",
        "- It builds 3/5/10 trading-day forward labels using T+1 open as entry and T+h close as exit.",
        "- It audits missing labels, extreme returns, adjustment effects, ST, suspension and limit-up/down execution risks.",
        "- It does not train a model, tune factors, or modify v7_locked.",
        "",
        "## Clean Panel Coverage",
        "",
        panel_cov.to_markdown(index=False),
        "",
        "## Panel Quality",
        "",
        panel_quality.to_markdown(index=False),
        "",
        "## Label Distribution",
        "",
        label_dist.to_markdown(index=False),
        "",
        "## Missing / Execution Risk Audit",
        "",
        missing_audit.to_markdown(index=False),
        "",
        "## Extreme Label Summary",
        "",
        extreme_summary.to_markdown(index=False) if not extreme_summary.empty else "No extreme labels above abs(20%) were found.",
        "",
        "## Initial Market Regime Distribution",
        "",
        regime_dist.to_markdown(index=False),
        "",
        "## Important Definitions",
        "",
        "- Primary label: `fwd_ret_{h}d_open = exit_adj_close(T+h) / entry_adj_open(T+1) - 1`.",
        "- `adj_*` fields use raw price multiplied by `adj_factor`; ratio returns are invariant to the absolute normalization of the factor.",
        "- Horizon uses full-market trading calendar dates. If the stock has no T+1 entry row or no T+h exit row, the label is missing and counted in the audit.",
        "- ST flag is derived from `namechange` active name intervals containing `ST`; this is a proxy and must be validated before model training.",
        "- `suspend_d` records are preserved as same-date flags. Cross-window suspension attribution is not used as a filter in this batch.",
        "",
        "## Next Step",
        "",
        "Proceed to Batch 3 factor IC and decile diagnostics only after reviewing these label distributions and deciding whether 3d, 5d, and/or 10d labels are sufficiently clean.",
    ]
    (REPORT_DIR / "batch2_conclusion.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="v9 Batch2: build clean daily panel and 3/5/10d swing label audit.")
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20260522")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    start = args.start.replace("-", "")[:8]
    end = args.end.replace("-", "")[:8]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    open_dates = read_open_dates(start, end)
    panel_cov = write_panel(open_dates, refresh=args.refresh)
    panel_cov.to_csv(REPORT_DIR / "batch2_panel_coverage.csv", index=False)

    panel_quality = panel_quality_audit()
    panel_quality.to_csv(REPORT_DIR / "batch2_panel_quality_audit.csv", index=False)

    label_dist, market_regime = build_labels(open_dates, refresh=args.refresh)
    label_dist.to_csv(REPORT_DIR / "batch2_label_distribution.csv", index=False)
    market_regime.to_csv(REPORT_DIR / "batch2_market_regime_initial.csv", index=False)

    label_cols = [
        "trade_date",
        "ts_code",
        "name_current",
        "industry",
        "entry_date",
        "entry_adj_open",
        "st_flag",
        "limit_up_close_flag",
        "limit_down_close_flag",
        "entry_st_flag",
        "entry_limit_up_open_flag",
        "listed_days",
    ]
    for h in HORIZONS:
        label_cols.extend([f"exit_date_{h}d", f"exit_adj_close_{h}d", f"fwd_ret_{h}d_open", f"fwd_ret_{h}d_close", f"raw_fwd_ret_{h}d_open", f"adj_factor_change_{h}d", f"exit_limit_down_close_flag_{h}d", f"exit_st_flag_{h}d"])
    labels = pd.read_parquet(LABEL_PATH, columns=label_cols)
    missing_audit = missing_label_audit(labels)
    missing_audit.to_csv(REPORT_DIR / "batch2_missing_label_audit.csv", index=False)
    by_year = by_year_distribution(labels)
    by_year.to_csv(REPORT_DIR / "batch2_label_distribution_by_year.csv", index=False)
    extreme = extreme_label_attribution(labels)
    extreme.to_csv(REPORT_DIR / "batch2_extreme_label_attribution.csv", index=False)
    if not extreme.empty:
        summary = extreme.groupby("horizon").agg(
            rows=("ts_code", "count"),
            positive=("fwd_ret", lambda x: int((x > 0).sum())),
            negative=("fwd_ret", lambda x: int((x < 0).sum())),
            new_stock_lt120=("new_stock_lt120", "sum"),
            adj_effect_gt5pct=("adj_effect_gt5pct", "sum"),
            any_st=("any_st", "sum"),
            entry_or_exit_limit=("entry_or_exit_limit", "sum"),
        ).reset_index()
    else:
        summary = pd.DataFrame(columns=["horizon", "rows", "positive", "negative", "new_stock_lt120", "adj_effect_gt5pct", "any_st", "entry_or_exit_limit"])
    summary.to_csv(REPORT_DIR / "batch2_extreme_label_summary.csv", index=False)

    write_conclusion(panel_cov, panel_quality, label_dist, missing_audit, summary, market_regime)

    manifest = pd.DataFrame(
        [
            {"file": str(PANEL_PATH), "role": "clean_daily_panel", "sha256": sha256_file(PANEL_PATH)},
            {"file": str(LABEL_PATH), "role": "swing_labels_3_5_10", "sha256": sha256_file(LABEL_PATH)},
            {"file": str(MARKET_REGIME_PATH), "role": "initial_market_regime", "sha256": sha256_file(MARKET_REGIME_PATH)},
        ]
    )
    manifest.to_csv(REPORT_DIR / "batch2_data_manifest.csv", index=False)

    output_hashes = []
    for path in sorted(REPORT_DIR.glob("batch2_*")):
        if path.is_file() and path.name != "batch2_file_sha256.csv":
            output_hashes.append({"file": str(path), "sha256": sha256_file(path)})
    pd.DataFrame(output_hashes).to_csv(REPORT_DIR / "batch2_file_sha256.csv", index=False)
    print(
        json.dumps(
            {
                "out_dir": str(REPORT_DIR),
                "panel_path": str(PANEL_PATH),
                "label_path": str(LABEL_PATH),
                "market_regime_path": str(MARKET_REGIME_PATH),
                "panel_rows": int(panel_cov["rows"].iloc[0]),
                "trade_days": int(panel_cov["trade_days"].iloc[0]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
