#!/usr/bin/env python3
"""Lock v9 raw data foundation before factor research.

This stage does not download data, build features, train models, or modify v7.
It creates an auditable manifest for the raw daily-level data already staged
locally and documents point-in-time limitations before downstream research.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data_tushare" / "raw"
REPORT_DIR = ROOT / "reports" / "tushare" / "v9_swing_research" / "stage0_data_foundation"


@dataclass(frozen=True)
class EndpointSpec:
    endpoint: str
    path: Path
    kind: str
    purpose: str
    point_in_time: str
    known_risk: str


DAILY_ENDPOINTS = [
    EndpointSpec(
        "daily",
        RAW_DIR / "daily",
        "daily_by_trade_date",
        "Daily OHLCV, pct_chg, volume and amount.",
        "T-day market data; usable after T close for T+1 swing research.",
        "Not usable for pre-close intraday decisions; raw prices must be joined with adj_factor for adjusted-return labels.",
    ),
    EndpointSpec(
        "adj_factor",
        RAW_DIR / "adj_factor",
        "daily_by_trade_date",
        "Daily adjustment factor used to compute adjusted prices.",
        "Factor observed for the trade_date; usable for historical adjusted labels after T close.",
        "Adjustment factor revisions can create back-adjusted history; raw/adjusted return gap must be audited.",
    ),
    EndpointSpec(
        "daily_basic",
        RAW_DIR / "daily_basic",
        "daily_by_trade_date",
        "Turnover, float shares, valuation and market-cap fields.",
        "T-day daily fundamentals snapshot; usable after T close for T+1 swing research.",
        "Field availability is lower than daily OHLCV; valuation fields may be missing or stale for some stocks.",
    ),
    EndpointSpec(
        "suspend_d",
        RAW_DIR / "suspend_d",
        "daily_by_trade_date",
        "Same-date suspension records returned by provider.",
        "Trade-date keyed event data; usable after provider publishes the record.",
        "Empty files mean no returned rows, not necessarily a full trading-state ledger for every stock.",
    ),
    EndpointSpec(
        "stk_limit",
        RAW_DIR / "stk_limit",
        "daily_by_trade_date",
        "Daily limit-up and limit-down prices.",
        "Trade-date keyed limit price data; usable after T close for execution-risk audit.",
        "Limit prices can be missing for a small share of rows; board-specific rules still need validation.",
    ),
]

BOOTSTRAP_ENDPOINTS = [
    EndpointSpec(
        "trade_cal",
        RAW_DIR / "bootstrap" / "trade_cal_20180101_20260522.parquet",
        "bootstrap_range",
        "Trading calendar and open-day list.",
        "Calendar data; no market-return future information when only used for date alignment.",
        "Future known exchange holidays are fine for scheduling, but labels must not use future prices.",
    ),
    EndpointSpec(
        "stock_basic",
        RAW_DIR / "bootstrap" / "stock_basic_all_status.parquet",
        "bootstrap_snapshot",
        "Static/security master fields: code, name, exchange, list status, list/delist dates.",
        "Partially point-in-time: list/delist dates are historical, but name/industry fields are current snapshot.",
        "Current industry/name can leak future classification; industry should be treated as non-point-in-time until validated.",
    ),
    EndpointSpec(
        "namechange",
        RAW_DIR / "bootstrap" / "namechange_20180101_20260522.parquet",
        "bootstrap_range",
        "Name-change intervals used as ST/name-history proxy.",
        "Interval data with start/end/announcement dates; can be converted to daily flags.",
        "ST derived from name text is only a proxy; announcement timing and exchange status should be validated before training.",
    ),
]


FIELD_DICTIONARY = {
    "trade_cal": {
        "exchange": "Exchange code.",
        "cal_date": "Calendar date.",
        "is_open": "Whether the exchange was open.",
        "pretrade_date": "Previous trading date.",
    },
    "stock_basic": {
        "ts_code": "Tushare security code.",
        "symbol": "Ticker symbol.",
        "name": "Current security name in stock_basic snapshot.",
        "area": "Listed company area.",
        "industry": "Current industry field in stock_basic snapshot.",
        "market": "Market board.",
        "exchange": "Exchange.",
        "list_status": "Listing status.",
        "list_date": "Listing date.",
        "delist_date": "Delisting date.",
    },
    "namechange": {
        "ts_code": "Tushare security code.",
        "name": "Historical name during interval.",
        "start_date": "Name-change effective start date.",
        "end_date": "Name-change effective end date.",
        "ann_date": "Announcement date if available.",
        "change_reason": "Reason for name change.",
    },
    "daily": {
        "ts_code": "Tushare security code.",
        "trade_date": "Trading date.",
        "open": "Daily open price.",
        "high": "Daily high price.",
        "low": "Daily low price.",
        "close": "Daily close price.",
        "pre_close": "Previous close price.",
        "change": "Daily price change.",
        "pct_chg": "Daily percent change.",
        "vol": "Daily volume.",
        "amount": "Daily transaction amount.",
    },
    "adj_factor": {
        "ts_code": "Tushare security code.",
        "trade_date": "Trading date.",
        "adj_factor": "Daily adjustment factor.",
    },
    "daily_basic": {
        "ts_code": "Tushare security code.",
        "trade_date": "Trading date.",
        "close": "Daily close in daily_basic endpoint.",
        "turnover_rate": "Turnover rate.",
        "turnover_rate_f": "Free-float turnover rate.",
        "volume_ratio": "Volume ratio.",
        "pe": "PE ratio.",
        "pe_ttm": "Trailing PE ratio.",
        "pb": "PB ratio.",
        "total_share": "Total shares.",
        "float_share": "Float shares.",
        "free_share": "Free-float shares.",
        "total_mv": "Total market value.",
        "circ_mv": "Circulating market value.",
    },
    "suspend_d": {
        "ts_code": "Tushare security code.",
        "trade_date": "Trading date.",
        "suspend_timing": "Suspension timing.",
        "suspend_type": "Suspension type.",
    },
    "stk_limit": {
        "trade_date": "Trading date.",
        "ts_code": "Tushare security code.",
        "up_limit": "Daily limit-up price.",
        "down_limit": "Daily limit-down price.",
    },
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def aggregate_sha(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for row in sorted(rows, key=lambda r: (str(r.get("endpoint", "")), str(r.get("trade_date", "")), str(r.get("path", "")))):
        h.update(
            f"{row.get('endpoint','')}|{row.get('trade_date','')}|{row.get('rows','')}|{row.get('size_bytes','')}|{row.get('sha256','')}\n".encode()
        )
    return h.hexdigest()


def parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def read_open_dates(start: str, end: str) -> list[str]:
    cal_path = RAW_DIR / "bootstrap" / "trade_cal_20180101_20260522.parquet"
    cal = pd.read_parquet(cal_path)
    cal["cal_date"] = cal["cal_date"].astype(str)
    open_dates = cal.loc[
        (cal["is_open"].astype(str).isin(["1", "True", "true"]))
        & cal["cal_date"].ge(start)
        & cal["cal_date"].le(end),
        "cal_date",
    ]
    return sorted(open_dates.tolist())


def build_daily_manifest(open_dates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_rows = []
    summary_rows = []
    expected = set(open_dates)
    for spec in DAILY_ENDPOINTS:
        endpoint_rows = []
        for path in sorted(spec.path.glob("trade_date=*.parquet")):
            trade_date = path.stem.split("=", 1)[-1]
            row = {
                "endpoint": spec.endpoint,
                "trade_date": trade_date,
                "path": str(path),
                "rows": parquet_rows(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            endpoint_rows.append(row)
            manifest_rows.append(row)
        available = {str(row["trade_date"]) for row in endpoint_rows}
        missing = sorted(expected - available)
        extra = sorted(available - expected)
        summary_rows.append(
            {
                "endpoint": spec.endpoint,
                "kind": spec.kind,
                "purpose": spec.purpose,
                "expected_open_dates": len(open_dates),
                "available_dates": len(available & expected),
                "missing_dates": len(missing),
                "extra_dates": len(extra),
                "first_available": min(available) if available else "",
                "last_available": max(available) if available else "",
                "first_missing": missing[0] if missing else "",
                "last_missing": missing[-1] if missing else "",
                "files": len(endpoint_rows),
                "rows": int(sum(row["rows"] for row in endpoint_rows)),
                "size_bytes": int(sum(row["size_bytes"] for row in endpoint_rows)),
                "aggregate_sha256": aggregate_sha(endpoint_rows),
                "point_in_time": spec.point_in_time,
                "known_risk": spec.known_risk,
            }
        )
    return pd.DataFrame(manifest_rows), pd.DataFrame(summary_rows)


def build_bootstrap_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary = []
    for spec in BOOTSTRAP_ENDPOINTS:
        path = spec.path
        row = {
            "endpoint": spec.endpoint,
            "trade_date": "",
            "path": str(path),
            "rows": parquet_rows(path) if path.exists() else 0,
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "sha256": sha256_file(path) if path.exists() else "",
        }
        rows.append(row)
        summary.append(
            {
                "endpoint": spec.endpoint,
                "kind": spec.kind,
                "purpose": spec.purpose,
                "files": 1 if path.exists() else 0,
                "rows": row["rows"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
                "columns": ",".join(parquet_columns(path)) if path.exists() else "",
                "point_in_time": spec.point_in_time,
                "known_risk": spec.known_risk,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary)


def build_data_dictionary() -> pd.DataFrame:
    spec_map = {spec.endpoint: spec for spec in DAILY_ENDPOINTS + BOOTSTRAP_ENDPOINTS}
    rows = []
    for endpoint, fields in FIELD_DICTIONARY.items():
        spec = spec_map[endpoint]
        for field, meaning in fields.items():
            rows.append(
                {
                    "endpoint": endpoint,
                    "field": field,
                    "meaning": meaning,
                    "source_interface": endpoint,
                    "point_in_time_status": spec.point_in_time,
                    "known_risk": spec.known_risk,
                }
            )
    return pd.DataFrame(rows)


def write_limitations(endpoint_summary: pd.DataFrame, bootstrap_summary: pd.DataFrame) -> None:
    lines = [
        "# v9 Stage 0 Data Foundation Limitations",
        "",
        "## What Is Locked",
        "",
        "- Raw daily-level files for `daily`, `adj_factor`, `daily_basic`, `suspend_d`, and `stk_limit` are hashed per file.",
        "- Bootstrap files for `trade_cal`, `stock_basic`, and `namechange` are hashed.",
        "- Coverage is checked against open trading dates from `trade_cal` for 2018-01-02 to 2026-05-22.",
        "",
        "## Point-In-Time Risks",
        "",
        "- `daily`, `daily_basic`, `adj_factor`, `suspend_d`, and `stk_limit` are keyed by trade date and should only be used after the relevant date is observable. They are suitable for T-close to T+1 swing research, not intraday pre-close decisions.",
        "- `stock_basic.industry` and `stock_basic.name` are current snapshot fields. They must not be treated as point-in-time industry/name history until independently validated.",
        "- `namechange` can create an ST/name-history proxy by effective interval, but name text alone is not a complete exchange status ledger. Announcement timing must be audited before using it as a training filter.",
        "- `adj_factor` can reflect back-adjusted history. The downstream label audit must keep raw-vs-adjusted return diagnostics.",
        "- `suspend_d` contains returned suspension records; absence of a row is not a full per-stock tradability ledger by itself.",
        "- No concept/theme membership data is locked in this stage. Any concept/industry crowding research must document its own point-in-time source before training.",
        "",
        "## Coverage Status",
        "",
        endpoint_summary[["endpoint", "expected_open_dates", "available_dates", "missing_dates", "first_available", "last_available", "aggregate_sha256"]].to_markdown(index=False),
        "",
        "## Bootstrap Files",
        "",
        bootstrap_summary[["endpoint", "files", "rows", "sha256", "known_risk"]].to_markdown(index=False),
    ]
    (REPORT_DIR / "stage0_limitations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_conclusion(endpoint_summary: pd.DataFrame, bootstrap_summary: pd.DataFrame) -> None:
    pass_coverage = int(endpoint_summary["missing_dates"].sum()) == 0
    total_raw_rows = int(endpoint_summary["rows"].sum())
    total_raw_files = int(endpoint_summary["files"].sum())
    lines = [
        "# v9 Stage 0 Data Foundation",
        "",
        "## Scope",
        "",
        "- This stage locks the raw data foundation before v9 factor IC diagnostics.",
        "- It does not download data, train models, tune parameters, modify labels, or modify `v7_locked`.",
        "",
        "## Objective Assessment",
        "",
        "This stage is necessary because Batch 1/2 already proved the data can be downloaded and transformed, but the raw sample itself was not yet frozen as an auditable foundation with per-file hashes and a data dictionary. Without this lock, later factor or model results would be hard to reproduce if raw files changed.",
        "",
        "## Raw Coverage",
        "",
        f"- Coverage pass: `{pass_coverage}`",
        f"- Raw daily endpoint files: `{total_raw_files}`",
        f"- Raw daily endpoint rows: `{total_raw_rows}`",
        "- Date range: `20180102` to `20260522`",
        "",
        endpoint_summary[["endpoint", "expected_open_dates", "available_dates", "missing_dates", "files", "rows", "aggregate_sha256"]].to_markdown(index=False),
        "",
        "## Bootstrap Coverage",
        "",
        bootstrap_summary[["endpoint", "files", "rows", "sha256"]].to_markdown(index=False),
        "",
        "## Decision",
        "",
        "- The raw daily data foundation is adequate to proceed to Batch 3 factor IC and decile diagnostics.",
        "- The dataset is not yet adequate for final model training using industry/concept features unless point-in-time industry/concept sources are added or current-snapshot leakage is explicitly excluded.",
        "- ST and suspension filters should remain audit flags first, not blind training filters, until their timing and completeness are validated.",
        "",
        "## Outputs",
        "",
        "- `stage0_raw_file_manifest.csv`: per-file raw manifest with SHA256.",
        "- `stage0_raw_endpoint_summary.csv`: per-endpoint coverage and aggregate hash.",
        "- `stage0_coverage_report.csv`: coverage pass/fail table.",
        "- `stage0_data_dictionary.csv`: field meanings and point-in-time status.",
        "- `stage0_limitations.md`: point-in-time and data-risk limitations.",
        "- `stage0_file_sha256.csv`: report file hashes.",
    ]
    (REPORT_DIR / "stage0_conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_hashes() -> None:
    rows = []
    for path in sorted(REPORT_DIR.glob("stage0_*")):
        if path.name == "stage0_file_sha256.csv" or not path.is_file():
            continue
        rows.append({"file": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    pd.DataFrame(rows).to_csv(REPORT_DIR / "stage0_file_sha256.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20260522")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    open_dates = read_open_dates(args.start, args.end)
    manifest, endpoint_summary = build_daily_manifest(open_dates)
    bootstrap_manifest, bootstrap_summary = build_bootstrap_manifest()

    raw_manifest = pd.concat([manifest, bootstrap_manifest], ignore_index=True)
    raw_manifest.to_csv(REPORT_DIR / "stage0_raw_file_manifest.csv", index=False)
    endpoint_summary.to_csv(REPORT_DIR / "stage0_raw_endpoint_summary.csv", index=False)
    bootstrap_summary.to_csv(REPORT_DIR / "stage0_bootstrap_summary.csv", index=False)
    endpoint_summary.assign(coverage_pass=endpoint_summary["missing_dates"].eq(0)).to_csv(
        REPORT_DIR / "stage0_coverage_report.csv", index=False
    )
    build_data_dictionary().to_csv(REPORT_DIR / "stage0_data_dictionary.csv", index=False)
    write_limitations(endpoint_summary, bootstrap_summary)
    write_conclusion(endpoint_summary, bootstrap_summary)
    write_report_hashes()

    print(
        {
            "out_dir": str(REPORT_DIR),
            "open_dates": len(open_dates),
            "raw_files": int(len(raw_manifest)),
            "daily_endpoint_missing_dates": int(endpoint_summary["missing_dates"].sum()),
            "raw_rows": int(endpoint_summary["rows"].sum()),
        }
    )


if __name__ == "__main__":
    main()
