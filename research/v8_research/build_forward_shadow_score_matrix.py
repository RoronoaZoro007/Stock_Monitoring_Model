#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LOCKED_DIR = ROOT / "locked_artifacts" / "v7_cap20_strong_label_003"
DEFAULT_MODEL_FILE = LOCKED_DIR / "model" / "model.pkl"
DEFAULT_FEATURE_FILE = ROOT / "data_tushare" / "clean" / "features" / "tail_dataset_top3000_strict_cap20.parquet"
DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "tushare" / "v8_forward_shadow"
FORBIDDEN_OUTPUT_COLUMNS = {
    "target_return",
    "target_win",
    "label_win",
    "label_static_003",
    "label_dynamic_amp10",
    "entry_vwap",
    "exit_time",
    "exit_reason",
    "next_trade_date",
}


def beijing_now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def load_model_bundle(path: Path) -> tuple[Any, list[str], str]:
    with path.open("rb") as fh:
        bundle = pickle.load(fh)
    if isinstance(bundle, dict):
        model = bundle.get("model")
        feature_columns = list(bundle.get("feature_columns") or [])
        label_col = str(bundle.get("label_col") or bundle.get("label_column") or "")
    else:
        model = bundle
        feature_columns = []
        label_col = ""
    if model is None:
        raise ValueError(f"model bundle has no model: {path}")
    if not feature_columns:
        feature_json = LOCKED_DIR / "config" / "feature_columns.json"
        feature_columns = json.loads(feature_json.read_text())
    return model, feature_columns, label_col


def read_features(path: Path, trade_date: str) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, compression="infer")
    if "code" in df.columns and "ts_code" not in df.columns:
        df = df.rename(columns={"code": "ts_code"})
    if "trade_date" not in df.columns:
        raise ValueError("feature file must contain trade_date")
    if "ts_code" not in df.columns:
        raise ValueError("feature file must contain ts_code or code")
    df["trade_date"] = df["trade_date"].astype(str)
    df["ts_code"] = df["ts_code"].astype(str)
    out = df[df["trade_date"].eq(str(trade_date))].copy()
    if out.empty:
        raise ValueError(f"no feature rows for trade_date={trade_date}")
    return out


def score_model(model: Any, feature_df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    missing = sorted(set(feature_columns) - set(feature_df.columns))
    if missing:
        raise ValueError(f"feature file missing locked v7 features: {missing[:20]}")
    x = feature_df[feature_columns].replace([np.inf, -np.inf], np.nan)
    if hasattr(model, "predict_proba"):
        pred = model.predict_proba(x)
        if pred.ndim == 2 and pred.shape[1] >= 2:
            return np.asarray(pred[:, 1], dtype=float)
    if hasattr(model, "predict"):
        return np.asarray(model.predict(x), dtype=float)
    raise TypeError("locked model has neither predict_proba nor predict")


def build_score_matrix(
    trade_date: str,
    feature_file: Path,
    model_file: Path,
    output_root: Path,
    data_max_timestamp: str | None,
    paper_reconstruction: bool,
) -> dict[str, Any]:
    model, feature_columns, label_col = load_model_bundle(model_file)
    features = read_features(feature_file, trade_date)
    features["score"] = score_model(model, features, feature_columns)
    if "market_tail_ret_median" not in features.columns:
        if "tail_ret_1420_1450" not in features.columns:
            raise ValueError("feature file missing market_tail_ret_median and tail_ret_1420_1450")
        features["market_tail_ret_median"] = pd.to_numeric(features["tail_ret_1420_1450"], errors="coerce").median()

    keep = [
        "trade_date",
        "ts_code",
        "name",
        "industry",
        "market",
        "score",
        "market_tail_ret_median",
        "tail_ret_1420_1450",
        "amount_sofar_log",
        "liquidity_rank",
    ]
    existing = [c for c in keep if c in features.columns]
    score = features[existing].copy()
    for col in keep:
        if col not in score.columns:
            score[col] = ""
    score = score[keep]
    score["score_generation_time_beijing"] = beijing_now()
    score["data_max_timestamp"] = data_max_timestamp or f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} 14:50:00"
    score["paper_reconstruction"] = bool(paper_reconstruction)
    leak_cols = sorted(FORBIDDEN_OUTPUT_COLUMNS & set(score.columns))
    if leak_cols:
        raise AssertionError(f"forbidden future/label columns leaked into score output: {leak_cols}")

    out_dir = output_root / "score_matrices"
    out_dir.mkdir(parents=True, exist_ok=True)
    score_path = out_dir / f"{trade_date}_score_matrix.csv"
    meta_path = out_dir / f"{trade_date}_score_matrix_meta.json"
    score.to_csv(score_path, index=False)
    max_ts = str(score["data_max_timestamp"].max())
    meta = {
        "trade_date": trade_date,
        "score_matrix_generated": True,
        "generation_time_beijing": beijing_now(),
        "physical_generation_note": "Generated by locked v7 inference; no model training.",
        "logical_data_max_timestamp": max_ts,
        "uses_15_or_next_day_for_score": False,
        "feature_count": len(feature_columns),
        "label_col_locked_reference": label_col,
        "rows": int(len(score)),
        "feature_file": str(feature_file),
        "model_file": str(model_file),
        "paper_reconstruction": bool(paper_reconstruction),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"score_path": str(score_path), "meta_path": str(meta_path), **meta}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build forward-shadow score matrix using locked v7 model only.")
    parser.add_argument("--trade-date", required=True, help="YYYYMMDD")
    parser.add_argument("--feature-file", default=str(DEFAULT_FEATURE_FILE))
    parser.add_argument("--model-file", default=str(DEFAULT_MODEL_FILE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--data-max-timestamp")
    parser.add_argument("--paper-reconstruction", action="store_true")
    args = parser.parse_args()
    result = build_score_matrix(
        trade_date=str(args.trade_date),
        feature_file=Path(args.feature_file),
        model_file=Path(args.model_file),
        output_root=Path(args.output_root),
        data_max_timestamp=args.data_max_timestamp,
        paper_reconstruction=bool(args.paper_reconstruction),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
