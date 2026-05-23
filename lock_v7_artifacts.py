#!/usr/bin/env python3
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from tushare_tail_model import FEATURE_COLUMNS, CostConfig, ExitConfig
from top3000_repaired_model_pipeline import BASE_PARAMS


ROOT = Path(__file__).resolve().parent
LOCK_ROOT = ROOT / "locked_artifacts" / "v7_cap20_strong_label_003"
V7_DIR = ROOT / "reports" / "tushare" / "full_top3000_model_compare" / "v7_cap20_strong_label_003"
AUDIT_DIR = ROOT / "reports" / "tushare" / "v7_cap20_audit"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def gzip_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fin, gzip.open(dst, "wb", compresslevel=9) as fout:
        shutil.copyfileobj(fin, fout)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> None:
    if LOCK_ROOT.exists():
        shutil.rmtree(LOCK_ROOT)
    (LOCK_ROOT / "code").mkdir(parents=True, exist_ok=True)

    code_files = [
        "tushare_tail_model.py",
        "top3000_repaired_model_pipeline.py",
        "top3000_data_quality_audit.py",
        "v7_cap20_audit.py",
        "stage_model_outputs.py",
        "top500_compare_data_quality_models.py",
        "requirements.txt",
    ]
    for name in code_files:
        copy_file(ROOT / name, LOCK_ROOT / "code" / name)

    copy_file(V7_DIR / "model.pkl", LOCK_ROOT / "model" / "model.pkl")
    copy_file(V7_DIR / "metrics.json", LOCK_ROOT / "model" / "metrics.json")
    copy_file(V7_DIR / "top10_validation.csv", LOCK_ROOT / "validation" / "top10_validation.csv")
    copy_file(V7_DIR / "validation_monthly_cash_included.csv", LOCK_ROOT / "validation" / "validation_monthly_cash_included.csv")
    gzip_copy(V7_DIR / "validation_predictions.csv", LOCK_ROOT / "validation" / "validation_predictions.csv.gz")

    if AUDIT_DIR.exists():
        for src in sorted(AUDIT_DIR.glob("*")):
            if src.is_file():
                copy_file(src, LOCK_ROOT / "audit" / src.name)

    feature_table = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "model_version": "v7_cap20_strong_label_003",
            "feature_set": "base_34",
        }
    )
    (LOCK_ROOT / "config").mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(LOCK_ROOT / "config" / "feature_columns.csv", index=False)
    write_json(
        LOCK_ROOT / "config" / "feature_columns.json",
        {"feature_columns": FEATURE_COLUMNS, "count": len(FEATURE_COLUMNS)},
    )
    write_json(
        LOCK_ROOT / "config" / "label_definition.json",
        {
            "label_column": "label_static_003",
            "definition": "int(target_return > 0.003)",
            "target_return_formula": "sell_vwap*(1-sell_slippage-sell_commission-stamp_tax)/(entry_vwap*(1+buy_slippage+buy_commission))-1",
            "entry_price": "当日 14:55 5分钟 bar VWAP",
            "exit_price": "次日 09:40/09:50/10:05/10:30 5分钟 bar VWAP，根据止盈止损规则选择",
            "cost_config": CostConfig().__dict__,
            "exit_config": ExitConfig().__dict__,
        },
    )
    write_json(
        LOCK_ROOT / "config" / "train_validation_split.json",
        {
            "validation_start": "20260224",
            "train_filter": "trade_date < 20260224",
            "validation_filter": "trade_date >= 20260224",
            "top_n": 10,
            "weighting": "equal_weight",
            "model_params": BASE_PARAMS,
            "random_state": 42,
        },
    )
    copy_file(ROOT / "data_tushare" / "manifests" / "liquid_top3000_20251120_20260213.csv", LOCK_ROOT / "config" / "liquid_top3000_20251120_20260213.csv")

    manifest_targets = [
        ROOT / "data_tushare" / "clean" / "daily_from_minutes_top3000.parquet",
        ROOT / "data_tushare" / "clean" / "daily_repaired_top3000.parquet",
        ROOT / "data_tushare" / "clean" / "features" / "tail_dataset_top3000_repaired_raw.parquet",
        ROOT / "data_tushare" / "clean" / "features" / "tail_dataset_top3000_event_clean.parquet",
        ROOT / "data_tushare" / "clean" / "features" / "tail_dataset_top3000_strict_cap20.parquet",
        ROOT / "data_tushare" / "clean" / "features" / "tail_dataset_top3000_model_frame.parquet",
        ROOT / "reports" / "tushare" / "full_top3000_data_quality" / "raw_audit.json",
        ROOT / "reports" / "tushare" / "full_top3000_model_compare" / "comparison_metrics.csv",
        V7_DIR / "validation_predictions.csv",
        V7_DIR / "top10_validation.csv",
        V7_DIR / "model.pkl",
    ]
    rows = []
    for path in manifest_targets:
        if not path.exists():
            continue
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "committed_to_git": str(path).startswith(str(LOCK_ROOT)),
            }
        )
    (LOCK_ROOT / "manifest").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(LOCK_ROOT / "manifest" / "data_hash_manifest.csv", index=False)

    readme = f"""# Locked v7 Artifact Bundle

Version: `v7_cap20_strong_label_003`

Generated at: {datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")}

This directory locks the code, model file, feature list, label definition,
train/validation split, validation outputs, and audit outputs used for the
current v7 review.

Large raw and cleaned parquet datasets are intentionally not committed because
they exceed normal GitHub repository limits. Their local paths, file sizes and
SHA256 hashes are recorded in `manifest/data_hash_manifest.csv`.

Reproduce locally:

```bash
.venv/bin/python top3000_repaired_model_pipeline.py
.venv/bin/python v7_cap20_audit.py
.venv/bin/python lock_v7_artifacts.py
```
"""
    (LOCK_ROOT / "README.md").write_text(readme, encoding="utf-8")

    locked_files = []
    for path in sorted(LOCK_ROOT.rglob("*")):
        if path.is_file():
            locked_files.append(
                {
                    "path": str(path.relative_to(LOCK_ROOT)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(locked_files).to_csv(LOCK_ROOT / "manifest" / "locked_files_manifest.csv", index=False)
    print(json.dumps({"lock_root": str(LOCK_ROOT), "files": len(locked_files)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
