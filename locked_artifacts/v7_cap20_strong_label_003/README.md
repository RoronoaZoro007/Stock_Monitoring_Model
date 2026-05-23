# Locked v7 Artifact Bundle

Version: `v7_cap20_strong_label_003`

Generated at: 2026-05-23T18:43:40+08:00

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
