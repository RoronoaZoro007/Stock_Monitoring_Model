# v9 Stage 0 Data Foundation

## Scope

- This stage locks the raw data foundation before v9 factor IC diagnostics.
- It does not download data, train models, tune parameters, modify labels, or modify `v7_locked`.

## Objective Assessment

This stage is necessary because Batch 1/2 already proved the data can be downloaded and transformed, but the raw sample itself was not yet frozen as an auditable foundation with per-file hashes and a data dictionary. Without this lock, later factor or model results would be hard to reproduce if raw files changed.

## Raw Coverage

- Coverage pass: `True`
- Raw daily endpoint files: `10160`
- Raw daily endpoint rows: `39670458`
- Date range: `20180102` to `20260522`

| endpoint    |   expected_open_dates |   available_dates |   missing_dates |   files |     rows | aggregate_sha256                                                 |
|:------------|----------------------:|------------------:|----------------:|--------:|---------:|:-----------------------------------------------------------------|
| daily       |                  2032 |              2032 |               0 |    2032 |  9306920 | b8a3d6a0ba9198059785e20ed8899defee805217cd1b0454e1a4b8ef332e7122 |
| adj_factor  |                  2032 |              2032 |               0 |    2032 |  9561479 | 2a05fb34d3f92676631858c963dd92f96fca6a54900d995f09b2cf3661629d5b |
| daily_basic |                  2032 |              2032 |               0 |    2032 |  9242185 | 402111c1bcb6b73a67b779e1e663a5cf96e4f6009bd73f935caa6592655a8b2d |
| suspend_d   |                  2032 |              2032 |               0 |    2032 |   109775 | c7d4e2b5437ca727cab6e32f1de80c77a14e882b95fc163e8bd3e029f023a26c |
| stk_limit   |                  2032 |              2032 |               0 |    2032 | 11450099 | a16324d9fcd8870f74a4f384c0ce409d4b3c3c356b25e28dc375a32528d26f44 |

## Bootstrap Coverage

| endpoint    |   files |   rows | sha256                                                           |
|:------------|--------:|-------:|:-----------------------------------------------------------------|
| trade_cal   |       1 |   3064 | 651a9d117adcc8d862ea5f9e25a927549106e6e87eda249dc24c4d2e71e75f66 |
| stock_basic |       1 |   5847 | a8a937c964c82c448f6dcec049fc5f5fb6a73652ed3452f372a0f022468877ce |
| namechange  |       1 |   5250 | 0511a853c86a58829da5953b633e0c00be3b7e93605cd6a63c6fdd7061f94872 |

## Decision

- The raw daily data foundation is adequate to proceed to Batch 3 factor IC and decile diagnostics.
- The dataset is not yet adequate for final model training using industry/concept features unless point-in-time industry/concept sources are added or current-snapshot leakage is explicitly excluded.
- ST and suspension filters should remain audit flags first, not blind training filters, until their timing and completeness are validated.

## Outputs

- `stage0_raw_file_manifest.csv`: per-file raw manifest with SHA256.
- `stage0_raw_endpoint_summary.csv`: per-endpoint coverage and aggregate hash.
- `stage0_coverage_report.csv`: coverage pass/fail table.
- `stage0_data_dictionary.csv`: field meanings and point-in-time status.
- `stage0_limitations.md`: point-in-time and data-risk limitations.
- `stage0_file_sha256.csv`: report file hashes.
