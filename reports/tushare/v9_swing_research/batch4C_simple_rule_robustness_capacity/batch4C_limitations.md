# Batch 4C Limitations

- This is a simple-rule audit, not a trading strategy and not a model training batch.
- No rules, TopN, labels, horizons, features or model files are changed.
- Returns are still gross or stress-adjusted daily cohort returns, not a full overlapping-capital portfolio ledger.
- `amount` is Tushare daily amount in thousand yuan; capacity calculations multiply it by 1000 to estimate yuan notional.
- `signal_amount` is available at T close and is used as an ex-ante capacity proxy.
- `entry_amount` is the T+1 full-day amount and is used only as post-trade execution-quality audit, not for signal selection.
- Capacity stress uses partial-fill cash drag: unfilled notional earns zero and is not redistributed to other names.
- Cost bps are stress assumptions only; this batch does not encode an official historical tax/commission schedule.
- Entry/exit flags are daily bar proxies and do not prove actual open/close executability.
- Industry classification remains current-snapshot and is not point-in-time clean.
