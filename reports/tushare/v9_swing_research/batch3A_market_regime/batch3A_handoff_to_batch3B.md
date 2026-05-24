# Handoff: Batch 3A to Batch 3B

## Completed Batch

- Completed: `Batch 3A market_regime`
- Output directory: `reports/tushare/v9_swing_research/batch3A_market_regime/`
- Formal regime file: `data_tushare/clean/v9/v9_market_regime_formal.parquet`
- No model training, factor IC, simple-rule backtest, or parameter optimization was performed.

## Core Conclusion

- Formal market-state labels are available for 2,032 trading days.
- Trend regime counts: `[{"value": "insufficient_history", "days": 60}, {"value": "neutral", "days": 1132}, {"value": "strong", "days": 556}, {"value": "weak", "days": 284}]`
- Volatility and crowding regimes use t-1 expanding quantile thresholds to avoid future distribution leakage.
- Industry crowding remains diagnostic because the underlying industry field is a current snapshot.

## Required Inputs For Next Step

- `data_tushare/clean/v9/v9_daily_panel.parquet`
- `data_tushare/clean/v9/v9_market_regime_formal.parquet`
- `reports/tushare/v9_swing_research/batch3A_market_regime/batch3A_market_regime_daily.csv`
- `reports/tushare/v9_swing_research/stage0_data_foundation/stage0_limitations.md`

## Next Step

Run `Batch 3B industry/theme features` only after accepting Batch 3A definitions.

Batch 3B should:

- Build industry daily strength, breadth, amount-share, and crowding features.
- Keep concept/theme features disabled unless point-in-time membership data is locked.
- Treat `stock_basic.industry` as diagnostic unless point-in-time classification is added.

## Blocked Actions

- Do not run Batch 3C factor IC until Batch 3B completes or is explicitly waived.
- Do not train models.
- Do not choose trading parameters from these market-state labels.
