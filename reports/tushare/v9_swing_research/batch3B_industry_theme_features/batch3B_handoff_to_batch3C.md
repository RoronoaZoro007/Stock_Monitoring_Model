# Handoff: Batch 3B to Batch 3C

## Completed Batch

- Completed: `Batch 3B industry/theme features`
- Output directory: `reports/tushare/v9_swing_research/batch3B_industry_theme_features/`
- Industry feature file: `data_tushare/clean/v9/v9_industry_daily_features.parquet`
- Stock-industry feature file: `data_tushare/clean/v9/v9_stock_industry_features.parquet`
- No factor IC, simple-rule backtest, model training, or parameter optimization was performed.

## Core Conclusion

- Industry daily feature rows: `225541`
- Stock-industry feature rows: `9306920`
- Industry count: `111`
- Top crowding industries by high-crowding days: `[{"industry": "电气设备", "crowding_high_days": 615, "avg_amount_share": 0.07012170464708939}, {"industry": "专用机械", "crowding_high_days": 572, "avg_amount_share": 0.030055692192144583}, {"industry": "半导体", "crowding_high_days": 562, "avg_amount_share": 0.05155231767824125}, {"industry": "通信设备", "crowding_high_days": 557, "avg_amount_share": 0.03886027528363976}, {"industry": "化工原料", "crowding_high_days": 515, "avg_amount_share": 0.032949345969648984}, {"industry": "元器件", "crowding_high_days": 512, "avg_amount_share": 0.0615982702338214}, {"industry": "陶瓷", "crowding_high_days": 506, "avg_amount_share": 0.00065890086081317}, {"industry": "摩托车", "crowding_high_days": 497, "avg_amount_share": 0.0014665831958821718}, {"industry": "玻璃", "crowding_high_days": 486, "avg_amount_share": 0.0037878562323220267}, {"industry": "新型电力", "crowding_high_days": 482, "avg_amount_share": 0.006457079035258962}]`
- Concept/theme features remain disabled because no point-in-time source is locked.
- Industry features are diagnostic-only unless point-in-time industry classification is added or leakage risk is explicitly accepted.

## Required Inputs For Next Step

- `data_tushare/clean/v9/v9_swing_labels_3_5_10.parquet`
- `data_tushare/clean/v9/v9_market_regime_formal.parquet`
- `data_tushare/clean/v9/v9_industry_daily_features.parquet`
- `data_tushare/clean/v9/v9_stock_industry_features.parquet`
- `reports/tushare/v9_swing_research/batch3B_industry_theme_features/batch3B_industry_feature_dictionary.csv`
- `reports/tushare/v9_swing_research/batch3B_industry_theme_features/batch3B_point_in_time_limitations.md`

## Next Step

Run `Batch 3C factor IC and decile diagnostics` only after accepting that industry snapshot fields are diagnostic-only.

Batch 3C should:

- Compute RankIC, ICIR, decile returns and Top-Bottom spreads for 3d/5d/10d labels.
- Split IC by market regime, year, industry, size and liquidity.
- Keep concept/theme factors disabled unless their data foundation is locked first.

## Blocked Actions

- Do not train models.
- Do not run simple-rule baseline before Batch 3C unless explicitly requested.
- Do not treat current-snapshot industry as a clean point-in-time training feature.
