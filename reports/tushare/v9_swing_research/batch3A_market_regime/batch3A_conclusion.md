# Batch 3A Market Regime Formalization

## Scope

- This batch formalizes market-state labels for v9 factor diagnostics.
- It does not run factor IC, build industry stock-level features, train models, tune parameters, or modify `v7_locked`.

## Definitions

- `trend_regime`: `strong` if `roll60_ew_ret >= +8%`, `weak` if `roll60_ew_ret <= -8%`, otherwise `neutral`.
- `extreme_selloff_flag`: `breadth_positive < 20%` or `ew_ret <= -3%`.
- `vol_regime`: `roll20_dispersion` versus t-1 expanding 20/80% quantile thresholds.
- `industry_crowding_regime`: `top5_industry_amount_share` versus t-1 expanding 20/80% quantile thresholds.
- `stock_concentration_regime`: `top100_amount_share` versus t-1 expanding 20/80% quantile thresholds.

## Trend Regime Distribution

| value                |   days |     ratio |
|:---------------------|-------:|----------:|
| insufficient_history |     60 | 0.0295276 |
| neutral              |   1132 | 0.557087  |
| strong               |    556 | 0.273622  |
| weak                 |    284 | 0.139764  |

## Volatility Regime Distribution

| value                |   days |     ratio |
|:---------------------|-------:|----------:|
| high_vol             |    493 | 0.242618  |
| insufficient_history |    272 | 0.133858  |
| low_vol              |    180 | 0.0885827 |
| normal_vol           |   1087 | 0.534941  |

## Industry Crowding Distribution

| value                |   days |     ratio |
|:---------------------|-------:|----------:|
| crowding_high        |    754 | 0.371063  |
| crowding_low         |    134 | 0.0659449 |
| crowding_normal      |    892 | 0.438976  |
| insufficient_history |    252 | 0.124016  |

## Threshold Sensitivity

|   threshold_abs |   strong_days |   weak_days |   neutral_days |   insufficient_history_days |
|----------------:|--------------:|------------:|---------------:|----------------------------:|
|            0.06 |           693 |         381 |            898 |                          60 |
|            0.08 |           556 |         284 |           1132 |                          60 |
|            0.1  |           429 |         211 |           1332 |                          60 |
|            0.12 |           338 |         160 |           1474 |                          60 |

## Gate Result

- Gate result: `pass`.
- Reason: market-state labels are available, definitions are documented, and no forward label performance was used to choose thresholds.

## Next Step

- Execute `Batch 3B industry/theme features`.
- Do not run factor IC until Batch 3B completes or the industry/theme step is explicitly waived.
