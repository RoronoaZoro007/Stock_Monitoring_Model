# Batch 3A Market Regime Limitations

## Scope

- This batch formalizes market-regime labels only.
- It does not run factor IC, build industry stock-level features, train models, or choose trading parameters.

## No Future Data Policy

- `trend_regime` uses current-day `roll60_ew_ret`, which is only suitable after the current trading day is known. It is for historical swing research labels, not same-day intraday decisions.
- `vol_regime`, `industry_crowding_regime`, and `stock_concentration_regime` use expanding quantile thresholds shifted by one day, so the threshold for day `t` uses only observations up to `t-1`.
- Fixed trend and selloff thresholds are pre-registered descriptive thresholds, not optimized on forward label performance.

## Data Limitations

- Industry crowding uses `stock_basic.industry`, which Stage 0 identified as a current snapshot field. It is acceptable for diagnostic grouping here, but not automatically acceptable as a training feature.
- Concept/theme crowding is not included because no point-in-time concept membership dataset has been locked.
- The first 252 trading days have insufficient history for expanding quantile regimes and are marked `insufficient_history` for those components.

## Downstream Use

- Batch 3C may use these market-state labels for split diagnostics.
- Batch 3B must still build and document industry-level features separately before any industry/crowding factor enters IC or training.
