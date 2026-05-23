# Batch 2B-R0 Limitations

Batch 2B-R0 is a market-regime gate diagnostic only. It does not retrain the
model, alter features, alter labels, alter the sell rule, alter TopN, or move
`v7_locked`.

Only three base strategies are tested: S0 original v7 Top10, S1 U2 filter-only
without refill, and S2 U3 filter-only without refill.

Only the pre-registered single market-state gates R1-R6 and C1-C4 are tested.
No multi-regime stacking, industry constraint, exit-rule optimization, dynamic
TopN, model retraining, or execution-aware label training is performed.

Matched random baselines use the same base strategy and the same number of
selected days as each gate, with 1000 random draws from the legacy validation
trading dates.

`condition_drop_top3_not_materially_negative` is operationalized as cumulative
return after dropping the largest 3 profit days being greater than or equal to
-2%. `condition_drawdown_not_materially_worse_than_base` is operationalized as
max drawdown no more than 2 percentage points worse than the corresponding base
strategy. These thresholds are diagnostic flags, not final parameters.

The legacy validation period must not be used as a final untouched test set.
No result in this folder is a live-trading recommendation.
