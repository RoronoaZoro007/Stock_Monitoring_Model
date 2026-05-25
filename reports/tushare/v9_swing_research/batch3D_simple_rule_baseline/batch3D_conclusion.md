# Batch 3D Simple Rule Baseline

## Scope

- This batch evaluates deterministic simple rules and matched random baselines.
- It does not train models, tune parameters, use concept/theme features, or modify v7_locked.
- All long-running steps are checkpointed and resumable; `--refresh` is required for full recompute.

## Method

- Primary TopN: `20` names per signal date.
- Return label: `fwd_ret_{h}d_open = exit_adj_close(T+h) / entry_adj_open(T+1) - 1`.
- Daily rule return: equal-weight mean return of selected names.
- Cumulative return: compounded daily cohort returns; this is not a full overlapping capital ledger.
- Costs, impact, capacity and partial fills are not included in this batch.

## Coverage

| horizon   |    rows |   screened_rows |   screened_valid_target_rows |   screened_valid_target_ratio |
|:----------|--------:|----------------:|-----------------------------:|------------------------------:|
| 3d        | 9306920 |         8639432 |                      8615311 |                      0.925689 |
| 5d        | 9306920 |         8639432 |                      8600989 |                      0.92415  |
| 10d       | 9306920 |         8639432 |                      8568101 |                      0.920616 |

## Gate

- Gate result: `pass_to_batch4A_plan_freeze`
- Eligible non-industry-snapshot rule rows passing pre-registered screen: `16`
- Eligibility screen: cumulative_return > 0, daily_win_rate > 50%, signal_days >= 1000, matched random percentile >= 95%, and no current-snapshot industry dependency.

## Top Rule Diagnostics

| rule_id                            | horizon   |   cumulative_return |   mean_daily_return |   daily_win_rate |   profit_factor |   max_drawdown |   rule_cumret_percentile | uses_industry_snapshot   | eligible_for_training_plan   |
|:-----------------------------------|:----------|--------------------:|--------------------:|-----------------:|----------------:|---------------:|-------------------------:|:-------------------------|:-----------------------------|
| low_log_total_mv                   | 10d       |         5.79346e+09 |          0.0151779  |         0.531652 |         1.69948 |      -0.996459 |                    1     | False                    | True                         |
| low_stock_amount_share_in_industry | 10d       |         5.23575e+07 |          0.0110172  |         0.549456 |         1.67102 |      -0.992681 |                    1     | False                    | True                         |
| low_log_amount                     | 10d       |         2.61956e+06 |          0.00947645 |         0.530168 |         1.58993 |      -0.995101 |                    1     | False                    | True                         |
| low_log_total_mv                   | 5d        |    210417           |          0.00796668 |         0.520967 |         1.49841 |      -0.949068 |                    1     | False                    | True                         |
| combo_low_size_low_liquidity       | 10d       |    132080           |          0.00776742 |         0.539565 |         1.46698 |      -0.990595 |                    1     | False                    | True                         |
| low_stock_amount_share_in_industry | 5d        |     13863.3         |          0.00576016 |         0.532807 |         1.49634 |      -0.883208 |                    1     | False                    | True                         |
| low_log_amount                     | 5d        |      3328.26        |          0.00496449 |         0.532314 |         1.44645 |      -0.902461 |                    1     | False                    | True                         |
| low_log_total_mv                   | 3d        |      2922.02        |          0.00497478 |         0.503696 |         1.42001 |      -0.809299 |                    1     | False                    | True                         |
| low_stock_ret_60d                  | 3d        |       415.017       |          0.00396204 |         0.511427 |         1.30024 |      -0.833411 |                    1     | False                    | True                         |
| low_stock_amount_share_in_industry | 3d        |       272.516       |          0.00328923 |         0.533268 |         1.38217 |      -0.715628 |                    1     | False                    | True                         |
| combo_low_liquidity_weak_momentum  | 10d       |    125973           |          0.00765149 |         0.555556 |         1.46928 |      -0.986671 |                    0.998 | False                    | True                         |
| combo_low_liquidity_weak_momentum  | 5d        |      1198.83        |          0.00448117 |         0.546009 |         1.38241 |      -0.914115 |                    0.996 | False                    | True                         |
| low_stock_ret_60d                  | 5d        |      1138.91        |          0.00514876 |         0.50788  |         1.29341 |      -0.940191 |                    0.996 | False                    | True                         |
| combo_low_size_low_liquidity       | 5d        |       797.916       |          0.00416351 |         0.544154 |         1.36508 |      -0.888126 |                    0.994 | False                    | True                         |
| low_stock_ret_20d                  | 3d        |       192.051       |          0.00342879 |         0.53559  |         1.26848 |      -0.735207 |                    0.994 | False                    | True                         |

## Interpretation

- If eligible rules exist, Batch 4A may freeze a model training plan, but no training should start yet.
- The strongest rules must be checked for costs, capacity, crowding and weak-market robustness before any forward tracking.
- Industry-snapshot rules remain diagnostic-only.

## Next Step

- Proceed to `Batch 4A model training plan freeze` only after reviewing this report.
- Do not start Batch 4B training directly.
