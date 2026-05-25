# Batch 4A Model Training Plan Freeze

## Scope

- This batch freezes the v9 swing model training plan.
- No model training, model scoring, hyperparameter optimization, or live/paper trading is performed.
- `v7_locked` remains untouched.

## Frozen Primary Horizon

- Primary horizon: `5d`.
- 3d and 10d remain diagnostic-only and cannot be used for Batch4B model selection.
- Reason: 5d has stronger IC/spread than 3d and lower tail/drawdown concern than 10d.

| horizon   | decision_role          |   screened_valid_target_ratio |   usable_candidate_factor_count |   usable_candidate_abs_ic_mean |   usable_candidate_spread_mean |   eligible_simple_rule_count | best_rule_id     |   best_rule_profit_factor |   best_rule_daily_win_rate |   least_bad_eligible_max_drawdown | decision_reason                                                                                                                             |
|:----------|:-----------------------|------------------------------:|--------------------------------:|-------------------------------:|-------------------------------:|-----------------------------:|:-----------------|--------------------------:|---------------------------:|----------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------|
| 3d        | diagnostic_only        |                      0.925689 |                               7 |                      0.0320328 |                     0.00176268 |                            5 | low_log_total_mv |                   1.42001 |                   0.503696 |                         -0.715628 | Kept diagnostic: better drawdown profile but thinner spread and lower IC than 5d.                                                           |
| 5d        | primary_training_label |                      0.92415  |                               7 |                      0.0404657 |                     0.00389873 |                            6 | low_log_total_mv |                   1.49841 |                   0.520967 |                         -0.883208 | Frozen as primary: stronger IC/spread than 3d, materially less extreme than 10d, and consistent with earlier Batch2 label audit compromise. |
| 10d       | diagnostic_only        |                      0.920616 |                               7 |                      0.0502857 |                     0.0078192  |                            5 | low_log_total_mv |                   1.69948 |                   0.531652 |                         -0.986671 | Kept diagnostic only: strongest gross cohort results but extreme drawdown and higher label tail risk.                                       |

## Frozen Label

- Primary target: `fwd_ret_5d_open`.
- Formula: `exit_adj_close_5d / entry_adj_open - 1`.
- Entry: T+1 adjusted open. Exit: T+5 adjusted close.
- `label_win_5d` is secondary diagnostic only.

## Frozen Primary Features

`stock_ret_5d`, `stock_ret_20d`, `stock_ret_60d`, `turnover_rate`, `log_amount`, `log_total_mv`

Conditional ablation-only features:

`stock_amount_share_in_industry`

Blocked fields:

- Current-snapshot industry fields and concept/theme fields are blocked from primary training.
- Market regime fields are used for evaluation grouping, not as primary model features in Batch4B.

## Frozen Splits

| split            |   start_date |   end_date |    rows |   screened_rows |   trade_days |   screened_valid_fwd_ret_3d_open_rows |   screened_valid_fwd_ret_3d_open_ratio |   screened_valid_fwd_ret_5d_open_rows |   screened_valid_fwd_ret_5d_open_ratio |   screened_valid_fwd_ret_10d_open_rows |   screened_valid_fwd_ret_10d_open_ratio |
|:-----------------|-------------:|-----------:|--------:|----------------:|-------------:|--------------------------------------:|---------------------------------------:|--------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|
| train            |     20180102 |   20221230 | 4947360 |         4511736 |         1215 |                               4505507 |                               0.998619 |                               4502499 |                               0.997953 |                                4496982 |                                0.99673  |
| validation       |     20230103 |   20241231 | 2552627 |         2418143 |          484 |                               2417068 |                               0.999555 |                               2416598 |                               0.999361 |                                2415781 |                                0.999023 |
| research_holdout |     20250102 |   20260522 | 1806933 |         1709553 |          333 |                               1692736 |                               0.990163 |                               1681892 |                               0.98382  |                                1655338 |                                0.968287 |

## Minimum Baselines To Beat

| rule_id                            | horizon   |   mean_daily_return |   daily_win_rate |   profit_factor |   max_drawdown |   rule_cumret_percentile |
|:-----------------------------------|:----------|--------------------:|-----------------:|----------------:|---------------:|-------------------------:|
| low_log_total_mv                   | 5d        |          0.00796668 |         0.520967 |         1.49841 |      -0.949068 |                    1     |
| low_stock_amount_share_in_industry | 5d        |          0.00576016 |         0.532807 |         1.49634 |      -0.883208 |                    1     |
| low_log_amount                     | 5d        |          0.00496449 |         0.532314 |         1.44645 |      -0.902461 |                    1     |
| combo_low_liquidity_weak_momentum  | 5d        |          0.00448117 |         0.546009 |         1.38241 |      -0.914115 |                    0.996 |
| combo_low_size_low_liquidity       | 5d        |          0.00416351 |         0.544154 |         1.36508 |      -0.888126 |                    0.994 |
| low_stock_ret_60d                  | 5d        |          0.00514876 |         0.50788  |         1.29341 |      -0.940191 |                    0.996 |

## Batch 4B Allowed Training Families

- Ridge / ElasticNet regression or rank model.
- Logistic / linear probability model for `label_win_5d` as secondary diagnostic.
- LightGBM rank/binary only with constrained complexity and fixed small search grid defined before execution.

## Batch 4B Mandatory Evaluation

- Compare against `batch4A_baseline_reference.csv` and matched random baseline.
- Report by year, split, market regime, size bucket, liquidity bucket and industry diagnostic group.
- Apply cost/capacity stress separately from label construction.
- Include profit concentration checks: remove largest 1/3/5/10 winning days.

## Blocked Actions

- Do not train before this plan is reviewed.
- Do not add features outside `batch4A_feature_freeze.csv`.
- Do not switch to 10d because its gross return is higher.
- Do not use concept/theme features until a point-in-time data foundation is locked.
