# Batch 4B Conclusion

## What Changed

- Trained frozen lightweight models from Batch 4A.
- Generated model registry, coefficients, prediction scores, Top20 daily returns and split/regime/year/industry diagnostics.

## What Did Not Change

- Did not change primary horizon, label or frozen feature list.
- Did not run hyperparameter search.
- Did not enable concept/theme fields.
- Did not move or modify `v7_locked`.

## Top Models

| model_id                            | model_family   | objective                   | training_role             | features                                                                                                            |   feature_count | model_file                                                                                                                                                                 |   validation_mean_daily_return |   validation_profit_factor |   validation_max_drawdown |   validation_auc |   validation_rank_ic |   holdout_mean_daily_return |   holdout_profit_factor |   holdout_max_drawdown |   holdout_auc |   holdout_rank_ic |
|:------------------------------------|:---------------|:----------------------------|:--------------------------|:--------------------------------------------------------------------------------------------------------------------|----------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------:|---------------------------:|--------------------------:|-----------------:|---------------------:|----------------------------:|------------------------:|-----------------------:|--------------:|------------------:|
| logistic_primary_sgd_l2             | logistic_sgd   | classification_label_win_5d | secondary_diagnostic      | stock_ret_5d, stock_ret_20d, stock_ret_60d, turnover_rate, log_amount, log_total_mv                                 |               6 | /Users/yuxiang.luo/Documents/Codex/2026-05-21/new-chat/reports/tushare/v9_swing_research/batch4B_lightweight_model_training/models/logistic_primary_sgd_l2.pkl             |                     0.00674787 |                    1.46658 |                 -0.899413 |         0.521    |            0.0821556 |                  0.0085222  |                 1.93892 |              -0.563034 |      0.532718 |         0.0712647 |
| ridge_primary_alpha1                | ridge          | regression_fwd_ret_5d_open  | primary_candidate         | stock_ret_5d, stock_ret_20d, stock_ret_60d, turnover_rate, log_amount, log_total_mv                                 |               6 | /Users/yuxiang.luo/Documents/Codex/2026-05-21/new-chat/reports/tushare/v9_swing_research/batch4B_lightweight_model_training/models/ridge_primary_alpha1.pkl                |                     0.00729549 |                    1.45922 |                 -0.931861 |         0.522031 |            0.0764446 |                  0.00943687 |                 1.87252 |              -0.636558 |      0.531621 |         0.0673911 |
| ridge_conditional_industry_ablation | ridge          | regression_fwd_ret_5d_open  | conditional_ablation_only | stock_ret_5d, stock_ret_20d, stock_ret_60d, turnover_rate, log_amount, log_total_mv, stock_amount_share_in_industry |               7 | /Users/yuxiang.luo/Documents/Codex/2026-05-21/new-chat/reports/tushare/v9_swing_research/batch4B_lightweight_model_training/models/ridge_conditional_industry_ablation.pkl |                     0.00633627 |                    1.39368 |                 -0.928018 |         0.521515 |            0.073617  |                  0.00990818 |                 1.92127 |              -0.634411 |      0.533415 |         0.0710277 |

## Gate

- Gate result: `review_required_before_batch5A`.
- Batch 4B is not considered passed unless the primary model also beats the frozen Batch3D 5d simple-rule baseline on validation.
- If accepted in a later review, next step is Batch 5A robustness and exposure audit, not walk-forward or paper tracking.
- If rejected, do not tune parameters on the same validation result; revise the plan explicitly in a new Batch 4A amendment.

## Key Limitations

- Research holdout is not a pristine final test because Batch3 diagnostics used full history.
- Returns remain gross cohort diagnostics; execution ledger and capacity are only stress diagnostics here.
- Low-size and low-liquidity exposure remains a central risk to audit in Batch 5A.
