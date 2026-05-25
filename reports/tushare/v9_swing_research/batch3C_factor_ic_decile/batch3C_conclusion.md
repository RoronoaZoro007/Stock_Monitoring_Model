# Batch 3C Factor IC and Decile Diagnostics

## Scope

- This batch computes RankIC, ICIR, decile returns and split diagnostics for 3d/5d/10d labels.
- It does not run simple-rule backtests, train models, tune parameters, or enable concept/theme factors.
- Long-running computations are checkpointed and resumable by default; use `--refresh` only when a full recompute is intended.

## Coverage

| horizon   |    rows |   valid_target_rows |   valid_target_ratio |   screened_rows |   screened_valid_target_rows |   screened_valid_target_ratio_of_total |
|:----------|--------:|--------------------:|---------------------:|----------------:|-----------------------------:|---------------------------------------:|
| 3d        | 9306920 |             9258989 |             0.99485  |         8639432 |                      8615311 |                               0.925689 |
| 5d        | 9306920 |             9241951 |             0.993019 |         8639432 |                      8600989 |                               0.92415  |
| 10d       | 9306920 |             9204385 |             0.988983 |         8639432 |                      8568101 |                               0.920616 |

## Candidate Screen

- Candidate screen is diagnostic: `ic_days >= 500`, `abs(mean_ic) >= 0.01`, `abs(positive_ic_rate - 0.5) >= 0.02`, and direction-adjusted Top-Bottom spread > 0.
- Candidate rows passing screen: `39`

## Top Diagnostics

| horizon   | factor                         | factor_group       |    mean_ic |     icir |   positive_ic_rate |   direction_adjusted_mean_spread | candidate_flag   |
|:----------|:-------------------------------|:-------------------|-----------:|---------:|-------------------:|---------------------------------:|:-----------------|
| 10d       | log_amount                     | liquidity          | -0.080928  | -8.60922 |           0.293769 |                      0.011033    | True             |
| 10d       | turnover_rate                  | liquidity          | -0.0674276 | -6.37574 |           0.348665 |                      0.00582592  | True             |
| 5d        | log_amount                     | liquidity          | -0.0632038 | -6.96777 |           0.318204 |                      0.00504959  | True             |
| 5d        | turnover_rate                  | liquidity          | -0.0541092 | -5.11366 |           0.373458 |                      0.0018269   | True             |
| 10d       | stock_ret_20d                  | price_momentum     | -0.0498196 | -5.57499 |           0.369131 |                      0.0067179   | True             |
| 10d       | stock_ret_60d                  | price_momentum     | -0.0496943 | -5.26337 |           0.373089 |                      0.00625154  | True             |
| 3d        | log_amount                     | liquidity          | -0.0491387 | -5.43605 |           0.354362 |                      0.00203624  | True             |
| 10d       | stock_amount_share_in_industry | liquidity_crowding | -0.0477577 | -5.97732 |           0.354599 |                      0.0103719   | True             |
| 10d       | stock_rel_industry_ret_20d     | relative_strength  | -0.0451551 | -6.9321  |           0.31968  |                      0.00691103  | True             |
| 3d        | turnover_rate                  | liquidity          | -0.0435075 | -4.05127 |           0.389847 |                      0.000352431 | True             |
| 5d        | stock_ret_20d                  | price_momentum     | -0.0412755 | -4.46013 |           0.386647 |                      0.00357209  | True             |
| 5d        | stock_ret_60d                  | price_momentum     | -0.0403573 | -4.09948 |           0.407219 |                      0.00326784  | True             |
| 10d       | stock_rel_industry_ret_60d     | relative_strength  | -0.0384644 | -5.57429 |           0.365953 |                      0.00590768  | True             |
| 5d        | stock_rel_industry_ret_20d     | relative_strength  | -0.0380941 | -5.71321 |           0.347285 |                      0.00378906  | True             |
| 5d        | stock_amount_share_in_industry | liquidity_crowding | -0.0356366 | -4.52105 |           0.369018 |                      0.00494784  | True             |

## Gate Result

- Gate result: `pass_to_batch3D`.
- Reason: factor diagnostics are complete. Batch 3D is still required before any model training.
- Main interpretation: the strongest rows are negative IC diagnostics, so Batch 3D should test inverse-ranked simple rules rather than assuming momentum continuation.

## Next Step

- Execute `Batch 3D simple rule baseline` using the candidate factor screen and random baselines.
- Do not train models yet.
