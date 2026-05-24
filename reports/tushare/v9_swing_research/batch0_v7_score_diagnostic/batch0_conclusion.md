# v9 Swing Research - Batch 0

## Purpose

This batch does not train a model and does not choose parameters. It only tests whether locked v7 scores contain measurable ranking information for 3/5/10 trading-day forward returns.

## Data Boundary

- Score source: locked v7 validation predictions.
- Daily source: repaired Top3000 daily data.
- Forward return: future close at T+h divided by v7 entry_vwap at T, minus 1.
- This is diagnostic only. The legacy validation window was already used in prior research, so it must not be treated as a final test set.

## Coverage

| horizon   |   sample_rows |   trading_days |   date_min |   date_max |   codes |
|:----------|--------------:|---------------:|-----------:|-----------:|--------:|
| 3d        |        164647 |             56 |   20260224 |   20260518 |    2999 |
| 5d        |        158775 |             54 |   20260224 |   20260514 |    2999 |
| 10d       |        144094 |             49 |   20260224 |   20260507 |    2999 |

## RankIC Summary

| metric            |   sample_rows |   trading_days |   daily_pearson_mean |   daily_pearson_median |   daily_spearman_mean |   daily_spearman_median |   positive_spearman_day_ratio |
|:------------------|--------------:|---------------:|---------------------:|-----------------------:|----------------------:|------------------------:|------------------------------:|
| target_return     |        170505 |             58 |            0.0278855 |              0.0266478 |             0.0120737 |              0.0133412  |                      0.517241 |
| fwd_ret_3d_entry  |        164647 |             56 |            0.0261218 |              0.0259313 |             0.0168031 |              0.0252216  |                      0.607143 |
| fwd_ret_5d_entry  |        158775 |             54 |            0.0314505 |              0.0378688 |             0.0197356 |              0.00347448 |                      0.5      |
| fwd_ret_10d_entry |        144094 |             49 |            0.047771  |              0.046881  |             0.0380166 |              0.036503   |                      0.653061 |

## 3d Initial Read

- Top score decile average return: 0.13%.
- Bottom score decile average return: -0.30%.
- Top-minus-bottom spread: 0.43%.
- Mean daily Spearman RankIC: 0.0168.

## 5d Initial Read

- Top score decile average return: 0.17%.
- Bottom score decile average return: -0.58%.
- Top-minus-bottom spread: 0.75%.
- Mean daily Spearman RankIC: 0.0197.

## 10d Initial Read

- Top score decile average return: 0.96%.
- Bottom score decile average return: -0.66%.
- Top-minus-bottom spread: 1.62%.
- Mean daily Spearman RankIC: 0.0380.

## Professional Judgment

v7 score was trained for an overnight/next-morning execution target, so it should not be assumed to be a swing-trading alpha. The only defensible use is as a candidate factor after measuring whether its cross-sectional ranking survives at 3/5/10 day horizons.

Top10 diagnostic table:

| horizon   | selection   |   sample_rows |   trading_days |   avg_signal_return |   median_signal_return |   signal_win_rate |   avg_daily_equal_weight_return |   daily_win_rate |   baseline_all_avg_daily_return |   excess_vs_all_daily |
|:----------|:------------|--------------:|---------------:|--------------------:|-----------------------:|------------------:|--------------------------------:|-----------------:|--------------------------------:|----------------------:|
| 3d        | top10       |           560 |             56 |         0.00205659  |            0.000203575 |          0.5      |                     0.00205659  |         0.589286 |                     0.000697823 |           0.00135877  |
| 5d        | top10       |           540 |             54 |        -0.000881061 |           -0.00747502  |          0.446296 |                    -0.000881061 |         0.5      |                     0.000787342 |          -0.0016684   |
| 10d       | top10       |           490 |             49 |         0.00771927  |           -0.0115452   |          0.455102 |                     0.00771927  |         0.55102  |                     0.00681474  |           0.000904534 |

Context diagnostic table:

| horizon   | group                |   sample_rows |   trading_days |   score_spearman_mean_by_day |   score_spearman_positive_day_ratio |   all_avg_return |   all_win_rate |   top10_rows |   top10_avg_return |   top10_win_rate |
|:----------|:---------------------|--------------:|---------------:|-----------------------------:|------------------------------------:|-----------------:|---------------:|-------------:|-------------------:|-----------------:|
| 3d        | all                  |        164647 |             56 |                    0.0168031 |                            0.607143 |      0.000647411 |       0.470273 |          560 |        0.00205659  |         0.5      |
| 3d        | U2_liquidity_top2500 |        137211 |             56 |                    0.0162425 |                            0.607143 |      0.000669575 |       0.470152 |          341 |       -0.000753775 |         0.495601 |
| 3d        | tail_down            |         67660 |             23 |                    0.0695191 |                            0.73913  |      0.0115111   |       0.569362 |          230 |        0.0176859   |         0.621739 |
| 3d        | U2_and_tail_down     |         56401 |             23 |                    0.0723659 |                            0.73913  |      0.0118509   |       0.570681 |          130 |        0.0182888   |         0.661538 |
| 5d        | all                  |        158775 |             54 |                    0.0197356 |                            0.5      |      0.000734949 |       0.458618 |          540 |       -0.000881061 |         0.446296 |
| 5d        | U2_liquidity_top2500 |        132321 |             54 |                    0.021097  |                            0.5      |      0.000720862 |       0.458506 |          333 |       -0.00435624  |         0.432432 |
| 5d        | tail_down            |         67659 |             23 |                    0.0748492 |                            0.695652 |      0.0130793   |       0.557768 |          230 |        0.018422    |         0.586957 |
| 5d        | U2_and_tail_down     |         56400 |             23 |                    0.0790005 |                            0.695652 |      0.0133499   |       0.559149 |          130 |        0.021134    |         0.607692 |
| 10d       | all                  |        144094 |             49 |                    0.0380166 |                            0.653061 |      0.00676226  |       0.469832 |          490 |        0.00771927  |         0.455102 |
| 10d       | U2_liquidity_top2500 |        120092 |             49 |                    0.0419815 |                            0.673469 |      0.00688407  |       0.469682 |          297 |        0.00761519  |         0.447811 |
| 10d       | tail_down            |         61778 |             21 |                    0.0542744 |                            0.714286 |      0.0142306   |       0.504953 |          210 |        0.0186057   |         0.480952 |
| 10d       | U2_and_tail_down     |         51499 |             21 |                    0.0606008 |                            0.761905 |      0.0145143   |       0.503893 |          117 |        0.0206115   |         0.495726 |

## Judgment

- The locked v7 score shows weak positive cross-sectional transfer to 3/5/10 day returns, strongest at 10d by RankIC and top-minus-bottom decile spread.
- The evidence is not strong enough to use v7 score as a standalone swing selector: Top10 5d is negative versus the all-candidate baseline, Top10 10d median return is negative, and the top score decile is not consistently the best decile.
- U2 liquidity filtering alone does not prove a stable swing improvement in this batch; it improves tradability but does not automatically improve v7 Top10 swing returns.
- tail_down and U2_and_tail_down look better in this legacy window, but this is diagnostic only and cannot be used to choose live parameters without forward unseen tracking.
- The professional interpretation is: v7 score can be kept as a candidate factor/control variable for v9 swing research, but v9 should not inherit v7 Top10 directly as its primary swing model.

## Next Step Gate

Proceed to v9 Batch 1 only as a diagnostic research step, not as optimization. Batch 1 should test whether v7 score remains useful after stricter robustness checks: monthly splits, industry neutrality, liquidity buckets, overlap-adjusted holding periods, cost/turnover estimates, and forward-only paper tracking once new data arrives.

No live, simulated, or real trading conclusion is made here.