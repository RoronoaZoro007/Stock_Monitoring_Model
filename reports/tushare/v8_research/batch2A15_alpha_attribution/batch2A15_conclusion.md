# Batch 2A-1.5 Alpha Attribution Conclusion

Generated at Beijing time: 2026-05-23T21:57:10+08:00.

## 1. Batch 2A-1.5 changed what?

It added alpha-attribution diagnostics for the locked v7 validation signals:
U2/U3 keep/drop attribution, U2/U3 filter-only diagnostics, liquidity-bucket
score effectiveness, cost-decay attribution and market-state grouping.

## 2. Batch 2A-1.5 did not change what?

It did not change v7_locked, model weights, features, labels, TopN main rule,
sell rules, execution-aware labels, Batch 2B, Batch 3, Batch 4 or Batch 5.

## 3. Was v7_locked modified?

No.

## 4. Was the model retrained?

No.

## 5. Were features, labels, TopN or sell rules modified?

No.

## 6. Does original v7 alpha come from U2/U3 kept or dropped groups?

10bp+impact standalone cumulative return:

| Group | Trades | CumRet | PF | v7 slot-weighted contribution |
|---|---:|---:|---:|---:|
| A1 in U2 | 323 | 0.063069 | 1.022362 | 0.005298 |
| A2 dropped by U2 | 257 | -0.055632 | 0.918019 | -0.017574 |
| A3 in U3 | 191 | 0.211970 | 1.100369 | 0.021258 |
| A4 dropped by U3 | 389 | -0.110408 | 0.901775 | -0.033175 |

This table should be read as attribution, not as an optimized rule.

## 7. Is U2/U3 filter-only better than Batch 2A-1 reselection?

10bp+impact cumulative return:

| Experiment | CumRet | PF | Avg daily holdings |
|---|---:|---:|---:|
| B1 U2 filter-only | 0.063069 | 1.022362 | 5.569 |
| Batch2A1 U2 reselection | -0.036710 | 0.943291 | 10.000 |
| B2 U3 filter-only | 0.211970 | 1.100369 | 3.293 |
| Batch2A1 U3 reselection | -0.042424 | 0.938997 | 10.000 |

Filter-only isolates the original v7 signals and does not refill dropped names.

## 8. Does v7 score still rank in high-liquidity buckets?

High avg_amount_60d bucket diagnostic. The `all_candidates_bucket_top10_by_score`
row tests whether the locked score can reselect inside the bucket; the
`v7_top10_trades_in_bucket` row measures the original v7 trades that happened
to fall in that bucket.

| liquidity_metric   | bucket   | sample_scope                         |   sample_count |   v7_top10_selected_count |   selected_trade_count |   score_mean |   score_target_pearson |   score_target_spearman |      auc |   cumulative_return_5bp |   cumulative_return_10bp |   cumulative_return_10bp_impact |   profit_factor |   max_drawdown |   trade_win_rate |   daily_win_rate |   avg_fill_ratio |   partial_fill_count |   zero_fill_count |   single_ticket_participation_median |   single_ticket_participation_p90 |   single_ticket_participation_p99 |
|:-------------------|:---------|:-------------------------------------|---------------:|--------------------------:|-----------------------:|-------------:|-----------------------:|------------------------:|---------:|------------------------:|-------------------------:|--------------------------------:|----------------:|---------------:|-----------------:|-----------------:|-----------------:|---------------------:|------------------:|-------------------------------------:|----------------------------------:|----------------------------------:|
| avg_amount_60d     | Q5_high  | all_candidates_bucket_top10_by_score |          34101 |                        53 |                    580 |     0.370345 |            -0.00379182 |              -0.0115432 | 0.52354  |               -0.105805 |               -0.156206  |                      -0.158678  |        0.787136 |      -0.161119 |         0.42069  |         0.482759 |                1 |                    0 |                 0 |                          0.000298591 |                       0.000989711 |                         0.0105758 |
| avg_amount_60d     | Q5_high  | v7_top10_trades_in_bucket            |             53 |                        53 |                     53 |     0.439531 |             0.106781   |               0.0886147 | 0.585714 |                0.134641 |                0.0999986 |                       0.0970824 |        1.15839  |      -0.13573  |         0.471698 |         0.293103 |                1 |                    0 |                 0 |                          0.000345689 |                       0.00916267  |                         0.0229816 |

Low avg_amount_60d bucket diagnostic:

| liquidity_metric   | bucket   | sample_scope                         |   sample_count |   v7_top10_selected_count |   selected_trade_count |   score_mean |   score_target_pearson |   score_target_spearman |      auc |   cumulative_return_5bp |   cumulative_return_10bp |   cumulative_return_10bp_impact |   profit_factor |   max_drawdown |   trade_win_rate |   daily_win_rate |   avg_fill_ratio |   partial_fill_count |   zero_fill_count |   single_ticket_participation_median |   single_ticket_participation_p90 |   single_ticket_participation_p99 |
|:-------------------|:---------|:-------------------------------------|---------------:|--------------------------:|-----------------------:|-------------:|-----------------------:|------------------------:|---------:|------------------------:|-------------------------:|--------------------------------:|----------------:|---------------:|-----------------:|-----------------:|-----------------:|---------------------:|------------------:|-------------------------------------:|----------------------------------:|----------------------------------:|
| avg_amount_60d     | Q1_low   | all_candidates_bucket_top10_by_score |          34101 |                       288 |                    580 |     0.370806 |              0.0211924 |               0.0439516 | 0.547059 |              -0.0135905 |               -0.0691883 |                      -0.0809612 |        0.841567 |      -0.121642 |         0.403448 |         0.465517 |                1 |                    0 |                 0 |                           0.00645341 |                         0.0176434 |                         0.0441193 |
| avg_amount_60d     | Q1_low   | v7_top10_trades_in_bucket            |            288 |                       288 |                    288 |     0.507902 |              0.0473075 |               0.0724745 | 0.546266 |              -0.0635071 |               -0.113636  |                      -0.124229  |        0.884922 |      -0.144792 |         0.416667 |         0.37931  |                1 |                    0 |                 0 |                           0.0070386  |                         0.0198353 |                         0.0445992 |

## 9. Is cost decay caused by a few high-friction trades or thin alpha overall?

For avg_amount_60d buckets, summed trade-return decay from 5bp to 10bp is
`0.580455`, and from 10bp to 10bp+impact is
`0.106471`. Positive 5bp-to-10bp decay appears in
`5` liquidity buckets. This indicates the main 5bp-to-10bp
collapse is broad transaction-cost pressure, while impact adds a smaller
liquidity-sensitive layer.

## 10. Is there a market state with positive 10bp+impact return?

| regime_field                         | regime_value   |   cumulative_return |   profit_factor |   max_drawdown |   regime_trade_days |
|:-------------------------------------|:---------------|--------------------:|----------------:|---------------:|--------------------:|
| market_tail_breadth_positive_qbucket | Q2             |          0.0922155  |         2.24754 |     -0.0118664 |                  11 |
| tail_direction                       | tail_down      |          0.0816383  |         1.46713 |     -0.0331281 |                  23 |
| market_tail_ret_median_qbucket       | Q2             |          0.0682308  |         1.79042 |     -0.0212601 |                  11 |
| market_dispersion_proxy_qbucket      | Q1_low         |          0.0502766  |         1.67516 |     -0.0165526 |                  12 |
| market_direction                     | market_down    |          0.0208414  |         1.08935 |     -0.0488724 |                  29 |
| tail_vol_regime                      | low_tail_vol   |          0.015895   |         1.07231 |     -0.0463616 |                  29 |
| market_tail_ret_median_qbucket       | Q1_low         |          0.0125511  |         1.15347 |     -0.0331281 |                  12 |
| market_tail_breadth_positive_qbucket | Q4             |          0.00656399 |         1.0658  |     -0.0444388 |                  11 |

These are diagnostics only. They are not selected parameters.

## 11. Should Batch 2B-R market-state filtering diagnostics start?

Yes, but only as diagnostics. Market-state grouping can be audited without
changing labels or retraining, and it directly tests whether v7 alpha is
conditional on market regime.

## 12. Should Batch 2A-2-L execution-aware label retraining start?

No. U/F non-retraining tests did not produce stable improvement, and this
attribution batch should be reviewed before any label change.

## 13. Should historical optimization pause and keep forward paper tracking?

Yes. Historical diagnostics can continue in narrow audit batches, but model
selection should not use the legacy 20260224-20260520 interval as a final test.
Forward paper tracking is needed before treating any filter as tradable.

## 14. Next step

Review Batch 2A-1.5 outputs first. The next audit candidate is Batch 2B-R
market-state diagnostics, not L retraining and not U+F combination optimization.
