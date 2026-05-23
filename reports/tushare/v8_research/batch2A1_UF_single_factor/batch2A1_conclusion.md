# Batch 2A-1 Conclusion

Generated at Beijing time: 2026-05-23T21:34:25+08:00.

## 1. Batch 2A-1 changed what?

It added non-retraining U and F single-factor experiments using locked v7 scores:
U reselects Top10 inside rolling liquidity universes, and F applies capacity
filters before taking Top10.

## 2. Batch 2A-1 did not change what?

It did not change v7_locked, model weights, features, labels, TopN, sell rules,
market regime filters, industry constraints, exit rules, model complexity, or
walk-forward setup. It did not start Batch 2A-2, Batch 2B, Batch 3, Batch 4 or
Batch 5.

## 3. Was v7_locked modified?

No.

## 4. Was the model retrained?

No.

## 5. Were features, labels, TopN or sell rules modified?

No.

## 6. Was a complete daily score matrix available?

Yes. The locked validation prediction file contains the complete daily score
matrix for the current Top3000 validation candidate set. Batch 2A-1 used it for
inference-only reselection. No model fit was performed.

## 7. Did U2/U3 improve under 10bp + impact?

U2 10bp+impact cumulative return: `-0.036710`,
PF `0.943291`. U3 10bp+impact cumulative return:
`-0.042424`, PF `0.938997`.

U cumulative-return table:

| experiment_id                |   10bp_one_way_slippage |   10bp_slippage_plus_impact |   5bp_one_way_slippage |
|:-----------------------------|------------------------:|----------------------------:|-----------------------:|
| U0_static_top3000_v7         |             -0.00322725 |                  -0.0137789 |              0.0563104 |
| U1_internal_roll_top3000_60d |             -0.00322725 |                  -0.0137789 |              0.0563104 |
| U2_internal_roll_top2500_60d |             -0.0283739  |                  -0.0367097 |              0.0296618 |
| U3_internal_roll_top2000_60d |             -0.0359069  |                  -0.0424241 |              0.0216788 |

## 8. Are U changes driven by fewer trades, concentration or liquidity?

Top10 remains daily Top10 when candidates are sufficient, so trade count does
not fall materially. The main change is candidate composition and liquidity
membership. Concentration diagnostics:

| experiment_id                |   total_trades |   unique_stocks |   unique_industries | top_industry   |   top_industry_trade_count |   top_industry_trade_share | top_stock   |   top_stock_trade_count |   top_stock_trade_share |   stock_hhi |   industry_hhi |   avg_v7_top10_overlap_ratio |
|:-----------------------------|---------------:|----------------:|--------------------:|:---------------|---------------------------:|---------------------------:|:------------|------------------------:|------------------------:|------------:|---------------:|-----------------------------:|
| U0_static_top3000_v7         |            580 |             396 |                  76 | 软件服务           |                         50 |                  0.0862069 | 001270.SZ   |                       9 |               0.0155172 |  0.00375743 |      0.0349108 |                     1        |
| U1_internal_roll_top3000_60d |            580 |             396 |                  76 | 软件服务           |                         50 |                  0.0862069 | 001270.SZ   |                       9 |               0.0155172 |  0.00375743 |      0.0349108 |                     1        |
| U2_internal_roll_top2500_60d |            580 |             424 |                  74 | 软件服务           |                         54 |                  0.0931034 | 001270.SZ   |                       9 |               0.0155172 |  0.00331748 |      0.038157  |                     0.556897 |
| U3_internal_roll_top2000_60d |            580 |             439 |                  79 | 电气设备           |                         64 |                  0.110345  | 001270.SZ   |                       9 |               0.0155172 |  0.00312128 |      0.0400951 |                     0.32931  |

## 9. Which F filters improve PF, drawdown and fill ratio?

Key F rows at capital `100000`, scenario `10bp+impact+10% participation+partial/zero`:

| experiment_id                                                             |   cumulative_return |   profit_factor |   max_drawdown |   avg_fill_ratio |   days_below_top10 |   filtered_removed_candidates_total |
|:--------------------------------------------------------------------------|--------------------:|----------------:|---------------:|-----------------:|-------------------:|------------------------------------:|
| F0_no_capacity_filter_none_capital_100000                                 |          -0.0141884 |        0.982832 |     -0.0754129 |         0.997998 |                  0 |                                   0 |
| F1_order_lte_10pct_ref_amount_A_tail_capital_100000                       |          -0.0141884 |        0.982832 |     -0.0754129 |         0.997998 |                  0 |                                   0 |
| F1_order_lte_10pct_ref_amount_B_tail_and_past20_exit_proxy_capital_100000 |          -0.0141884 |        0.982832 |     -0.0754129 |         0.997998 |                  0 |                                   0 |
| F2_order_lte_5pct_ref_amount_A_tail_capital_100000                        |          -0.0141884 |        0.982832 |     -0.0754129 |         0.997998 |                  0 |                                   0 |
| F2_order_lte_5pct_ref_amount_B_tail_and_past20_exit_proxy_capital_100000  |          -0.0141884 |        0.982832 |     -0.0754129 |         0.997998 |                  0 |                                   0 |
| F3_order_lte_3pct_ref_amount_A_tail_capital_100000                        |          -0.0141884 |        0.982832 |     -0.0754129 |         0.997998 |                  0 |                                   0 |
| F3_order_lte_3pct_ref_amount_B_tail_and_past20_exit_proxy_capital_100000  |          -0.0141884 |        0.982832 |     -0.0754129 |         0.997998 |                  0 |                                   0 |
| F4_order_lte_1pct_ref_amount_A_tail_capital_100000                        |          -0.0226786 |        0.966665 |     -0.0754129 |         0.997998 |                  0 |                                   3 |
| F4_order_lte_1pct_ref_amount_B_tail_and_past20_exit_proxy_capital_100000  |          -0.0226786 |        0.966665 |     -0.0754129 |         0.997998 |                  0 |                                   3 |

## 10. Is capacity filtering only reducing trades rather than improving signal quality?

In this run, capacity filtering removed candidates but did not reduce any day
below Top10. Therefore it did not reduce executed trade count in the main Top10
setup. Any apparent improvement or deterioration should still be treated as a
candidate-composition and capacity-screening effect unless it also improves PF,
drawdown and fill quality consistently across capital levels.

## 11. Are there days with fewer than 10 candidates?

No. All tested F filters still had at least 10 eligible candidates on every validation trading day.

Daily shortage counts are in `batch2A1_F_capacity_filter_events.csv`. Aggregate
sample:

| experiment_id                                                              |   days_below_top10 |
|:---------------------------------------------------------------------------|-------------------:|
| F0_no_capacity_filter_none_capital_100000                                  |                  0 |
| F0_no_capacity_filter_none_capital_1000000                                 |                  0 |
| F0_no_capacity_filter_none_capital_300000                                  |                  0 |
| F0_no_capacity_filter_none_capital_500000                                  |                  0 |
| F0_no_capacity_filter_none_capital_5000000                                 |                  0 |
| F1_order_lte_10pct_ref_amount_A_tail_capital_100000                        |                  0 |
| F1_order_lte_10pct_ref_amount_A_tail_capital_1000000                       |                  0 |
| F1_order_lte_10pct_ref_amount_A_tail_capital_300000                        |                  0 |
| F1_order_lte_10pct_ref_amount_A_tail_capital_500000                        |                  0 |
| F1_order_lte_10pct_ref_amount_A_tail_capital_5000000                       |                  0 |
| F1_order_lte_10pct_ref_amount_B_tail_and_past20_exit_proxy_capital_100000  |                  0 |
| F1_order_lte_10pct_ref_amount_B_tail_and_past20_exit_proxy_capital_1000000 |                  0 |

## 12. Is return concentrated in a few days?

Yes, the summary files include cumulative return after dropping the largest 1,
3 and 5 profit days. These concentration columns should be used before treating
any old-validation-period gain as robust.

## 13. Should Batch 2A-2 L execution-aware label retraining start?

No.

## 14. If L starts, which universe and F filter should be used?

Recommended universe: `none`. Recommended F filter:
`none`.

Reason: Batch 2A-1 shows U/F filters improve some fill-quality or concentration diagnostics, but 10bp+impact+participation partial/zero-fill results remain weak on the legacy interval. Starting L retraining now would risk fitting labels to a fragile execution edge.

