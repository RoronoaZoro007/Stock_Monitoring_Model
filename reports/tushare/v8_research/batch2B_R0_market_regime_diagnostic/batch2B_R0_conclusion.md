# Batch 2B-R0 Market Regime Gate Diagnostic Conclusion

Generated at Beijing time: 2026-05-23T22:10:25+08:00.

## 1. Batch 2B-R0 changed what?

It added a pre-registered market-regime gate diagnostic for S0 original v7
Top10, S1 U2 filter-only without refill, and S2 U3 filter-only without refill.
It also added matched random baselines for every S + R/C combination.

## 2. Batch 2B-R0 did not change what?

It did not change v7_locked, model weights, features, labels, TopN main rule,
sell rules, execution-aware labels, Batch 3, Batch 4 or Batch 5.

## 3. Was v7_locked modified?

No.

## 4. Was the model retrained?

No.

## 5. Were features, labels or sell rules modified?

No.

## 6. Which market-state combinations are positive after 10bp+impact?

| strategy_id                 | regime_id                   |   gate_day_count |   cumulative_return |   profit_factor |   max_drawdown |   cum_return_drop_top3_profit_days |   actual_percentile_vs_random |   avg_daily_holdings |
|:----------------------------|:----------------------------|-----------------:|--------------------:|----------------:|---------------:|-----------------------------------:|------------------------------:|---------------------:|
| S2_U3_filter_only_no_refill | R4_tail_breadth_Q2          |               11 |          0.22746    |        1.96456  |     -0.0098333 |                         0.0461013  |                         0.992 |              3.90909 |
| S2_U3_filter_only_no_refill | R1_tail_down                |               23 |          0.188534   |        1.56138  |     -0.0678061 |                         0.0129264  |                         0.868 |              2.47826 |
| S1_U2_filter_only_no_refill | R1_tail_down                |               23 |          0.140979   |        1.8959   |     -0.0332185 |                         0.0480321  |                         0.948 |              5.08696 |
| S2_U3_filter_only_no_refill | C2_market_up                |               29 |          0.132485   |        0.957292 |     -0.0933009 |                        -0.0534907  |                         0.63  |              3.37931 |
| S1_U2_filter_only_no_refill | R4_tail_breadth_Q2          |               11 |          0.121097   |        2.51463  |     -0.0109707 |                         0.0308589  |                         0.97  |              6.63636 |
| S1_U2_filter_only_no_refill | R6_market_dispersion_Q1_low |               12 |          0.119776   |        1.64754  |     -0.0345575 |                         0.010966   |                         0.973 |              4.75    |
| S0_v7_original_top10        | R4_tail_breadth_Q2          |               11 |          0.0922155  |        2.24754  |     -0.0118664 |                         0.0159661  |                         0.985 |             10       |
| S0_v7_original_top10        | R1_tail_down                |               23 |          0.0816383  |        1.46713  |     -0.0331281 |                         0.00506737 |                         0.957 |             10       |
| S2_U3_filter_only_no_refill | R2_market_down              |               29 |          0.0701864  |        1.25258  |     -0.0753878 |                        -0.03811    |                         0.388 |              3.2069  |
| S0_v7_original_top10        | R5_tail_ret_median_Q2       |               11 |          0.0682308  |        1.79042  |     -0.0212601 |                        -0.00634419 |                         0.954 |             10       |
| S2_U3_filter_only_no_refill | R6_market_dispersion_Q1_low |               12 |          0.0677355  |        1.50525  |     -0.0635853 |                        -0.0329191  |                         0.675 |              2.33333 |
| S1_U2_filter_only_no_refill | R5_tail_ret_median_Q2       |               11 |          0.0672933  |        1.93312  |     -0.03557   |                        -0.010763   |                         0.863 |              7       |
| S0_v7_original_top10        | R6_market_dispersion_Q1_low |               12 |          0.0502766  |        1.67516  |     -0.0165526 |                        -0.00444869 |                         0.891 |             10       |
| S1_U2_filter_only_no_refill | C2_market_up                |               29 |          0.0472801  |        0.881462 |     -0.071363  |                        -0.0650489  |                         0.604 |              5.72414 |
| S2_U3_filter_only_no_refill | R5_tail_ret_median_Q2       |               11 |          0.0414115  |        1.3468   |     -0.0626976 |                        -0.0491023  |                         0.549 |              4.27273 |
| S1_U2_filter_only_no_refill | C4_tail_ret_median_Q5_high  |               12 |          0.025203   |        0.780205 |     -0.0240235 |                        -0.0573276  |                         0.589 |              6.16667 |
| S2_U3_filter_only_no_refill | C4_tail_ret_median_Q5_high  |               12 |          0.0225559  |        0.864582 |     -0.0392059 |                        -0.068861   |                         0.421 |              4.33333 |
| S0_v7_original_top10        | R2_market_down              |               29 |          0.0208414  |        1.08935  |     -0.0488724 |                        -0.0415216  |                         0.712 |             10       |
| S1_U2_filter_only_no_refill | R3_low_tail_vol             |               29 |          0.0202836  |        1.15448  |     -0.070097  |                        -0.0585883  |                         0.436 |              5.68966 |
| S2_U3_filter_only_no_refill | C1_tail_up                  |               35 |          0.0197188  |        0.946088 |     -0.0995391 |                        -0.101181   |                         0.148 |              3.82857 |
| S0_v7_original_top10        | R3_low_tail_vol             |               29 |          0.015895   |        1.07231  |     -0.0463616 |                        -0.0386803  |                         0.684 |             10       |
| S1_U2_filter_only_no_refill | R2_market_down              |               29 |          0.0150763  |        1.19157  |     -0.0618241 |                        -0.0600271  |                         0.426 |              5.41379 |
| S2_U3_filter_only_no_refill | R3_low_tail_vol             |               29 |          0.00333629 |        1.08606  |     -0.127919  |                        -0.130106   |                         0.157 |              3.06897 |

## 7. Which combinations remain stable after dropping the largest 3 profit days?

| strategy_id                 | regime_id                   |   gate_day_count |   cumulative_return |   profit_factor |   max_drawdown |   cum_return_drop_top3_profit_days |   actual_percentile_vs_random |   avg_daily_holdings |
|:----------------------------|:----------------------------|-----------------:|--------------------:|----------------:|---------------:|-----------------------------------:|------------------------------:|---------------------:|
| S2_U3_filter_only_no_refill | R4_tail_breadth_Q2          |               11 |           0.22746   |         1.96456 |     -0.0098333 |                         0.0461013  |                         0.992 |              3.90909 |
| S2_U3_filter_only_no_refill | R1_tail_down                |               23 |           0.188534  |         1.56138 |     -0.0678061 |                         0.0129264  |                         0.868 |              2.47826 |
| S1_U2_filter_only_no_refill | R1_tail_down                |               23 |           0.140979  |         1.8959  |     -0.0332185 |                         0.0480321  |                         0.948 |              5.08696 |
| S1_U2_filter_only_no_refill | R4_tail_breadth_Q2          |               11 |           0.121097  |         2.51463 |     -0.0109707 |                         0.0308589  |                         0.97  |              6.63636 |
| S1_U2_filter_only_no_refill | R6_market_dispersion_Q1_low |               12 |           0.119776  |         1.64754 |     -0.0345575 |                         0.010966   |                         0.973 |              4.75    |
| S0_v7_original_top10        | R4_tail_breadth_Q2          |               11 |           0.0922155 |         2.24754 |     -0.0118664 |                         0.0159661  |                         0.985 |             10       |
| S0_v7_original_top10        | R1_tail_down                |               23 |           0.0816383 |         1.46713 |     -0.0331281 |                         0.00506737 |                         0.957 |             10       |
| S0_v7_original_top10        | R5_tail_ret_median_Q2       |               11 |           0.0682308 |         1.79042 |     -0.0212601 |                        -0.00634419 |                         0.954 |             10       |
| S1_U2_filter_only_no_refill | R5_tail_ret_median_Q2       |               11 |           0.0672933 |         1.93312 |     -0.03557   |                        -0.010763   |                         0.863 |              7       |
| S0_v7_original_top10        | R6_market_dispersion_Q1_low |               12 |           0.0502766 |         1.67516 |     -0.0165526 |                        -0.00444869 |                         0.891 |             10       |

## 8. Which combinations are small-sample observations?

| strategy_id                 | regime_id                   |   gate_day_count |   cumulative_return |   profit_factor |   max_drawdown |   cum_return_drop_top3_profit_days |   actual_percentile_vs_random |   avg_daily_holdings |
|:----------------------------|:----------------------------|-----------------:|--------------------:|----------------:|---------------:|-----------------------------------:|------------------------------:|---------------------:|
| S2_U3_filter_only_no_refill | R4_tail_breadth_Q2          |               11 |           0.22746   |        1.96456  |     -0.0098333 |                         0.0461013  |                         0.992 |              3.90909 |
| S1_U2_filter_only_no_refill | R4_tail_breadth_Q2          |               11 |           0.121097  |        2.51463  |     -0.0109707 |                         0.0308589  |                         0.97  |              6.63636 |
| S1_U2_filter_only_no_refill | R6_market_dispersion_Q1_low |               12 |           0.119776  |        1.64754  |     -0.0345575 |                         0.010966   |                         0.973 |              4.75    |
| S0_v7_original_top10        | R4_tail_breadth_Q2          |               11 |           0.0922155 |        2.24754  |     -0.0118664 |                         0.0159661  |                         0.985 |             10       |
| S0_v7_original_top10        | R5_tail_ret_median_Q2       |               11 |           0.0682308 |        1.79042  |     -0.0212601 |                        -0.00634419 |                         0.954 |             10       |
| S2_U3_filter_only_no_refill | R6_market_dispersion_Q1_low |               12 |           0.0677355 |        1.50525  |     -0.0635853 |                        -0.0329191  |                         0.675 |              2.33333 |
| S1_U2_filter_only_no_refill | R5_tail_ret_median_Q2       |               11 |           0.0672933 |        1.93312  |     -0.03557   |                        -0.010763   |                         0.863 |              7       |
| S0_v7_original_top10        | R6_market_dispersion_Q1_low |               12 |           0.0502766 |        1.67516  |     -0.0165526 |                        -0.00444869 |                         0.891 |             10       |
| S2_U3_filter_only_no_refill | R5_tail_ret_median_Q2       |               11 |           0.0414115 |        1.3468   |     -0.0626976 |                        -0.0491023  |                         0.549 |              4.27273 |
| S1_U2_filter_only_no_refill | C4_tail_ret_median_Q5_high  |               12 |           0.025203  |        0.780205 |     -0.0240235 |                        -0.0573276  |                         0.589 |              6.16667 |
| S2_U3_filter_only_no_refill | C4_tail_ret_median_Q5_high  |               12 |           0.0225559 |        0.864582 |     -0.0392059 |                        -0.068861   |                         0.421 |              4.33333 |
| S1_U2_filter_only_no_refill | C3_tail_breadth_Q5_high     |               12 |          -0.0085299 |        0.560558 |     -0.0302659 |                        -0.0791076  |                         0.324 |              5.58333 |
| S0_v7_original_top10        | C4_tail_ret_median_Q5_high  |               12 |          -0.0252264 |        0.792309 |     -0.0438251 |                        -0.0648776  |                         0.281 |             10       |
| S2_U3_filter_only_no_refill | C3_tail_breadth_Q5_high     |               12 |          -0.0294922 |        0.569356 |     -0.0471709 |                        -0.103435   |                         0.16  |              3.83333 |
| S0_v7_original_top10        | C3_tail_breadth_Q5_high     |               12 |          -0.0475523 |        0.601147 |     -0.0479981 |                        -0.0705523  |                         0.139 |             10       |

## 9. Which combinations are significantly better than matched random baseline?

| strategy_id                 | regime_id                   |   gate_day_count |   cumulative_return |   profit_factor |   max_drawdown |   cum_return_drop_top3_profit_days |   actual_percentile_vs_random |   avg_daily_holdings |
|:----------------------------|:----------------------------|-----------------:|--------------------:|----------------:|---------------:|-----------------------------------:|------------------------------:|---------------------:|
| S2_U3_filter_only_no_refill | R4_tail_breadth_Q2          |               11 |           0.22746   |         1.96456 |     -0.0098333 |                         0.0461013  |                         0.992 |              3.90909 |
| S0_v7_original_top10        | R4_tail_breadth_Q2          |               11 |           0.0922155 |         2.24754 |     -0.0118664 |                         0.0159661  |                         0.985 |             10       |
| S1_U2_filter_only_no_refill | R6_market_dispersion_Q1_low |               12 |           0.119776  |         1.64754 |     -0.0345575 |                         0.010966   |                         0.973 |              4.75    |
| S1_U2_filter_only_no_refill | R4_tail_breadth_Q2          |               11 |           0.121097  |         2.51463 |     -0.0109707 |                         0.0308589  |                         0.97  |              6.63636 |
| S0_v7_original_top10        | R1_tail_down                |               23 |           0.0816383 |         1.46713 |     -0.0331281 |                         0.00506737 |                         0.957 |             10       |
| S0_v7_original_top10        | R5_tail_ret_median_Q2       |               11 |           0.0682308 |         1.79042 |     -0.0212601 |                        -0.00634419 |                         0.954 |             10       |
| S1_U2_filter_only_no_refill | R1_tail_down                |               23 |           0.140979  |         1.8959  |     -0.0332185 |                         0.0480321  |                         0.948 |              5.08696 |

## 10. Should a forward shadow candidate list be built?

Yes, but only for paper tracking. The qualifying rows are listed in `batch2B_R0_candidate_screening.csv`.

Forward-shadow candidates:

| strategy_id                 | regime_id    |   gate_day_count |   cumulative_return |   profit_factor |   max_drawdown |   cum_return_drop_top3_profit_days |   actual_percentile_vs_random |   avg_daily_holdings | candidate_status         |
|:----------------------------|:-------------|-----------------:|--------------------:|----------------:|---------------:|-----------------------------------:|------------------------------:|---------------------:|:-------------------------|
| S0_v7_original_top10        | R1_tail_down |               23 |           0.0816383 |         1.46713 |     -0.0331281 |                         0.00506737 |                         0.957 |             10       | forward_shadow_candidate |
| S1_U2_filter_only_no_refill | R1_tail_down |               23 |           0.140979  |         1.8959  |     -0.0332185 |                         0.0480321  |                         0.948 |              5.08696 | forward_shadow_candidate |

## 11. Is L label retraining still not recommended?

Yes. This batch is diagnostic and does not provide enough evidence to start
execution-aware label retraining.

## 12. Should further historical optimization pause and move to forward paper tracking?

Yes. Historical diagnostics can continue only as narrow audits. Strategy
selection should move to forward paper tracking because the legacy validation
period is already heavily inspected.

## 13. Live-trading status

No direct live-trading recommendation is made.
