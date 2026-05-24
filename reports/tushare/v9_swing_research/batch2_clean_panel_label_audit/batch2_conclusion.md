# v9 Swing Research - Batch 2 Clean Panel + Label Audit

## Scope

- This batch merges daily, adj_factor, daily_basic, suspend_d, stk_limit, stock_basic and namechange into a clean daily panel.
- It builds 3/5/10 trading-day forward labels using T+1 open as entry and T+h close as exit.
- It audits missing labels, extreme returns, adjustment effects, ST, suspension and limit-up/down execution risks.
- It does not train a model, tune factors, or modify v7_locked.

## Clean Panel Coverage

|    rows |   trade_days |   codes |   date_min |   date_max |   min_codes_per_day |   median_codes_per_day |   max_codes_per_day |   duplicate_code_dates |
|--------:|-------------:|--------:|-----------:|-----------:|--------------------:|-----------------------:|--------------------:|-----------------------:|
| 9306920 |         2032 |    5755 |   20180102 |   20260522 |                3177 |                 4793.5 |                5504 |                      0 |

## Panel Quality

| field                   |   missing_rows |   missing_ratio |
|:------------------------|---------------:|----------------:|
| open                    |              0 |      0          |
| close                   |              0 |      0          |
| adj_factor              |              0 |      0          |
| basic_close             |          64735 |      0.00695558 |
| up_limit                |          65524 |      0.00704035 |
| down_limit              |          65524 |      0.00704035 |
| total_mv                |          64735 |      0.00695558 |
| turnover_rate           |          64735 |      0.00695558 |
| duplicate_code_date     |              0 |      0          |
| non_positive_price_rows |              0 |      0          |

## Label Distribution

| horizon   |    rows |   valid_labels |   missing_labels |   valid_ratio |       mean |       median |       std |       p01 |        p05 |        p25 |       p75 |       p95 |      p99 |   win_rate |   abs_gt_20pct |   abs_gt_20pct_ratio |   raw_adj_mean_gap |
|:----------|--------:|---------------:|-----------------:|--------------:|-----------:|-------------:|----------:|----------:|-----------:|-----------:|----------:|----------:|---------:|-----------:|---------------:|---------------------:|-------------------:|
| 3d        | 9306920 |        9258989 |            47931 |      0.99485  | 0.00231981 | -0.000673854 | 0.0540946 | -0.12926  | -0.0738317 | -0.025     | 0.0247423 | 0.0882841 | 0.178622 |   0.482587 |          81910 |           0.00880098 |        0.000320101 |
| 5d        | 9306920 |        9241951 |            64969 |      0.993019 | 0.00345782 | -0.00113379  | 0.071328  | -0.163192 | -0.0954733 | -0.033     | 0.0325203 | 0.115942  | 0.235514 |   0.483977 |         178722 |           0.0192031  |        0.000639642 |
| 10d       | 9306920 |        9204385 |           102535 |      0.988983 | 0.00617309 | -0.00165153  | 0.101601  | -0.210289 | -0.131009  | -0.0478558 | 0.0471584 | 0.166102  | 0.338505 |   0.486235 |         431055 |           0.0463155  |        0.00144926  |

## Missing / Execution Risk Audit

| horizon   |    rows |   missing_entry |   missing_exit |   missing_label |   current_st_rows |   entry_st_rows |   exit_st_rows |   entry_limit_up_open_rows |   exit_limit_down_rows |   current_limit_up_close_rows |   current_limit_down_close_rows |   new_stock_lt120_rows |
|:----------|--------:|----------------:|---------------:|----------------:|------------------:|----------------:|---------------:|---------------------------:|-----------------------:|------------------------------:|--------------------------------:|-----------------------:|
| 3d        | 9306920 |           22503 |          40923 |           47931 |            273673 |          272296 |         271908 |                      29401 |                  59587 |                        145801 |                           61520 |                 234297 |
| 5d        | 9306920 |           22503 |          57622 |           64969 |            273673 |          272296 |         271621 |                      29401 |                  59275 |                        145801 |                           61520 |                 234297 |
| 10d       | 9306920 |           22503 |          94452 |          102535 |            273673 |          272296 |         271217 |                      29401 |                  58783 |                        145801 |                           61520 |                 234297 |

## Extreme Label Summary

| horizon   |   rows |   positive |   negative |   new_stock_lt120 |   adj_effect_gt5pct |   any_st |   entry_or_exit_limit |
|:----------|-------:|-----------:|-----------:|------------------:|--------------------:|---------:|----------------------:|
| 10d       | 431055 |     317483 |     113572 |             19858 |                2488 |    21706 |                 26182 |
| 3d        |  81910 |      69562 |      12348 |              7386 |                 164 |      955 |                  9901 |
| 5d        | 178722 |     140576 |      38146 |             11023 |                 490 |     5607 |                 15311 |

## Initial Market Regime Distribution

| regime   |   days |
|:---------|-------:|
| neutral  |    848 |
| weak     |    592 |
| strong   |    592 |

## Important Definitions

- Primary label: `fwd_ret_{h}d_open = exit_adj_close(T+h) / entry_adj_open(T+1) - 1`.
- `adj_*` fields use raw price multiplied by `adj_factor`; ratio returns are invariant to the absolute normalization of the factor.
- Horizon uses full-market trading calendar dates. If the stock has no T+1 entry row or no T+h exit row, the label is missing and counted in the audit.
- ST flag is derived from `namechange` active name intervals containing `ST`; this is a proxy and must be validated before model training.
- `suspend_d` records are preserved as same-date flags. Cross-window suspension attribution is not used as a filter in this batch.

## Next Step

Proceed to Batch 3 factor IC and decile diagnostics only after reviewing these label distributions and deciding whether 3d, 5d, and/or 10d labels are sufficiently clean.