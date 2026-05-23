# Batch 1 Conclusion

Generated at Beijing time: 2026-05-23T21:15:23+08:00.

## 1. Batch 1 changed what?

Batch 1 added hard-check audits for universe construction, extreme target-return
attribution and execution ledger realism. It generated only audit outputs under
`reports/tushare/v8_research/batch1_hard_checks/`.

## 2. Batch 1 did not change what?

It did not change v7 data, model, features, labels, TopN, sell rules, or any
training process. It did not start Batch 2A, Batch 2B, Batch 3, Batch 4 or Batch 5.

## 3. Was v7_locked modified?

No.

## 4. Was the model retrained?

No.

## 5. Were features, labels, TopN or sell rules modified?

No.

## 6. U: How different are static Top3000 and rolling liquidity universes?

| universe_id                  |   avg_overlap |   avg_removed |   min_overlap |
|:-----------------------------|--------------:|--------------:|--------------:|
| U1_internal_roll_top3000_60d |      1        |         0     |      1        |
| U2_internal_roll_top2500_60d |      0.850456 |       439.741 |      0.840619 |
| U3_internal_roll_top2000_60d |      0.680365 |       939.741 |      0.672495 |

## 7. U: What is v7 original Top10 retention inside rolling pools?

| universe_id                  |   avg_retention |   min_retention |   total_removed |
|:-----------------------------|----------------:|----------------:|----------------:|
| U1_internal_roll_top3000_60d |        1        |             1   |               0 |
| U2_internal_roll_top2500_60d |        0.556897 |             0.1 |             257 |
| U3_internal_roll_top2000_60d |        0.32931  |             0   |             389 |

## 8. C: What mainly causes extreme target_return?

| primary_attribution                        |   rows |
|:-------------------------------------------|-------:|
| special_20pct_limit_board                  |    288 |
| unknown                                    |     88 |
| subnew_stock_365d                          |     10 |
| new_stock_180d                             |      3 |
| confirmed_dirty_event_clean_removed        |      1 |
| suspension_or_resume_gap_gt7_calendar_days |      1 |

## 9. C: What is the C0/C1/C2 effect?

Sample policy summary:

| policy_id           | policy_name                                                      |   sample_rows |   removed_vs_C0 |   extreme_abs_gt20 |   positive_extreme |   negative_extreme |   train_extreme |   legacy_validation_extreme |
|:--------------------|:-----------------------------------------------------------------|--------------:|----------------:|-------------------:|-------------------:|-------------------:|----------------:|----------------------------:|
| C0_full_no_cap      | No event removal and no cap20 removal                            |       1324814 |               0 |                391 |                104 |                287 |             341 |                          50 |
| C1_event_clean_only | Only confirmed dirty event removed                               |       1324813 |               1 |                390 |                104 |                286 |             341 |                          49 |
| C2_cap20            | Confirmed dirty event removed and abs(target_return)>20% removed |       1324423 |             391 |                  0 |                  0 |                  0 |               0 |                           0 |

Existing v7 trade impact:

| policy_id           |   cumulative_return |   profit_factor |   removed_v7_trades |   removed_extreme_v7_trades |
|:--------------------|--------------------:|----------------:|--------------------:|----------------------------:|
| C0_full_no_cap      |           0.0563104 |         1.11989 |                   0 |                           0 |
| C1_event_clean_only |           0.0563104 |         1.11989 |                   0 |                           0 |
| C2_cap20            |           0.0563104 |         1.11989 |                   0 |                           0 |

## 10. E: Is v7 still positive after 10bp + impact + participation + partial/zero fill?

At `capital=100000`, `participation_cap=10%`, E5 cumulative return is
`-0.012279`, PF is `0.986399`.

## 11. E: How is fill quality across capital sizes?

|    capital |   avg_fill |   min_return |   max_return |
|-----------:|-----------:|-------------:|-------------:|
| 100000     |   0.978641 |   -0.0442806 |   -0.0122789 |
| 300000     |   0.887405 |   -0.0594071 |   -0.0255105 |
| 500000     |   0.81285  |   -0.0664783 |   -0.0360206 |
|      1e+06 |   0.673567 |   -0.070459  |   -0.0419203 |
|      5e+06 |   0.320189 |   -0.0687821 |   -0.0187236 |

## 12. E: Are there non-executable, zero-fill or deferred-sell issues?

E5 includes limit/suspension checks and zero sell-bar deferral. The known
`603268.SH 20260416` zero exit amount event is deferred to the next positive
amount bar. Full event details are in `batch1_603268_zero_exit_analysis.md`.

## 13. Should Batch 2A start?

Yes, but only after accepting that Batch 2A is still an audit/research step, not a deployment step.

## 14. What should Batch 2A prioritize among U/L/F?

Priority should be `U` first, then `F`, then `L`.

Basis:

- `U`: rolling liquidity membership materially changes the candidate set and can
  be audited without retraining or label changes.
- `F`: execution and capacity stress show that transaction friction can erase
  the v7 edge, so factor robustness should be judged under fixed execution costs.
- `L`: label changes are higher risk because they require retraining and a fresh
  out-of-sample protocol; they should wait until the universe and execution
  measurement harness is accepted.

This recommendation is based on audit stability and implementation risk, not on
choosing the highest old-validation-period return.
