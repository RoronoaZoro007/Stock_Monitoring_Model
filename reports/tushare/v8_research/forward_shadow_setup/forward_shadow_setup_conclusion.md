# Forward Shadow Setup Conclusion

Generated at Beijing time: 2026-05-23T22:25:57+08:00.

## 1. What Changed?

Batch F0 locked four forward shadow tracking lines, wrote the `tail_down` and
U2 definitions, created immutable candidate configuration files, added a
paper-only runner, and documented the forward output schema and evaluation plan.

## 2. What Did Not Change?

It did not change v7_locked, model weights, features, labels, sell rules, TopN
main ranking, execution-aware labels, Batch 3, Batch 4 or Batch 5.

## 3. Was v7_locked Modified?

No.

## 4. Was The Model Retrained?

No.

## 5. Were Features, Labels Or Sell Rules Modified?

No.

## 6. Is tail_down Available Between 14:50 And 14:55?

Yes under the locked v7 feature lineage. `tail_down` uses
`market_tail_ret_median < 0`, where `market_tail_ret_median` is the current-day
cross-sectional median of `tail_ret_1420_1450`. That field uses 14:20 and 14:50
bars only and does not use 15:00, next-day or future data.

Operational caveat: future vendor latency must be checked daily. If the 14:50
score matrix is not available before the 14:55 freeze, tail-down candidates must
record a no-trade reason for that day.

## 7. Does U2 Membership Use Only t-1 And Earlier Data?

Yes. U2 uses prior daily amount only:

```text
amount.shift(1).rolling(60, min_periods=20).mean()
```

The rank is computed inside the existing downloaded Top3000 matrix. It is not an
all-A-share unbiased Top2500.

## 8. Are The Four Lines Locked?

Yes.

| Role | Strategy |
|---|---|
| Control 1 | `S0_v7_original_top10` |
| Control 2 | `S1_U2_filter_only_no_refill` |
| Candidate 1 | `S0_v7_original_top10_tail_down` |
| Candidate 2 | `S1_U2_filter_only_no_refill_tail_down` |

## 9. Were Immutable Candidate Configs Generated?

Yes. The four YAML files in this folder are the locked forward-shadow configs.
They explicitly prohibit auto-ordering and auto-parameter changes.

## 10. Was The Daily Paper-Tracking Output Structure Established?

Yes. `forward_shadow_runner.py` writes:

- `reports/tushare/v8_forward_shadow/daily_signals/YYYYMMDD_signals.csv`
- `reports/tushare/v8_forward_shadow/daily_ledgers/YYYYMMDD_ledgers.csv`
- `reports/tushare/v8_forward_shadow/daily_execution_quality/YYYYMMDD_execution_quality.csv`
- cumulative summary/status/trade/execution-quality CSVs under
  `reports/tushare/v8_forward_shadow/`

## 11. Are There Any Blockers?

No blocker for paper tracking. The remaining operational dependency is that the
daily v7 score matrix must be produced from <=14:50 data and available before
14:55. If that fails on a given day, the affected candidate lines must record a
no-trade reason instead of trading on stale or late data.

## 12. Is L Label Retraining Still Not Recommended?

Yes. This setup does not create evidence for execution-aware label retraining.

## 13. Is Live Trading Still Not Recommended?

Yes. This is paper tracking only and does not authorize simulation trading or
live trading.

## 14. Next Step

Only start daily forward shadow tracking. Do not continue historical parameter
optimization and do not start L retraining or Batch 3 combination research.
