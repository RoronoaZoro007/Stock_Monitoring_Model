# R1 tail_down Definition

## Exact Definition

`R1_tail_down = true` when:

```text
market_tail_ret_median < 0
```

In the locked v7 feature code, `market_tail_ret_median` is computed as:

```text
median(tail_ret_1420_1450 over the current candidate matrix)
tail_ret_1420_1450 = close_1450 / close_1420 - 1
```

Code evidence:

- `locked_artifacts/v7_cap20_strong_label_003/code/tushare_tail_model.py`
  computes `tail_ret_1420_1450` from 14:20 and 14:50 bars, then computes
  `market_tail_ret_median=("tail_ret_1420_1450", "median")`.
- `locked_artifacts/v7_cap20_strong_label_003/code/v7_cap20_audit.py`
  records the lineage as "当日全候选 <=14:50" and marks it available before
  signal time if full candidate real-time minute data is available.
- Batch 2A-1.5 and Batch 2B-R0 use the same gate direction:
  `tail_down` when `market_tail_ret_median < 0`.

## Fields Used

| Field | Meaning | Data Range |
|---|---|---|
| `tail_ret_1420_1450` | Each stock's 14:20 to 14:50 return | Current day 14:20 and 14:50 bars |
| `market_tail_ret_median` | Cross-sectional median of `tail_ret_1420_1450` | Current candidate matrix, <=14:50 |

## Signal-Time Availability

The gate is intended to be evaluated after the 14:50 bar is complete and before
the 14:55 entry bar is frozen.

It does not use:

- 15:00 close.
- Full-day high/low/close/amount after 14:50.
- Next-day prices.
- Future labels or target returns.

## Live Operational Requirement

`tail_down` is eligible for forward shadow tracking only when the daily score
file was generated from features with a maximum data timestamp of 14:50 or
earlier. The runner must freeze signals before the 14:55 entry bar.

The historical files prove the feature lineage. They do not prove future vendor
latency. If the data vendor cannot deliver the 14:50 bar and full candidate
cross-section before 14:55 on a live day, Candidate 1 and Candidate 2 must be
marked `no_trade_reason=tail_down_unavailable_before_freeze` for that day.

## Mapping To Existing Fields

| Existing Field | Forward Shadow Use |
|---|---|
| `market_tail_ret_median` | Direct input to `tail_down_flag` |
| `tail_ret_1420_1450` | Per-stock input to market median |

## Conclusion

Within the locked v7 feature definition, `tail_down` is computable after 14:50
and before 14:55 without using 15:00, next-day or future data. It can enter
paper tracking, subject to a daily operational check that the 14:50 feature file
was actually available before signal freeze.
