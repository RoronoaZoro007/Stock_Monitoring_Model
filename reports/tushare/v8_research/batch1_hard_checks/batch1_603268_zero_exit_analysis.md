# 603268.SH Zero Exit Analysis

Signal: `603268.SH`, trade date `20260416`, intended exit `20260417 10:30`.

| Item | Value |
|---|---:|
| Original v7 target_return | -0.0019985011 |
| Original intended exit bar amount | 0.00 |
| Original gross return reconstructed from v7 | 0.0000000000 |
| E5 zero-exit deferred? | True |
| E5 effective exit date | 20260420 |
| E5 effective exit time | 09:30 |
| E5 effective exit amount | 78616990.00 |
| E5 effective gross return after deferral | 0.0224625759 |
| Gross return change from zero-exit deferral | 0.0224625759 |

Answers:

1. Original v7 backtest used the locked `target_return`; it did not reject this
   trade because the intended 10:30 sell bar amount was zero.
2. Batch 1 E5 treats a zero sell bar amount as non-executable at that bar.
3. E5 defers the sell to the next available 5-minute bar with positive amount.
4. The gross-return impact is shown in the table above.
5. Other exit samples with amount `<=100000` are exported to
   `batch1_low_exit_amount_samples.csv`.
6. Buy samples with amount `<=100000` are exported to
   `batch1_low_entry_amount_samples.csv`.

