# Forward Shadow Evaluation Plan

## Locked Lines

The forward shadow system tracks four lines:

| Line | Role | Logic |
|---|---|---|
| Control 1 | Control | `S0_v7_original_top10` |
| Control 2 | Control | `S1_U2_filter_only_no_refill` |
| Candidate 1 | Candidate | `S0_v7_original_top10 + R1_tail_down` |
| Candidate 2 | Candidate | `S1_U2_filter_only_no_refill + R1_tail_down` |

No line is a live-trading recommendation. All lines are paper-tracking only.

## Tracking Horizon

Minimum tracking horizon:

1. 60 new trading days.
2. At least 20 `tail_down` trigger days.

If 60 trading days pass and `tail_down` trigger days are still fewer than 20,
continue until the earlier of:

1. 90 new trading days.
2. 20 `tail_down` trigger days.

## Daily Process

1. Build the locked v7 daily score matrix after the 14:50 bar is complete.
2. Verify the score matrix uses only data available at or before 14:50.
3. Compute `tail_down_flag = market_tail_ret_median < 0`.
4. Compute U2 membership from `t-1` and earlier daily amount.
5. Freeze all four line signals before the 14:55 entry bar.
6. Write daily signal, ledger placeholder and execution-quality placeholder files.
7. After the exit window is observable, settle Ledger A and Ledger B without
   changing candidate rules.

## Ledgers

Ledger A: `theoretical_5bp_ledger`

- Uses the v7 theoretical 5bp one-way slippage cost basis.
- Used as continuity check versus locked v7 backtest.

Ledger B: `execution_10bp_impact_ledger`

- Uses 10bp one-way slippage plus impact and execution-quality accounting.
- Used as the main forward paper-tracking ledger.

## Future Evaluation Standards

Candidate 1 and Candidate 2 need to satisfy all of the following in the future
paper-tracking period before any further research step:

1. `10bp + impact` cumulative return > 0.
2. PF > 1.05, preferably > 1.10.
3. Maximum drawdown not materially worse than legacy observation.
4. `tail_down` trigger days >= 20.
5. Return after removing largest 3 profit days not materially negative.
6. Better than matched random baseline over the same period.
7. Candidate 2 not materially worse than Candidate 1.
8. Average daily holdings not too low for usable execution.
9. Execution quality does not materially deteriorate.

These are future evaluation standards only. They must not be used to alter the
currently locked candidates.

## Matched Random Baseline

For each candidate evaluation checkpoint:

1. Let `N` be the number of `tail_down` trigger days in the checkpoint period.
2. Randomly draw `N` days from the same forward period.
3. Use the same base strategy and holding-count logic.
4. Repeat 1000 times.
5. Report the actual return percentile versus the random distribution.

The random baseline is only for stage evaluation. It must not be used for daily
parameter changes.

## Stop Conditions

Forward shadow tracking should not be interpreted as live readiness. Any move
from paper tracking to simulation or live execution requires a separate approval
batch and a new audit of data latency, execution, risk limits and compliance.
