# Batch 4D Limitations

- This batch is a deeper audit of two pre-selected simple rules, not a model or live trading system.
- It does not add rules, change TopN, tune parameters, change labels or use validation results to select a new rule.
- The ledger uses a 20% capital sleeve per signal day to avoid hidden leverage from overlapping 5-day holds.
- Mark-to-market uses adjusted daily prices; intraday slippage and true order book execution are not modeled.
- Capacity uses signal-day amount, which is ex-ante at post-close signal time. T+1 entry amount is not used for selection.
- Fill policy is partial-fill cash drag: unfilled cash remains idle and is not reallocated.
- Entry limit-up/suspension blocks are handled in execution scenarios, but exit limit-down/suspension is flagged rather than delayed.
- Research holdout contains no weak trend-regime signal days, so weak-market robustness remains unresolved.
- Current-snapshot industry classification remains a point-in-time caveat for industry concentration reporting.
