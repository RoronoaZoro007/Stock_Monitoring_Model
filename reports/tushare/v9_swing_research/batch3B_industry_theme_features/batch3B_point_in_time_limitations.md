# Batch 3B Point-In-Time Limitations

## Scope

- This batch builds industry-level and stock-relative industry diagnostics.
- It does not compute factor IC, run simple-rule backtests, train models, or choose parameters.

## Industry Field Risk

- `stock_basic.industry` is a current snapshot field according to Stage 0.
- Therefore industry features are allowed for diagnostics and grouping, but are not automatically approved as training features.
- If training needs industry classification, a point-in-time industry source should be locked or the leakage risk must be explicitly accepted.

## Concept / Theme Features

- Concept/theme features are disabled in this batch.
- Reason: no point-in-time historical concept membership or concept index dataset has been locked.
- Any future concept feature work must start with a separate data-foundation batch before IC or training.

## Timing

- Return, breadth, amount-share and crowding features use same-day close/amount information; they are suitable for post-close swing research, not intraday pre-close signal generation.
- Industry crowding score uses current amount share compared with the same industry's prior 252 trading days, with a 60-day minimum history. The percentile window excludes the current day and future days.
