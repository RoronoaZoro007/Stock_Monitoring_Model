# Batch 3C Factor IC Limitations

## Scope

- This batch computes factor RankIC, ICIR, decile returns and split diagnostics only.
- It does not train models, run simple-rule backtests, choose trading parameters, or enable concept/theme factors.

## Universe

- `all_valid` keeps all rows with valid target and factor values.
- Primary `screened` universe additionally excludes ST rows, suspended rows, stocks listed for less than 120 days, and current-day close limit-up/limit-down rows.
- This is a diagnostic tradability screen, not a final trading universe.

## Point-In-Time

- Factors use T-day close/day-level data and are suitable for post-close swing research, not intraday decisions.
- Industry factors remain diagnostic because `stock_basic.industry` is a current snapshot field.
- Concept/theme factors are disabled because no point-in-time concept membership data is locked.

## Interpretation

- Positive IC means higher factor values rank with higher future returns.
- Negative IC can still be economically useful as an inverse rank, but must be validated against simple-rule baselines in Batch 3D.
- IC significance alone is not a trading result; costs, turnover and execution remain untested here.
