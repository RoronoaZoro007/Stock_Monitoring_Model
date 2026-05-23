# U2 Definition

## Exact Definition

`U2` is the existing downloaded Top3000 matrix filtered to the internal rolling
Top2500 by prior liquidity.

For each trading day `t`:

```text
avg_amount_60d(t, stock) = mean(amount over prior trading days only)
rank_amount_60d(t, stock) = descending rank of avg_amount_60d within that day's current Top3000 matrix
in_U2(t, stock) = rank_amount_60d <= 2500
```

The implemented audit code uses:

```text
amount.shift(1).rolling(60, min_periods=20).mean()
```

Therefore day `t` uses only `t-1` and earlier daily amount.

## What U2 Is Not

U2 is not an all-A-share unbiased rolling Top2500. It is only a rolling
liquidity filter inside the already downloaded v7 Top3000 candidate matrix.

## Data Timing

U2 membership can be known before 14:50 on day `t` because it uses only
historical daily amount through `t-1`.

It does not use:

- Day `t` intraday amount.
- Day `t` close or full-day amount.
- Day `t+1` data.
- Any validation-period future data beyond the current trade date.

## Insufficient History

The current audit implementation uses `min_periods=20`. If a stock has fewer
than 20 valid prior daily amount observations, `avg_amount_60d` is `NaN`; the
stock does not receive a valid rank and is not included in U2.

If a stock has 20 to 59 valid prior observations, the mean of available valid
prior amounts is used. This is a practical rolling-history rule for newer or
recently resumed names.

## Missing Amount

Missing `amount` values are not forward-filled or imputed. Rolling mean requires
at least 20 non-null prior observations. Stocks failing this requirement are
excluded from U2 for that date.

## Forward Shadow Use

S1 and Candidate 2 first compute the original v7 Top10, then keep only names
with `in_U2=true`. They do not refill dropped stocks. If no names remain, the
strategy records a no-trade day for that line.
