# Batch 2A-1 U Limitations

The U experiment uses the existing v7 Top3000 validation score matrix. It is an
internal rolling liquidity filter inside the downloaded Top3000, not an unbiased
all-A-share daily rolling Top3000.

The score matrix is complete for the legacy validation interval, so U0/U1/U2/U3
are reselected daily by locked v7 `score` without retraining, refitting,
feature changes, label changes, TopN changes or sell-rule changes.

Rolling liquidity uses `avg_amount_60d` computed with `t-1` and earlier daily
amount. It does not use future liquidity or post-validation data.

U1 is retained only as a sanity check because Batch 1 showed it is identical to
U0 inside the current Top3000 matrix.
