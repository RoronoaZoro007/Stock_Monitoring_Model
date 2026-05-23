# Batch 1 Universe Limitations

Current data only covers the previously downloaded v7 Top3000 universe. Therefore
Batch 1 can only audit rolling liquidity filters **inside the existing Top3000**.
It must not be described as an unbiased all-A-share daily rolling Top3000.

U0 is the locked v7 static Top3000 candidate set available in the validation
score matrix. U1/U2/U3 are internal rolling liquidity filters using only `t-1`
and earlier `avg_amount_60d`.

A full daily score matrix exists for the locked validation period, so Batch 1
does reselect Top10 inside U0/U1/U2/U3 for audit comparison. This is not model
retraining and does not change v7_locked.

The period `20260224-20260520` remains a legacy validation interval and is not
used here for final model selection.
