# Batch 2A-1 F Limitations

F uses locked v7 scores and only changes candidate filtering before taking daily
Top10. It does not retrain, refit, alter features, alter labels, alter TopN, or
alter sell rules.

Reference amount A:

`tail_1430_1450_amount = expm1(amount_sofar_log) * tail_amount_share`

Reference amount B:

`min(tail_1430_1450_amount, past20_exit_window_amount_proxy)`

`past20_exit_window_amount_proxy` is computed from raw 5-minute bars as the
rolling mean of prior 20 trading days' `09:30-10:30` amount. It is shifted by
one trading day, so day t uses only t-1 and earlier data. Current proxy coverage
inside the validation candidate matrix is `1.0000`.

F is a single-factor capacity filter experiment. U+F combinations are not a
formal Batch 2A-1 conclusion and are reserved for Batch 3.
