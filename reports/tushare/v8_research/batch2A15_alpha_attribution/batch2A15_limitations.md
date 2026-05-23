# Batch 2A-1.5 Limitations

This batch is attribution-only. It does not optimize parameters, retrain the
model, alter labels, alter features, change TopN, change sell rules, or claim a
new tradable v8 strategy.

U diagnostics are computed inside the existing downloaded v7 Top3000 matrix.
They are not all-A-share unbiased rolling-universe tests.

`market_dispersion_proxy` and `market_tail_vol_proxy` are diagnostic proxies
derived from the validation candidate matrix. They were not v7 model features
and are not used to train or choose a production rule here.

The legacy validation interval remains unsuitable as a final untouched test set.
