# Batch 0 Conclusion

## 1. Batch 0 changed what?

Batch 0 added a unified evaluator at `research/v8_research/evaluate_strategy.py`
and generated local evaluation outputs under `reports/tushare/v8_research/batch0_evaluator/`.
It standardizes metric calculation, cost/slippage scenarios, execution flags,
capacity/participation scenarios, random Top10 baseline and fixed simple rule baselines.

## 2. Batch 0 did not change what?

It did not change the locked v7 model, features, label, TopN, exit rule, universe,
or any training data. It did not start Batch 1 to Batch 5.

## 3. Was `v7_locked` modified?

No.

## 4. Was the model retrained?

No.

## 5. Were features, label, TopN or exit rule modified?

No.

## 6. Were the locked v7 core results reproduced?

Yes.

| Metric | Locked reference | Batch 0 reproduced | Difference |
|---|---:|---:|---:|
| 5bp cumulative return | 0.0563104482 | 0.0563104482 | 0.000000000000 |
| PF | 1.1198887096 | 1.1198887096 | 0.000000000000 |
| AUC | 0.5334428793 | 0.5334428793 | 0.000000000000 |
| 10bp cumulative return | -0.0032272496 | -0.0032272496 | 0.000000000000 |

## 7. If there is a difference, what caused it?

The reproduced core metrics are numerically equal within floating-point tolerance.
No material difference was observed.

## 8. Can this evaluator be used for later batches?

Yes for Batch 1, Batch 2A, Batch 2B, Batch 3 and Batch 4 style comparisons,
provided each later batch writes signals or scores in the same schema. It already
accepts signal/score/candidate inputs, cost settings, slippage settings, impact
cost, capital levels, participation caps, partial/zero fill and limit/suspension
execution flags. Walk-forward in Batch 5 can use the same evaluator per fold,
but the fold runner should be added separately when that batch is approved.

## 9. Should we enter Batch 1?

Yes, after this Batch 0 output is reviewed. The reason is that the evaluator
now reproduces v7 exactly and gives a fixed measurement harness. Batch 1 should
still be accepted separately before execution, and it must not optimize against
`20260224-20260520` as a final test set.

## Additional Batch 0 facts

| Item | Value |
|---|---:|
| Selected trades | 580 |
| Trading days | 58 |
| Random baseline runs | 1000 |
| Random baseline cumulative return mean | -0.0879748879 |
| v7 percentile vs random cumulative return | 0.9990 |
| Best fixed simple benchmark in this report | ai_model_score_top10 |
| Best fixed simple benchmark cumulative return | 0.0563104482 |

