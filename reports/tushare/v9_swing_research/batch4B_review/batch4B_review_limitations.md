# Batch 4B Review Limitations

- This batch does not retrain or rescore models; it only audits existing Batch4B model outputs.
- Simple-rule selected trades are reconstructed with the frozen Batch3D rule definitions because Batch3D originally persisted daily returns, not trade-level selections.
- Review returns remain gross cohort returns; they are not execution-ledger returns and do not include fees, slippage, impact cost, overlapping capital usage, or capacity constraints.
- `amount` and `total_mv` are reported in their source data units and used as capacity proxies only.
- Research holdout is not a pristine final test because earlier diagnostic batches inspected the full history.
- Current-snapshot industry classification remains a known limitation; this review does not convert it into point-in-time industry membership.
