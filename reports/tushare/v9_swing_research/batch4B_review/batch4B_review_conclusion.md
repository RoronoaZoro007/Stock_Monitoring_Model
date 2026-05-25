# Batch 4B Review

## Scope

- This batch audits why Batch4B lightweight models did not beat Batch3D simple rules.
- No model retraining, parameter search, feature change, label change, horizon change, or v7_locked modification was performed.

## Core Validation Comparison

- Best validation simple rule: `low_log_total_mv`.
- Best validation model: `ridge_primary_alpha1`.
- Mean daily return gap, model minus rule: `-0.013211`.
- Best model vs best rule average overlap ratio: `0.1098`.
- Best model vs best rule replacement return difference: `-0.013211`.
- Best model validation size_q1 share: `0.9863`.
- Best model validation liquidity_q1 share: `1.0000`.
- Reconstructed Batch3D simple-rule trade details exactly match Batch4B reference split summaries; see `batch4B_review_reconstruction_parity.csv`.
- Best model top coefficients: `[{'feature': 'log_total_mv', 'coefficient': -0.0030572107061743, 'abs_coefficient': 0.0030572107061743}, {'feature': 'log_amount', 'coefficient': -0.0020490605384111, 'abs_coefficient': 0.0020490605384111}, {'feature': 'turnover_rate', 'coefficient': -0.0011147715849801, 'abs_coefficient': 0.0011147715849801}, {'feature': 'stock_ret_60d', 'coefficient': -0.0010739160934463, 'abs_coefficient': 0.0010739160934463}, {'feature': 'stock_ret_5d', 'coefficient': -0.0008905680733732, 'abs_coefficient': 0.0008905680733732}, {'feature': 'stock_ret_20d', 'coefficient': -0.0002614363911561, 'abs_coefficient': 0.0002614363911561}]`.

## Interpretation

- The model did not create a clearly independent alpha layer. Its strongest coefficients and selections are dominated by low market value, low amount and low turnover exposures.
- The simple rules are already a direct expression of the same low-size/low-liquidity/reversal signal. Batch4B models mostly repackage that exposure, then replace part of the simple-rule basket with weaker names.
- The validation advantage of simple rules is not explained by missing model complexity; it is visible under the same gross cohort label and same Top20 count.
- Low-size/low-liquidity concentration is also a capacity risk, so simply training a more complex model is not justified until the simple-rule exposure passes cost/capacity robustness.

## Data Foundation Check

- Stage0 raw manifests and Batch2 clean panel/label audit already exist and were used as the data foundation.
- This review did not find evidence that the Batch4B underperformance is primarily caused by raw data download or cleaning failure.
- Remaining data caveats are unchanged: no independent vendor cross-check, current-snapshot industry limitation, and gross-return rather than executable-ledger returns.

## Gate

- Gate result: `block_batch5A_and_complex_models`.
- Recommendation: do not enter Batch5A for the current lightweight model set.
- Recommendation: do not train more complex models yet.
- Next practical review should be simple-rule robustness/capacity audit or a new Batch4A amendment with a clearly different objective. It should not tune on the same validation gap.
