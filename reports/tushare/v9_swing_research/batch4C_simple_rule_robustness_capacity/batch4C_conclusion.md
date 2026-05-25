# Batch 4C Simple Rule Robustness and Capacity Audit

## Scope

- Audited Batch3D minimum-to-beat simple rules after Batch4B models failed to beat them.
- No model retraining, rule search, parameter tuning, feature change, label change, horizon change or v7_locked modification was performed.

## Core Findings

- Best research-holdout gross rule: `low_log_total_mv`, mean daily return `0.010211`.
- Best research-holdout capacity-stressed rule under signal_amount, 1,000,000 yuan capital, 5% participation cap, 40bps roundtrip cost: `low_log_total_mv`, mean daily return `0.006211`, avg fill ratio `1.0000`.
- Under a stricter signal_amount, 5,000,000 yuan capital, 1% participation cap, 40bps roundtrip cost stress, best research-holdout rule: `low_stock_ret_60d`, mean daily return `0.005923`, avg fill ratio `0.9809`.
- Research-holdout has no `weak` trend-regime days, so weak-market robustness cannot be validated in the most recent holdout. Validation weak-regime best rule: `low_stock_ret_60d`, mean daily return `0.024350`.
- Basic robustness screen pass count: `0` of `6` rules.

## Interpretation

- Simple rules are stronger than Batch4B models under gross cohort comparison, but their edge is still highly exposed to small-cap / low-liquidity implementation risk.
- Capacity is acceptable for small capital under 5% daily-amount participation, but stricter 1% participation at 5,000,000 yuan materially reduces fill ratio for the lower-liquidity rules.
- Weak-market and profit-concentration diagnostics must be treated as gating evidence before any forward paper tracking.
- Passing this audit would only justify a deeper simple-rule audit; it is not live readiness and not permission to resume complex model training.

## Gate

- Gate result: `review_required_no_forward_tracking`.
- If accepted, the next step should be a deeper simple-rule audit with a full overlapping-capital ledger, execution assumptions and point-in-time industry caveat review.
- Do not enter model Batch5A for the current Batch4B models.
