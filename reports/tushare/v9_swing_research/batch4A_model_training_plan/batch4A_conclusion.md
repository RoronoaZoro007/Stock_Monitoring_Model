# Batch 4A Conclusion

## What Changed

- Froze the v9 swing model training plan.
- Froze primary horizon, label, feature set, split config, baseline references and evaluation gates.
- Updated the project handoff to Batch 4B.

## What Did Not Change

- No model was trained.
- No feature was recomputed.
- No label was changed.
- No historical parameter optimization was performed.
- `v7_locked` was not modified or moved.

## Frozen Decisions

- Primary horizon: `5d`.
- Primary label: `fwd_ret_5d_open`.
- Primary training features: `6`.
- Conditional ablation-only features: `1`.
- Blocked/diagnostic-only features: `14`.
- Minimum-to-beat Batch3D baselines for 5d: `6`.

## Data-Supported Rationale

- 5d retains 7 usable Batch3C candidate factors; 6 are frozen as primary training features and 1 industry-dependent factor is kept as conditional ablation-only.
- 3d remains useful as a lower-tail-risk diagnostic but has thinner IC/spread.
- 10d remains diagnostic because the strongest gross returns are paired with near-total drawdowns in simple rules.
- Batch4B must prove improvement over simple rules, not merely reproduce low-size/low-liquidity exposure.

## Gate

- Gate result: `pass_to_batch4B_after_review`.
- Next step, if accepted: `Batch 4B lightweight model training` using only the frozen plan.
- Do not start Batch 4B automatically from this script.
