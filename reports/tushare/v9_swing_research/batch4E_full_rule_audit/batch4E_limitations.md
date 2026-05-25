# Batch 4E Limitations

- This is a full audit of two pre-selected simple rules, not an optimization batch.
- It does not train models, add rules, tune thresholds, change labels, change TopN, or modify v7_locked.
- Exit-delay simulation uses daily close data, not intraday order book data.
- If a blocked exit remains unresolved after 5 trading days, the position is marked at the last finite close inside the delay window and explicitly flagged as unresolved.
- Weak-market validation uses the frozen `trend_regime=weak` labels. The latest research holdout has no weak signal days, so truly recent weak-market validation is unavailable.
- Validation weak-market samples have already been observed in earlier audits; they are stress evidence, not final unseen proof.
- Industry fields remain current-snapshot caveats inherited from previous v9 batches.
