# Batch 3D Limitations

- This batch is a simple-rule baseline, not a model and not a trading strategy.
- No model training, hyperparameter optimization, feature selection by returns, concept/theme data or live trading is performed.
- Primary return is gross cohort return: each signal date selects Top20 equal-weight names and uses `T+1 open -> T+h close` labels.
- Cohort compounding is diagnostic and does not model overlapping capital usage across 3/5/10 trading-day holding periods.
- Transaction costs, impact cost, partial fills, stop rules and capacity are not included in this batch.
- Industry-derived rules use current-snapshot industry classification and are diagnostic-only until point-in-time industry data is locked.
- Random baselines are matched to each rule's eligible pool, active dates and selected count.
- Long-running work is checkpointed under `_checkpoints/`; rerun without `--refresh` resumes completed checkpoints.
