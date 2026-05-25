# Handoff: Batch 4A to Batch 4B

## Completed Batch

- Completed: `Batch 4A model training plan freeze`.
- Output directory: `reports/tushare/v9_swing_research/batch4A_model_training_plan/`.
- No model training, hyperparameter search, prediction generation, or trading was performed.

## Frozen Plan

- Primary horizon: `5d`.
- Primary label: `fwd_ret_5d_open`.
- Primary features: `stock_ret_5d, stock_ret_20d, stock_ret_60d, turnover_rate, log_amount, log_total_mv`.
- Split: chronological train 20180102-20221230, validation 20230103-20241231, research_holdout 20250102-20260522.
- Baseline: Batch3D 5d simple rules and matched random baseline.

## Required Inputs For Next Step

- `reports/tushare/v9_swing_research/batch4A_model_training_plan/batch4A_model_training_plan.md`
- `reports/tushare/v9_swing_research/batch4A_model_training_plan/batch4A_feature_freeze.csv`
- `reports/tushare/v9_swing_research/batch4A_model_training_plan/batch4A_label_freeze.yaml`
- `reports/tushare/v9_swing_research/batch4A_model_training_plan/batch4A_split_config.yaml`
- `reports/tushare/v9_swing_research/batch4A_model_training_plan/batch4A_baseline_reference.csv`

## Blocked Actions

- Do not add non-frozen features in Batch4B.
- Do not switch primary horizon after seeing Batch4B results.
- Do not use current-snapshot industry or concept/theme fields as primary training features.
- Do not treat `research_holdout` as a pristine final test, because Batch3 diagnostics already viewed full history.
- Do not proceed to live or paper tracking from Batch4A.
