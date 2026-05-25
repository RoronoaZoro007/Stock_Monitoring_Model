# Handoff: Batch 3D to Batch 4A

## Completed Batch

- Completed: `Batch 3D simple rule baseline`
- Output directory: `reports/tushare/v9_swing_research/batch3D_simple_rule_baseline/`
- No model training, hyperparameter search, concept/theme feature enablement, or live/sim trading was performed.
- Long-running computations support checkpoint/resume by default.

## Core Conclusion

- Gate result: `pass_to_batch4A_plan_freeze`
- Deterministic rule rows: `54`
- Matched random rows: `54`
- Top rule diagnostics: `[{"rule_id": "low_log_total_mv", "horizon": "10d", "cumulative_return": 5793460113.173466, "mean_daily_return": 0.015177921908974884, "daily_win_rate": 0.5316518298714145, "profit_factor": 1.6994775397228894, "max_drawdown": -0.996459142863129, "rule_cumret_percentile": 1.0, "uses_industry_snapshot": false, "eligible_for_training_plan": true}, {"rule_id": "low_stock_amount_share_in_industry", "horizon": "10d", "cumulative_return": 52357512.53313346, "mean_daily_return": 0.011017191198098821, "daily_win_rate": 0.549455984174085, "profit_factor": 1.6710208911641868, "max_drawdown": -0.9926810269270534, "rule_cumret_percentile": 1.0, "uses_industry_snapshot": false, "eligible_for_training_plan": true}, {"rule_id": "low_log_amount", "horizon": "10d", "cumulative_return": 2619562.4400753924, "mean_daily_return": 0.009476454131167935, "daily_win_rate": 0.5301681503461919, "profit_factor": 1.5899255566389687, "max_drawdown": -0.9951006788518085, "rule_cumret_percentile": 1.0, "uses_industry_snapshot": false, "eligible_for_training_plan": true}, {"rule_id": "low_log_total_mv", "horizon": "5d", "cumulative_return": 210416.81947605, "mean_daily_return": 0.00796667721496163, "daily_win_rate": 0.5209669462259496, "profit_factor": 1.498413343216287, "max_drawdown": -0.9490676611415909, "rule_cumret_percentile": 1.0, "uses_industry_snapshot": false, "eligible_for_training_plan": true}, {"rule_id": "combo_low_size_low_liquidity", "horizon": "10d", "cumulative_return": 132079.86014452938, "mean_daily_return": 0.007767416178109583, "daily_win_rate": 0.539564787339268, "profit_factor": 1.466978008827714, "max_drawdown": -0.990594991635384, "rule_cumret_percentile": 1.0, "uses_industry_snapshot": false, "eligible_for_training_plan": true}, {"rule_id": "low_stock_amount_share_in_industry", "horizon": "5d", "cumulative_return": 13863.297402321035, "mean_daily_return": 0.0057601616152103685, "daily_win_rate": 0.5328071040947213, "profit_factor": 1.4963370844565367, "max_drawdown": -0.883207702552175, "rule_cumret_percentile": 1.0, "uses_industry_snapshot": false, "eligible_for_training_plan": true}, {"rule_id": "low_log_amount", "horizon": "5d", "cumulative_return": 3328.258391085165, "mean_daily_return": 0.004964489070742337, "daily_win_rate": 0.5323137641835225, "profit_factor": 1.4464532395983063, "max_drawdown": -0.9024614423285247, "rule_cumret_percentile": 1.0, "uses_industry_snapshot": false, "eligible_for_training_plan": true}, {"rule_id": "low_log_total_mv", "horizon": "3d", "cumulative_return": 2922.0204815757, "mean_daily_return": 0.004974783994604902, "daily_win_rate": 0.5036964021685559, "profit_factor": 1.4200107319183415, "max_drawdown": -0.8092994394650468, "rule_cumret_percentile": 1.0, "uses_industry_snapshot": false, "eligible_for_training_plan": true}, {"rule_id": "low_stock_ret_60d", "horizon": "3d", "cumulative_return": 415.01702387211975, "mean_daily_return": 0.003962036164868288, "daily_win_rate": 0.5114271203656678, "profit_factor": 1.3002437392958868, "max_drawdown": -0.8334109691098514, "rule_cumret_percentile": 1.0, "uses_industry_snapshot": false, "eligible_for_training_plan": true}, {"rule_id": "low_stock_amount_share_in_industry", "horizon": "3d", "cumulative_return": 272.5155966373837, "mean_daily_return": 0.0032892318898613024, "daily_win_rate": 0.5332676195170034, "profit_factor": 1.382173688505725, "max_drawdown": -0.7156280356416418, "rule_cumret_percentile": 1.0, "uses_industry_snapshot": false, "eligible_for_training_plan": true}, {"rule_id": "combo_low_liquidity_weak_momentum", "horizon": "10d", "cumulative_return": 125972.9274723249, "mean_daily_return": 0.007651491877062872, "daily_win_rate": 0.5555555555555556, "profit_factor": 1.4692804496166734, "max_drawdown": -0.9866710185012929, "rule_cumret_percentile": 0.998, "uses_industry_snapshot": false, "eligible_for_training_plan": true}, {"rule_id": "combo_low_liquidity_weak_momentum", "horizon": "5d", "cumulative_return": 1198.8287140007444, "mean_daily_return": 0.004481165725271738, "daily_win_rate": 0.5460091509913574, "profit_factor": 1.3824080466605486, "max_drawdown": -0.9141151551298855, "rule_cumret_percentile": 0.996, "uses_industry_snapshot": false, "eligible_for_training_plan": true}, {"rule_id": "low_stock_ret_60d", "horizon": "5d", "cumulative_return": 1138.9051216875944, "mean_daily_return": 0.005148756985119134, "daily_win_rate": 0.5078800203355364, "profit_factor": 1.293406236330062, "max_drawdown": -0.9401913787283557, "rule_cumret_percentile": 0.996, "uses_industry_snapshot": false, "eligible_for_training_plan": true}, {"rule_id": "combo_low_size_low_liquidity", "horizon": "5d", "cumulative_return": 797.9159853960557, "mean_daily_return": 0.0041635089761313145, "daily_win_rate": 0.5441539220522941, "profit_factor": 1.365081115757021, "max_drawdown": -0.8881259261654656, "rule_cumret_percentile": 0.994, "uses_industry_snapshot": false, "eligible_for_training_plan": true}, {"rule_id": "low_stock_ret_20d", "horizon": "3d", "cumulative_return": 192.051406218964, "mean_daily_return": 0.0034287915814195597, "daily_win_rate": 0.5355898456943753, "profit_factor": 1.2684782416391944, "max_drawdown": -0.7352071065780218, "rule_cumret_percentile": 0.994, "uses_industry_snapshot": false, "eligible_for_training_plan": true}]`
- Main caveat: results are gross cohort diagnostics and do not include overlapping capital ledger, costs, impact or capacity.
- Industry snapshot rules remain diagnostic-only.

## Required Inputs For Next Step

- `reports/tushare/v9_swing_research/batch3D_simple_rule_baseline/batch3D_rule_backtest_summary.csv`
- `reports/tushare/v9_swing_research/batch3D_simple_rule_baseline/batch3D_random_baseline_summary.csv`
- `reports/tushare/v9_swing_research/batch3D_simple_rule_baseline/batch3D_rule_config.yaml`
- `reports/tushare/v9_swing_research/batch3D_simple_rule_baseline/batch3D_limitations.md`

## Next Step

If accepted, proceed only to `Batch 4A model training plan freeze`, not direct training.
Batch 4A must freeze horizon, features, label, split, baseline references and allowed training fields before any model is trained.

## Blocked Actions

- Do not train models before Batch 4A is accepted.
- Do not treat current-snapshot industry rules as clean point-in-time training features.
- Do not use these gross cohort returns as executable strategy returns.
