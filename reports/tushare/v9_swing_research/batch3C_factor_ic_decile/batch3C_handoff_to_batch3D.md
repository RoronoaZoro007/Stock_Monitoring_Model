# Handoff: Batch 3C to Batch 3D

## Completed Batch

- Completed: `Batch 3C factor IC and decile diagnostics`
- Output directory: `reports/tushare/v9_swing_research/batch3C_factor_ic_decile/`
- No simple-rule backtest, model training, or parameter optimization was performed.
- Long-running steps are checkpointed under `_checkpoints/`; rerun without `--refresh` resumes completed chunks, while `--refresh` clears checkpoints and recomputes from scratch.

## Core Conclusion

- Candidate factor rows by pre-registered screen: `39`
- Top candidate diagnostics: `[{"horizon": "10d", "factor": "log_amount", "mean_ic": -0.08092797953451496, "icir": -8.609218296569827, "positive_ic_rate": 0.29376854599406527, "direction_adjusted_mean_spread": 0.01103299492813537}, {"horizon": "10d", "factor": "turnover_rate", "mean_ic": -0.06742755096012658, "icir": -6.375743524869499, "positive_ic_rate": 0.3486646884272997, "direction_adjusted_mean_spread": 0.005825915989627664}, {"horizon": "5d", "factor": "log_amount", "mean_ic": -0.0632037543090167, "icir": -6.967771251706737, "positive_ic_rate": 0.31820424272323633, "direction_adjusted_mean_spread": 0.005049589922424069}, {"horizon": "5d", "factor": "turnover_rate", "mean_ic": -0.05410916718260954, "icir": -5.113659098997315, "positive_ic_rate": 0.3734583127775037, "direction_adjusted_mean_spread": 0.001826901460606969}, {"horizon": "10d", "factor": "stock_ret_20d", "mean_ic": -0.04981956450806205, "icir": -5.574994927817758, "positive_ic_rate": 0.3691308691308691, "direction_adjusted_mean_spread": 0.006717904044662261}, {"horizon": "10d", "factor": "stock_ret_60d", "mean_ic": -0.04969425749294563, "icir": -5.2633742589158485, "positive_ic_rate": 0.3730886850152905, "direction_adjusted_mean_spread": 0.006251536804348985}, {"horizon": "3d", "factor": "log_amount", "mean_ic": -0.04913865959051971, "icir": -5.43605330344435, "positive_ic_rate": 0.354361754558896, "direction_adjusted_mean_spread": 0.0020362423721772575}, {"horizon": "10d", "factor": "stock_amount_share_in_industry", "mean_ic": -0.047757715883366726, "icir": -5.977317604065891, "positive_ic_rate": 0.3545994065281899, "direction_adjusted_mean_spread": 0.010371926563187814}, {"horizon": "10d", "factor": "stock_rel_industry_ret_20d", "mean_ic": -0.045155085194168575, "icir": -6.932100570410619, "positive_ic_rate": 0.3196803196803197, "direction_adjusted_mean_spread": 0.006911028686252084}, {"horizon": "3d", "factor": "turnover_rate", "mean_ic": -0.0435075162103312, "icir": -4.051274582020062, "positive_ic_rate": 0.38984721537703304, "direction_adjusted_mean_spread": 0.0003524307546356714}, {"horizon": "5d", "factor": "stock_ret_20d", "mean_ic": -0.041275498686429754, "icir": -4.460130831686827, "positive_ic_rate": 0.3866467364225212, "direction_adjusted_mean_spread": 0.003572093168558866}, {"horizon": "5d", "factor": "stock_ret_60d", "mean_ic": -0.04035734912095861, "icir": -4.099482777713901, "positive_ic_rate": 0.4072191154041688, "direction_adjusted_mean_spread": 0.0032678404341276775}, {"horizon": "10d", "factor": "stock_rel_industry_ret_60d", "mean_ic": -0.038464399828459804, "icir": -5.574287807382815, "positive_ic_rate": 0.3659531090723751, "direction_adjusted_mean_spread": 0.00590767832584814}, {"horizon": "5d", "factor": "stock_rel_industry_ret_20d", "mean_ic": -0.038094133277598134, "icir": -5.713210675860623, "positive_ic_rate": 0.34728450423517687, "direction_adjusted_mean_spread": 0.0037890577842933986}, {"horizon": "5d", "factor": "stock_amount_share_in_industry", "mean_ic": -0.03563659610332773, "icir": -4.521049640378811, "positive_ic_rate": 0.36901825357671436, "direction_adjusted_mean_spread": 0.004947836234120521}, {"horizon": "10d", "factor": "log_total_mv", "mean_ic": -0.03424122029807453, "icir": -2.999322905920413, "positive_ic_rate": 0.39119683481701284, "direction_adjusted_mean_spread": 0.010703784635616333}, {"horizon": "3d", "factor": "stock_ret_20d", "mean_ic": -0.03299899375436742, "icir": -3.4663285093974543, "positive_ic_rate": 0.4116475858636137, "direction_adjusted_mean_spread": 0.0014151915337535794}, {"horizon": "5d", "factor": "stock_rel_industry_ret_60d", "mean_ic": -0.03195710964921819, "icir": -4.514040542586672, "positive_ic_rate": 0.38942552109811895, "direction_adjusted_mean_spread": 0.0031622407543799836}, {"horizon": "3d", "factor": "stock_ret_60d", "mean_ic": -0.03125837848178513, "icir": -3.1046790493827388, "positive_ic_rate": 0.4245810055865922, "direction_adjusted_mean_spread": 0.0013923118358983922}, {"horizon": "3d", "factor": "stock_rel_industry_ret_20d", "mean_ic": -0.03037557936729157, "icir": -4.4724216093930655, "positive_ic_rate": 0.3782976605276257, "direction_adjusted_mean_spread": 0.0017211424099747982}]`
- Concept/theme factors remain disabled.
- Industry snapshot factors remain diagnostic-only unless point-in-time industry classification is added or leakage risk is explicitly accepted.
- The strongest diagnostics are mostly negative IC signals: low liquidity, low turnover, low stock amount share inside industry, and weak medium-term momentum rank higher for future returns.

## Required Inputs For Next Step

- `reports/tushare/v9_swing_research/batch3C_factor_ic_decile/batch3C_candidate_factor_screen.csv`
- `reports/tushare/v9_swing_research/batch3C_factor_ic_decile/batch3C_rankic_summary.csv`
- `reports/tushare/v9_swing_research/batch3C_factor_ic_decile/batch3C_decile_return_summary.csv`
- `reports/tushare/v9_swing_research/batch3C_factor_ic_decile/batch3C_top_bottom_spread.csv`
- `data_tushare/clean/v9/v9_swing_labels_3_5_10.parquet`
- `data_tushare/clean/v9/v9_stock_industry_features.parquet`

## Next Step

Run `Batch 3D simple rule baseline` to test whether simple rules built from the strongest IC factors can beat random baselines.
Batch 3D must also be implemented with checkpoint/resume support before any long-running computation.

## Blocked Actions

- Do not train models.
- Do not freeze model features until Batch 3D passes.
- Do not enable concept/theme factors without a point-in-time data foundation.
