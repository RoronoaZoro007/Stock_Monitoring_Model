# v7_cap20_strong_label_003 审计包

生成目录：`/Users/yuxiang.luo/Documents/Codex/2026-05-21/new-chat/reports/tushare/v7_cap20_audit`

## P0 硬伤结论

| 项 | 结论 | 证据文件 |
|---|---|---|
| 模型类型 | 尾盘隔夜选股模型，不是纯日内尾盘套利 | `p0_execution_contract.csv` |
| 交易成本 | 5.63% 为基础成本 + 5bp 单边滑点后的净收益 | `p0_p1_cost_slippage_stress.csv` |
| 股票池 | 静态 Top3000，排名区间 20251120-20260213；不含验证期 20260224 之后，但对早期训练期存在股票池前视 | `p0_universe_audit.csv` |
| 验证集调参 | v7 是看过 v1-v8 验证结果后选出的；当前没有完全未使用测试集 | `p0_model_versions_v1_v8.csv` |
| 未来函数 | 34 个基础因子表逐项列出；未使用当日 15:00 后或次日数据，但市场截面因子需要 14:50 前全市场数据 | `p0_feature_lineage.csv` |
| 日线修复 | 2173 个 raw daily 缺失交易日已由分钟线聚合修复 | `p0_daily_repair_2173_missing_days.csv` |
| 极端收益 | 原始修复特征 `abs(target_return)>20%` 共 391 行，正 104、负 287；主清洗后剩 390 行，正 104、负 286 | `p0_extreme_target_return_raw_391.csv`, `p0_extreme_target_return_391.csv` |
| 涨跌停/成交 | 当前仅过滤 14:50 近涨跌停样本，未显式模拟涨停买不进/跌停卖不出/冲击成本 | `p0_limit_suspend_fill_diagnostics.csv` |

## P1 模拟盘准入结论

| 项 | 当前结果 |
|---|---:|
| v7 验证期累计收益 | 5.63% |
| v7 验证期最大回撤 | -6.14% |
| v7 验证期 PF | 1.1199 |
| 随机 Top10 累计收益 95 分位 | -0.52% |
| 模型累计收益在 1000 次随机中的百分位 | 0.999 |
| 最大回撤区间 | 20260305 至 20260320 |

模拟盘前必须补：完全未使用测试集、涨跌停可成交性、冲击成本、滚动前向验证。

## P2 优化文件索引

- 因子表：`p0_feature_lineage.csv`
- 因子重要性：`p2_feature_importance_permutation_auc_sample50k.csv`
- 因子组消融：`p2_feature_ablation.csv`
- 简单规则基准：`p2_rule_benchmarks.csv`
- 容量测算：`p2_capacity_summary.csv`
- 最大回撤拆解：`p2_max_drawdown_summary.json`
