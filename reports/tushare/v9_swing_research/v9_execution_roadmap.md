# v9 Swing Research Execution Roadmap

本文档用于锁定 v9 中短线选股研究的阶段、依赖关系、验收口径和下一步动作。  
后续任何上下文压缩、换会话或恢复任务时，优先读取本文档、最新 `stage_status.csv` 和各阶段 `*_conclusion.md`。

## 0. 硬性边界

- 不修改 `v7_locked`。
- 不移动 `v7_locked` commit。
- 不在未完成因子 IC、简单规则基准和稳健性检查前训练新模型。
- 不用 2026-02-24 至 2026-05-20 的旧验证结果做最终参数选择。
- 不把当前阶段结论包装成实盘建议。
- 每个阶段完成后必须输出：当前分支、当前 commit、v7_locked 是否未移动、输出目录、核心结论、是否通过进入下一阶段。
- 所有长任务默认必须优先支持可中断续跑：分块 checkpoint、默认 resume、只有显式 `--refresh` 才全量重算。

## 1. 阶段与 Batch 映射

| 阶段 | 目的 | 对应 Batch | 当前状态 | 关键输出 | 下一步条件 |
|---|---|---|---|---|---|
| Stage 0 数据底座锁定 | 确认 raw 数据可审计、可复现 | `stage0_data_foundation` | 已完成 | raw manifest、coverage、字段字典、limitations、hash | 缺失交易日为 0，风险已写明 |
| Stage 1 历史日频补齐 | 补齐 2018 至今全 A 日线/复权/市值/停牌/涨跌停 | `batch1_history_daily` | 已完成 | API smoke test、download summary、coverage after | 5 个日频 endpoint 全部 2032/2032 |
| Stage 2 clean panel + label audit | 合并日频面板，生成 3d/5d/10d forward label 并审计 | `batch2_clean_panel_label_audit` | 已完成 | clean panel、labels、缺失/极端收益/市场状态初稿 | 标签有效率、极端收益和执行风险已量化 |
| Stage 3 市场状态标记 | 解决牛市训练、熊市失效风险 | `batch3A_market_regime` | 已完成 | 正式 regime 标签、阈值说明、分布、敏感性审计、handoff | 市场状态定义不使用未来收益调参 |
| Stage 4 行业/概念特征工程 | 刻画行业强度、拥挤度、抱团风险 | `batch3B_industry_theme_features` | 已完成 | 行业强度、行业成交集中度、个股行业暴露、概念禁用说明、handoff | 行业字段为当前快照，仅诊断；概念数据源未锁定，不启用 |
| Stage 5 因子 IC 与分层诊断 | 训练前判断因子是否有排序能力 | `batch3C_factor_ic_decile` | 已完成 | RankIC、ICIR、10 桶收益、Top-Bottom、分状态/行业/市值/流动性 IC | 已通过进入 Batch 3D；候选仅作简单规则基准输入，不可直接训练 |
| Stage 6 简单规则基准 | 训练前必须打败简单规则 | 建议 `batch3D_simple_rule_baseline` 或独立 Batch4 | 未开始 | 趋势、反转、量价、行业动量、流动性等规则基准 | 新模型必须显著优于简单规则才进入训练 |
| Stage 7 轻量模型研究 | 只在 Stage 5/6 通过后训练 | 后续 Batch4 | 禁止当前执行 | Logistic/Ridge/LightGBM rank model | 不能跳过前置 IC 和基准 |
| Stage 8 稳健性与前向纸面 | 牛熊震荡、抱团、高拥挤、未见数据验证 | 后续 Batch5/6 | 禁止当前执行 | walk-forward、forward paper tracking | 历史结果不能直接转实盘 |

## 2. 已完成内容

### Stage 0 数据底座锁定

输出目录：`reports/tushare/v9_swing_research/stage0_data_foundation/`

已锁定：

| endpoint | 覆盖 | 缺失 | 行数 |
|---|---:|---:|---:|
| daily | 2032/2032 | 0 | 9,306,920 |
| adj_factor | 2032/2032 | 0 | 9,561,479 |
| daily_basic | 2032/2032 | 0 | 9,242,185 |
| suspend_d | 2032/2032 | 0 | 109,775 |
| stk_limit | 2032/2032 | 0 | 11,450,099 |

重要限制：

- `stock_basic.industry` 和 `stock_basic.name` 是当前快照，不可直接当 point-in-time 行业历史。
- `namechange` 只能作为 ST proxy，不能当完整交易所状态表。
- 概念/题材成分数据尚未锁定，不得直接进入训练。

### Stage 1 历史日频补齐

输出目录：`reports/tushare/v9_swing_research/batch1_history_daily/`

结论：

- 已完成 2018-01-02 至 2026-05-22 全交易日下载。
- `daily / adj_factor / daily_basic / suspend_d / stk_limit` 覆盖均为 2032/2032。

### Stage 2 clean panel + label audit

输出目录：`reports/tushare/v9_swing_research/batch2_clean_panel_label_audit/`

核心结果：

| 周期 | 标签有效率 | 均值 | 中位数 | 胜率 | `abs(ret)>20%` |
|---|---:|---:|---:|---:|---:|
| 3d | 99.485% | 0.232% | -0.067% | 48.26% | 0.88% |
| 5d | 99.302% | 0.346% | -0.113% | 48.40% | 1.92% |
| 10d | 98.898% | 0.617% | -0.165% | 48.62% | 4.63% |

客观含义：

- 3d 标签最干净，但收益窗口短，可能噪声较高。
- 5d 是当前最合理的折中候选。
- 10d 均值更高，但极端样本更多，后续必须重点看是否由牛市、题材抱团或少数极端行情驱动。

### Stage 3 市场状态正式标记

输出目录：`reports/tushare/v9_swing_research/batch3A_market_regime/`

核心定义：

- `trend_regime`: `roll60_ew_ret >= +8%` 为 `strong`，`<= -8%` 为 `weak`，其余为 `neutral`。
- `extreme_selloff_flag`: `breadth_positive < 20%` 或 `ew_ret <= -3%`。
- `vol_regime`: `roll20_dispersion` 相对 t-1 扩展 20/80 分位。
- `industry_crowding_regime`: `top5_industry_amount_share` 相对 t-1 扩展 20/80 分位。
- `stock_concentration_regime`: `top100_amount_share` 相对 t-1 扩展 20/80 分位。

核心结果：

| 状态 | 天数 | 占比 |
|---|---:|---:|
| insufficient_history | 60 | 2.95% |
| neutral | 1132 | 55.71% |
| strong | 556 | 27.36% |
| weak | 284 | 13.98% |

重要限制：

- `trend_regime` 使用当日收盘后的市场数据，适合历史 swing 研究，不适合盘中实时判定。
- 波动和拥挤阈值使用 t-1 扩展分位，避免未来分布泄露。
- 行业拥挤仍为诊断字段，因为 Stage 0 已确认 `stock_basic.industry` 是当前快照字段。

Handoff：

- `reports/tushare/v9_swing_research/batch3A_market_regime/batch3A_handoff_to_batch3B.md`
- `reports/tushare/v9_swing_research/v9_current_handoff.md`

### Stage 4 行业/概念特征工程

输出目录：`reports/tushare/v9_swing_research/batch3B_industry_theme_features/`

核心产物：

- `data_tushare/clean/v9/v9_industry_daily_features.parquet`
- `data_tushare/clean/v9/v9_stock_industry_features.parquet`
- `reports/tushare/v9_swing_research/batch3B_industry_theme_features/batch3B_industry_feature_dictionary.csv`
- `reports/tushare/v9_swing_research/batch3B_industry_theme_features/batch3B_point_in_time_limitations.md`

核心结果：

| 项目 | 数值 |
|---|---:|
| 行业日频特征行数 | 225,541 |
| 个股相对行业特征行数 | 9,306,920 |
| 交易日 | 2,032 |
| 行业数 | 111 |

行业拥挤状态：

| 状态 | 行数 |
|---|---:|
| crowding_normal | 116,855 |
| crowding_low | 57,143 |
| crowding_high | 44,883 |
| insufficient_history | 6,660 |

重要限制：

- `stock_basic.industry` 是当前快照字段，本批次行业特征仅作为诊断和分组输入，不能默认作为无偏训练特征。
- `industry_crowding_score` 是行业相对自身过去 252 个交易日成交占比的拥挤分位，不是行业之间绝对成交额排名。
- 概念/题材特征保持禁用；没有 point-in-time 概念成分数据前，不进入 IC 或训练。

Handoff：

- `reports/tushare/v9_swing_research/batch3B_industry_theme_features/batch3B_handoff_to_batch3C.md`
- `reports/tushare/v9_swing_research/v9_current_handoff.md`

### Stage 5 因子 IC 与分层诊断

输出目录：`reports/tushare/v9_swing_research/batch3C_factor_ic_decile/`

执行说明：

- 本批次只做 RankIC、ICIR、decile、Top-Bottom spread 和分层 IC 诊断。
- 未训练模型、未跑简单规则回测、未启用概念/题材特征、未调参。
- 已实现可中断续跑：长步骤按日期块或 `horizon + factor` 写入 `_checkpoints/`；默认不带 `--refresh` 时自动跳过已完成块。

覆盖情况：

| 周期 | 标签有效行 | 筛选后有效行 | 筛选后有效率 |
|---|---:|---:|---:|
| 3d | 9,258,989 | 8,615,311 | 92.5689% |
| 5d | 9,241,951 | 8,600,989 | 92.4150% |
| 10d | 9,204,385 | 8,568,101 | 92.0616% |

核心结果：

| 项目 | 数值 |
|---|---:|
| 面板总行数 | 9,306,920 |
| screened universe 行数 | 8,639,432 |
| 预注册因子数量 | 21 |
| 通过候选筛选的 horizon-factor 行 | 39 |

Top 诊断结果显示：

- 10d `log_amount`：mean IC = -0.0809，ICIR = -8.6092，Top-Bottom direction-adjusted spread = 1.1033%。
- 10d `turnover_rate`：mean IC = -0.0674，ICIR = -6.3757。
- 5d `log_amount`：mean IC = -0.0632，ICIR = -6.9678。
- 主要有效方向是低成交额、低换手、低个股行业成交占比，以及部分中期弱动量/弱相对强度的反向排序。

重要限制：

- IC 是排序诊断，不是交易收益；不能直接作为实盘或模型训练结论。
- 行业字段仍为当前快照来源，行业相关因子只允许诊断，不能默认作为无偏训练字段。
- 负 IC 因子可能代表反向选择价值，但必须由 Batch 3D 简单规则基准验证交易意义、成本、换手和稳定性。

Handoff：

- `reports/tushare/v9_swing_research/batch3C_factor_ic_decile/batch3C_handoff_to_batch3D.md`
- `reports/tushare/v9_swing_research/v9_current_handoff.md`

## 3. Batch 3 不应该一次性做完的原因

用户提出的 Stage 3 至 Stage 6 都与 “Batch 3 因子 IC 和分层诊断” 有关，但不是同一个动作。

正确关系：

1. Stage 3 市场状态标记是 Batch 3C 分市场状态 IC 的前置输入。
2. Stage 4 行业/概念特征是 Batch 3C 分行业/拥挤度 IC 的前置输入。
3. Stage 5 因子 IC 是核心诊断阶段。
4. Stage 6 简单规则基准是训练模型前的最低门槛。

因此 Batch 3 应拆为：

| 子批次 | 名称 | 是否训练 | 是否调参 | 输出用途 |
|---|---|---:|---:|---|
| Batch 3A | market regime formalization | 否 | 否 | 生成正式市场状态标签 |
| Batch 3B | industry/theme feature engineering | 否 | 否 | 生成行业强度和拥挤度特征 |
| Batch 3C | factor IC and decile diagnostics | 否 | 否 | 判断哪些因子有排序能力 |
| Batch 3D | simple rule baseline | 否 | 否 | 判断是否值得进入模型训练 |

## 4. 下一步准确执行步骤

### Step 3A: 市场状态正式标记

目标：把 Batch2 的市场状态初稿升级成正式、可审计、不可事后调参的市场状态表。

输入：

- `data_tushare/clean/v9/v9_daily_panel.parquet`
- `data_tushare/clean/v9/v9_market_regime_initial.parquet`
- Stage0 raw manifest 和 limitations

预注册状态：

| 状态 | 计算方式 |
|---|---|
| strong / weak / neutral | 全市场等权 60 日收益的固定分位或固定阈值 |
| extreme_selloff | 当日上涨家数比例 < 20% 或等权收益落入历史低分位 |
| high_vol / low_vol | 全市场收益横截面标准差分位 |
| crowding_high / crowding_low | Top 行业成交额占比、Top 50/100/300 个股成交额占比 |

必须输出：

- `batch3A_market_regime_summary.csv`
- `batch3A_market_regime_daily.csv`
- `batch3A_regime_transition.csv`
- `batch3A_regime_limitations.md`
- `batch3A_conclusion.md`
- `batch3A_file_sha256.csv`

验收问题：

- 状态定义是否只用当日及以前数据？
- 每个状态样本数是否足够？
- 是否能覆盖 2018、2022、2023 等弱市/震荡区间？
- 是否存在明显用未来收益定义状态的问题？

完成后下一步提示：

> Batch 3A 已完成。请确认市场状态定义和样本分布是否可接受。若通过，下一步执行 Batch 3B 行业/题材特征工程；若不通过，只允许修改市场状态定义文档和重新审计，不进入 IC。

### Step 3B: 行业/概念特征工程

目标：刻画行业强度、成交集中度、抱团和拥挤度。

输入：

- `v9_daily_panel.parquet`
- Batch3A 市场状态表
- `stock_basic.industry` 当前快照，仅可作为诊断字段，不可默认 point-in-time

先做行业，概念暂缓：

- 行业强度：行业内等权 5/20/60 日收益。
- 行业广度：行业内上涨比例。
- 行业成交占比：行业成交额 / 全市场成交额。
- 行业拥挤度：行业成交占比的滚动分位。
- 个股行业相对强弱：个股收益 - 所属行业收益。

概念/题材处理原则：

- 如果没有历史 point-in-time 概念成分，不能训练使用。
- 可以在报告中标记“概念特征缺失”，后续另起数据源锁定批次。

必须输出：

- `batch3B_industry_feature_dictionary.csv`
- `batch3B_industry_daily_features.parquet`
- `batch3B_stock_industry_features.parquet`
- `batch3B_industry_crowding_summary.csv`
- `batch3B_point_in_time_limitations.md`
- `batch3B_conclusion.md`
- `batch3B_file_sha256.csv`

验收问题：

- 行业字段是否存在未来信息风险？
- 行业强度和拥挤度是否按日期滚动计算？
- 是否明确概念特征未启用或单独锁定？

完成后下一步提示：

> Batch 3B 已完成。请确认行业字段使用限制。若接受当前快照行业只做诊断，不做训练特征，则进入 Batch 3C 因子 IC；若需要 point-in-time 行业/概念，需要先补数据，不能直接 IC。

### Step 3C: 因子 IC 和分层诊断

目标：训练前判断因子是否真的有排序能力。

候选因子组：

| 因子组 | 示例 |
|---|---|
| 动量 | 5/20/60 日收益、突破均线、相对行业强度 |
| 反转 | 1/3/5 日短期反转 |
| 波动 | 20/60 日波动、振幅、回撤 |
| 流动性 | 20/60 日成交额、换手率、成交额变化 |
| 市值 | total_mv、circ_mv、log_mv |
| 交易约束 | ST、停牌、涨停/跌停、上市天数 |
| 行业拥挤 | 行业成交占比、行业成交分位 |

对每个 horizon 分别做：

- 3d forward return
- 5d forward return
- 10d forward return

必须输出：

- `batch3C_factor_dictionary.csv`
- `batch3C_rankic_summary.csv`
- `batch3C_ic_by_year.csv`
- `batch3C_ic_by_regime.csv`
- `batch3C_ic_by_industry.csv`
- `batch3C_ic_by_size_liquidity.csv`
- `batch3C_decile_return_summary.csv`
- `batch3C_top_bottom_spread.csv`
- `batch3C_factor_limitations.md`
- `batch3C_conclusion.md`
- `batch3C_file_sha256.csv`

验收问题：

- IC 是否稳定，而不是只来自 2024-2026 牛市？
- 3d/5d/10d 哪个 horizon 的 ICIR 最稳？
- 收益分桶是否单调？
- Top-Bottom spread 是否足够覆盖成本？
- 因子是否只在小票、低流动性或少数行业有效？
- 弱市/熊市下是否完全失效？

完成后下一步提示：

> Batch 3C 已完成。若没有任何因子在跨年份、跨市场状态下稳定为正，不进入训练；若存在可解释且稳定的因子，进入 Batch 3D 简单规则基准。

### Step 3D: 简单规则基准

目标：训练模型前，先证明复杂模型有必要。

基准规则：

| 规则 | 含义 |
|---|---|
| random | 同股票池随机选股 |
| momentum_20d | 20 日动量 TopN |
| reversal_3d | 3 日超跌反转 TopN |
| industry_momentum | 强行业内选强股 |
| liquidity_momentum | 流动性过滤 + 动量 |
| low_vol_quality_proxy | 低波动 + 流动性 |
| crowding_avoid | 避开高拥挤行业 |

必须输出：

- `batch3D_rule_config.yaml`
- `batch3D_rule_backtest_summary.csv`
- `batch3D_rule_by_regime.csv`
- `batch3D_rule_by_horizon.csv`
- `batch3D_random_baseline_summary.csv`
- `batch3D_conclusion.md`
- `batch3D_file_sha256.csv`

验收问题：

- 简单规则是否已经能获得足够稳定收益？
- 如果简单规则不稳定，模型训练是否有意义？
- 新模型未来至少要打败哪些规则？

完成后下一步提示：

> Batch 3D 已完成。只有当因子 IC 和简单规则均支持存在可重复 alpha 时，才建议进入轻量模型训练 Batch 4；否则暂停训练，补数据或转为 paper diagnostics。

### Step 4A: 轻量模型训练准备

进入条件：

- Batch 3A 市场状态已通过。
- Batch 3B 行业/拥挤度特征已通过，且已明确哪些字段可训练、哪些只能诊断。
- Batch 3C 至少存在一组跨年份、跨市场状态可解释的候选因子。
- Batch 3D 至少存在一个简单规则基准，且后续模型必须以它作为最低对照。

目标：在训练前冻结训练配置，避免看结果后反复改特征、改标签、改样本。

必须预注册：

| 项目 | 要求 |
|---|---|
| 目标周期 | 从 3d/5d/10d 中选择，依据 Batch 3C/3D，不依据训练后收益 |
| 股票池 | 明确全 A、流动性过滤、ST/停牌/上市天数过滤 |
| 标签 | 固定 forward return 或分类 label，不在训练后修改 |
| 特征列表 | 仅来自 Batch 3C 通过的因子；行业/概念字段需说明 point-in-time 状态 |
| 切分 | 时间序列切分，不随机打散 |
| 成本口径 | 至少包含无成本、基础成本、滑点/冲击成本 |
| 基准 | Batch 3D 的简单规则 + random baseline |
| 禁止事项 | 禁止用旧验证期收益挑模型；禁止训练后临时删失败样本 |

必须输出：

- `batch4A_model_training_plan.md`
- `batch4A_feature_freeze.csv`
- `batch4A_label_freeze.yaml`
- `batch4A_split_config.yaml`
- `batch4A_baseline_reference.csv`
- `batch4A_conclusion.md`
- `batch4A_file_sha256.csv`

验收问题：

- 是否能解释为什么选择该 horizon？
- 是否所有训练字段都能追溯到前置批次？
- 是否明确哪些字段不能作为训练特征？
- 是否存在未来信息或幸存者偏差风险？

完成后下一步提示：

> Batch 4A 已完成。请确认训练计划、特征冻结和切分方式。若通过，才允许执行 Batch 4B 轻量模型训练；若不通过，只修改计划，不训练。

### Step 4B: 轻量模型训练

目标：只训练可解释、可复现、低复杂度模型，验证是否优于简单规则。

允许模型：

| 模型 | 用途 |
|---|---|
| Logistic / Linear probability | 分类胜率基准，便于解释 |
| Ridge / ElasticNet | 连续收益或 rank score |
| LightGBM rank / binary | 非线性基准，但必须限制复杂度 |

禁止模型：

- 深度学习模型。
- 高复杂度 stacking。
- 自动化超参大搜索。
- 用 legacy 验证期反复挑参数。

训练方式：

- 时间序列训练/验证，不随机打散。
- 所有 scaler/imputer/encoder 只在训练集 fit。
- 每次训练必须保存 config、特征列表、随机种子、模型文件、预测结果和日志。
- 训练只使用 Batch 4A 冻结的样本、特征和标签。

必须输出：

- `batch4B_train_config.yaml`
- `batch4B_model_registry.csv`
- `batch4B_feature_importance.csv`
- `batch4B_prediction_scores.parquet`
- `batch4B_validation_metrics.csv`
- `batch4B_by_regime_metrics.csv`
- `batch4B_by_year_metrics.csv`
- `batch4B_by_industry_metrics.csv`
- `batch4B_vs_simple_baseline.csv`
- `batch4B_conclusion.md`
- `batch4B_file_sha256.csv`

验收问题：

- 是否显著优于 Batch 3D 简单规则和 random baseline？
- 是否只是牛市阶段有效？
- 是否只依赖少数行业、少数股票或低流动性样本？
- 是否在成本后仍有正收益或至少有稳定排序能力？
- 是否存在训练/验证指标明显背离？

完成后下一步提示：

> Batch 4B 已完成。若模型未稳定打败简单规则，不进入鲁棒性审计；若通过，冻结候选模型进入 Batch 5A 鲁棒性与暴露审计。

### Step 5A: 鲁棒性和暴露审计

目标：判断模型是否只是数据挖掘、牛市 beta、题材抱团或流动性幻觉。

审计维度：

| 维度 | 输出 |
|---|---|
| 时间 | 年度、季度、滚动窗口收益/IC |
| 市场状态 | strong/weak/neutral/extreme_selloff/high_vol/crowding_high |
| 行业 | 信号数量、收益贡献、最大暴露、行业集中度 |
| 个股 | 重复入选、收益贡献、最大亏损贡献 |
| 市值/流动性 | 分桶收益、成交容量、冲击成本敏感性 |
| 题材拥挤 | 行业成交占比高低分组；概念仅在 point-in-time 数据可用时做 |
| 极端样本 | 剔除疑似脏数据、保留真实极端波动后的差异 |
| 收益集中 | 去掉最大 1/3/5/10 个盈利日后表现 |
| 成本压力 | 5bp/10bp/20bp 滑点，冲击成本，参与率上限 |

必须输出：

- `batch5A_robustness_summary.csv`
- `batch5A_by_regime.csv`
- `batch5A_by_year_quarter.csv`
- `batch5A_industry_exposure.csv`
- `batch5A_stock_concentration.csv`
- `batch5A_size_liquidity_capacity.csv`
- `batch5A_cost_sensitivity.csv`
- `batch5A_profit_concentration.csv`
- `batch5A_failure_case_review.csv`
- `batch5A_conclusion.md`
- `batch5A_file_sha256.csv`

验收问题：

- 弱市/熊市是否完全失效？
- 收益是否集中在少数日期、少数股票或少数行业？
- 成本和容量压力下是否仍有安全垫？
- 是否明显依赖高拥挤题材？
- 去掉最大盈利日后是否仍成立？

完成后下一步提示：

> Batch 5A 已完成。若鲁棒性不过关，模型冻结为研究失败样本，不进入 walk-forward；若通过，进入 Batch 5B walk-forward。

### Step 5B: Walk-forward 前向滚动验证

目标：用历史上的滚动前向方式模拟“未来未知”，而不是只看一次固定切分。

要求：

- 训练窗口、验证窗口、步长预先固定。
- 每期只使用当期之前数据训练。
- 每期都输出独立指标，不只输出平均值。
- 不允许根据某几期表现临时改参数。

必须输出：

- `batch5B_walk_forward_config.yaml`
- `batch5B_period_metrics.csv`
- `batch5B_period_predictions.parquet`
- `batch5B_period_trade_summary.csv`
- `batch5B_failure_periods.md`
- `batch5B_conclusion.md`
- `batch5B_file_sha256.csv`

验收问题：

- 多数滚动期是否有效，而不是一两期贡献？
- 失效期是否集中在某类市场状态？
- 平均收益、IC、PF、最大回撤是否可接受？
- 与简单规则相比是否仍有增量？

完成后下一步提示：

> Batch 5B 已完成。若 walk-forward 不稳定，不进入 forward paper tracking；若稳定，冻结候选策略进入 Batch 6 未来纸面跟踪。

### Step 6: Forward paper tracking

目标：在未来未见数据中验证候选策略，不再历史调参。

要求：

- 冻结模型、特征、标签、股票池、成本口径、TopN、调仓周期。
- 每日或每个调仓日生成 paper-only 信号。
- 不接券商、不自动下单、不手动跟单。
- 至少跟踪 60 个新交易日；若信号稀疏，则延长至 90 个交易日或满足最低信号数。
- 每阶段评估 matched random baseline。

必须输出：

- `forward_model_candidate_config.yaml`
- `forward_daily_signals/YYYYMMDD.csv`
- `forward_trade_ledger.csv`
- `forward_execution_quality.csv`
- `forward_periodic_evaluation.csv`
- `forward_random_baseline.csv`
- `forward_conclusion.md`

验收问题：

- 未见数据中是否仍优于 random 和简单规则？
- 成本后是否仍为正？
- 是否依赖少数日期？
- 行业/个股暴露是否可控？
- 是否仍只是 paper candidate，而非实盘建议？

完成后下一步提示：

> Forward paper tracking 阶段仅能给出是否继续观察或终止研究的建议，不直接进入实盘。若要模拟盘或实盘，需要另建风控、交易、合规和人工确认流程。

## 5. 每阶段 Handoff 固化规则

每个阶段完成后，必须同时写两类 handoff 文件：

| 文件 | 作用 |
|---|---|
| `reports/tushare/v9_swing_research/{batch_dir}/{batch}_handoff_to_{next_batch}.md` | 保存本阶段结论、下一阶段输入、禁止事项和进入条件 |
| `reports/tushare/v9_swing_research/v9_current_handoff.md` | 始终覆盖为最新 handoff，下一次恢复任务时优先读取 |

handoff 文件必须至少包含：

1. 已完成 batch 名称和输出目录。
2. 本阶段核心结论。
3. 本阶段明确没有做什么。
4. 下一阶段必须读取的文件。
5. 下一阶段允许做什么。
6. 下一阶段禁止做什么。
7. 是否需要用户确认后继续。

当前最新 handoff：

- `reports/tushare/v9_swing_research/v9_current_handoff.md`
- 内容来自 `Batch 3C -> Batch 3D`

## 6. 每阶段完成后的固定回复模板

每个阶段完成后，必须按以下结构回复：

1. 当前分支
2. 当前 commit
3. `v7_locked` 是否未移动
4. 本阶段做了什么
5. 本阶段没有做什么
6. 输出目录
7. 核心数据结果
8. 关键风险和限制
9. 是否通过进入下一阶段
10. 下一阶段具体要做什么
11. 是否需要用户确认后再继续

## 7. 当前推荐下一步

当前已经完成 Stage 0、Stage 1、Stage 2、Stage 3、Stage 4、Stage 5。  
下一步不应直接训练模型。  
推荐执行：

> Batch 3D: simple rule baseline

原因：

- Batch 3C 已证明部分低流动性、低换手和反向动量因子存在排序诊断信号。
- 这些诊断还不是收益证明，必须先转化为简单规则并和 random/simple baselines 对比。
- 概念/题材特征仍未锁定 point-in-time 数据源，Batch 3D 仍必须保持禁用。
