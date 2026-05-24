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

## 1. 阶段与 Batch 映射

| 阶段 | 目的 | 对应 Batch | 当前状态 | 关键输出 | 下一步条件 |
|---|---|---|---|---|---|
| Stage 0 数据底座锁定 | 确认 raw 数据可审计、可复现 | `stage0_data_foundation` | 已完成 | raw manifest、coverage、字段字典、limitations、hash | 缺失交易日为 0，风险已写明 |
| Stage 1 历史日频补齐 | 补齐 2018 至今全 A 日线/复权/市值/停牌/涨跌停 | `batch1_history_daily` | 已完成 | API smoke test、download summary、coverage after | 5 个日频 endpoint 全部 2032/2032 |
| Stage 2 clean panel + label audit | 合并日频面板，生成 3d/5d/10d forward label 并审计 | `batch2_clean_panel_label_audit` | 已完成 | clean panel、labels、缺失/极端收益/市场状态初稿 | 标签有效率、极端收益和执行风险已量化 |
| Stage 3 市场状态标记 | 解决牛市训练、熊市失效风险 | 建议 `batch3A_market_regime` | 未开始，Batch2 只有初稿 | 正式 regime 标签、阈值说明、分布、敏感性审计 | 市场状态定义不使用未来收益调参 |
| Stage 4 行业/概念特征工程 | 刻画行业强度、拥挤度、抱团风险 | 建议 `batch3B_industry_theme_features` | 未开始 | 行业强度、行业成交集中度、个股行业暴露；概念如无 point-in-time 数据则延后 | 明确行业字段是否 point-in-time；概念数据源未锁定则不能训练使用 |
| Stage 5 因子 IC 与分层诊断 | 训练前判断因子是否有排序能力 | 建议 `batch3C_factor_ic_decile` | 未开始 | RankIC、ICIR、10 桶收益、Top-Bottom、分状态/行业/市值/流动性 IC | 只有稳定、可解释、非单一环境驱动的因子进入候选 |
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

## 5. 每阶段完成后的固定回复模板

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

## 6. 当前推荐下一步

当前已经完成 Stage 0、Stage 1、Stage 2。  
下一步不应直接训练模型，也不应直接做简单规则回测。  
推荐执行：

> Batch 3A: market regime formalization

原因：

- Batch 3C 的“分市场状态 IC”依赖正式市场状态标签。
- Batch 3B 的行业拥挤度也需要和市场状态一起解释。
- 先做正式 market regime，可以防止后续因子只在牛市有效却被误判为长期 alpha。
