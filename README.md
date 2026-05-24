# A 股尾盘隔夜模型研究框架

这个目录包含一个可复现的尾盘买入、次日早盘卖出研究框架。默认使用 BaoStock 的 5 分钟 K 线做历史研究，用 AkShare/BaoStock 做数据可得性校验。

## 核心设计

- 信号时间：交易日 14:50，只使用 14:50 及之前的 5 分钟 K 线和前一交易日以前的日线统计。
- 模拟买入：14:55 这一根 5 分钟 K 线的 VWAP，加买入滑点和佣金。
- 模拟卖出：次日 09:35、09:45、10:00 定时检查，满足止盈/止损/走弱条件则下一根 5 分钟 K 线 VWAP 卖出；否则 10:30 强制退出。
- 回测切分：最近两年数据中，最后三个月完全留作验证集；训练期内部做月度 walk-forward 回测，验证集只在最终模型固定后使用。
- 排名依据：模型预测胜率、历史期望收益、风控过滤共同决定；报告会输出每日 Top 10。

## 快速运行

```bash
source .venv/bin/activate
python tail_strategy_research.py audit-data
python tail_strategy_research.py build-dataset --start 2024-05-21 --end 2026-05-20 --max-symbols 200
python tail_strategy_research.py train-evaluate
```

`--max-symbols` 用于快速验证流程。生产级全市场运行去掉该参数：

```bash
python tail_strategy_research.py build-dataset --start 2024-05-21 --end 2026-05-20
python tail_strategy_research.py train-evaluate
```

全市场两年 5 分钟数据约为 1 亿行以上，脚本会按股票缓存特征，支持断点续跑。

BaoStock 串行取数较慢，可用固定随机种子做子样本验证：

```bash
python tail_strategy_research.py build-dataset --start 2024-05-21 --end 2026-05-20 --max-symbols 80 --sample-seed 20260521 --workers 3
python tail_strategy_research.py train-evaluate --validation-start 2026-02-21 --random-seeds 200
```

本机已完成上述 80 只随机样本验证。结果见 `reports/report.md`，结论是不满足实盘门槛：最近三个月验证期 Top10 日均收益为 -0.2099%，累计收益 -11.75%，Profit Factor 0.77，没有跑赢随机 Top10 的 95 分位。

## 输出

- `data/features/*.parquet`：单股票特征缓存。
- `data/model_dataset.parquet`：合并后的建模样本。
- `reports/data_audit.json`：数据源覆盖检查。
- `reports/backtest_walk_forward.csv`：训练期月度滚动回测预测。
- `reports/validation_predictions.csv`：最近三个月验证集逐笔预测。
- `reports/top10_validation.csv`：验证期每日胜率排名前 10。
- `reports/metrics.json`：核心指标。
- `reports/report.md`：研究报告。

## 重要限制

免费 5 分钟 K 线只能近似实盘成交，无法还原逐笔盘口、排队成交和真实冲击成本。实盘前应接入券商 Level-1/Level-2 或付费历史分钟/逐笔数据重新验证。

## Tushare 代理数据下载

已新增 `tushare_data_pipeline.py`，用于通过第三方 Tushare 代理做可断点的数据回填。Token 不应写进代码，使用环境变量：

```bash
export TUSHARE_PROXY_URL=http://tsy.xiaodefa.cn
export TUSHARE_TOKEN=你的56位key
```

初始化两年 5 分钟数据任务，按由近到远顺序下载：

```bash
python tushare_data_pipeline.py init \
  --start 2024-05-21 \
  --end 2026-05-21 \
  --freq 5min \
  --direction near-first \
  --include-enhanced \
  --include-memberships
```

执行下载，支持中断后继续：

```bash
python tushare_data_pipeline.py run \
  --requests-per-minute 120 \
  --max-requests 9000 \
  --timeout 60
```

查看进度和校验文件：

```bash
python tushare_data_pipeline.py status
python tushare_data_pipeline.py audit-files --limit 200
```

当前已初始化 `data_tushare/manifests/download_tasks.sqlite3`：共 `35322` 个任务，已完成 `12` 个烟囱测试任务，剩余任务可继续断点下载。

## Forward Shadow Paper Tracking

当前可运行分支提供 v7 locked 后续纸面跟踪入口。该入口只做未来未见数据的
paper tracking，不训练、不调参、不下单。

新 Mac 环境一条命令运行：

```bash
git clone -b codex/forward-shadow-live-runner git@github.com:RoronoaZoro007/Stock_Monitoring_Model.git
cd Stock_Monitoring_Model

export TUSHARE_TOKEN='你的 Tushare token'
export WXPUSHER_APP_TOKEN='你的 WxPusher app token'
export WXPUSHER_TOPIC_ID='44635'
export TUSHARE_PROXY_URL='http://tsy.xiaodefa.cn'

./scripts/run_forward_shadow_live.sh
```

本地可视化控制台：

```bash
./scripts/run_forward_shadow_dashboard.sh
```

打开 `http://127.0.0.1:8788/` 后可以选择执行日期和执行模式：

- 模式1：模拟时间点快跑。仍调用真实数据接口和真实处理流程，但不等待 09:25/14:50 等墙上时间，适合历史漏跑补跑。
- 模式2：真实时间点模式。按北京时间节点等待执行，适合当天 paper tracking。
- `TUSHARE_TOKEN`、`WXPUSHER_APP_TOKEN` 和 WxPusher Topic/GroupId 可以在页面临时输入；留空则使用本机环境变量。页面输入只注入本次任务进程，不写入文件。
- 如果任务失败，页面会显示失败状态、返回码和最近错误日志，便于定位。
- 重复跑同一交易日时，可以勾选 `run_id 输出目录` 隔离本次产物，或勾选 `保留本次 run 快照` 把关键 CSV/JSON/日志复制到快照目录；`强制重新下载分钟线` 会重新请求分钟 bar 并按 `ts_code + trade_time` 覆盖去重。
- 页面会展示 T-1 纸面买入、今日卖出提示/退出记录、今日尾盘选股和 14:55 纸面买入价。消息推送支持 `关键交易 + 失败`、`只推买卖提示`、`只推失败`、`所有节点`、`不推送`。

指定交易日：

```bash
TRADE_DATE=20260525 ./scripts/run_forward_shadow_live.sh
```

详细说明见：

```text
reports/tushare/v8_research/forward_shadow_setup/forward_shadow_live_runbook.md
```
