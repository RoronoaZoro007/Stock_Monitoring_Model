# Forward Shadow Dashboard Runbook

This dashboard is a local paper-only control panel for the locked forward shadow runner.
It does not retrain models, change labels/features, connect to brokers, or place orders.

## Start

```bash
export TUSHARE_PROXY_URL='http://tsy.xiaodefa.cn'

./scripts/run_forward_shadow_dashboard.sh
```

`TUSHARE_TOKEN`, `WXPUSHER_APP_TOKEN`, and WxPusher Topic/GroupId can be entered
on the dashboard page. If left blank, the dashboard uses environment variables.
Page-entered tokens are injected only into the child runner process and are not
written to config files, logs, status JSON, or API responses.

Open:

```text
http://127.0.0.1:8788/
```

## Modes

| Mode | Dashboard label | Runner behavior | Intended use |
|---|---|---|---|
| `fast_replay` | 模式1：模拟时间点快跑 | Adds `--no-wait`; still calls the real provider and real processing chain | Historical catch-up or operational drill |
| `live_time` | 模式2：真实时间点模式 | Does not add `--no-wait`; waits for Beijing wall-clock schedule | Same-day forward shadow tracking |

In live mode, if the dashboard is started after a scheduled checkpoint, the runner catches up that checkpoint immediately. For true same-day use, start the dashboard and job before 09:25 Beijing time.

## Repeated Runs

When the same trade date is run more than once, the default runner reuses the
same dated output files. Use these dashboard options when you need auditability:

| Option | Behavior |
|---|---|
| `run_id 输出目录` | Writes this job under `OUTPUT_ROOT/runs/RUN_ID`, isolating it from prior runs. |
| `保留本次 run 快照` | Copies lightweight outputs, logs, signal files, score files, feature files, and guard reports into `OUTPUT_ROOT/run_snapshots/RUN_ID` after the job exits. Raw minute parquet files are not copied. |
| `强制重新下载分钟线` | Adds minute-fetch `--refresh`, so existing minute bars are ignored for fetch planning and requested again. Written minute parquet remains deduped by `ts_code + trade_time` with the latest row kept. |

If both `run_id 输出目录` and `保留本次 run 快照` are enabled, the run outputs are isolated
under `runs/RUN_ID` and a lightweight copy is also written under `run_snapshots/RUN_ID`.

## Progress Files

The dashboard reads these runner outputs:

```text
reports/tushare/v8_forward_shadow/live_runner/YYYYMMDD_live_runner_status.json
reports/tushare/v8_forward_shadow/live_runner/YYYYMMDD_live_runner_steps.csv
reports/tushare/v8_forward_shadow/forward_shadow_candidate_status.csv
reports/tushare/v8_forward_shadow/daily_signals/YYYYMMDD_signals.csv
reports/tushare/v8_forward_shadow/daily_entry_prices/YYYYMMDD_entry_prices.csv
```

The live runner writes `live_runner_status.json` before waiting, before running a checkpoint, and after each successful checkpoint, so the page can show whether it is waiting or executing.

## Guardrails

- The page starts only the paper-only forward shadow runner.
- Tokens are read from environment variables or temporary page input and are not displayed by status APIs.
- The dashboard binds to `127.0.0.1` by default.
- Stopping a job terminates the runner process only; it does not delete existing output files.
- Candidate YAML, tail_down definition, U2 definition, v7 model, features, labels, TopN and exit rules are not modified by this dashboard.
- If the child runner fails, the page shows failed status, return code, and recent log lines for diagnosis.
- Repeated-run controls change only output isolation, snapshotting, and provider fetch refresh behavior. They do not change model inference or tracking rules.

## Dashboard Views

The dashboard surfaces the operational state required for paper tracking:

- `当前节点` and `正在做什么`: the live runner step and a human-readable description of the current action.
- `T-1 纸面持仓与今日卖出`: prior-day paper entry rows, same-day sell recommendations, exit execution records, and per-line settlement summary.
- `今日尾盘选股与买入价`: current-day frozen buy candidates and the 14:55 paper entry VWAP once recorded.
- `四线路状态`: S0, S1, S0+R1, and S1+R1 route activation and selected counts.
- Changing `执行日期` or `输出目录` refreshes the displayed data for that selection. The `刷新页面数据` button also reloads the selected date/output-root explicitly.

## Notification Policy

The dashboard and CLI support `NOTIFICATION_POLICY`:

| Policy | Behavior |
|---|---|
| `key_events` | Default. Push trade-critical buy/sell/entry messages and failures. |
| `trade_only` | Push buy/sell/entry paper-tracking messages and failures. |
| `failures_only` | Push failures only. |
| `all_steps` | Push every runner checkpoint plus trade messages. |
| `none` | Do not push messages. |

Buy notifications list the current route, stock, rank, score, expected buy time, and paper entry VWAP after the 14:55 bar is recorded.
Sell notifications list the checkpoint, expected sell time, stock, recommended sell price, and later the recorded paper exit VWAP.
