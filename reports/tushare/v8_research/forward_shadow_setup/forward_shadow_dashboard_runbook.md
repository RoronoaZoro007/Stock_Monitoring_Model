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
