# Forward Shadow Runtime Runbook

This runbook describes the paper-only forward shadow runtime. It does not
authorize live trading, simulation trading, manual order following, model
retraining, label retraining or historical parameter optimization.

## Scope

The runtime does three things:

1. Schedules paper-tracking checkpoints.
2. Runs locked v7 inference and the locked four-line forward shadow freeze.
3. Records SQLite state, JSONL logs and optional WxPusher notifications.

It does not connect to a broker and does not place orders.

## Sensitive Configuration

Do not write tokens into tracked files.

Use environment variables:

```bash
export WXPUSHER_APP_TOKEN="replace_with_wxpusher_app_token"
export TUSHARE_TOKEN="..."
export TUSHARE_PROXY_URL="http://tsy.xiaodefa.cn"
```

The checked-in config stores only the non-sensitive WxPusher topic id:

```json
"topic_ids": [44635]
```

The default config keeps `wxpusher.dry_run=true`. To actually push messages,
run the runtime with `--send-notifications` and make sure
`WXPUSHER_APP_TOKEN` is present in the environment.

## Files

| File | Purpose |
|---|---|
| `research/v8_research/forward_shadow_runtime.py` | Long-running paper-only scheduler and task runner. |
| `research/v8_research/build_forward_shadow_daily_features.py` | Builds one trade day's <=14:50 forward-only feature file. |
| `research/v8_research/build_forward_shadow_score_matrix.py` | Locked v7 inference entry. No training. |
| `research/v8_research/forward_shadow_runtime_config.example.json` | Runtime config template. |
| `research/v8_research/forward_shadow_data_guard.py` | Provider data guard for auction and minute-bar fetches. |
| `research/v8_research/forward_shadow_exit_monitor.py` | T+1 locked v7 paper exit monitor and sell-alert notifier. |
| `data_tushare/forward_shadow_runtime.sqlite3` | Runtime checkpoint state. Local only. |
| `logs/forward_shadow/YYYYMMDD.jsonl` | Structured runtime log. Local only. |

## Daily Timeline

| Time | Checkpoint | Expected behavior |
|---|---|---|
| 07:20 | `startup_check` | Check branch, `v7_locked` commit and paper-only mode. |
| 07:30 | `premarket_check` | Check daily file, locked model and locked definitions exist. |
| 09:25:30 | `auction_guard_0925` | Check/fetch T-1 close auction and T-day open auction. |
| 09:35:05 | `fetch_exit_0935_bar` | Fetch the 09:35 bar for T-1 paper positions. |
| 09:35:50 | `exit_check_0935` | Use the 09:35 bar to decide whether to recommend a 09:40 paper exit. |
| 09:40:05 | `fetch_exit_0940_bar` | Fetch the 09:40 bar for paper exit VWAP. |
| 09:40:50 | `exit_exec_0940` | Record the paper VWAP for exits recommended at 09:35. |
| 09:45:05 | `fetch_exit_0945_bar` | Fetch the 09:45 bar for T-1 paper positions. |
| 09:45:50 | `exit_check_0945` | Use the 09:45 bar to decide whether to recommend a 09:50 paper exit. |
| 09:50:05 | `fetch_exit_0950_bar` | Fetch the 09:50 bar for paper exit VWAP. |
| 09:50:50 | `exit_exec_0950` | Record the paper VWAP for exits recommended at 09:45. |
| 10:00:05 | `fetch_exit_1000_bar` | Fetch the 10:00 bar for T-1 paper positions. |
| 10:00:50 | `exit_check_1000` | Use the 10:00 bar to decide whether to recommend a 10:05 paper exit. |
| 10:05:05 | `fetch_exit_1005_bar` | Fetch the 10:05 bar for paper exit VWAP. |
| 10:05:50 | `exit_exec_1005` | Record the paper VWAP for exits recommended at 10:00. |
| 10:25:05 | `fetch_exit_1025_bar` | Fetch the 10:25 bar for default-exit pre-alert reference. |
| 10:25:50 | `exit_prealert_1025` | Paper-only pre-alert for still-open positions that will default-exit at 10:30. |
| 10:30:05 | `fetch_exit_1030_bar` | Fetch the 10:30 bar for default paper exit VWAP. |
| 10:30:50 | `exit_default_1030` | Default paper exit for positions not already exited. |
| 10:35 | `prior_trade_settlement` | Generate prior signal-date settlement report. |
| 14:30:00 | `fetch_tail_until_1430` | Fetch Top3000 09:30-14:30 bars in batched provider requests. |
| 14:35:05 | `fetch_tail_1435_bar` | Fetch Top3000 14:35 bar. |
| 14:40:05 | `fetch_tail_1440_bar` | Fetch Top3000 14:40 bar. |
| 14:45:05 | `fetch_tail_1445_bar` | Fetch Top3000 14:45 bar. |
| 14:50:05 | `fetch_tail_1450_bar` | Fetch Top3000 14:50 bar before scoring. |
| 14:50:50 | `build_forward_features` | Build one-day features from data no later than 14:50. |
| 14:51:20 | `build_score_matrix` | Run locked v7 inference using configured feature file. |
| 14:51:50 | `freeze_signals` | Freeze S0/S1/S0_tail_down/S1_tail_down signals. |
| 14:52:10 | `latency_guard` | Assert score timestamp and freeze-time constraints. |
| 14:55:05 | `fetch_tail_1455_bar` | Fetch Top3000 14:55 bar for paper entry VWAP observation. |
| 14:55:50 | `record_entry_1455_vwap` | Write `{T}_entry_prices.csv`, which T+1 exit monitoring reads. |
| 14:56:00 | `entry_observation_check` | Check frozen signal and paper entry VWAP files exist. |

## T+1 Paper Exit Monitor

The exit monitor keeps the locked v7 exit rule unchanged:

1. At 09:35, sell at 09:40 if `stop_loss` or `take_profit_0935` triggers.
2. At 09:45, sell at 09:50 if `stop_loss`, `take_profit_0945` or `weak_open`
   triggers.
3. At 10:00, sell at 10:05 if `stop_loss` or `not_recovered_1000` triggers.
4. Otherwise sell at 10:30 by `time_exit`.

The sell-alert price is a paper-only reference price available at the decision
checkpoint. For 09:35/09:45/10:00 decisions this is the check-bar close. The
actual paper exit VWAP is recorded only after the expected execution bar is
observable. No broker connection or order action is performed.

The 10:25 default-exit pre-alert is only a paper reminder for positions that
have not triggered earlier exit rules. It does not add a new exit rule; the
locked v7 default exit remains 10:30.

## Data Guards

Auction guard:

```bash
.venv/bin/python research/v8_research/forward_shadow_data_guard.py \
  auction-guard \
  --trade-date YYYYMMDD \
  --prior-trade-date YYYYMMDD \
  --output-root reports/tushare/v8_forward_shadow
```

Morning exit bar fetch:

```bash
.venv/bin/python research/v8_research/forward_shadow_data_guard.py \
  minute-fetch \
  --trade-date YYYYMMDD \
  --signal-date PRIOR_YYYYMMDD \
  --output-root reports/tushare/v8_forward_shadow \
  --codes-source prior_entries \
  --bar-time 09:35 \
  --mode bar \
  --retry-until-complete-seconds 40
```

Tail Top3000 catch-up:

```bash
.venv/bin/python research/v8_research/forward_shadow_data_guard.py \
  minute-fetch \
  --trade-date YYYYMMDD \
  --output-root reports/tushare/v8_forward_shadow \
  --codes-source top3000 \
  --bar-time 14:30 \
  --mode until \
  --batch-size 160
```

The guard writes reports under
`reports/tushare/v8_forward_shadow/data_guards/`. If all required local bars
already exist, it can complete without calling the provider. If data is missing,
provider credentials must be present in environment variables.

Paper entry recording:

```bash
.venv/bin/python research/v8_research/forward_shadow_data_guard.py \
  record-entry \
  --trade-date YYYYMMDD \
  --output-root reports/tushare/v8_forward_shadow
```

This writes `reports/tushare/v8_forward_shadow/daily_entry_prices/YYYYMMDD_entry_prices.csv`.
The T+1 exit monitor treats that file as the source of paper entry VWAP and
will not infer a buy price from the frozen signal file alone.

Manual dry-run example:

```bash
.venv/bin/python research/v8_research/forward_shadow_exit_monitor.py \
  --signal-date 20260521 \
  --settlement-date 20260522 \
  --checkpoint check_0935 \
  --output-root reports/tushare/v8_forward_shadow \
  --dry-run-push
```

Notification-enabled example:

```bash
export WXPUSHER_APP_TOKEN="replace_with_wxpusher_app_token"
.venv/bin/python research/v8_research/forward_shadow_exit_monitor.py \
  --signal-date 20260521 \
  --settlement-date 20260522 \
  --checkpoint check_0935 \
  --output-root reports/tushare/v8_forward_shadow \
  --send-notifications
```

## Important Production Caveat

The runtime can call a data fetch command and a score command, but true
production readiness still depends on a validated intraday data path:

1. 14:50 bar must be available quickly enough.
2. The incremental cleaner must only use data at or before 14:50 for scoring.
3. The score matrix must freeze before 14:55.
4. Late data must produce `no_trade` or a timestamp violation, not a backfilled
   signal.

The default `download_1450_bar` command is disabled because provider latency and
intraday availability must be measured before making it part of the live
paper-tracking path.

## Initialize

```bash
.venv/bin/python research/v8_research/forward_shadow_runtime.py \
  --config research/v8_research/forward_shadow_runtime_config.example.json \
  init-db
```

## Dry-Run One Checkpoint

```bash
.venv/bin/python research/v8_research/forward_shadow_runtime.py \
  --config research/v8_research/forward_shadow_runtime_config.example.json \
  run-once \
  --trade-date 20260521 \
  --checkpoint startup_check \
  --dry-run
```

## Build A Score Matrix For A Replay Date

```bash
.venv/bin/python research/v8_research/build_forward_shadow_daily_features.py \
  --trade-date 20260521

.venv/bin/python research/v8_research/build_forward_shadow_score_matrix.py \
  --trade-date 20260521 \
  --feature-file reports/tushare/v8_forward_shadow/test_features/20260521_forward_features.csv \
  --paper-reconstruction
```

This only performs locked v7 inference. It does not train a model.

## Replay/Test Mode

Replay mode separates physical run time from the logical trade-date time. For
example, on 2026-05-24 before market open, the following command validates the
2026-05-22 paper flow as if the 14:50 bar had just completed:

```bash
.venv/bin/python research/v8_research/forward_shadow_runtime.py \
  --config research/v8_research/forward_shadow_runtime_config.example.json \
  replay-day \
  --trade-date 20260522
```

Replay mode does not change model rules. It still requires the score matrix to
declare `logical_data_max_timestamp <= YYYY-MM-DD 14:50:00`. The only relaxed
guard is the physical wall-clock freeze check, because the command is being run
after the historical 14:55 deadline.

Default replay checkpoints:

1. `startup_check`
2. `premarket_check`
3. `build_forward_features`
4. `build_score_matrix`
5. `freeze_signals`
6. `latency_guard`
7. `entry_observation_check`

The optional downloader can be included only when the data-provider credentials
are configured and the user explicitly asks to test downloading:

```bash
.venv/bin/python research/v8_research/forward_shadow_runtime.py \
  --config research/v8_research/forward_shadow_runtime_config.example.json \
  replay-day \
  --trade-date 20260522 \
  --include-download
```

The single-day downloader uses `download_top3000_day.py`, with a default batch
size of 160 symbols per request. This is designed to stay under the documented
single-request row cap for one day of 5-minute bars. If the target day's raw
minute files already exist, the downloader records them as already present
instead of re-pulling them unless `--refresh` is wired into the command config.

For a trade date such as 20260522, replay mode can validate signal generation
and entry observation. It cannot settle the next-day exit unless the next
trading day's required exit-window data is already available.

## Run Signal Freeze For A Date

```bash
.venv/bin/python research/v8_research/forward_shadow_runtime.py \
  --config research/v8_research/forward_shadow_runtime_config.example.json \
  run-once \
  --trade-date 20260521 \
  --checkpoint freeze_signals
```

## Run Daemon

Dry-run mode:

```bash
.venv/bin/python research/v8_research/forward_shadow_runtime.py \
  --config research/v8_research/forward_shadow_runtime_config.example.json \
  daemon \
  --trade-date YYYYMMDD \
  --dry-run
```

Notification-enabled mode:

```bash
export WXPUSHER_APP_TOKEN="replace_with_wxpusher_app_token"
.venv/bin/python research/v8_research/forward_shadow_runtime.py \
  --config research/v8_research/forward_shadow_runtime_config.example.json \
  daemon \
  --trade-date YYYYMMDD \
  --send-notifications
```

## Query Status

```bash
.venv/bin/python research/v8_research/forward_shadow_runtime.py \
  --config research/v8_research/forward_shadow_runtime_config.example.json \
  status \
  --trade-date YYYYMMDD
```

## WxPusher Smoke Test

Dry run:

```bash
.venv/bin/python research/v8_research/forward_shadow_runtime.py \
  --config research/v8_research/forward_shadow_runtime_config.example.json \
  send-test \
  --trade-date YYYYMMDD
```

Actual push:

```bash
export WXPUSHER_APP_TOKEN="replace_with_wxpusher_app_token"
.venv/bin/python research/v8_research/forward_shadow_runtime.py \
  --config research/v8_research/forward_shadow_runtime_config.example.json \
  send-test \
  --trade-date YYYYMMDD \
  --send-notifications
```

## Guardrails

- `runtime.paper_only` must be true.
- `v7_locked` must equal
  `f688eec575af4667681d87b5b2c1fca72754e399`.
- `data_max_timestamp` must be no later than 14:50 for signal generation.
- Signal freeze later than 14:55 is a violation unless the run is explicitly a
  replay.
- Frozen daily signal files must not be edited to improve outcomes.
- WxPusher failure is logged and does not change signal output.
