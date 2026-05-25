# Forward Shadow Live Paper Tracking Runbook

This branch is for paper-only forward shadow tracking of the locked v7 routes.
It does not train, tune, place orders, connect to a broker, or change any locked
strategy definition.

## One-command Run

On a fresh macOS machine:

```bash
git clone -b codex/forward-shadow-live-runner git@github.com:RoronoaZoro007/Stock_Monitoring_Model.git
cd Stock_Monitoring_Model

export TUSHARE_TOKEN='replace_with_your_token'
export WXPUSHER_APP_TOKEN='replace_with_your_wxpusher_app_token'
export WXPUSHER_TOPIC_ID='44635'
export TUSHARE_PROXY_URL='https://tt.xiaodefa.cn'

./scripts/run_forward_shadow_live.sh
```

For a specific trade date:

```bash
TRADE_DATE=20260525 ./scripts/run_forward_shadow_live.sh
```

For a dry operational drill that does not wait for wall-clock times:

```bash
TRADE_DATE=20260525 NO_WAIT=1 SEND_NOTIFICATIONS=0 ./scripts/run_forward_shadow_live.sh
```

## What The Runner Does

1. Creates `.venv` if missing and installs `requirements.txt`.
2. Validates `TUSHARE_TOKEN`; validates `WXPUSHER_APP_TOKEN` when
   `SEND_NOTIFICATIONS=1`.
3. Downloads the minimum provider data needed to build the current-day forward
   baseline:
   - `trade_cal`
   - `stock_basic`
   - recent `daily`
   - recent `moneyflow` and `moneyflow_ths` unless `SKIP_MONEYFLOW=1`
4. Builds `data_tushare/clean/daily_repaired_top3000.parquet` locally.
   The row for the live trade date is synthetic and uses only `T-1` and earlier
   daily information.
5. Checks whether the prior trading day's paper entry file exists. If it does
   not, the runner reconstructs the prior day's paper signal and 14:55 paper
   entry from provider data.
6. Waits for the live-day schedule:
   - 09:25:30 auction guard
   - 09:35 to 10:30 prior-day paper exit checks
   - 14:30 to 15:00 live-day tail download, feature build, score, signal
     freeze, and paper entry recording
7. Sends paper-only WxPusher messages at key checkpoints.

## Locked Routes

The runner tracks exactly the four locked forward-shadow lines:

| line | role |
|---|---|
| `S0_v7_original_top10` | control |
| `S1_U2_filter_only_no_refill` | control |
| `S0_v7_original_top10_tail_down` | forward-shadow candidate |
| `S1_U2_filter_only_no_refill_tail_down` | forward-shadow candidate |

Definitions remain locked:

- `tail_down = market_tail_ret_median < 0`
- `market_tail_ret_median = median(tail_ret_1420_1450)`
- `tail_ret_1420_1450 = close_1450 / close_1420 - 1`
- `U2 = existing Top3000 internal rolling Top2500 by amount.shift(1).rolling(60, min_periods=20).mean()`

## Outputs

Main runtime outputs are written under:

```text
reports/tushare/v8_forward_shadow/
```

Key files:

```text
daily_signals/YYYYMMDD_signals.csv
daily_entry_prices/YYYYMMDD_entry_prices.csv
daily_exit_recommendations/SIGNALDATE_SETTLEMENTDATE_exit_recommendations.csv
daily_exit_execution/SIGNALDATE_SETTLEMENTDATE_exit_execution.csv
daily_ledgers/SIGNALDATE_SETTLEMENTDATE_exit_ledgers.csv
live_runner/YYYYMMDD_live_runner_steps.csv
```

Provider raw and generated local state are written under `data_tushare/`.
They are intentionally ignored by Git.

## Environment Variables

| variable | required | default |
|---|---:|---|
| `TUSHARE_TOKEN` | yes | none |
| `TUSHARE_PROXY_URL` | no | `https://tt.xiaodefa.cn` |
| `WXPUSHER_APP_TOKEN` | yes if sending | none |
| `WXPUSHER_TOPIC_ID` | no | `44635` |
| `SEND_NOTIFICATIONS` | no | `1` |
| `TRADE_DATE` | no | current Beijing date |
| `REQUESTS_PER_MINUTE` | no | `120` |
| `BATCH_SIZE` | no | `160` |
| `LOOKBACK_TRADING_DAYS` | no | `90` |
| `NO_WAIT` | no | `0` |
| `SKIP_MONEYFLOW` | no | `0` |

## Guardrails

- No broker API.
- No live or simulated orders.
- No retraining.
- No feature, label, target, TopN, tail_down, U2, or exit-rule changes.
- No use of 15:00 or next-day data for 14:55 signal generation.
- No token written to files or committed to Git.
