#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  ./scripts/run_forward_shadow_live.sh

Environment:
  TUSHARE_TOKEN=...              required
  WXPUSHER_APP_TOKEN=...         required when SEND_NOTIFICATIONS=1
  TUSHARE_PROXY_URL=...          default https://tt.xiaodefa.cn
  TRADE_DATE=YYYYMMDD            default current Beijing date
  SEND_NOTIFICATIONS=1|0         default 1
  WXPUSHER_TOPIC_ID=44635        default 44635
  NOTIFICATION_POLICY=key_events all_steps|key_events|trade_only|failures_only|none
  PRIOR_INPUT_POLICY=reuse_or_rebuild reuse_or_rebuild|reuse_only|force_rebuild
  PRIOR_SEED_ROOT=reports/tushare/v8_forward_shadow
  NO_WAIT=1                      execute schedule immediately, drill only
  REQUESTS_PER_MINUTE=120        default 120
  BATCH_SIZE=160                 default 160
  PREFLIGHT_TIMEOUT=8            default 8 seconds per connectivity probe
USAGE
  exit 0
fi

if [[ -z "${TUSHARE_TOKEN:-}" ]]; then
  echo "ERROR: TUSHARE_TOKEN is not set. Export it in your shell; do not write it into files." >&2
  exit 2
fi

if [[ "${SEND_NOTIFICATIONS:-1}" == "1" && -z "${WXPUSHER_APP_TOKEN:-}" ]]; then
  echo "ERROR: WXPUSHER_APP_TOKEN is not set while SEND_NOTIFICATIONS=1." >&2
  exit 2
fi

export TUSHARE_PROXY_URL="${TUSHARE_PROXY_URL:-https://tt.xiaodefa.cn}"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt

TRADE_DATE="${TRADE_DATE:-auto}"
OUTPUT_ROOT="${OUTPUT_ROOT:-reports/tushare/v8_forward_shadow}"
REQUESTS_PER_MINUTE="${REQUESTS_PER_MINUTE:-120}"
BATCH_SIZE="${BATCH_SIZE:-160}"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-8}"
LOOKBACK_TRADING_DAYS="${LOOKBACK_TRADING_DAYS:-90}"
TOPIC_ID="${WXPUSHER_TOPIC_ID:-44635}"
NOTIFICATION_POLICY="${NOTIFICATION_POLICY:-key_events}"
PRIOR_INPUT_POLICY="${PRIOR_INPUT_POLICY:-reuse_or_rebuild}"
PRIOR_SEED_ROOT="${PRIOR_SEED_ROOT:-reports/tushare/v8_forward_shadow}"

ARGS=(
  --trade-date "$TRADE_DATE"
  --output-root "$OUTPUT_ROOT"
  --requests-per-minute "$REQUESTS_PER_MINUTE"
  --batch-size "$BATCH_SIZE"
  --preflight-timeout "$PREFLIGHT_TIMEOUT"
  --lookback-trading-days "$LOOKBACK_TRADING_DAYS"
  --topic-id "$TOPIC_ID"
  --notification-policy "$NOTIFICATION_POLICY"
  --prior-input-policy "$PRIOR_INPUT_POLICY"
  --prior-seed-root "$PRIOR_SEED_ROOT"
)

if [[ "${SEND_NOTIFICATIONS:-1}" == "1" ]]; then
  ARGS+=(--send-notifications)
fi

if [[ "${NO_WAIT:-0}" == "1" ]]; then
  ARGS+=(--no-wait)
fi

if [[ "${SKIP_MONEYFLOW:-0}" == "1" ]]; then
  ARGS+=(--skip-moneyflow)
fi

exec python research/v8_research/forward_shadow_live_orchestrator.py "${ARGS[@]}"
