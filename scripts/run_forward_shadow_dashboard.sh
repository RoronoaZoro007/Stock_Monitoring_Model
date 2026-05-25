#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  ./scripts/run_forward_shadow_dashboard.sh

Environment:
  TUSHARE_TOKEN=...              required before starting a job from the page
  WXPUSHER_APP_TOKEN=...         required when the page starts a job with notifications enabled
  TUSHARE_PROXY_URL=...          default https://tt.xiaodefa.cn
  DASHBOARD_HOST=127.0.0.1       default local-only bind
  DASHBOARD_PORT=8788            default dashboard port
  OUTPUT_ROOT=reports/tushare/v8_forward_shadow
  WXPUSHER_TOPIC_ID=44635
USAGE
  exit 0
fi

export TUSHARE_PROXY_URL="${TUSHARE_PROXY_URL:-https://tt.xiaodefa.cn}"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt

HOST="${DASHBOARD_HOST:-127.0.0.1}"
PORT="${DASHBOARD_PORT:-8788}"
OUTPUT_ROOT="${OUTPUT_ROOT:-reports/tushare/v8_forward_shadow}"
TOPIC_ID="${WXPUSHER_TOPIC_ID:-44635}"

exec python research/v8_research/forward_shadow_dashboard.py \
  --host "$HOST" \
  --port "$PORT" \
  --output-root "$OUTPUT_ROOT" \
  --topic-id "$TOPIC_ID"
