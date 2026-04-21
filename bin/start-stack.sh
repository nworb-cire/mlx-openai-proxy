#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
HOME_DIR="${HOME:-$(eval echo ~$(whoami))}"
export PATH="${HOME_DIR}/.lmstudio/bin:${HOME_DIR}/.local/bin:/Library/Frameworks/Python.framework/Versions/3.12/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PYTHONUNBUFFERED="1"

LM_BIN="${LM_BIN:-${HOME_DIR}/.lmstudio/bin/lms}"
PROXY_BIN="${PROXY_BIN:-${REPO_ROOT}/.venv/bin/mlx-openai-proxy}"
BACKEND_PORT="8097"
PROXY_PORT="8080"
PRIMARY_MODEL_KEY="google/gemma-4-e2b"
PRIMARY_MODEL_ID="gemma4:e2b"
PRIMARY_CONTEXT_LENGTH="8192"
PRIMARY_MODEL_PARALLEL="2"
SECONDARY_MODEL_ID="gemma4:26b"

"$LM_BIN" server start -p "$BACKEND_PORT" --bind 127.0.0.1 >/dev/null 2>&1 || true

for _ in {1..30}; do
  if curl -fsS "http://127.0.0.1:${BACKEND_PORT}/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if "$LM_BIN" ps --json | grep -q "\"identifier\":\"${SECONDARY_MODEL_ID}\""; then
  "$LM_BIN" unload "$SECONDARY_MODEL_ID" >/dev/null 2>&1 || true
fi

if ! "$LM_BIN" ps --json | grep -q "\"identifier\":\"${PRIMARY_MODEL_ID}\""; then
  "$LM_BIN" load "$PRIMARY_MODEL_KEY" --identifier "$PRIMARY_MODEL_ID" -c "$PRIMARY_CONTEXT_LENGTH" --parallel "$PRIMARY_MODEL_PARALLEL" -y >/dev/null 2>&1
fi

exec env \
  MLX_PROXY_BACKEND_BASE_URL="http://127.0.0.1:${BACKEND_PORT}/v1" \
  MLX_PROXY_ACTIVE_REQUEST_TIMEOUT_SECONDS="600" \
  MLX_PROXY_MAX_UPSTREAM_CONCURRENCY="2" \
  MLX_PROXY_LOG_LEVEL="INFO" \
  MLX_PROXY_METRICS_DB_PATH="${MLX_PROXY_METRICS_DB_PATH:-${REPO_ROOT}/data/metrics.db}" \
  "$PROXY_BIN" --host 0.0.0.0 --port "$PROXY_PORT"
