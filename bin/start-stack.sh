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
MODEL_CONFIG_PATH="${MLX_PROXY_MODEL_CONFIG_PATH:-${REPO_ROOT}/config/models.json}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

MODEL_INFO="$("$PYTHON_BIN" - "$MODEL_CONFIG_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    models = json.load(handle)
if not models:
    raise SystemExit("model config must contain at least one model")
first = models[0]
aliases = " ".join(model["alias"] for model in models[1:])
print(first["key"])
print(first["alias"])
print(first.get("context_length", 8192))
print(first.get("parallel", 1))
print(aliases)
PY
)"
PRIMARY_MODEL_KEY="${${(f)MODEL_INFO}[1]}"
PRIMARY_MODEL_ID="${${(f)MODEL_INFO}[2]}"
PRIMARY_CONTEXT_LENGTH="${${(f)MODEL_INFO}[3]}"
PRIMARY_MODEL_PARALLEL="${${(f)MODEL_INFO}[4]}"
OTHER_MODEL_IDS="${${(f)MODEL_INFO}[5]}"

"$LM_BIN" server start -p "$BACKEND_PORT" --bind 127.0.0.1 >/dev/null 2>&1 || true

for _ in {1..30}; do
  if curl -fsS "http://127.0.0.1:${BACKEND_PORT}/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

for MODEL_ID in ${(z)OTHER_MODEL_IDS}; do
  if "$LM_BIN" ps --json | grep -q "\"identifier\":\"${MODEL_ID}\""; then
    "$LM_BIN" unload "$MODEL_ID" >/dev/null 2>&1 || true
  fi
done

if ! "$LM_BIN" ps --json | grep -q "\"identifier\":\"${PRIMARY_MODEL_ID}\""; then
  "$LM_BIN" load "$PRIMARY_MODEL_KEY" --identifier "$PRIMARY_MODEL_ID" -c "$PRIMARY_CONTEXT_LENGTH" --parallel "$PRIMARY_MODEL_PARALLEL" -y >/dev/null 2>&1
fi

exec env \
  MLX_PROXY_BACKEND_BASE_URL="http://127.0.0.1:${BACKEND_PORT}/v1" \
  MLX_PROXY_ACTIVE_REQUEST_TIMEOUT_SECONDS="600" \
  MLX_PROXY_MAX_UPSTREAM_CONCURRENCY="2" \
  MLX_PROXY_MODEL_CONFIG_PATH="$MODEL_CONFIG_PATH" \
  MLX_PROXY_LOG_LEVEL="INFO" \
  MLX_PROXY_METRICS_DB_PATH="${MLX_PROXY_METRICS_DB_PATH:-${REPO_ROOT}/data/metrics.db}" \
  "$PROXY_BIN" --host 0.0.0.0 --port "$PROXY_PORT"
