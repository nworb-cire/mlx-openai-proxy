# MLX OpenAI Proxy

Transparent OpenAI-compatible middleware for local MLX-native backends on Apple Silicon.

## Goals

- Preserve the client-facing OpenAI API surface.
- Keep normal streaming behavior for ordinary requests.
- Preserve visible reasoning on reasoning-capable models.
- Make structured output safer by using a two-phase "reason, then format" path.

## Endpoints

- `/v1/chat/completions`
- `/v1/responses`
- `/v1/models`
- `/healthz`

## Run

```bash
git clone git@github.com:nworb-cire/mlx-openai-proxy.git
cd mlx-openai-proxy
uv venv
source .venv/bin/activate
uv pip install -e .
mlx-openai-proxy
```

Or use the included launch script:

```bash
./bin/start-stack.sh
```

Environment variables:

- `MLX_PROXY_BACKEND_BASE_URL=http://127.0.0.1:8080/v1`
- `MLX_PROXY_HOST=0.0.0.0`
- `MLX_PROXY_PORT=8090`
- `MLX_PROXY_METRICS_DB_PATH=./data/metrics.db`
- `LM_BIN=~/.lmstudio/bin/lms`
- `PROXY_BIN=./.venv/bin/mlx-openai-proxy`
- `MLX_PROXY_REASONING_VISIBILITY=compatible`
- `MLX_PROXY_SCHEMA_MODE=auto`
