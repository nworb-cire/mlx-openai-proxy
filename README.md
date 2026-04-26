# MLX OpenAI Proxy

MLX OpenAI Proxy is a compatibility layer that lets OpenAI-style clients talk to local MLX model backends on Apple Silicon.

## What It Does

It sits between an application and a local model server, translating requests in a way that keeps the OpenAI API shape familiar to the caller. The goal is to make local models easier to use with existing tools, SDKs, and workflows that already expect OpenAI-compatible behavior.

The proxy is especially useful when you want to run models locally without rewriting the rest of your stack. It helps preserve normal chat and streaming patterns, and it adds safer handling for responses that need to match a strict structure.

In practice, this project is meant to sit on top of `LM Studio`. `LM Studio` handles loading and serving the local MLX-backed models, while this proxy adds the OpenAI-compatible interface, request routing, and observability layer in front of it.

## Quickstart

Prerequisites:

- Apple Silicon Mac
- `LM Studio` with the `lms` CLI installed
- `uv` for Python dependency management

Install the project and development dependencies:

```sh
uv sync --extra dev
```

Run the test suite:

```sh
uv run pytest
```

Start `LM Studio` and the proxy together:

```sh
bin/start-stack.sh
```

By default, the helper script starts the `LM Studio` backend on `127.0.0.1:8097` and the OpenAI-compatible proxy on `0.0.0.0:8080`.

List available models:

```sh
curl http://127.0.0.1:8080/v1/models
```

Send a chat completion request:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma4:e2b",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}]
  }'
```

Open the dashboard at:

```text
http://127.0.0.1:8080/admin/dashboard
```

## Configuration

Runtime settings are read from environment variables with the `MLX_PROXY_` prefix. You can copy `.env.example` to `.env` for local overrides when running the Python entrypoint directly.

| Setting | Default | Purpose |
| --- | --- | --- |
| `MLX_PROXY_HOST` | `0.0.0.0` | Interface the proxy binds to. |
| `MLX_PROXY_PORT` | `8090` | Port used by the Python entrypoint. `bin/start-stack.sh` overrides this to `8080`. |
| `MLX_PROXY_BACKEND_BASE_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible backend URL. `bin/start-stack.sh` starts LM Studio on `127.0.0.1:8097` and points the proxy there. |
| `MLX_PROXY_MODEL_CONFIG_PATH` | `config/models.json` | JSON file containing the served model aliases and LM Studio model keys. |
| `MLX_PROXY_METRICS_DB_PATH` | `~/.local/share/mlx-openai-proxy/metrics.db` | SQLite database used for request history and dashboard metrics. |
| `MLX_PROXY_LM_STUDIO_BIN` | discovered `lms` or `~/.lmstudio/bin/lms` | LM Studio CLI used for model residency management. |
| `MLX_PROXY_MAX_UPSTREAM_CONCURRENCY` | `2` | Maximum number of concurrent upstream model requests. |
| `MLX_PROXY_BACKEND_TIMEOUT_SECONDS` | `600` | HTTP timeout for backend requests. |
| `MLX_PROXY_ACTIVE_REQUEST_TIMEOUT_SECONDS` | `600` | Maximum active request runtime before the proxy returns a timeout. |
| `MLX_PROXY_LOG_LEVEL` | `INFO` | Python logging level. |

## What It Is Used For

Use this project when you want to:

- point OpenAI-compatible applications at local MLX-hosted models
- keep a familiar API surface while changing the backend
- support structured-output workflows more reliably
- observe local model traffic through a lightweight dashboard and request metrics
- manage a small local serving stack where `LM Studio` runs the models and the proxy coordinates how clients reach them

## Model Routing

This repo also supports serving multiple configured models through one OpenAI-compatible API surface. When a request targets a configured model that is not currently loaded, the proxy switches `LM Studio` to that model on demand.

The served model set is configured as a JSON list. Each entry controls the LM Studio model key, exposed alias, context length, and parallelism.

To add or remove a served model, edit `config/models.json` and restart the proxy.

You can also point the proxy at another model config file with `MLX_PROXY_MODEL_CONFIG_PATH`.

## In Practice

This project is meant for developers building local AI workflows on Apple hardware. It is not the model-serving engine itself; it is the layer that sits in front of `LM Studio` to make local inference feel more like a drop-in OpenAI-style service.
