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
