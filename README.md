# MLX OpenAI Proxy

MLX OpenAI Proxy is a compatibility layer that lets OpenAI-style clients talk to local MLX model backends on Apple Silicon.

## What It Does

It sits between an application and a local model server, translating requests in a way that keeps the OpenAI API shape familiar to the caller. The goal is to make local models easier to use with existing tools, SDKs, and workflows that already expect OpenAI-compatible behavior.

The proxy is especially useful when you want to run models locally without rewriting the rest of your stack. It helps preserve normal chat and streaming patterns, and it adds safer handling for responses that need to match a strict structure.

In practice, this project is meant to sit on top of `LM Studio`. `LM Studio` handles loading and serving the local MLX-backed models, while this proxy adds the OpenAI-compatible interface, request routing, and observability layer in front of it.

## What It Is Used For

Use this project when you want to:

- point OpenAI-compatible applications at local MLX-hosted models
- keep a familiar API surface while changing the backend
- support structured-output workflows more reliably
- observe local model traffic through a lightweight dashboard and request metrics
- manage a small local serving stack where `LM Studio` runs the models and the proxy coordinates how clients reach them

## Burst Workflow

This repo also includes a burst workflow for handling requests that should run on a larger or different model than the normal default.

At a high level, the proxy keeps a default model ready for ordinary traffic. When a request targets the burst model, the proxy can switch `LM Studio` over to that model, serve the queued burst work, and then return to the default model when the burst traffic is finished. The purpose is to make it practical to keep a lighter everyday model available while still being able to temporarily "burst" into a heavier model when needed.

## In Practice

This project is meant for developers building local AI workflows on Apple hardware. It is not the model-serving engine itself; it is the layer that sits in front of `LM Studio` to make local inference feel more like a drop-in OpenAI-style service.
