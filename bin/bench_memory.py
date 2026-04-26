#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROMPT = (
    "Summarize the tradeoffs in local LLM serving on Apple Silicon. "
    "Discuss memory pressure, concurrency, context length, and scheduling. "
    "Keep the answer practical and concrete."
)


def run_command(args: list[str]) -> str:
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        command = " ".join(args)
        raise RuntimeError(detail or f"command failed: {command}")
    return result.stdout


def run_optional(args: list[str]) -> str:
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    return result.stdout


def wait_for_http(url: str, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def parse_lms_ps(lms_bin: str) -> list[dict[str, Any]]:
    output = run_command([lms_bin, "ps", "--json"])
    data = json.loads(output or "[]")
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected lms ps payload: {output}")
    return [item for item in data if isinstance(item, dict)]


def unload_all_models(lms_bin: str) -> None:
    for item in parse_lms_ps(lms_bin):
        identifier = item.get("identifier")
        if isinstance(identifier, str):
            subprocess.run(
                [lms_bin, "unload", identifier],
                check=False,
                capture_output=True,
                text=True,
            )


def load_model(
    *,
    lms_bin: str,
    model_key: str,
    alias: str,
    context_length: int,
    parallel: int,
) -> None:
    unload_all_models(lms_bin)
    run_command(
        [
            lms_bin,
            "load",
            model_key,
            "--identifier",
            alias,
            "-c",
            str(context_length),
            "--parallel",
            str(parallel),
            "-y",
        ]
    )


def parse_vm_stat() -> dict[str, int]:
    output = run_command(["vm_stat"])
    lines = output.splitlines()
    page_size = 16384
    if lines and "page size of " in lines[0]:
        page_size = int(lines[0].split("page size of ", 1)[1].split(" bytes", 1)[0])
    metrics: dict[str, int] = {"page_size": page_size}
    for line in lines[1:]:
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        cleaned = raw_value.strip().rstrip(".").replace('"', "").replace(",", "")
        if cleaned.isdigit():
            metrics[key.strip().lower().replace(" ", "_")] = int(cleaned)
    return metrics


def parse_memory_pressure_free_percent() -> int | None:
    output = run_command(["memory_pressure", "-Q"])
    for line in output.splitlines():
        if "free percentage" not in line:
            continue
        value = line.rsplit(":", 1)[-1].strip().rstrip("%")
        try:
            return int(value)
        except ValueError:
            return None
    return None


def find_listener_pid(port: int) -> int | None:
    result = subprocess.run(
        ["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return int(lines[0]) if lines else None


def process_tree_rss_bytes(root_pid: int | None) -> int | None:
    if root_pid is None:
        return None
    output = run_command(["ps", "-axo", "pid=,ppid=,rss="])
    children_by_parent: dict[int, list[int]] = {}
    rss_by_pid: dict[int, int] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        pid, ppid, rss_kib = map(int, parts)
        children_by_parent.setdefault(ppid, []).append(pid)
        rss_by_pid[pid] = rss_kib

    total_rss_kib = 0
    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total_rss_kib += rss_by_pid.get(pid, 0)
        stack.extend(children_by_parent.get(pid, []))
    return total_rss_kib * 1024


@dataclass
class MemorySample:
    label: str
    timestamp: float
    free_percent: int | None
    pages_free: int | None
    pages_speculative: int | None
    pages_stored_in_compressor: int | None
    pageouts: int | None
    swapouts: int | None
    server_rss_bytes: int | None


def sample_memory(label: str, backend_port: int) -> MemorySample:
    vm = parse_vm_stat()
    return MemorySample(
        label=label,
        timestamp=time.time(),
        free_percent=parse_memory_pressure_free_percent(),
        pages_free=vm.get("pages_free"),
        pages_speculative=vm.get("pages_speculative"),
        pages_stored_in_compressor=vm.get("pages_stored_in_compressor"),
        pageouts=vm.get("pageouts"),
        swapouts=vm.get("swapouts"),
        server_rss_bytes=process_tree_rss_bytes(find_listener_pid(backend_port)),
    )


class Sampler:
    def __init__(self, *, backend_port: int, interval_seconds: float) -> None:
        self.backend_port = backend_port
        self.interval_seconds = interval_seconds
        self.samples: list[MemorySample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.samples.append(sample_memory("request_start", self.backend_port))
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self.samples.append(sample_memory("request_end", self.backend_port))

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.samples.append(sample_memory("request_running", self.backend_port))


@dataclass
class RequestResult:
    index: int
    ok: bool
    latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    error: str | None


def make_payload(model: str, max_tokens: int, request_index: int) -> bytes:
    body = {
        "model": model,
        "stream": False,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": "You are a precise benchmark assistant."},
            {
                "role": "user",
                "content": f"{PROMPT}\n\nBenchmark request {request_index}.",
            },
        ],
    }
    return json.dumps(body).encode("utf-8")


def execute_request(
    *,
    url: str,
    model: str,
    max_tokens: int,
    request_index: int,
    timeout_seconds: float,
    start_event: threading.Event,
) -> RequestResult:
    start_event.wait()
    request = urllib.request.Request(
        url,
        data=make_payload(model, max_tokens, request_index),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
        usage = payload.get("usage") if isinstance(payload, dict) else None
        return RequestResult(
            index=request_index,
            ok=True,
            latency_seconds=time.perf_counter() - started,
            prompt_tokens=usage.get("prompt_tokens")
            if isinstance(usage, dict)
            else None,
            completion_tokens=usage.get("completion_tokens")
            if isinstance(usage, dict)
            else None,
            error=None,
        )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return RequestResult(
            index=request_index,
            ok=False,
            latency_seconds=time.perf_counter() - started,
            prompt_tokens=None,
            completion_tokens=None,
            error=detail.strip() or str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return RequestResult(
            index=request_index,
            ok=False,
            latency_seconds=time.perf_counter() - started,
            prompt_tokens=None,
            completion_tokens=None,
            error=str(exc),
        )


@dataclass
class LevelResult:
    model_alias: str
    model_key: str
    context_length: int
    parallel: int
    idle_rss_gb: float | None
    max_request_rss_gb: float | None
    request_rss_delta_gb: float | None
    min_free_percent: int | None
    delta_pageouts: int | None
    delta_swapouts: int | None
    completed: int
    failed: int
    latency_p50_seconds: float | None
    latency_p95_seconds: float | None
    prompt_tokens_total: int
    completion_tokens_total: int
    samples: list[MemorySample]
    requests: list[RequestResult]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def run_request_level(
    *,
    backend_port: int,
    model_alias: str,
    parallel: int,
    max_tokens: int,
    timeout_seconds: float,
) -> LevelResult:
    sampler = Sampler(backend_port=backend_port, interval_seconds=0.5)
    sampler.start()
    start_event = threading.Event()
    request_url = f"http://127.0.0.1:{backend_port}/v1/chat/completions"
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = [
            executor.submit(
                execute_request,
                url=request_url,
                model=model_alias,
                max_tokens=max_tokens,
                request_index=index,
                timeout_seconds=timeout_seconds,
                start_event=start_event,
            )
            for index in range(1, parallel + 1)
        ]
        start_event.set()
        requests = [future.result() for future in futures]
    sampler.stop()

    succeeded = [item for item in requests if item.ok]
    latencies = [item.latency_seconds for item in succeeded]
    rss_values = [
        item.server_rss_bytes
        for item in sampler.samples
        if item.server_rss_bytes is not None
    ]
    free_percents = [
        item.free_percent for item in sampler.samples if item.free_percent is not None
    ]
    pageouts = [item.pageouts for item in sampler.samples if item.pageouts is not None]
    swapouts = [item.swapouts for item in sampler.samples if item.swapouts is not None]
    idle_rss = sampler.samples[0].server_rss_bytes
    max_rss = max(rss_values) if rss_values else None

    return LevelResult(
        model_alias=model_alias,
        model_key="",
        context_length=0,
        parallel=parallel,
        idle_rss_gb=(idle_rss / (1024**3) if idle_rss is not None else None),
        max_request_rss_gb=(max_rss / (1024**3) if max_rss is not None else None),
        request_rss_delta_gb=(
            (max_rss - idle_rss) / (1024**3)
            if max_rss is not None and idle_rss is not None
            else None
        ),
        min_free_percent=min(free_percents) if free_percents else None,
        delta_pageouts=(pageouts[-1] - pageouts[0] if len(pageouts) >= 2 else None),
        delta_swapouts=(swapouts[-1] - swapouts[0] if len(swapouts) >= 2 else None),
        completed=len(succeeded),
        failed=len(requests) - len(succeeded),
        latency_p50_seconds=percentile(latencies, 0.50),
        latency_p95_seconds=percentile(latencies, 0.95),
        prompt_tokens_total=sum(item.prompt_tokens or 0 for item in succeeded),
        completion_tokens_total=sum(item.completion_tokens or 0 for item in succeeded),
        samples=sampler.samples,
        requests=requests,
    )


def parse_model_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be formatted as alias=key")
    alias, key = value.split("=", 1)
    if not alias or not key:
        raise argparse.ArgumentTypeError("model must include both alias and key")
    return alias, key


def print_summary(results: list[LevelResult]) -> None:
    print()
    print(
        "model       | parallel | ok/total | idle_gb | max_gb | delta_gb | min_free% | p50_s | p95_s"
    )
    print("-" * 96)
    for result in results:
        total = result.completed + result.failed
        idle = f"{result.idle_rss_gb:.2f}" if result.idle_rss_gb is not None else "-"
        max_rss = (
            f"{result.max_request_rss_gb:.2f}"
            if result.max_request_rss_gb is not None
            else "-"
        )
        delta = (
            f"{result.request_rss_delta_gb:.2f}"
            if result.request_rss_delta_gb is not None
            else "-"
        )
        free = (
            str(result.min_free_percent) if result.min_free_percent is not None else "-"
        )
        p50 = (
            f"{result.latency_p50_seconds:.1f}"
            if result.latency_p50_seconds is not None
            else "-"
        )
        p95 = (
            f"{result.latency_p95_seconds:.1f}"
            if result.latency_p95_seconds is not None
            else "-"
        )
        print(
            f"{result.model_alias:<11} | "
            f"{result.parallel:>8} | "
            f"{result.completed:>2}/{total:<5} | "
            f"{idle:>7} | "
            f"{max_rss:>6} | "
            f"{delta:>8} | "
            f"{free:>9} | "
            f"{p50:>5} | "
            f"{p95:>5}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark LM Studio memory usage for configured models"
    )
    parser.add_argument(
        "--lms-bin", default=str(Path.home() / ".lmstudio" / "bin" / "lms")
    )
    parser.add_argument("--backend-port", type=int, default=8097)
    parser.add_argument("--model", action="append", type=parse_model_arg)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--parallel-levels", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--settle-seconds", type=float, default=8.0)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    models = args.model or [
        ("gemma4:e2b", "google/gemma-4-e2b"),
        ("gemma4:26b", "google/gemma-4-26b-a4b"),
    ]
    output_json = (
        Path(args.output_json)
        if args.output_json
        else Path("data") / "benchmarks" / f"memory-{int(time.time())}.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)

    run_optional(
        [
            args.lms_bin,
            "server",
            "start",
            "-p",
            str(args.backend_port),
            "--bind",
            "127.0.0.1",
        ]
    )
    wait_for_http(f"http://127.0.0.1:{args.backend_port}/v1/models", timeout_seconds=60)

    results: list[LevelResult] = []
    errors: list[dict[str, Any]] = []
    baseline = sample_memory("baseline", args.backend_port)
    for alias, key in models:
        for parallel in args.parallel_levels:
            print(f"\n==> Loading {alias} parallel={parallel}")
            try:
                load_model(
                    lms_bin=args.lms_bin,
                    model_key=key,
                    alias=alias,
                    context_length=args.context_length,
                    parallel=parallel,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "model_alias": alias,
                        "model_key": key,
                        "context_length": args.context_length,
                        "parallel": parallel,
                        "error": str(exc),
                    }
                )
                print(f"Load failed: {exc}")
                continue
            time.sleep(args.settle_seconds)
            loaded = sample_memory("loaded_idle", args.backend_port)
            print(
                f"Idle RSS: {loaded.server_rss_bytes / (1024**3):.2f} GiB"
                if loaded.server_rss_bytes is not None
                else "Idle RSS: unknown"
            )
            result = run_request_level(
                backend_port=args.backend_port,
                model_alias=alias,
                parallel=parallel,
                max_tokens=args.max_tokens,
                timeout_seconds=args.request_timeout_seconds,
            )
            result.model_key = key
            result.context_length = args.context_length
            result.samples.insert(0, loaded)
            results.append(result)
            if result.max_request_rss_gb is not None:
                print(
                    f"Completed {result.completed}/{result.completed + result.failed}; "
                    f"max RSS {result.max_request_rss_gb:.2f} GiB; "
                    f"delta {result.request_rss_delta_gb:.2f} GiB"
                    if result.request_rss_delta_gb is not None
                    else f"Completed {result.completed}/{result.completed + result.failed}; max RSS {result.max_request_rss_gb:.2f} GiB"
                )

    output_json.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "baseline": asdict(baseline),
                "results": [asdict(result) for result in results],
                "errors": errors,
            },
            indent=2,
        )
    )
    print_summary(results)
    print(f"\nSaved JSON report to {output_json}")


if __name__ == "__main__":
    main()
