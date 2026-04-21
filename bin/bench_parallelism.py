#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROMPT = """Write a detailed technical brief about running local language models on Apple Silicon.

Requirements:
- Use 10 numbered sections.
- Each section must be 2 full paragraphs.
- Include concrete discussion of memory pressure, model parallelism, throughput, latency, scheduling, and failure modes.
- Keep the tone practical and specific.
- Do not use bullet points inside the sections.
- End with a short conclusion paragraph.
"""

REFERENCE_BLOCK = """
Reference material:
Apple Silicon uses unified memory, so local inference workloads often compete across CPU, GPU, file cache, and application memory rather than staying in isolated pools. A serving stack that looks healthy by process RSS alone can still run into pressure when the GPU allocator grows or when the operating system starts compressing pages aggressively.

In practical local serving, throughput is shaped by at least two distinct phases. The first is prompt processing, sometimes called prefill, where the model digests the request context before any output token appears. The second is decode, where the model emits new tokens autoregressively. Prefill tends to benefit from batching and can scale differently from decode, especially once concurrent requests are in flight.

Operators usually care about three things at once: how many concurrent requests the model can accept, how much latency each caller experiences, and whether the machine remains stable under sustained pressure. A concurrency setting that technically works may still be a poor production choice if tail latency doubles or if swap activity starts rising.

When a local proxy sits in front of the model server, its own admission control can hide the true model capacity. If the proxy only allows a small number of upstream requests, the benchmark may mostly measure queueing in the proxy rather than the backend model scheduler. Any useful stress test needs those limits aligned.

LM Studio exposes model loading options such as context length and parallel request slots. Changing the parallel setting can affect both how many requests execute at once and how the backend schedules work internally. The best setting depends on the specific model, quantization, prompt length, completion length, and available memory headroom.

Reasoning-capable models can make timing analysis less obvious because they may emit hidden or visible reasoning tokens before the final assistant answer. For benchmarking server capacity, those reasoning tokens still count as decode work because they consume generation time and backend resources even when the user interface chooses not to display them.

Stable local serving requires looking beyond success or failure. Useful diagnostics include time to first token, total completion time, prompt token counts, completion token counts, pageouts, swapouts, compressed memory growth, and whether the machine retains a comfortable free-memory margin throughout the run.
""".strip()


def run_command(args: list[str]) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
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


def ensure_model_parallel(
    *,
    lms_bin: str,
    model_key: str,
    alias: str,
    context_length: int,
    parallel: int,
) -> None:
    loaded = parse_lms_ps(lms_bin)
    for item in loaded:
        if item.get("identifier") == alias and int(item.get("parallel", 0) or 0) == parallel:
            return

    for item in loaded:
        identifier = item.get("identifier")
        if isinstance(identifier, str):
            subprocess.run([lms_bin, "unload", identifier], check=False, capture_output=True, text=True)

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
    if lines:
        marker = "page size of "
        if marker in lines[0]:
            page_size = int(lines[0].split(marker, 1)[1].split(" bytes", 1)[0])
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
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    return int(lines[0])


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
class Sample:
    timestamp: float
    free_percent: int | None
    pages_free: int | None
    pages_speculative: int | None
    pages_stored_in_compressor: int | None
    pageouts: int | None
    swapouts: int | None
    server_rss_bytes: int | None


class Sampler:
    def __init__(self, *, backend_port: int, interval_seconds: float) -> None:
        self.backend_port = backend_port
        self.interval_seconds = interval_seconds
        self.samples: list[Sample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.samples.append(self.sample_once())
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self.samples.append(self.sample_once())

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.samples.append(self.sample_once())

    def sample_once(self) -> Sample:
        vm = parse_vm_stat()
        return Sample(
            timestamp=time.time(),
            free_percent=parse_memory_pressure_free_percent(),
            pages_free=vm.get("pages_free"),
            pages_speculative=vm.get("pages_speculative"),
            pages_stored_in_compressor=vm.get("pages_stored_in_compressor"),
            pageouts=vm.get("pageouts"),
            swapouts=vm.get("swapouts"),
            server_rss_bytes=process_tree_rss_bytes(find_listener_pid(self.backend_port)),
        )


@dataclass
class RequestResult:
    index: int
    ok: bool
    latency_seconds: float
    time_to_first_decode_seconds: float | None
    status_code: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    total_tokens: int | None
    input_tps: float | None
    decode_tps: float | None
    response_chars: int | None
    error: str | None


@dataclass
class BenchmarkResult:
    parallel: int
    batch_wall_seconds: float
    completed: int
    failed: int
    latency_p50_seconds: float | None
    latency_p95_seconds: float | None
    avg_latency_seconds: float | None
    ttft_p50_seconds: float | None
    ttft_p95_seconds: float | None
    prompt_tokens_total: int
    completion_tokens_total: int
    aggregate_input_tps: float | None
    aggregate_decode_tps: float | None
    input_tps_p50: float | None
    input_tps_p95: float | None
    decode_tps_p50: float | None
    decode_tps_p95: float | None
    min_free_percent: int | None
    start_free_percent: int | None
    end_free_percent: int | None
    max_server_rss_gb: float | None
    delta_pageouts: int | None
    delta_swapouts: int | None
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
    fraction = index - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def make_payload(model: str, max_tokens: int, request_index: int) -> bytes:
    long_reference = "\n\n".join(f"Document {i}:\n{REFERENCE_BLOCK}" for i in range(1, 13))
    body = {
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise but thorough technical writer.",
            },
            {
                "role": "user",
                "content": (
                    f"{PROMPT}\n\n"
                    f"Use the following reference corpus and synthesize it rather than quoting it directly.\n\n"
                    f"{long_reference}\n\n"
                    f"Include a one-sentence note that this is benchmark run {request_index}."
                ),
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
    payload = make_payload(model, max_tokens, request_index)
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            usage: dict[str, Any] = {}
            response_chars = 0
            first_decode_seconds: float | None = None
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                if not isinstance(event, dict):
                    continue
                if "usage" in event and isinstance(event["usage"], dict):
                    usage = event["usage"]
                choices = event.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta", {})
                if not isinstance(delta, dict):
                    continue
                reasoning = delta.get("reasoning_content")
                content = delta.get("content")
                has_decode = (
                    isinstance(reasoning, str)
                    and bool(reasoning)
                    or isinstance(content, str)
                    and bool(content)
                )
                if has_decode and first_decode_seconds is None:
                    first_decode_seconds = time.perf_counter() - started
                if isinstance(reasoning, str):
                    response_chars += len(reasoning)
                if isinstance(content, str):
                    response_chars += len(content)
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            total_latency = time.perf_counter() - started
            if (
                first_decode_seconds is None
                or not isinstance(prompt_tokens, int)
                or not isinstance(completion_tokens, int)
                or response_chars <= 0
            ):
                return RequestResult(
                    index=request_index,
                    ok=False,
                    latency_seconds=total_latency,
                    time_to_first_decode_seconds=first_decode_seconds,
                    status_code=getattr(response, "status", 200),
                    prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
                    completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
                    reasoning_tokens=reasoning_tokens if isinstance(reasoning_tokens, int) else None,
                    total_tokens=usage.get("total_tokens") if isinstance(usage.get("total_tokens"), int) else None,
                    input_tps=None,
                    decode_tps=None,
                    response_chars=response_chars,
                    error="stream ended without usable decode tokens and usage",
                )
            input_tps = None
            if isinstance(prompt_tokens, int) and first_decode_seconds and first_decode_seconds > 0:
                input_tps = prompt_tokens / first_decode_seconds
            decode_tps = None
            decode_window = total_latency - first_decode_seconds if first_decode_seconds is not None else None
            if isinstance(completion_tokens, int) and decode_window and decode_window > 0:
                decode_tps = completion_tokens / decode_window
            return RequestResult(
                index=request_index,
                ok=True,
                latency_seconds=total_latency,
                time_to_first_decode_seconds=first_decode_seconds,
                status_code=getattr(response, "status", 200),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
                total_tokens=usage.get("total_tokens"),
                input_tps=input_tps,
                decode_tps=decode_tps,
                response_chars=response_chars,
                error=None,
            )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return RequestResult(
            index=request_index,
            ok=False,
            latency_seconds=time.perf_counter() - started,
            time_to_first_decode_seconds=None,
            status_code=exc.code,
            prompt_tokens=None,
            completion_tokens=None,
            reasoning_tokens=None,
            total_tokens=None,
            input_tps=None,
            decode_tps=None,
            response_chars=None,
            error=detail.strip() or str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return RequestResult(
            index=request_index,
            ok=False,
            latency_seconds=time.perf_counter() - started,
            time_to_first_decode_seconds=None,
            status_code=None,
            prompt_tokens=None,
            completion_tokens=None,
            reasoning_tokens=None,
            total_tokens=None,
            input_tps=None,
            decode_tps=None,
            response_chars=None,
            error=str(exc),
        )


def start_proxy(
    *,
    proxy_bin: str,
    proxy_port: int,
    backend_port: int,
    model_parallel: int,
    metrics_db_path: Path,
    log_path: Path,
) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env.update(
        {
            "MLX_PROXY_BACKEND_BASE_URL": f"http://127.0.0.1:{backend_port}/v1",
            "MLX_PROXY_MAX_UPSTREAM_CONCURRENCY": str(model_parallel),
            "MLX_PROXY_DEFAULT_MODEL_PARALLEL": str(model_parallel),
            "MLX_PROXY_METRICS_DB_PATH": str(metrics_db_path),
            "MLX_PROXY_LOG_LEVEL": "WARNING",
        }
    )
    log_handle = log_path.open("wb")
    process = subprocess.Popen(
        [proxy_bin, "--host", "127.0.0.1", "--port", str(proxy_port)],
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    process._benchmark_log_handle = log_handle  # type: ignore[attr-defined]
    wait_for_http(f"http://127.0.0.1:{proxy_port}/healthz", timeout_seconds=30)
    return process


def stop_proxy(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    log_handle = getattr(process, "_benchmark_log_handle", None)
    if log_handle is not None:
        log_handle.close()


def run_benchmark_level(
    *,
    parallel: int,
    model: str,
    proxy_url: str,
    max_tokens: int,
    request_timeout_seconds: float,
    backend_port: int,
) -> BenchmarkResult:
    sampler = Sampler(backend_port=backend_port, interval_seconds=1.0)
    sampler.start()
    start_event = threading.Event()
    request_url = f"{proxy_url.rstrip('/')}/v1/chat/completions"
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = [
            executor.submit(
                execute_request,
                url=request_url,
                model=model,
                max_tokens=max_tokens,
                request_index=index,
                timeout_seconds=request_timeout_seconds,
                start_event=start_event,
            )
            for index in range(1, parallel + 1)
        ]
        start_event.set()
        requests = [future.result() for future in futures]
    batch_wall_seconds = time.perf_counter() - started
    sampler.stop()

    succeeded = [item for item in requests if item.ok]
    latencies = [item.latency_seconds for item in succeeded]
    ttfts = [item.time_to_first_decode_seconds for item in succeeded if item.time_to_first_decode_seconds is not None]
    prompt_tokens_total = sum(item.prompt_tokens or 0 for item in succeeded)
    completion_tokens_total = sum(item.completion_tokens or 0 for item in succeeded)
    input_tps_values = [item.input_tps for item in succeeded if item.input_tps is not None]
    decode_tps_values = [item.decode_tps for item in succeeded if item.decode_tps is not None]

    samples = sampler.samples
    free_percents = [item.free_percent for item in samples if item.free_percent is not None]
    rss_values = [item.server_rss_bytes for item in samples if item.server_rss_bytes is not None]
    pageouts = [item.pageouts for item in samples if item.pageouts is not None]
    swapouts = [item.swapouts for item in samples if item.swapouts is not None]

    return BenchmarkResult(
        parallel=parallel,
        batch_wall_seconds=batch_wall_seconds,
        completed=len(succeeded),
        failed=len(requests) - len(succeeded),
        latency_p50_seconds=percentile(latencies, 0.50),
        latency_p95_seconds=percentile(latencies, 0.95),
        avg_latency_seconds=(statistics.fmean(latencies) if latencies else None),
        ttft_p50_seconds=percentile(ttfts, 0.50),
        ttft_p95_seconds=percentile(ttfts, 0.95),
        prompt_tokens_total=prompt_tokens_total,
        completion_tokens_total=completion_tokens_total,
        aggregate_input_tps=(
            prompt_tokens_total / max(ttfts) if ttfts and prompt_tokens_total > 0 else None
        ),
        aggregate_decode_tps=(
            completion_tokens_total / max(1e-9, batch_wall_seconds - min(ttfts))
            if ttfts and completion_tokens_total > 0 and batch_wall_seconds > min(ttfts)
            else None
        ),
        input_tps_p50=percentile(input_tps_values, 0.50),
        input_tps_p95=percentile(input_tps_values, 0.95),
        decode_tps_p50=percentile(decode_tps_values, 0.50),
        decode_tps_p95=percentile(decode_tps_values, 0.95),
        min_free_percent=min(free_percents) if free_percents else None,
        start_free_percent=samples[0].free_percent if samples else None,
        end_free_percent=samples[-1].free_percent if samples else None,
        max_server_rss_gb=(max(rss_values) / (1024**3) if rss_values else None),
        delta_pageouts=(pageouts[-1] - pageouts[0] if len(pageouts) >= 2 else None),
        delta_swapouts=(swapouts[-1] - swapouts[0] if len(swapouts) >= 2 else None),
        requests=requests,
    )


def print_summary(results: list[BenchmarkResult]) -> None:
    print()
    print(
        "parallel | ok/total | wall_s | ttft_p50 | in_tps | dec_tps | agg_in_tps | agg_dec_tps | min_free% | rss_gb"
    )
    print("-" * 116)
    for result in results:
        total = result.completed + result.failed
        ttft = f"{result.ttft_p50_seconds:.1f}" if result.ttft_p50_seconds is not None else "-"
        in_tps = f"{result.input_tps_p50:.1f}" if result.input_tps_p50 is not None else "-"
        dec_tps = f"{result.decode_tps_p50:.1f}" if result.decode_tps_p50 is not None else "-"
        agg_in = f"{result.aggregate_input_tps:.1f}" if result.aggregate_input_tps is not None else "-"
        agg_dec = f"{result.aggregate_decode_tps:.1f}" if result.aggregate_decode_tps is not None else "-"
        rss = f"{result.max_server_rss_gb:.2f}" if result.max_server_rss_gb is not None else "-"
        free = str(result.min_free_percent) if result.min_free_percent is not None else "-"
        print(
            f"{result.parallel:>8} | "
            f"{result.completed:>2}/{total:<5} | "
            f"{result.batch_wall_seconds:>6.1f} | "
            f"{ttft:>8} | "
            f"{in_tps:>6} | "
            f"{dec_tps:>7} | "
            f"{agg_in:>10} | "
            f"{agg_dec:>11} | "
            f"{free:>9} | "
            f"{rss:>6}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LM Studio model parallelism through the proxy")
    parser.add_argument("--lms-bin", default=str(Path.home() / ".lmstudio" / "bin" / "lms"))
    parser.add_argument("--proxy-bin", default=str(Path.cwd() / ".venv" / "bin" / "mlx-openai-proxy"))
    parser.add_argument("--backend-port", type=int, default=8097)
    parser.add_argument("--proxy-port", type=int, default=8090)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--model-key", default="google/gemma-4-e2b")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--parallel-levels", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--request-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    results: list[BenchmarkResult] = []
    output_json = (
        Path(args.output_json)
        if args.output_json
        else Path("data") / "benchmarks" / f"parallelism-{int(time.time())}.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    log_dir = output_json.parent

    for parallel in args.parallel_levels:
        print(f"\n==> Testing parallel={parallel}")
        ensure_model_parallel(
            lms_bin=args.lms_bin,
            model_key=args.model_key,
            alias=args.model,
            context_length=args.context_length,
            parallel=parallel,
        )
        metrics_db_path = log_dir / f"benchmark-metrics-p{parallel}.db"
        proxy_log_path = log_dir / f"benchmark-proxy-p{parallel}.log"
        proxy_process = start_proxy(
            proxy_bin=args.proxy_bin,
            proxy_port=args.proxy_port,
            backend_port=args.backend_port,
            model_parallel=parallel,
            metrics_db_path=metrics_db_path,
            log_path=proxy_log_path,
        )
        try:
            result = run_benchmark_level(
                parallel=parallel,
                model=args.model,
                proxy_url=f"http://127.0.0.1:{args.proxy_port}",
                max_tokens=args.max_tokens,
                request_timeout_seconds=args.request_timeout_seconds,
                backend_port=args.backend_port,
            )
            results.append(result)
            print(
                f"Completed {result.completed}/{result.completed + result.failed} "
                f"requests in {result.batch_wall_seconds:.1f}s; "
                f"max server RSS {result.max_server_rss_gb:.2f} GiB"
                if result.max_server_rss_gb is not None
                else f"Completed {result.completed}/{result.completed + result.failed} requests in {result.batch_wall_seconds:.1f}s"
            )
        finally:
            stop_proxy(proxy_process)

    output_json.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "results": [asdict(result) for result in results],
            },
            indent=2,
        )
    )
    print_summary(results)
    print(f"\nSaved JSON report to {output_json}")


if __name__ == "__main__":
    main()
