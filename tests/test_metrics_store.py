from pathlib import Path

from mlx_openai_proxy.metrics_store import MetricsStore


def test_metrics_store_tracks_active_and_history(tmp_path: Path) -> None:
    store = MetricsStore(str(tmp_path / "metrics.db"))
    request_id = store.start_request(
        {
            "path": "/v1/chat/completions",
            "model": "gemma4:26b",
            "stream": False,
            "execution_path": "passthrough",
            "classification_reason": "no_schema",
            "has_schema": False,
            "has_images": False,
            "asks_for_reasoning": False,
            "input_messages": 1,
            "input_chars": 42,
            "input_image_count": 0,
        }
    )
    assert len(store.get_active_requests()) == 1

    store.complete_request(
        request_id,
        prompt_tokens=10,
        completion_tokens=20,
        reasoning_tokens=5,
        output_chars=30,
        reasoning_chars=12,
        service_duration_ms=250,
    )
    assert len(store.get_active_requests()) == 0
    history = store.get_history(limit=10)
    assert history[0]["prompt_tokens"] == 10
    assert history[0]["service_duration_ms"] == 250
    assert "path" not in history[0]
    assert history[0]["model"] == "gemma4:26b"


def test_metrics_store_derives_queue_and_service_from_service_start(tmp_path: Path) -> None:
    store = MetricsStore(str(tmp_path / "metrics.db"))
    request_id = store.start_request(
        {
            "path": "/v1/chat/completions",
            "model": "gemma4:26b",
            "stream": False,
            "execution_path": "passthrough",
            "classification_reason": "no_schema",
            "has_schema": False,
            "has_images": False,
            "asks_for_reasoning": False,
            "input_messages": 1,
            "input_chars": 10,
            "input_image_count": 0,
            "started_at": 100.0,
            "service_started_at": 103.0,
        }
    )

    original_time = __import__("time").time
    try:
        __import__("time").time = lambda: 107.5
        store.complete_request(request_id, prompt_tokens=1, completion_tokens=2)
    finally:
        __import__("time").time = original_time

    history = store.get_history(limit=1)
    assert history[0]["queue_duration_ms"] == 3000
    assert history[0]["service_duration_ms"] == 4500
    summary = store.get_summary()
    assert summary["avg_queue_duration_ms"] == 3000
    assert summary["avg_service_duration_ms"] == 4500


def test_metrics_store_hides_legacy_timing_split(tmp_path: Path) -> None:
    store = MetricsStore(str(tmp_path / "metrics.db"))
    request_id = store.start_request(
        {
            "path": "/v1/chat/completions",
            "model": "gemma4:26b",
            "stream": False,
            "execution_path": "passthrough",
            "classification_reason": "no_schema",
            "has_schema": False,
            "has_images": False,
            "asks_for_reasoning": False,
            "input_messages": 1,
            "input_chars": 10,
            "input_image_count": 0,
            "started_at": 200.0,
        }
    )

    original_time = __import__("time").time
    try:
        __import__("time").time = lambda: 205.0
        store.complete_request(
            request_id,
            prompt_tokens=1,
            completion_tokens=2,
            service_duration_ms=0,
            queue_duration_ms=5000,
        )
    finally:
        __import__("time").time = original_time

    history = store.get_history(limit=1)
    assert history[0]["service_duration_ms"] is None
    assert history[0]["queue_duration_ms"] is None


def test_metrics_store_reports_live_progress_estimates(tmp_path: Path) -> None:
    store = MetricsStore(str(tmp_path / "metrics.db"))
    request_id = store.start_request(
        {
            "path": "/v1/chat/completions",
            "model": "gemma4:26b",
            "stream": True,
            "execution_path": "passthrough",
            "classification_reason": "no_schema",
            "has_schema": False,
            "has_images": False,
            "asks_for_reasoning": True,
            "input_messages": 1,
            "input_chars": 12,
            "input_image_count": 0,
        }
    )
    store.append_progress(request_id, reasoning_delta="abcdef", output_delta="wxyz")

    active = store.get_active_requests()
    assert active[0]["live_reasoning_chars"] == 6
    assert active[0]["live_output_chars"] == 4
    assert active[0]["live_reasoning_tokens_est"] == 2
    assert active[0]["live_output_tokens_est"] == 1
