from mlx_openai_proxy.classifier import ExecutionPath, classify_chat_request
from mlx_openai_proxy.config import StructuredMode


def test_passthrough_without_schema() -> None:
    body = {
        "model": "gemma4:26b",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    result = classify_chat_request(body, StructuredMode.AUTO)
    assert result.execution_path == ExecutionPath.PASSTHROUGH


def test_schema_defaults_to_fast_path_without_reasoning() -> None:
    body = {
        "model": "gemma4:26b",
        "messages": [{"role": "user", "content": "What is 17 + 24? Return JSON."}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "sum", "schema": {"type": "object"}},
        },
    }
    result = classify_chat_request(body, StructuredMode.AUTO)
    assert result.execution_path == ExecutionPath.STRICT_STRUCTURED_FAST_PATH
    assert result.asks_for_reasoning is False
    assert result.reason == "no_reasoning_requested"


def test_schema_uses_two_phase_when_reasoning_is_requested() -> None:
    body = {
        "model": "gemma4:26b",
        "messages": [{"role": "user", "content": "What is 17 + 24? Return JSON."}],
        "reasoning_effort": "low",
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "sum", "schema": {"type": "object"}},
        },
    }
    result = classify_chat_request(body, StructuredMode.AUTO)
    assert result.execution_path == ExecutionPath.REASON_THEN_STRUCTURE
    assert result.asks_for_reasoning is True
    assert result.reason == "reasoning_requested"


def test_schema_reasoning_none_uses_fast_path() -> None:
    body = {
        "model": "gemma4:26b",
        "messages": [{"role": "user", "content": "Think this through. Return JSON."}],
        "reasoning_effort": "none",
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "sum", "schema": {"type": "object"}},
        },
    }
    result = classify_chat_request(body, StructuredMode.AUTO)
    assert result.execution_path == ExecutionPath.STRICT_STRUCTURED_FAST_PATH
    assert result.asks_for_reasoning is False
