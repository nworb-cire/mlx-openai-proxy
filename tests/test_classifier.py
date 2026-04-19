from mlx_openai_proxy.classifier import ExecutionPath, classify_chat_request
from mlx_openai_proxy.config import StructuredMode


def test_passthrough_without_schema() -> None:
    body = {
        "model": "gemma4:26b",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    result = classify_chat_request(body, StructuredMode.AUTO)
    assert result.execution_path == ExecutionPath.PASSTHROUGH


def test_schema_defaults_to_two_phase() -> None:
    body = {
        "model": "gemma4:26b",
        "messages": [{"role": "user", "content": "What is 17 + 24? Return JSON."}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "sum", "schema": {"type": "object"}},
        },
    }
    result = classify_chat_request(body, StructuredMode.AUTO)
    assert result.execution_path == ExecutionPath.REASON_THEN_STRUCTURE
