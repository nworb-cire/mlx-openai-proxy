import pytest

from mlx_openai_proxy.schema_utils import normalize_json_text, validate_json_text


def test_normalize_json_text_strips_fences() -> None:
    assert normalize_json_text("```json\n{\"a\": 1}\n```") == "{\"a\":1}"


def test_validate_json_text_checks_schema() -> None:
    schema = {
        "type": "object",
        "properties": {"sum": {"type": "integer"}},
        "required": ["sum"],
        "additionalProperties": False,
    }
    parsed, normalized = validate_json_text("{\"sum\": 41}", schema)
    assert parsed == {"sum": 41}
    assert normalized == "{\"sum\":41}"

    with pytest.raises(Exception):
        validate_json_text("{\"answer\": 41}", schema)
