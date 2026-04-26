from mlx_openai_proxy.service import GemmaThoughtStreamParser


def _parse(*chunks: str) -> tuple[str, str]:
    parser = GemmaThoughtStreamParser()
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    for chunk in chunks:
        reasoning, content = parser.feed(chunk)
        reasoning_parts.extend(reasoning)
        content_parts.extend(content)
    final_reasoning, final_content = parser.finish()
    reasoning_parts.extend(final_reasoning)
    content_parts.extend(final_content)
    return "".join(reasoning_parts), "".join(content_parts)


def test_parser_handles_exact_gemma_thought_tags() -> None:
    reasoning, content = _parse("<|channel>thoughthello<channel|>world")
    assert reasoning == "hello"
    assert content == "world"


def test_parser_handles_leading_newline_before_thought_tag() -> None:
    reasoning, content = _parse("\n<|channel>thoughthello<channel|>world")
    assert reasoning == "\nhello"
    assert content == "world"


def test_parser_handles_split_tag_with_leading_whitespace() -> None:
    reasoning, content = _parse(
        "  ", "<|channel>thought", "hello", "<channel|>", "world"
    )
    assert reasoning == "  hello"
    assert content == "world"


def test_parser_preserves_plain_content_when_no_tag_exists() -> None:
    reasoning, content = _parse("  hello world")
    assert reasoning == ""
    assert content == "  hello world"
