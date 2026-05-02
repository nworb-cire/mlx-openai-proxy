from mlx_openai_proxy.dashboard import dashboard_html


def test_dashboard_rolling_average_uses_invocation_timestamps() -> None:
    html = dashboard_html()

    assert "const ROLLING_STEP_SECONDS" not in html
    assert "for (let sample = firstSample;" not in html
    assert "entry.points.forEach((point, index) => {" in html
    assert (
        "points.push({ x: point.x, y: sum / windowCount, model: entry.model });" in html
    )
    assert 'id="loaded-model"' in html
    assert "summary.loaded_model || 'unknown'" in html


def test_dashboard_formats_input_by_endpoint_type() -> None:
    html = dashboard_html()

    assert "function fmtInput(item)" in html
    assert "path === '/v1/audio/transcriptions'" in html
    assert "path === '/v1/responses'" in html
    assert "path === '/v1/chat/completions'" in html
    assert "parts.push(`${fmtNumber(item.input_audio_seconds, 1)}s audio`);" in html
    assert "parts.push(fmtCount(messages, 'input', 'inputs'));" in html
    assert "parts.push(fmtCount(messages, 'msg', 'msgs'));" in html
    assert "`${item.input_chars || 0} chars / ${item.input_messages || 0} msgs`" not in html
