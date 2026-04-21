from mlx_openai_proxy.dashboard import dashboard_html


def test_dashboard_rolling_average_uses_invocation_timestamps() -> None:
    html = dashboard_html()

    assert "const ROLLING_STEP_SECONDS" not in html
    assert "for (let sample = firstSample;" not in html
    assert "entry.points.forEach((point, index) => {" in html
    assert "points.push({ x: point.x, y: sum / windowCount, model: entry.model });" in html
    assert 'id="loaded-model"' in html
    assert "summary.loaded_model || 'unknown'" in html
