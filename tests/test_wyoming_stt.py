from __future__ import annotations

from mlx_openai_proxy.asr import ResidentAsrService
from mlx_openai_proxy.config import ConfiguredAsr
from mlx_openai_proxy.metrics_store import MetricsStore
from mlx_openai_proxy.wyoming_stt import PARAKEET_V3_LANGUAGES, build_info

from test_asr import FakeAsrRuntime


def test_build_info_advertises_parakeet_v3(tmp_path) -> None:
    asr = ResidentAsrService(
        ConfiguredAsr(),
        FakeAsrRuntime(),
        MetricsStore(str(tmp_path / "metrics.db")),
    )
    info = build_info(asr)
    model = info.asr[0].models[0]
    assert info.asr[0].name == "parakeet-mlx"
    assert model.name == "parakeet:tdt-0.6b-v3"
    assert set(model.languages) == set(PARAKEET_V3_LANGUAGES)
    assert "en" in model.languages
