from __future__ import annotations

import json

from mlx_openai_proxy.config import Settings


def test_settings_load_models_from_config_file(tmp_path) -> None:
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            [
                {
                    "alias": "small",
                    "key": "local/small",
                    "context_length": 4096,
                    "parallel": 3,
                },
                {
                    "alias": "large",
                    "key": "local/large",
                    "context_length": 8192,
                    "parallel": 1,
                },
            ]
        )
    )

    settings = Settings(model_config_path=str(config_path))

    assert [model.alias for model in settings.models] == ["small", "large"]
    assert settings.default_model_alias == "small"
    assert settings.models[0].parallel == 3
