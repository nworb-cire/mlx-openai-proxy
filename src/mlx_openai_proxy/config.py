from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from shutil import which

from pydantic import BaseModel, Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ReasoningVisibility(StrEnum):
    OFF = "off"
    COMPATIBLE = "compatible"
    FULL = "full"


class StructuredMode(StrEnum):
    AUTO = "auto"
    STRICT_FAST_PATH = "strict_fast_path"
    REASON_THEN_STRUCTURE = "reason_then_structure"


class ConfiguredModel(BaseModel):
    alias: str
    key: str
    context_length: int = Field(default=8192)
    parallel: int = Field(default=1)


def _default_metrics_db_path() -> str:
    return str(Path.home() / ".local" / "share" / "mlx-openai-proxy" / "metrics.db")


def _default_lm_studio_bin() -> str:
    return which("lms") or str(Path.home() / ".lmstudio" / "bin" / "lms")


def _default_model_config_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "config" / "models.json")


def _default_models() -> list[ConfiguredModel]:
    return _load_models_from_path(Path(_default_model_config_path()))


def _load_models_from_path(path: Path) -> list[ConfiguredModel]:
    if not path.exists():
        return [
            ConfiguredModel(
                alias="gemma4:e2b",
                key="google/gemma-4-e2b",
                context_length=8192,
                parallel=2,
            ),
            ConfiguredModel(
                alias="gemma4:26b",
                key="google/gemma-4-26b-a4b",
                context_length=8192,
                parallel=1,
            ),
        ]
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"model config must be a JSON list: {path}")
    return [ConfiguredModel.model_validate(item) for item in data]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MLX_PROXY_",
        env_file=".env",
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8090)
    metrics_db_path: str = Field(default_factory=_default_metrics_db_path)

    backend_base_url: str = Field(default="http://127.0.0.1:8080/v1")
    backend_timeout_seconds: float = Field(default=600.0)
    active_request_timeout_seconds: float = Field(default=600.0)
    max_upstream_concurrency: int = Field(default=2)
    lm_studio_bin: str = Field(default_factory=_default_lm_studio_bin)

    model_config_path: str = Field(default_factory=_default_model_config_path)
    models: list[ConfiguredModel] = Field(default_factory=_default_models)

    reasoning_visibility: ReasoningVisibility = Field(
        default=ReasoningVisibility.COMPATIBLE
    )
    schema_mode: StructuredMode = Field(default=StructuredMode.AUTO)
    phase2_max_tokens: int = Field(default=1024)
    phase2_temperature: float = Field(default=0.0)
    phase2_top_p: float = Field(default=1.0)
    phase2_top_k: int = Field(default=1)
    max_repair_attempts: int = Field(default=1)
    final_chunk_size: int = Field(default=24)

    enable_image_file_urls: bool = Field(default=True)
    max_inline_image_bytes: int = Field(default=8 * 1024 * 1024)

    log_level: str = Field(default="INFO")

    @field_validator("models")
    @classmethod
    def validate_models(cls, models: list[ConfiguredModel]) -> list[ConfiguredModel]:
        if not models:
            raise ValueError("at least one model must be configured")
        aliases = [model.alias.lower() for model in models]
        if len(aliases) != len(set(aliases)):
            raise ValueError("model aliases must be unique")
        keys = [model.key.lower() for model in models]
        if len(keys) != len(set(keys)):
            raise ValueError("model keys must be unique")
        return models

    def model_post_init(self, __context: object) -> None:
        field_set = self.model_fields_set
        if "models" not in field_set and "model_config_path" in field_set:
            self.models = self.validate_models(
                _load_models_from_path(Path(self.model_config_path))
            )

    @computed_field
    @property
    def default_model_alias(self) -> str:
        return self.models[0].alias
