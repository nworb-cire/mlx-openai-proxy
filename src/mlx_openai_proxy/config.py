from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from shutil import which

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ReasoningVisibility(StrEnum):
    OFF = "off"
    COMPATIBLE = "compatible"
    FULL = "full"


class StructuredMode(StrEnum):
    AUTO = "auto"
    STRICT_FAST_PATH = "strict_fast_path"
    REASON_THEN_STRUCTURE = "reason_then_structure"


def _default_metrics_db_path() -> str:
    return str(Path.home() / ".local" / "share" / "mlx-openai-proxy" / "metrics.db")


def _default_lm_studio_bin() -> str:
    return which("lms") or str(Path.home() / ".lmstudio" / "bin" / "lms")


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
    max_upstream_concurrency: int = Field(default=2)
    lm_studio_bin: str = Field(default_factory=_default_lm_studio_bin)

    default_model_alias: str = Field(default="gemma4:e2b")
    default_model_key: str = Field(default="google/gemma-4-e2b")
    default_model_context_length: int = Field(default=8192)
    default_model_parallel: int = Field(default=2)

    burst_model_alias: str = Field(default="gemma4:26b")
    burst_model_key: str = Field(default="google/gemma-4-26b-a4b")
    burst_model_context_length: int = Field(default=8192)
    burst_model_parallel: int = Field(default=1)

    reasoning_visibility: ReasoningVisibility = Field(default=ReasoningVisibility.COMPATIBLE)
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
