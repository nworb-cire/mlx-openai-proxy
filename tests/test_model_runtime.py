from __future__ import annotations

import pytest

from mlx_openai_proxy.config import ConfiguredModel, Settings
from mlx_openai_proxy.model_runtime import ModelRuntimeManager


class FakeRuntime(ModelRuntimeManager):
    def __init__(self, settings: Settings, loaded_aliases: list[str]) -> None:
        super().__init__(settings)
        self.loaded_aliases = list(loaded_aliases)
        self.commands: list[tuple[str, ...]] = []

    async def _run_lms(self, *args: str) -> str:
        self.commands.append(args)
        if args[:2] == ("ps", "--json"):
            items = [{"identifier": alias} for alias in self.loaded_aliases]
            import json

            return json.dumps(items)
        if args and args[0] == "unload":
            self.loaded_aliases = [alias for alias in self.loaded_aliases if alias != args[1]]
            return ""
        if args and args[0] == "load":
            alias = args[args.index("--identifier") + 1]
            self.loaded_aliases = [alias]
            return ""
        raise AssertionError(f"unexpected lms command: {args}")


@pytest.mark.asyncio
async def test_switch_to_unloads_extra_loaded_models_when_target_already_loaded() -> None:
    settings = Settings()
    runtime = FakeRuntime(settings, loaded_aliases=["gemma4:26b", "gemma4:e2b"])

    await runtime.switch_to("gemma4:26b")

    assert runtime.loaded_aliases == ["gemma4:26b"]
    assert ("unload", "gemma4:e2b") in runtime.commands


@pytest.mark.asyncio
async def test_normalize_residency_keeps_existing_single_loaded_model() -> None:
    settings = Settings()
    runtime = FakeRuntime(settings, loaded_aliases=["gemma4:26b"])

    chosen = await runtime.normalize_residency()

    assert chosen == "gemma4:26b"
    assert runtime.loaded_aliases == ["gemma4:26b"]


@pytest.mark.asyncio
async def test_normalize_residency_loads_default_when_nothing_loaded() -> None:
    settings = Settings()
    runtime = FakeRuntime(settings, loaded_aliases=[])

    chosen = await runtime.normalize_residency()

    assert chosen == settings.default_model_alias
    assert runtime.loaded_aliases == [settings.default_model_alias]


def test_default_model_list_is_advertised() -> None:
    runtime = FakeRuntime(Settings(), loaded_aliases=[])

    model_ids = [item["id"] for item in runtime.advertised_models()["data"]]

    assert model_ids == ["gemma4:e2b", "gemma4:26b"]


def test_model_list_can_be_overridden() -> None:
    settings = Settings(
        models=[
            ConfiguredModel(alias="small", key="local/small", parallel=2),
            ConfiguredModel(alias="large", key="local/large"),
        ]
    )
    runtime = FakeRuntime(settings, loaded_aliases=[])

    model_ids = [item["id"] for item in runtime.advertised_models()["data"]]

    assert model_ids == ["small", "large"]
    assert runtime.normalize_alias("local/large") == "large"


@pytest.mark.asyncio
async def test_switch_to_configured_model_loads_its_spec() -> None:
    settings = Settings(
        models=[
            ConfiguredModel(alias="small", key="local/small", parallel=2),
            ConfiguredModel(alias="large", key="local/large", context_length=4096),
        ]
    )
    runtime = FakeRuntime(settings, loaded_aliases=["small"])

    await runtime.switch_to("large")

    assert runtime.loaded_aliases == ["large"]
    assert (
        "load",
        "local/large",
        "--identifier",
        "large",
        "-c",
        "4096",
        "--parallel",
        "1",
        "-y",
    ) in runtime.commands
