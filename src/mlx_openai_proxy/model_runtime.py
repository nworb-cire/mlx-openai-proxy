from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from .config import Settings


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    key: str
    context_length: int
    parallel: int


@dataclass(frozen=True)
class LoadedModel:
    alias: str
    context_length: int | None
    parallel: int | None


def configured_model_specs(settings: Settings) -> dict[str, ModelSpec]:
    return {
        model.alias: ModelSpec(
            alias=model.alias,
            key=model.key,
            context_length=model.context_length,
            parallel=model.parallel,
        )
        for model in settings.models
    }


class ModelRuntimeError(RuntimeError):
    pass


class ModelRuntimeManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._specs = configured_model_specs(settings)
        self._aliases_by_input = {}
        for alias, spec in self._specs.items():
            self._aliases_by_input[alias.lower()] = alias
            self._aliases_by_input[spec.key.lower()] = alias
        self._active_alias = settings.default_model_alias
        self._lock = asyncio.Lock()

    def normalize_alias(self, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("model must be a string")
        alias = self._aliases_by_input.get(value.lower())
        if alias is None:
            supported = ", ".join(sorted(self._specs))
            raise ValueError(
                f"unsupported model '{value}'; supported models: {supported}"
            )
        return alias

    def advertised_models(self) -> dict[str, Any]:
        data = [
            {"id": alias, "object": "model", "owned_by": "organization_owner"}
            for alias in self._specs
        ]
        return {"object": "list", "data": data}

    def concurrency_for(self, alias: str) -> int:
        return max(1, self._specs[alias].parallel)

    def current_active_alias(self) -> str | None:
        return self._active_alias

    async def current_loaded_aliases(self) -> list[str]:
        return await self.list_loaded_aliases()

    async def current_loaded_alias(self) -> str | None:
        loaded_aliases = await self.list_loaded_aliases()
        if loaded_aliases:
            return loaded_aliases[0]
        return None

    async def normalize_residency(self, preferred_alias: str | None = None) -> str:
        loaded_aliases = await self.list_loaded_aliases()
        known_loaded = [alias for alias in loaded_aliases if alias in self._specs]

        if preferred_alias is not None:
            target_alias = self.normalize_alias(preferred_alias)
        elif known_loaded:
            target_alias = known_loaded[0]
        else:
            target_alias = self.settings.default_model_alias

        await self.switch_to(target_alias)
        return target_alias

    async def switch_to(self, alias: str) -> None:
        alias = self.normalize_alias(alias)
        async with self._lock:
            await self._switch_to_locked(alias)

    async def _switch_to_locked(self, alias: str) -> None:
        loaded_models = await self.list_loaded_models()
        loaded_aliases = [model.alias for model in loaded_models]
        spec = self._specs[alias]
        loaded_target = next(
            (model for model in loaded_models if model.alias == alias), None
        )
        if loaded_target is not None and self._loaded_model_matches_spec(
            loaded_target, spec
        ):
            for loaded_model in loaded_models:
                if loaded_model.alias != alias:
                    await self._run_lms("unload", loaded_model.alias)
            self._active_alias = alias
            return

        previous_alias = next(
            (item for item in loaded_aliases if item in self._specs), None
        )
        for loaded_alias in loaded_aliases:
            await self._run_lms("unload", loaded_alias)

        try:
            await self._run_lms(
                "load",
                spec.key,
                "--identifier",
                spec.alias,
                "-c",
                str(spec.context_length),
                "--parallel",
                str(spec.parallel),
                "-y",
            )
        except Exception as exc:
            loaded_after_failure: list[str] = []
            try:
                loaded_after_failure = await self.list_loaded_aliases()
            except Exception:
                loaded_after_failure = []
            if alias in loaded_after_failure:
                self._active_alias = alias
                return
            if previous_alias is not None and previous_alias != alias:
                if previous_alias in loaded_after_failure:
                    self._active_alias = previous_alias
                    raise ModelRuntimeError(
                        f"failed to load model '{alias}': {exc}"
                    ) from exc
                previous_spec = self._specs[previous_alias]
                try:
                    await self._run_lms(
                        "load",
                        previous_spec.key,
                        "--identifier",
                        previous_spec.alias,
                        "-c",
                        str(previous_spec.context_length),
                        "--parallel",
                        str(previous_spec.parallel),
                        "-y",
                    )
                    self._active_alias = previous_alias
                except Exception as rollback_exc:
                    raise ModelRuntimeError(
                        f"failed to load model '{alias}': {exc}; "
                        f"also failed to restore '{previous_alias}': {rollback_exc}"
                    ) from exc
            raise ModelRuntimeError(f"failed to load model '{alias}': {exc}") from exc
        self._active_alias = alias

    @staticmethod
    def _loaded_model_matches_spec(model: LoadedModel, spec: ModelSpec) -> bool:
        if model.context_length is not None and model.context_length != spec.context_length:
            return False
        if model.parallel is not None and model.parallel != spec.parallel:
            return False
        return True

    async def list_loaded_aliases(self) -> list[str]:
        return [model.alias for model in await self.list_loaded_models()]

    async def list_loaded_models(self) -> list[LoadedModel]:
        output = await self._run_lms("ps", "--json")
        try:
            items = json.loads(output or "[]")
        except json.JSONDecodeError as exc:
            raise ModelRuntimeError(
                f"failed to parse 'lms ps --json': {output}"
            ) from exc
        loaded: list[LoadedModel] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            identifier = item.get("identifier")
            if isinstance(identifier, str):
                loaded.append(
                    LoadedModel(
                        alias=identifier,
                        context_length=self._optional_int(item.get("contextLength")),
                        parallel=self._optional_int(item.get("parallel")),
                    )
                )
        if loaded:
            self._active_alias = loaded[0].alias
        return loaded

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None

    async def _run_lms(self, *args: str) -> str:
        process = await asyncio.create_subprocess_exec(
            self.settings.lm_studio_bin,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            detail = (stderr or stdout).decode("utf-8", errors="replace").strip()
            raise ModelRuntimeError(detail or f"lms command failed: {' '.join(args)}")
        return stdout.decode("utf-8", errors="replace")
