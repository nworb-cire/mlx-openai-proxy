from __future__ import annotations

import asyncio
import logging
from functools import partial

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

from .asr import ResidentAsrService

_LOGGER = logging.getLogger(__name__)

PARAKEET_V3_LANGUAGES = [
    "bg",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fi",
    "fr",
    "de",
    "el",
    "hu",
    "it",
    "lv",
    "lt",
    "mt",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "es",
    "sv",
    "ru",
    "uk",
]


def build_info(asr: ResidentAsrService) -> Info:
    model_name = asr.config.alias
    attribution = Attribution(
        name="NVIDIA",
        url="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
    )
    return Info(
        asr=[
            AsrProgram(
                name="parakeet-mlx",
                description="Local Parakeet speech recognition on Apple MLX",
                attribution=attribution,
                installed=True,
                version="0.5.1",
                models=[
                    AsrModel(
                        name=model_name,
                        description="Parakeet TDT 0.6B v3",
                        attribution=attribution,
                        installed=True,
                        languages=PARAKEET_V3_LANGUAGES,
                        version="v3",
                    )
                ],
                supports_transcript_streaming=False,
                requires_external_vad=True,
            )
        ]
    )


class ParakeetEventHandler(AsyncEventHandler):
    def __init__(
        self,
        info: Info,
        asr: ResidentAsrService,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._info_event = info.event()
        self._asr = asr
        self._converter = AudioChunkConverter(rate=16000, width=2, channels=1)
        self._audio = bytearray()
        self._language: str | None = None

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info_event)
            return True

        if Transcribe.is_type(event.type):
            request = Transcribe.from_event(event)
            self._language = request.language
            return True

        if AudioChunk.is_type(event.type):
            chunk = self._converter.convert(AudioChunk.from_event(event))
            self._audio.extend(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            result = await self._asr.transcribe_pcm(bytes(self._audio), 16000)
            await self.write_event(
                Transcript(text=result.text, language=self._language).event()
            )
            return False

        return True


class WyomingSttServer:
    def __init__(self, uri: str, asr: ResidentAsrService) -> None:
        self.uri = uri
        self.asr = asr
        self.info = build_info(asr)
        self.server = AsyncServer.from_uri(uri)

    async def start(self) -> None:
        await self.server.start(partial(ParakeetEventHandler, self.info, self.asr))
        _LOGGER.info("Wyoming Parakeet STT listening on %s", self.uri)

    async def close(self) -> None:
        await self.server.stop()
        await asyncio.sleep(0)
