from __future__ import annotations

import asyncio
import audioop
import base64
import io
import subprocess
import tempfile
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Any, Protocol

from fastapi import UploadFile, WebSocket, WebSocketDisconnect

from .config import ConfiguredAsr
from .metrics_store import MetricsStore


class AsrError(RuntimeError):
    pass


@dataclass(frozen=True)
class Transcript:
    text: str


class StreamingTranscriber(Protocol):
    def add_pcm(self, pcm: bytes, sample_rate: int) -> str:
        ...

    def close(self) -> None:
        ...


class AsrRuntime(Protocol):
    @property
    def alias(self) -> str:
        ...

    async def load(self) -> None:
        ...

    async def close(self) -> None:
        ...

    async def transcribe_pcm(self, pcm: bytes, sample_rate: int) -> Transcript:
        ...

    def create_stream(self) -> StreamingTranscriber:
        ...


class ParakeetAsrRuntime:
    def __init__(self, config: ConfiguredAsr) -> None:
        self.config = config
        self._model: Any | None = None
        self._sample_rate: int | None = None

    @property
    def alias(self) -> str:
        return self.config.alias

    async def load(self) -> None:
        if self._model is not None:
            return
        try:
            from parakeet_mlx import from_pretrained
        except ImportError as exc:
            raise AsrError(
                "parakeet-mlx is required for ASR; install project dependencies"
            ) from exc

        self._model = await asyncio.to_thread(from_pretrained, self.config.model_id)
        self._sample_rate = int(
            getattr(
                getattr(self._model, "preprocessor_config", None),
                "sample_rate",
                self.config.input_sample_rate,
            )
        )

    async def close(self) -> None:
        self._model = None

    async def transcribe_pcm(self, pcm: bytes, sample_rate: int) -> Transcript:
        await self.load()
        assert self._model is not None
        target_rate = self._target_sample_rate()
        audio = resample_pcm16(pcm, sample_rate, target_rate)
        result = await asyncio.to_thread(
            self._transcribe_wav_bytes,
            encode_wav_bytes(audio, target_rate),
        )
        return Transcript(text=str(getattr(result, "text", "")).strip())

    def _transcribe_wav_bytes(self, content: bytes) -> Any:
        assert self._model is not None
        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            handle.write(content)
            handle.flush()
            return self._model.transcribe(handle.name)

    def create_stream(self) -> StreamingTranscriber:
        if self._model is None:
            raise AsrError("ASR model is not loaded")
        return ParakeetStreamingTranscriber(
            model=self._model,
            sample_rate=self._target_sample_rate(),
        )

    def _target_sample_rate(self) -> int:
        return int(self._sample_rate or self.config.input_sample_rate)


class ParakeetStreamingTranscriber:
    def __init__(self, model: Any, sample_rate: int) -> None:
        self._model = model
        self._sample_rate = sample_rate
        self._manager = model.transcribe_stream(context_size=(256, 256))
        self._transcriber = self._manager.__enter__()

    def add_pcm(self, pcm: bytes, sample_rate: int) -> str:
        audio = pcm16_to_float32(resample_pcm16(pcm, sample_rate, self._sample_rate))
        self._transcriber.add_audio(audio)
        result = getattr(self._transcriber, "result", None)
        return str(getattr(result, "text", "") or "").strip()

    def close(self) -> None:
        self._manager.__exit__(None, None, None)


class ResidentAsrService:
    def __init__(
        self,
        config: ConfiguredAsr,
        runtime: AsrRuntime,
        metrics: MetricsStore,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.metrics = metrics
        self._semaphore = asyncio.Semaphore(max(1, config.max_concurrency))

    async def load(self) -> None:
        await self.runtime.load()

    async def close(self) -> None:
        await self.runtime.close()

    def advertised_model(self) -> dict[str, str]:
        return {
            "id": self.config.alias,
            "object": "model",
            "owned_by": "organization_owner",
        }

    def health(self) -> dict[str, object]:
        return {
            "loaded": True,
            "model": self.config.alias,
            "max_concurrency": self.config.max_concurrency,
        }

    async def transcribe_upload(
        self,
        *,
        upload: UploadFile,
        model: str | None,
        response_format: str | None,
    ) -> dict[str, Any] | str:
        if model and model != self.config.alias and model != self.config.model_id:
            raise ValueError(f"unsupported ASR model '{model}'")
        fmt = response_format or "json"
        if fmt not in {"json", "text"}:
            raise ValueError("response_format must be one of: json, text")

        content = await upload.read()
        pcm, sample_rate = decode_audio_bytes(content, upload.filename or "")
        request_id = self.metrics.start_request(
            {
                "path": "/v1/audio/transcriptions",
                "model": self.config.alias,
                "stream": False,
                "execution_path": "asr",
                "classification_reason": "audio_transcription",
                "has_schema": False,
                "has_images": False,
                "asks_for_reasoning": False,
                "input_messages": 0,
                "input_chars": 0,
                "input_image_count": 0,
                "input_audio_seconds": len(pcm) / 2 / sample_rate if sample_rate else 0,
            }
        )
        try:
            async with self._semaphore:
                self.metrics.update_request(request_id, service_started_at=time.time())
                transcript = await self.runtime.transcribe_pcm(pcm, sample_rate)
            self.metrics.complete_request(request_id, output_chars=len(transcript.text))
        except Exception as exc:
            self.metrics.fail_request(request_id, error_message=str(exc))
            raise

        if fmt == "text":
            return transcript.text
        return {"text": transcript.text}

    async def realtime(self, websocket: WebSocket) -> None:
        await websocket.accept()
        session = RealtimeSession(self.config, self.runtime, self._semaphore, websocket)
        await session.run()


class RealtimeSession:
    def __init__(
        self,
        config: ConfiguredAsr,
        runtime: AsrRuntime,
        semaphore: asyncio.Semaphore,
        websocket: WebSocket,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.semaphore = semaphore
        self.websocket = websocket
        self.input_sample_rate = config.input_sample_rate
        self.server_vad = config.vad.enabled
        self.buffer = bytearray()
        self.prefix = deque[bytes]()
        self.prefix_bytes = int(
            self.input_sample_rate
            * 2
            * max(0, config.vad.prefix_padding_ms)
            / 1000
        )
        self.speech_started = False
        self.silence_ms = 0
        self.last_text = ""
        self.stream: StreamingTranscriber | None = None

    async def run(self) -> None:
        if self.semaphore.locked():
            await self.websocket.send_json(
                {"type": "error", "error": {"message": "ASR realtime session busy"}}
            )
            await self.websocket.close(code=1013)
            return

        async with self.semaphore:
            self.stream = self.runtime.create_stream()
            await self.websocket.send_json(
                {
                    "type": "session.created",
                    "session": {
                        "model": self.config.alias,
                        "input_audio_format": "pcm16",
                        "turn_detection": self._turn_detection_payload(),
                    },
                }
            )
            try:
                while True:
                    event = await self.websocket.receive_json()
                    await self._handle_event(event)
            except WebSocketDisconnect:
                return
            except Exception as exc:
                await self.websocket.send_json(
                    {"type": "error", "error": {"message": str(exc)}}
                )
                await self.websocket.close(code=1011)
            finally:
                if self.stream is not None:
                    self.stream.close()

    async def _handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "session.update":
            session = event.get("session") or {}
            turn_detection = session.get("turn_detection", "missing")
            if turn_detection is None:
                self.server_vad = False
            elif isinstance(turn_detection, dict):
                detection_type = turn_detection.get("type")
                if detection_type not in {None, "server_vad"}:
                    raise ValueError("only server_vad turn detection is supported")
                self.server_vad = True
            await self.websocket.send_json(
                {
                    "type": "session.updated",
                    "session": {
                        "model": self.config.alias,
                        "input_audio_format": "pcm16",
                        "turn_detection": self._turn_detection_payload(),
                    },
                }
            )
            return

        if event_type == "input_audio_buffer.append":
            audio = event.get("audio")
            if not isinstance(audio, str):
                raise ValueError("input_audio_buffer.append requires base64 audio")
            pcm = base64.b64decode(audio)
            await self._append_pcm(pcm)
            return

        if event_type == "input_audio_buffer.commit":
            await self._commit()
            return

        if event_type == "input_audio_buffer.clear":
            self._clear()
            await self.websocket.send_json({"type": "input_audio_buffer.cleared"})
            return

        await self.websocket.send_json(
            {"type": "error", "error": {"message": f"unsupported event: {event_type}"}}
        )

    async def _append_pcm(self, pcm: bytes) -> None:
        if len(pcm) % 2:
            pcm = pcm[:-1]
        self._remember_prefix(pcm)

        if self.server_vad:
            rms = audioop.rms(pcm, 2) if pcm else 0
            duration_ms = int((len(pcm) / 2 / self.input_sample_rate) * 1000)
            if rms >= self.config.vad.threshold:
                if not self.speech_started:
                    self.speech_started = True
                    self.buffer.extend(b"".join(self.prefix))
                    await self.websocket.send_json(
                        {"type": "input_audio_buffer.speech_started"}
                    )
                self.silence_ms = 0
            elif self.speech_started:
                self.silence_ms += duration_ms

            if self.speech_started:
                self.buffer.extend(pcm)
                await self._stream_delta(pcm)
                if self.silence_ms >= self.config.vad.silence_duration_ms:
                    await self.websocket.send_json(
                        {"type": "input_audio_buffer.speech_stopped"}
                    )
                    await self._commit()
            return

        self.buffer.extend(pcm)
        await self._stream_delta(pcm)

    async def _stream_delta(self, pcm: bytes) -> None:
        if self.stream is None:
            return
        text = self.stream.add_pcm(pcm, self.input_sample_rate)
        if text and text != self.last_text:
            delta = text[len(self.last_text) :] if text.startswith(self.last_text) else text
            self.last_text = text
            await self.websocket.send_json(
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "delta": delta,
                }
            )

    async def _commit(self) -> None:
        pcm = bytes(self.buffer)
        self._clear()
        await self.websocket.send_json({"type": "input_audio_buffer.committed"})
        if not pcm:
            await self.websocket.send_json(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "",
                }
            )
            return
        transcript = await self.runtime.transcribe_pcm(pcm, self.input_sample_rate)
        await self.websocket.send_json(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": transcript.text,
            }
        )

    def _clear(self) -> None:
        self.buffer.clear()
        self.speech_started = False
        self.silence_ms = 0
        self.last_text = ""
        self._reset_stream()

    def _reset_stream(self) -> None:
        if self.stream is not None:
            self.stream.close()
        self.stream = self.runtime.create_stream()

    def _remember_prefix(self, pcm: bytes) -> None:
        if self.prefix_bytes <= 0:
            return
        self.prefix.append(pcm)
        total = sum(len(item) for item in self.prefix)
        while total > self.prefix_bytes and self.prefix:
            removed = self.prefix.popleft()
            total -= len(removed)

    def _turn_detection_payload(self) -> dict[str, Any] | None:
        if not self.server_vad:
            return None
        return {
            "type": "server_vad",
            "threshold": self.config.vad.threshold,
            "silence_duration_ms": self.config.vad.silence_duration_ms,
            "prefix_padding_ms": self.config.vad.prefix_padding_ms,
        }


def decode_audio_bytes(content: bytes, filename: str) -> tuple[bytes, int]:
    suffix = Path(filename).suffix.lower()
    if suffix in {"", ".wav"}:
        try:
            with wave.open(io.BytesIO(content), "rb") as handle:
                channels = handle.getnchannels()
                width = handle.getsampwidth()
                sample_rate = handle.getframerate()
                frames = handle.readframes(handle.getnframes())
        except wave.Error:
            if suffix == "":
                return content, 24000
            return decode_with_ffmpeg(content, suffix)
        if width != 2:
            raise ValueError("only 16-bit PCM WAV audio is supported")
        if channels > 1:
            frames = audioop.tomono(frames, width, 0.5, 0.5)
        return frames, sample_rate

    return decode_with_ffmpeg(content, suffix)


def decode_with_ffmpeg(content: bytes, suffix: str) -> tuple[bytes, int]:
    ffmpeg = which("ffmpeg")
    if ffmpeg is None:
        raise ValueError(
            "audio format requires ffmpeg; install ffmpeg or upload WAV/PCM16 audio"
        )
    target_rate = 24000
    with tempfile.NamedTemporaryFile(suffix=suffix or ".audio") as handle:
        handle.write(content)
        handle.flush()
        process = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                handle.name,
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                str(target_rate),
                "pipe:1",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"failed to decode audio with ffmpeg: {detail}")
    return process.stdout, target_rate


def resample_pcm16(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    if source_rate == target_rate:
        return pcm
    converted, _ = audioop.ratecv(pcm, 2, 1, source_rate, target_rate, None)
    return converted


def encode_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return output.getvalue()


def pcm16_to_float32(pcm: bytes) -> Any:
    import numpy as np

    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
