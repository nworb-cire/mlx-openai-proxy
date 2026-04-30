from __future__ import annotations

import base64
import io
import json
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from mlx_openai_proxy.asr import Transcript
from mlx_openai_proxy.config import ConfiguredAsr, Settings
from mlx_openai_proxy.main import create_app


class FakeStream:
    def __init__(self) -> None:
        self.closed = False

    def add_pcm(self, pcm: bytes, sample_rate: int) -> str:
        return "draft" if pcm else ""

    def close(self) -> None:
        self.closed = True


class FakeAsrRuntime:
    def __init__(self) -> None:
        self.loaded = 0
        self.closed = 0
        self.transcriptions: list[tuple[bytes, int]] = []
        self.stream = FakeStream()

    @property
    def alias(self) -> str:
        return "parakeet:tdt-0.6b-v3"

    async def load(self) -> None:
        self.loaded += 1

    async def close(self) -> None:
        self.closed += 1

    async def transcribe_pcm(self, pcm: bytes, sample_rate: int) -> Transcript:
        self.transcriptions.append((pcm, sample_rate))
        return Transcript(text="hello world")

    def create_stream(self) -> FakeStream:
        return self.stream


def build_settings(tmp_path: Path, **overrides: object) -> Settings:
    data = {
        "metrics_db_path": str(tmp_path / "metrics.db"),
        "asr": ConfiguredAsr(
            vad={
                "enabled": True,
                "threshold": 500,
                "silence_duration_ms": 50,
                "prefix_padding_ms": 0,
            }
        ),
    }
    data.update(overrides)
    return Settings(**data)


def wav_bytes(pcm: bytes, sample_rate: int = 24000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return output.getvalue()


def tone_bytes(samples: int = 2400, amplitude: int = 1200) -> bytes:
    return b"".join(int(amplitude).to_bytes(2, "little", signed=True) for _ in range(samples))


def silence_bytes(samples: int = 2400) -> bytes:
    return b"\x00\x00" * samples


def test_asr_config_loads_from_file(tmp_path: Path) -> None:
    asr_path = tmp_path / "asr.json"
    asr_path.write_text(
        json.dumps(
            {
                "alias": "parakeet:test",
                "model_id": "local/test",
                "max_concurrency": 2,
                "input_sample_rate": 16000,
                "vad": {"threshold": 123},
            }
        )
    )

    from mlx_openai_proxy.config import _load_asr_from_path

    config = _load_asr_from_path(asr_path)

    assert config.alias == "parakeet:test"
    assert config.model_id == "local/test"
    assert config.max_concurrency == 2
    assert config.input_sample_rate == 16000
    assert config.vad.threshold == 123


def test_audio_transcriptions_json_and_models(tmp_path: Path) -> None:
    runtime = FakeAsrRuntime()
    app = create_app(build_settings(tmp_path), asr_runtime=runtime)

    with TestClient(app) as client:
        models = client.get("/v1/models").json()["data"]
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("speech.wav", wav_bytes(tone_bytes()), "audio/wav")},
            data={"model": "parakeet:tdt-0.6b-v3"},
        )

    assert runtime.loaded == 1
    assert response.status_code == 200
    assert response.json() == {"text": "hello world"}
    assert "parakeet:tdt-0.6b-v3" in [item["id"] for item in models]


def test_audio_transcriptions_text_format(tmp_path: Path) -> None:
    app = create_app(build_settings(tmp_path), asr_runtime=FakeAsrRuntime())

    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("speech.wav", wav_bytes(tone_bytes()), "audio/wav")},
            data={"response_format": "text"},
        )

    assert response.status_code == 200
    assert response.text == "hello world"


def test_realtime_server_vad_auto_commits(tmp_path: Path) -> None:
    app = create_app(build_settings(tmp_path), asr_runtime=FakeAsrRuntime())

    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/realtime?model=parakeet:tdt-0.6b-v3"
        ) as websocket:
            assert websocket.receive_json()["type"] == "session.created"
            websocket.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(tone_bytes()).decode("ascii"),
                }
            )
            events = [websocket.receive_json(), websocket.receive_json()]
            websocket.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(silence_bytes()).decode("ascii"),
                }
            )
            events.extend(websocket.receive_json() for _ in range(3))

    event_types = [event["type"] for event in events]
    assert "input_audio_buffer.speech_started" in event_types
    assert "conversation.item.input_audio_transcription.delta" in event_types
    assert "input_audio_buffer.speech_stopped" in event_types
    assert "input_audio_buffer.committed" in event_types
    assert events[-1] == {
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": "hello world",
    }


def test_realtime_manual_commit(tmp_path: Path) -> None:
    app = create_app(build_settings(tmp_path), asr_runtime=FakeAsrRuntime())

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as websocket:
            assert websocket.receive_json()["type"] == "session.created"
            websocket.send_json(
                {"type": "session.update", "session": {"turn_detection": None}}
            )
            assert websocket.receive_json()["session"]["turn_detection"] is None
            websocket.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(tone_bytes()).decode("ascii"),
                }
            )
            assert (
                websocket.receive_json()["type"]
                == "conversation.item.input_audio_transcription.delta"
            )
            websocket.send_json({"type": "input_audio_buffer.commit"})
            assert websocket.receive_json()["type"] == "input_audio_buffer.committed"
            assert websocket.receive_json()["transcript"] == "hello world"
