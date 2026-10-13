"""Runtime settings, read from ARCS_* environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _load_env_file(path: Path) -> None:
    """Read KEY=VALUE lines from backend/.env, where API keys are kept out of git.

    Variables already set in the real environment win.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _env(name: str, default: str) -> str:
    return os.getenv(f"ARCS_{name}", default)


@dataclass(frozen=True)
class Settings:
    # Which implementation to use: "mediapipe" or "fake"
    face_provider: str
    # Speech provider used unless a request asks for another: "whisper" (on this
    # machine), "deepgram" (cloud) or "fake"
    stt_provider: str

    # Where downloaded model files are cached
    model_dir: Path
    max_faces: int

    # tiny.en is the fastest; base.en / small.en are more accurate but slower
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    whisper_threads: int
    # Spoken language code, or "" to auto-detect (needs a multilingual model, e.g. "base")
    stt_language: str

    # Empty means the Deepgram provider is unavailable
    deepgram_api_key: str
    deepgram_url: str
    deepgram_model: str
    # Milliseconds of silence after which Deepgram ends an utterance
    deepgram_endpointing_ms: int

    # The caption_* settings apply to the on-device provider only
    # Seconds between interim caption updates
    caption_interval_s: float
    # Seconds of trailing silence that end an utterance
    caption_silence_s: float
    # Longest utterance before a caption is forced final
    caption_max_utterance_s: float
    # Loudness (RMS, 0-1) above which a chunk of audio counts as speech
    caption_speech_rms: float

    cors_origins: list[str]


def load_settings() -> Settings:
    _load_env_file(BACKEND_DIR / ".env")
    return Settings(
        face_provider=_env("FACE_PROVIDER", "mediapipe"),
        stt_provider=_env("STT_PROVIDER", "whisper"),
        model_dir=Path(_env("MODEL_DIR", str(BACKEND_DIR / "models"))),
        max_faces=int(_env("MAX_FACES", "4")),
        whisper_model=_env("WHISPER_MODEL", "tiny.en"),
        whisper_device=_env("WHISPER_DEVICE", "auto"),
        whisper_compute_type=_env("WHISPER_COMPUTE_TYPE", "int8"),
        whisper_threads=int(_env("WHISPER_THREADS", str(os.cpu_count() or 4))),
        stt_language=_env("STT_LANGUAGE", "en"),
        deepgram_api_key=_env("DEEPGRAM_API_KEY", ""),
        deepgram_url=_env("DEEPGRAM_URL", "wss://api.deepgram.com/v1/listen"),
        deepgram_model=_env("DEEPGRAM_MODEL", "nova-3"),
        deepgram_endpointing_ms=int(_env("DEEPGRAM_ENDPOINTING_MS", "300")),
        caption_interval_s=float(_env("CAPTION_INTERVAL_S", "0.4")),
        caption_silence_s=float(_env("CAPTION_SILENCE_S", "0.7")),
        caption_max_utterance_s=float(_env("CAPTION_MAX_UTTERANCE_S", "12")),
        caption_speech_rms=float(_env("CAPTION_SPEECH_RMS", "0.01")),
        cors_origins=_env("CORS_ORIGINS", "*").split(","),
    )
