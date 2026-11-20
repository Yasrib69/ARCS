"""The functions the rest of the app calls. The HTTP server is a thin layer over these.

Models are loaded on first use and shared; the choice of provider comes from
the settings (see config.py).
"""

import time
from functools import lru_cache

import cv2
import numpy as np

from .audio import SAMPLE_RATE, decode_wav
from .config import load_settings
from .face import FaceDetector, FaceTracker
from .schemas import Face, Transcript
from .stt import CaptionSession, CaptionStream, LocalCaptionSession, SpeechToText


@lru_cache
def get_face_detector() -> FaceDetector:
    settings = load_settings()
    if settings.face_provider == "mediapipe":
        from .face.mediapipe_detector import MediaPipeFaceDetector

        return MediaPipeFaceDetector(settings.model_dir, settings.max_faces)
    if settings.face_provider == "fake":
        from .face.fake import FakeFaceDetector

        return FakeFaceDetector()
    raise ValueError(f"Unknown ARCS_FACE_PROVIDER: {settings.face_provider!r}")


@lru_cache
def get_speech_to_text() -> SpeechToText:
    """The speech model that runs on this machine."""
    settings = load_settings()
    if settings.stt_provider == "fake":
        from .stt.fake import FakeSpeechToText

        return FakeSpeechToText()
    from .stt.whisper_local import WhisperSpeechToText

    return WhisperSpeechToText(
        settings.whisper_model,
        settings.whisper_device,
        settings.whisper_compute_type,
        settings.whisper_threads,
        settings.stt_language,
    )


def stt_providers() -> list[str]:
    """The speech providers that can be used right now."""
    settings = load_settings()
    if settings.stt_provider == "fake":
        return ["fake"]
    return ["whisper"] + (["deepgram"] if settings.deepgram_api_key else [])


def decode_image(data: bytes) -> np.ndarray:
    """Decode an encoded image (JPEG, PNG, ...) to HxWx3 uint8 RGB."""
    if not data:
        raise ValueError("No image data received")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Not a readable image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def detect_faces(image: bytes) -> list[Face]:
    """Find the faces in a single image. Ids are only meaningful within this one result."""
    return FaceTracker(get_face_detector()).update(decode_image(image))


def open_face_tracker() -> FaceTracker:
    """Start tracking a video stream; feed it frames with `tracker.update(decode_image(frame))`."""
    return FaceTracker(get_face_detector())


def transcribe_wav(wav: bytes) -> Transcript:
    """Transcribe a complete 16-bit PCM WAV recording, on this machine."""
    audio = decode_wav(wav)
    started = time.perf_counter()
    text = get_speech_to_text().transcribe(audio)
    return Transcript(
        utterance_id=0,
        text=text,
        is_final=True,
        audio_ms=round(len(audio) / SAMPLE_RATE * 1000),
        process_ms=round((time.perf_counter() - started) * 1000, 1),
    )


def open_caption_session(
    sample_rate: int = SAMPLE_RATE, provider: str | None = None
) -> CaptionSession:
    """Start captioning a live audio stream; see CaptionSession for how to drive it.

    `provider` picks the speech provider for this one stream; by default it is
    the one in the settings.
    """
    settings = load_settings()
    provider = provider or settings.stt_provider
    if provider == "deepgram" and not settings.deepgram_api_key:
        raise ValueError("Cloud captions need a Deepgram API key (ARCS_DEEPGRAM_API_KEY)")
    if provider not in stt_providers():
        raise ValueError(f"Unknown speech provider: {provider!r}")

    if provider == "deepgram":
        from .stt.deepgram import DeepgramCaptionSession

        return DeepgramCaptionSession(
            settings.deepgram_api_key,
            sample_rate=sample_rate,
            url=settings.deepgram_url,
            model=settings.deepgram_model,
            language=settings.stt_language,
            endpointing_ms=settings.deepgram_endpointing_ms,
        )
    return LocalCaptionSession(open_caption_stream(sample_rate), settings.caption_interval_s)


def open_caption_stream(sample_rate: int = SAMPLE_RATE) -> CaptionStream:
    """The on-device captioner behind a local session; usable directly without asyncio."""
    settings = load_settings()
    return CaptionStream(
        get_speech_to_text(),
        sample_rate=sample_rate,
        silence_s=settings.caption_silence_s,
        max_utterance_s=settings.caption_max_utterance_s,
        speech_rms=settings.caption_speech_rms,
    )
