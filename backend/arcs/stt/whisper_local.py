"""Speech-to-text with Whisper running on this machine (via faster-whisper)."""

import threading

import numpy as np
from faster_whisper import WhisperModel

from .base import SpeechToText


class WhisperSpeechToText(SpeechToText):
    def __init__(self, model: str, device: str, compute_type: str, threads: int, language: str):
        # Downloads the model on first use
        self._model = WhisperModel(
            model, device=device, compute_type=compute_type, cpu_threads=threads
        )
        self._language = language or None
        # One transcription at a time, so concurrent clients don't fight over the CPU
        self._lock = threading.Lock()

    def transcribe(self, audio: np.ndarray) -> str:
        with self._lock:
            segments, _ = self._model.transcribe(
                audio,
                language=self._language,
                beam_size=1,
                # No retries at higher temperatures: they can take seconds on a clipped word
                temperature=0.0,
                without_timestamps=True,
                # Whisper invents text for silence unless non-speech is filtered out
                vad_filter=True,
                condition_on_previous_text=False,
            )
            return " ".join(segment.text.strip() for segment in segments).strip()
