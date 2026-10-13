"""Turns a live audio stream into interim and final captions."""

import threading
import time

import numpy as np

from ..audio import SAMPLE_RATE, pcm16_to_float, resample
from ..schemas import Transcript
from .base import SpeechToText

# Audio kept from before speech starts, so the first word isn't clipped
PREROLL_S = 0.3


class CaptionStream:
    """Holds the state for one audio stream. Create one per microphone/connection.

    Audio accumulates into the current utterance. Each `poll` re-transcribes the
    utterance so far (an interim caption); once the speaker pauses, the caption
    is marked final and the next utterance starts.
    """

    def __init__(
        self,
        engine: SpeechToText,
        *,
        sample_rate: int = SAMPLE_RATE,
        silence_s: float = 0.7,
        max_utterance_s: float = 12.0,
        speech_rms: float = 0.01,
    ):
        self._engine = engine
        self._sample_rate = sample_rate
        self._silence_s = silence_s
        self._max_samples = int(max_utterance_s * SAMPLE_RATE)
        self._speech_rms = speech_rms
        self._preroll = int(PREROLL_S * SAMPLE_RATE)

        self._lock = threading.Lock()
        # The buffer array is replaced, never modified in place, so a reference
        # taken under the lock stays a valid snapshot
        self._buffer = np.zeros(0, dtype=np.float32)
        # Index just past the last loud sample; 0 means no speech yet
        self._speech_end = 0
        # Value of _speech_end at the last transcription
        self._transcribed_end = 0
        self._utterance_id = 0
        self._last_text = ""
        self._last_process_ms = 0.0

    def add_audio(self, pcm: bytes) -> None:
        """Append 16-bit little-endian mono PCM. Cheap; safe to call from the event loop."""
        audio = resample(pcm16_to_float(pcm), self._sample_rate)
        if not len(audio):
            return
        is_speech = float(np.sqrt(np.mean(audio**2))) >= self._speech_rms
        with self._lock:
            self._buffer = np.concatenate((self._buffer, audio))
            if is_speech:
                self._speech_end = len(self._buffer)
            elif not self._speech_end:
                self._buffer = self._buffer[-self._preroll :]

    def poll(self, flush: bool = False) -> Transcript | None:
        """Transcribe if there is something new. Runs the model, so call it off the event loop.

        `flush` forces whatever has been said so far to be returned as final.
        """
        with self._lock:
            audio = self._buffer
            if not self._speech_end:
                return None
            silence_s = (len(audio) - self._speech_end) / SAMPLE_RATE
            is_final = flush or silence_s >= self._silence_s or len(audio) >= self._max_samples
            unchanged = self._speech_end == self._transcribed_end
            if not is_final and unchanged:
                return None
            self._transcribed_end = self._speech_end
            # Leave out trailing silence; models tend to invent words for it
            spoken = audio[: self._speech_end + self._preroll]

        if not unchanged:
            # Otherwise nothing was said since the last interim, so its result still stands
            started = time.perf_counter()
            self._last_text = self._engine.transcribe(spoken)
            self._last_process_ms = (time.perf_counter() - started) * 1000
        text = self._last_text

        with self._lock:
            utterance_id = self._utterance_id
            if is_final:
                # Keep anything that arrived while the model was running
                self._buffer = self._buffer[len(audio) :]
                self._speech_end = max(0, self._speech_end - len(audio))
                self._transcribed_end = 0
                self._utterance_id += 1
                if not self._speech_end:
                    self._buffer = self._buffer[-self._preroll :]

        if not text:
            return None
        return Transcript(
            utterance_id=utterance_id,
            text=text,
            is_final=is_final,
            audio_ms=round(len(spoken) / SAMPLE_RATE * 1000),
            process_ms=round(self._last_process_ms, 1),
        )
