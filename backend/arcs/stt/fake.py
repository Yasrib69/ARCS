"""Stand-in transcriber for tests and for frontend work without the real model."""

import numpy as np

from ..audio import SAMPLE_RATE
from .base import SpeechToText


class FakeSpeechToText(SpeechToText):
    def transcribe(self, audio: np.ndarray) -> str:
        return f"[{len(audio) / SAMPLE_RATE:.1f}s of speech]"
