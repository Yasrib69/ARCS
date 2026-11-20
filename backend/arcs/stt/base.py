"""Interface every speech-to-text provider implements."""

from abc import ABC, abstractmethod

import numpy as np


class SpeechToText(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> str:
        """Return the words spoken in `audio` (mono float32 in [-1, 1] at 16 kHz)."""
