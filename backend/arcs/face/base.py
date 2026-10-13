"""Interface every face detection provider implements."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    # Bounding box as fractions of the frame (0-1), origin top-left
    x: float
    y: float
    width: float
    height: float
    # 0 = mouth closed, 1 = wide open
    mouth_open: float


class FaceDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Find faces in one frame. `image` is HxWx3 uint8 RGB."""
