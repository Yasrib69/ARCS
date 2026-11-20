"""Stand-in detector for tests and for frontend work without the real model."""

import numpy as np

from .base import Detection, FaceDetector


class FakeFaceDetector(FaceDetector):
    def detect(self, image: np.ndarray) -> list[Detection]:
        return [Detection(x=0.35, y=0.25, width=0.3, height=0.4, mouth_open=0.0)]
