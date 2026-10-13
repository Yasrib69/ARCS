"""Face detection with MediaPipe's Face Landmarker."""

import threading
import urllib.request
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from .base import Detection, FaceDetector

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_model(model_dir: Path) -> Path:
    path = model_dir / "face_landmarker.task"
    if not path.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, partial)
        partial.replace(path)
    return path


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


class MediaPipeFaceDetector(FaceDetector):
    def __init__(self, model_dir: Path, max_faces: int):
        # The model is passed as bytes because MediaPipe mishandles some Windows paths
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_buffer=_ensure_model(model_dir).read_bytes()
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_faces,
            output_face_blendshapes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._lock = threading.Lock()

    def detect(self, image: np.ndarray) -> list[Detection]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image))
        with self._lock:
            result = self._landmarker.detect(mp_image)

        detections = []
        for landmarks, blendshapes in zip(result.face_landmarks, result.face_blendshapes):
            xs = [point.x for point in landmarks]
            ys = [point.y for point in landmarks]
            left, top = _clamp(min(xs)), _clamp(min(ys))
            right, bottom = _clamp(max(xs)), _clamp(max(ys))
            jaw_open = next((b.score for b in blendshapes if b.category_name == "jawOpen"), 0.0)
            detections.append(
                Detection(
                    x=left,
                    y=top,
                    width=right - left,
                    height=bottom - top,
                    mouth_open=float(jaw_open),
                )
            )
        return detections
