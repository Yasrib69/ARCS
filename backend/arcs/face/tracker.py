"""Follows faces from frame to frame and estimates who is speaking."""

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..schemas import Face
from .base import Detection, FaceDetector
from .speaker import SpeakerFusion

# Minimum overlap for a detection to count as the same face as last frame
MIN_IOU = 0.2


@dataclass
class _Track:
    id: int
    box: Detection
    last_seen: float
    # Recent (time, mouth_open) samples
    mouth: deque = field(default_factory=deque)


def _iou(a: Detection, b: Detection) -> float:
    overlap_w = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
    overlap_h = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
    if overlap_w <= 0 or overlap_h <= 0:
        return 0.0
    intersection = overlap_w * overlap_h
    return intersection / (a.width * a.height + b.width * b.height - intersection)


class FaceTracker:
    """Holds the state for one video stream. Create one per camera/connection."""

    def __init__(
        self,
        detector: FaceDetector,
        *,
        forget_after_s: float = 1.0,
        activity_window_s: float = 0.8,
        speaking_threshold: float = 0.08,
        fusion: SpeakerFusion | None = None,
    ):
        self._detector = detector
        self._forget_after_s = forget_after_s
        self._activity_window_s = activity_window_s
        self._speaking_threshold = speaking_threshold
        self._tracks: list[_Track] = []
        self._next_id = 0
        # Optional: matches mouth movement to microphone loudness to find the real speaker
        self._fusion = fusion
        self.active_speaker_id: int | None = None

    def push_audio_level(self, rms: float, now: float | None = None) -> None:
        """Tell the tracker how loud the microphone is right now (RMS, 0-1)."""
        if self._fusion:
            self._fusion.add_audio(rms, time.monotonic() if now is None else now)

    def update(self, image: np.ndarray, now: float | None = None) -> list[Face]:
        """Process the next frame (HxWx3 uint8 RGB) and return the faces in it."""
        now = time.monotonic() if now is None else now
        detections = self._detector.detect(image)

        self._tracks = [t for t in self._tracks if now - t.last_seen <= self._forget_after_s]
        unmatched = list(self._tracks)
        faces = []
        for detection in detections:
            track = max(unmatched, key=lambda t: _iou(t.box, detection), default=None)
            if track is None or _iou(track.box, detection) < MIN_IOU:
                track = _Track(id=self._next_id, box=detection, last_seen=now)
                self._next_id += 1
                self._tracks.append(track)
            else:
                unmatched.remove(track)
            track.box = detection
            track.last_seen = now

            # Speech shows up as the mouth opening and closing, so activity is
            # the range of mouth_open over the last moment, not its current value
            track.mouth.append((now, detection.mouth_open))
            while track.mouth[0][0] < now - self._activity_window_s:
                track.mouth.popleft()
            values = [value for _, value in track.mouth]
            activity = min(1.0, max(values) - min(values))
            if self._fusion:
                self._fusion.observe(track.id, detection.mouth_open, now)

            faces.append(
                Face(
                    id=track.id,
                    x=detection.x,
                    y=detection.y,
                    width=detection.width,
                    height=detection.height,
                    mouth_activity=activity,
                    speaking=activity >= self._speaking_threshold,
                )
            )
        if self._fusion:
            self.active_speaker_id, scores = self._fusion.update([f.id for f in faces], now)
            for face in faces:
                face.speaker_score = round(scores.get(face.id, 0.0), 3)
        return faces
