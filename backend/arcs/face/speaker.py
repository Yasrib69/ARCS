"""Decides who is talking by matching mouth movement to the sound.

Mouth movement alone is a weak clue: a listener who nods, smiles or chews moves their
mouth too. Speech has a stronger signature: the loudness of the voice rises and falls
in step with the speaker's mouth. For every face this keeps a short history of how much
the mouth moved, compares it with the history of the microphone loudness, and the face
whose lips best follow the sound is the speaker.

The class has no camera or microphone dependencies, so it can be tested with plain numbers.
Feed it with `observe` (one mouth sample per face per frame) and `add_audio` (one loudness
sample per audio chunk), then call `update` to get the current speaker.
"""

import threading
from collections import deque

import numpy as np

# Mouth movement smaller than this between two frames is detector jitter, not speech
JITTER = 0.01


def _weighted_corr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Pearson correlation where the samples with a larger weight count more."""
    w = w / w.sum()
    a = a - (w * a).sum()
    b = b - (w * b).sum()
    var = (w * a * a).sum() * (w * b * b).sum()
    return float((w * a * b).sum() / np.sqrt(var)) if var > 1e-12 else 0.0


class SpeakerFusion:
    def __init__(
        self,
        *,
        window_s: float = 1.0,
        grid_hz: float = 10.0,
        min_audio_rms: float = 0.01,
        max_lag_s: float = 0.1,
        min_score: float = 0.35,
        motion_floor: float = 0.04,
        switch_margin: float = 0.1,
        switch_hold_s: float = 0.1,
        release_s: float = 1.5,
        recency_s: float = 0.25,
    ):
        self._window_s = window_s
        self._dt = 1.0 / grid_hz
        self._bins = round(window_s * grid_hz)
        self._min_audio_rms = min_audio_rms
        self._max_lag = round(max_lag_s * grid_hz)
        self._min_score = min_score
        self._motion_floor = motion_floor
        self._switch_margin = switch_margin
        self._switch_hold_s = switch_hold_s
        self._release_s = release_s
        # Recent samples count more than old ones, so the lock follows a change of speaker quickly
        self._weights = np.exp(-(self._bins - 1 - np.arange(self._bins)) * self._dt / recency_s)

        self._lock = threading.Lock()
        self._audio: deque = deque()  # (time, rms)
        self._mouths: dict[int, deque] = {}  # face id -> (time, mouth_open)
        self._active: int | None = None
        self._last_qualified = 0.0
        self._challenger: int | None = None
        self._challenger_since = 0.0

    # ---------- input ----------

    def add_audio(self, rms: float, now: float) -> None:
        """One microphone loudness sample (RMS, 0-1), stamped with the time it arrived."""
        with self._lock:
            self._audio.append((now, rms))
            self._trim(self._audio, now)

    def observe(self, face_id: int, mouth_open: float, now: float) -> None:
        """One mouth sample (0 closed, 1 wide open) for a face in the current frame."""
        with self._lock:
            history = self._mouths.setdefault(face_id, deque())
            history.append((now, mouth_open))
            self._trim(history, now)

    def _trim(self, history: deque, now: float) -> None:
        while history and history[0][0] < now - self._window_s - 0.5:
            history.popleft()

    # ---------- decision ----------

    def update(self, face_ids: list[int], now: float) -> tuple[int | None, dict[int, float]]:
        """Return (speaker id or None, score 0-1 for each face in `face_ids`)."""
        with self._lock:
            for gone in [i for i in self._mouths if i not in face_ids]:
                if not self._mouths[gone] or self._mouths[gone][-1][0] < now - self._window_s:
                    del self._mouths[gone]
            audio = self._audio_series(now)
            scores = {i: self._score(i, audio, now) for i in face_ids}

            # Someone can only be speaking while there is sound
            recent = [r for t, r in self._audio if t >= now - 0.5]
            if not recent or max(recent) < self._min_audio_rms:
                scores = {i: 0.0 for i in face_ids}
            return self._choose(scores, now), scores

    def _grid(self, now: float) -> np.ndarray:
        return now - self._window_s + np.arange(self._bins + 1) * self._dt

    def _audio_series(self, now: float) -> np.ndarray:
        edges = self._grid(now)
        series = np.zeros(self._bins)
        last = 0.0
        for k in range(self._bins):
            values = [r for t, r in self._audio if edges[k] <= t < edges[k + 1]]
            if values:
                last = float(np.mean(values))
            series[k] = last
        return series

    def _motion_series(self, face_id: int, now: float) -> np.ndarray | None:
        history = list(self._mouths.get(face_id, ()))
        if len(history) < 3:
            return None
        edges = self._grid(now)
        series = np.zeros(self._bins)
        for (_, before), (t, after) in zip(history, history[1:]):
            k = int((t - edges[0]) // self._dt)
            if 0 <= k < self._bins:
                series[k] += max(0.0, abs(after - before) - JITTER)
        return series

    def _score(self, face_id: int, audio: np.ndarray, now: float) -> float:
        motion = self._motion_series(face_id, now)
        if motion is None or audio.std() < 1e-6 or motion.std() < 1e-6:
            return 0.0
        best = 0.0
        for lag in range(-self._max_lag, self._max_lag + 1):
            a = audio[max(0, -lag) : self._bins - max(0, lag)]
            m = motion[max(0, lag) : self._bins - max(0, -lag)]
            w = self._weights[max(0, lag) : self._bins - max(0, -lag)]
            if len(a) < 5:
                continue
            best = max(best, _weighted_corr(a, m, w))
        # A face that barely moves cannot be the speaker, however well it lines up
        weight = min(1.0, float((self._weights * motion).sum() / self._weights.sum()) / self._motion_floor)
        return max(0.0, min(1.0, best * weight))

    def _choose(self, scores: dict[int, float], now: float) -> int | None:
        best = max(scores, key=scores.get, default=None)
        qualified = best is not None and scores[best] >= self._min_score
        if qualified:
            self._last_qualified = now

        if self._active is not None and self._active not in scores:
            self._active = None
        if self._active is None:
            if qualified:
                self._active = best
            return self._active

        current = scores[self._active]
        if qualified and best != self._active and scores[best] > current + self._switch_margin:
            # Someone else leads clearly; switch once that has held for a moment
            if self._challenger != best:
                self._challenger, self._challenger_since = best, now
            if now - self._challenger_since >= self._switch_hold_s:
                self._active, self._challenger = best, None
        else:
            self._challenger = None

        # Silence: let go of the speaker after a while
        if now - self._last_qualified > self._release_s:
            self._active = None
        return self._active
