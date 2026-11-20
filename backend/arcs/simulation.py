"""A simulated conversation, for demonstrating and measuring the speaker lock without a camera.

A `Scene` is a pretend room: several people, who talks when, how loud they are, and which of
the others are fidgeting (chewing, nodding) with their mouths while someone else speaks.
It produces the same signals the real system gets, a mouth opening per face per frame and a
microphone loudness, and feeds them through the real `FaceTracker` and `SpeakerFusion`.

Because the scene knows who is really talking, it can score the result. Note that this
checks the logic, not the real world: real webcams and rooms are noisier than this model.
"""

import bisect
import math
import random
from dataclasses import dataclass

import numpy as np

from .config import load_settings
from .face import Detection, FaceDetector, FaceTracker, SpeakerFusion
from .schemas import FacesMessage, Transcript

FRAME_S = 1 / 20  # simulated camera frame interval
AUDIO_S = 0.1  # simulated microphone chunk interval
AUDIO_DELAY = 0.08  # the sound reaches the page a little after the picture does
VIDEO_DELAY = 0.03

# loudness name -> (microphone RMS, how wide the jaw opens)
LOUDNESS = {"soft": (0.03, 0.16), "normal": (0.08, 0.28), "loud": (0.20, 0.42)}


@dataclass(frozen=True)
class Turn:
    speaker: int
    start: float
    end: float
    loudness: str = "normal"
    text: str = ""


class _Pulses:
    """Smooth on/off pulses at fixed times, e.g. the words of a sentence."""

    def __init__(self, spans: list[tuple[float, float]]):
        self.spans = spans
        self._starts = [a for a, _ in spans]

    def level(self, t: float, ramp: float = 0.04) -> float:
        i = bisect.bisect_right(self._starts, t) - 1
        if i < 0 or t > self.spans[i][1]:
            return 0.0
        a, b = self.spans[i]
        return max(0.0, min(1.0, (t - a) / ramp, (b - t) / ramp))


def _words(rng: random.Random, start: float, end: float) -> list[tuple[float, float]]:
    spans, t = [], start + 0.1
    while t < end - 0.2:
        length = rng.uniform(0.25, 0.7)
        spans.append((t, min(end, t + length)))
        t += length + rng.uniform(0.04, 0.2)
    return spans


class Scene:
    def __init__(
        self,
        faces: int,
        turns: list[Turn],
        fidget: dict[int, tuple[float, float, float]] | None = None,
        seed: int = 0,
    ):
        """`fidget` maps a face id to (jaw amplitude, min Hz, max Hz) of the mouth movement
        it makes while someone else is talking."""
        rng = random.Random(seed)
        self.faces = faces
        self.turns = turns
        self.duration = max(t.end for t in turns) + 1.0
        self._noise = np.random.default_rng(seed)
        self._turn_starts = [t.start for t in turns]
        self._turn_words = [_Pulses(_words(rng, t.start, t.end)) for t in turns]
        self._turn_syllable = [(rng.uniform(4.2, 6.0), rng.uniform(0, 6.28)) for _ in turns]

        self._fidget = {}
        for face, (amp, low, high) in (fidget or {}).items():
            spans, bursts, t = [], [], rng.uniform(0, 1)
            while t < self.duration:
                length = rng.uniform(0.6, 2.2)
                spans.append((t, t + length))
                bursts.append((rng.uniform(low, high), rng.uniform(0, 6.28)))
                t += length + rng.uniform(0.3, 1.2)
            self._fidget[face] = (amp, _Pulses(spans), bursts)

    def _turn_at(self, t: float) -> int | None:
        i = bisect.bisect_right(self._turn_starts, t) - 1
        return i if i >= 0 and t <= self.turns[i].end else None

    def truth(self, t: float) -> int | None:
        """Who is really talking at time t."""
        i = self._turn_at(t % self.duration)
        return None if i is None else self.turns[i].speaker

    def loudness_name(self, t: float) -> str | None:
        i = self._turn_at(t % self.duration)
        return None if i is None else self.turns[i].loudness

    def audio(self, t: float) -> float:
        """Microphone loudness (RMS) as the page would report it at time t."""
        t = t % self.duration - AUDIO_DELAY
        level = 0.0
        i = self._turn_at(t)
        if i is not None:
            rms, _ = LOUDNESS[self.turns[i].loudness]
            hz, phase = self._turn_syllable[i]
            wobble = 0.6 * math.sin(2 * math.pi * 0.7 * t)
            syllable = 0.5 + 0.5 * math.sin(2 * math.pi * hz * t + phase + wobble)
            level = rms * self._turn_words[i].level(t) * (0.45 + 0.55 * syllable)
            level *= 1 + self._noise.normal(0, 0.1)
        return max(0.0, level + abs(self._noise.normal(0, 0.002)))

    def mouth(self, face: int, t: float) -> float:
        """How open this face's mouth is at time t (0-1)."""
        t = t % self.duration - VIDEO_DELAY
        value = 0.04
        i = self._turn_at(t)
        if i is not None and self.turns[i].speaker == face:
            _, amp = LOUDNESS[self.turns[i].loudness]
            hz, phase = self._turn_syllable[i]
            syllable = 0.5 + 0.5 * math.sin(2 * math.pi * hz * t + phase)
            value += amp * self._turn_words[i].level(t) * syllable
        elif face in self._fidget:
            amp, pulses, bursts = self._fidget[face]
            j = bisect.bisect_right(pulses._starts, t) - 1
            if j >= 0:
                hz, phase = bursts[j]
                value += amp * pulses.level(t) * (0.5 + 0.5 * math.sin(2 * math.pi * hz * t + phase))
        return max(0.0, min(1.0, value + self._noise.normal(0, 0.01)))

    def detections(self, t: float) -> list[Detection]:
        out = []
        for face in range(self.faces):
            centre = (face + 1) / (self.faces + 1) + 0.006 * math.sin(t * 0.8 + face)
            out.append(Detection(centre - 0.1, 0.25, 0.2, 0.34, self.mouth(face, t)))
        return out


class SceneDetector(FaceDetector):
    """Plays a Scene back as if it were a camera; set `.t` before each frame."""

    def __init__(self, scene: Scene):
        self.scene = scene
        self.t = 0.0

    def detect(self, image):
        return self.scene.detections(self.t)


def _tracker(scene: Scene) -> tuple[FaceTracker, SceneDetector]:
    settings = load_settings()
    detector = SceneDetector(scene)
    fusion = SpeakerFusion(
        window_s=settings.speaker_window_s,
        min_audio_rms=settings.caption_speech_rms,
        min_score=settings.speaker_min_score,
        motion_floor=settings.speaker_motion_floor,
    )
    return FaceTracker(detector, fusion=fusion), detector


def _mouth_only_pick(faces, previous: int | None) -> int | None:
    """What the page did before the speaker lock: whoever's mouth moves most."""
    talking = sorted((f for f in faces if f.speaking), key=lambda f: -f.mouth_activity)
    return talking[0].id if talking else previous


# =========
# SCORING
# =========

_BLANK = np.zeros((1, 1, 3), dtype=np.uint8)


def run_scene(scene: Scene) -> dict:
    """Run a whole scene through the tracker and score who it thought was talking."""
    tracker, detector = _tracker(scene)
    t, next_audio = 0.0, 0.0
    lock = naive = None  # sticky picks, as the page keeps the caption on the last speaker
    frames = lock_ok = naive_ok = 0
    reached: dict[int, float] = {}  # turn index -> seconds until the lock reached the speaker

    while t < scene.duration:
        while next_audio <= t:
            tracker.push_audio_level(scene.audio(next_audio), now=next_audio)
            next_audio += AUDIO_S
        detector.t = t
        faces = tracker.update(_BLANK, now=t)
        if tracker.active_speaker_id is not None:
            lock = tracker.active_speaker_id
        naive = _mouth_only_pick(faces, naive)

        truth = scene.truth(t)
        if truth is not None:
            frames += 1
            lock_ok += lock == truth
            naive_ok += naive == truth
            i = scene._turn_at(t)
            if lock == truth and i not in reached:
                reached[i] = t - scene.turns[i].start
        t += FRAME_S

    delays = [reached.get(i, scene.turns[i].end - scene.turns[i].start) for i in range(len(scene.turns))]
    return {
        "frames": frames,
        "speaker_lock_accuracy": lock_ok / max(frames, 1),
        "mouth_only_accuracy": naive_ok / max(frames, 1),
        "mean_seconds_to_lock": sum(delays) / len(delays),
        "turns_missed": sum(1 for i in range(len(scene.turns)) if i not in reached),
    }


def _turns(rng: random.Random, faces: int, count: int, length: tuple[float, float],
           loudness: tuple[str, ...] = ("normal",)) -> list[Turn]:
    turns, t, last = [], 0.5, -1
    for _ in range(count):
        speaker = rng.choice([f for f in range(faces) if f != last])
        end = t + rng.uniform(*length)
        turns.append(Turn(speaker, t, end, rng.choice(loudness)))
        t, last = end + rng.uniform(0.5, 1.0), speaker
    return turns


def _scenarios(seed: int) -> dict[str, Scene]:
    rng = random.Random(seed)
    chew = (0.25, 1.5, 3.2)  # slow chewing / nodding
    return {
        "Two people taking turns": Scene(2, _turns(rng, 2, 6, (3, 5)), {0: (0.1, 1.5, 3.2), 1: (0.1, 1.5, 3.2)}, seed),
        "Listener fidgeting": Scene(2, _turns(rng, 2, 6, (3, 5)), {0: chew, 1: chew}, seed),
        "Three people": Scene(3, _turns(rng, 3, 7, (2.5, 4.5)), {0: chew, 1: chew, 2: chew}, seed),
        "Quiet speaker, busy listeners": Scene(
            3, _turns(rng, 3, 7, (2.5, 4.5), ("soft",)), {0: (0.3, 1.5, 3.2), 1: (0.3, 1.5, 3.2), 2: (0.3, 1.5, 3.2)}, seed),
        "Rapid turns": Scene(3, _turns(rng, 3, 10, (1.5, 2.5), ("normal", "loud")), {0: chew, 1: chew, 2: chew}, seed),
        "Hard: listeners mouthing along": Scene(
            2, _turns(rng, 2, 6, (3, 5)), {0: (0.3, 4.2, 6.0), 1: (0.3, 4.2, 6.0)}, seed),
    }


def evaluate(seeds: int = 4, first_seed: int = 0) -> dict:
    """Score the speaker lock against the old mouth-only method on simulated conversations."""
    results = {}
    for name in _scenarios(0):
        runs = [run_scene(_scenarios(seed)[name]) for seed in range(first_seed, first_seed + seeds)]
        results[name] = {
            key: round(sum(r[key] for r in runs) / len(runs), 3)
            for key in ("speaker_lock_accuracy", "mouth_only_accuracy", "mean_seconds_to_lock")
        }
        results[name]["turns_missed"] = sum(r["turns_missed"] for r in runs)

    overall = {
        key: round(sum(r[key] for r in results.values()) / len(results), 3)
        for key in ("speaker_lock_accuracy", "mouth_only_accuracy", "mean_seconds_to_lock")
    }
    return {"scenarios": results, "overall": overall, "runs_per_scenario": seeds, "simulated": True}


# =========
# LIVE DEMO
# =========

DEMO_TURNS = [
    Turn(0, 0.5, 5.5, "normal", "Welcome everyone. Today I am showing ARCS, live captions that follow whoever is talking."),
    Turn(1, 6.3, 10.8, "normal", "Nice. But how does it know it is me talking and not someone else?"),
    Turn(0, 11.6, 16.2, "normal", "It matches each person's lip movement to the sound coming from the microphone."),
    Turn(2, 17.0, 20.6, "loud", "That sounds really useful for deaf and hard of hearing students!"),
    Turn(0, 21.4, 25.6, "soft", "And the captions change size to show how loudly something was said."),
    Turn(1, 26.4, 30.0, "normal", "Got it. Thanks for the demo."),
]
DEMO_FIDGET = {0: (0.2, 1.5, 3.2), 1: (0.32, 1.5, 3.2), 2: (0.32, 1.5, 3.2)}


class LiveDemo:
    """A looping 3-person conversation, played in real time for the demo page."""

    def __init__(self, seed: int = 7):
        self.scene = Scene(3, DEMO_TURNS, DEMO_FIDGET, seed)
        self._tracker, self._detector = _tracker(self.scene)
        self._next_audio = 0.0
        self._naive: int | None = None
        self._sent: dict[int, int] = {}  # utterance id -> words already sent
        self._level = 0.0

    def step(self, now: float) -> list[dict]:
        """Everything to send to the page at time `now` (seconds, always increasing)."""
        scene_t = now % self.scene.duration
        while self._next_audio <= now:
            self._level = self.scene.audio(self._next_audio)
            self._tracker.push_audio_level(self._level, now=self._next_audio)
            self._next_audio += AUDIO_S
        self._detector.t = now
        faces = self._tracker.update(_BLANK, now=now)
        self._naive = _mouth_only_pick(faces, self._naive)
        messages = [
            FacesMessage(faces=faces, process_ms=0.0, active_speaker_id=self._tracker.active_speaker_id).model_dump(),
            {
                "type": "sim_status",
                "truth": self.scene.truth(now),
                "mouth_only": self._naive,
                "audio_level": round(self._level, 4),
                "loudness": self.scene.loudness_name(now),
                "mouths": {str(i): round(self.scene.mouth(i, now), 3) for i in range(self.scene.faces)},
            },
        ]
        messages += self._captions(now, scene_t)
        return messages

    def _captions(self, now: float, scene_t: float) -> list[dict]:
        loop = int(now // self.scene.duration)
        out = []
        for index, turn in enumerate(self.scene.turns):
            uid = loop * len(self.scene.turns) + index
            if scene_t < turn.start or self._sent.get(uid) == -1:
                continue
            words = turn.text.split()
            done = scene_t >= turn.end
            progress = 1.0 if done else (scene_t - turn.start) / (turn.end - turn.start)
            shown = min(len(words), max(1, math.ceil(len(words) * progress)))
            if done or shown != self._sent.get(uid, 0):
                self._sent[uid] = -1 if done else shown
                out.append(
                    Transcript(
                        utterance_id=uid,
                        text=" ".join(words[:shown]),
                        is_final=done,
                        audio_ms=round(min(scene_t, turn.end) * 1000),
                        process_ms=0.0,
                    ).model_dump()
                )
        return out
