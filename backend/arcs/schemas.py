"""JSON shapes exchanged with the frontend."""

from typing import Literal

from pydantic import BaseModel


class Face(BaseModel):
    # Stable for as long as the same face stays in view
    id: int
    # Bounding box as fractions of the frame (0-1), origin at the top-left of
    # the image as sent, i.e. not mirrored
    x: float
    y: float
    width: float
    height: float
    # How much the mouth has been moving recently (0-1)
    mouth_activity: float
    speaking: bool
    # How well this face's mouth movement follows the sound (0-1). Only filled in once the
    # client reports microphone loudness; the highest score is the speaker.
    speaker_score: float = 0.0


class FacesMessage(BaseModel):
    type: Literal["faces"] = "faces"
    faces: list[Face]
    process_ms: float
    # The face matching the sound, or None when nobody is clearly talking
    active_speaker_id: int | None = None


class Transcript(BaseModel):
    type: Literal["transcript"] = "transcript"
    # Interim results for one utterance share an id; the next utterance gets a new one
    utterance_id: int
    text: str
    # False while the speaker is still talking and the text may change
    is_final: bool
    # Length of the audio this text covers
    audio_ms: int
    # On-device: how long the model took. Cloud: how far the text trails the audio sent.
    process_ms: float


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    message: str
