"""Audio helpers. Everything downstream works on mono float32 at 16 kHz."""

import io
import wave

import numpy as np

SAMPLE_RATE = 16000


def pcm16_to_float(data: bytes) -> np.ndarray:
    """Convert 16-bit little-endian PCM bytes to float32 samples in [-1, 1]."""
    data = data[: len(data) - len(data) % 2]
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def resample(audio: np.ndarray, source_rate: int, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    if source_rate == target_rate or not len(audio):
        return audio
    target_len = round(len(audio) * target_rate / source_rate)
    positions = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)


def decode_wav(data: bytes) -> np.ndarray:
    """Decode a 16-bit PCM WAV file to mono float32 at 16 kHz."""
    try:
        with wave.open(io.BytesIO(data)) as wav:
            if wav.getsampwidth() != 2:
                raise ValueError("WAV must be 16-bit PCM")
            channels = wav.getnchannels()
            rate = wav.getframerate()
            audio = pcm16_to_float(wav.readframes(wav.getnframes()))
    except (wave.Error, EOFError) as e:
        raise ValueError(f"Not a readable WAV file: {e}") from e
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return resample(audio, rate)
