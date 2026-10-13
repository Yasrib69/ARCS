import asyncio
import http
import json
import time

import cv2
import numpy as np
import pytest
import websockets
from fastapi.testclient import TestClient

from arcs import service
from arcs.audio import SAMPLE_RATE
from arcs.face import Detection, FaceDetector, FaceTracker
from arcs.stt import CaptionStream
from arcs.stt.deepgram import DeepgramCaptionSession
from arcs.stt.fake import FakeSpeechToText


def pcm(seconds: float, loud: bool, rate: int = SAMPLE_RATE) -> bytes:
    samples = np.full(int(seconds * rate), 8000 if loud else 0, dtype="<i2")
    return samples.tobytes()


def jpeg() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ARCS_FACE_PROVIDER", "fake")
    monkeypatch.setenv("ARCS_STT_PROVIDER", "fake")
    monkeypatch.setenv("ARCS_CAPTION_INTERVAL_S", "0.05")
    service.get_face_detector.cache_clear()
    service.get_speech_to_text.cache_clear()
    from arcs import server

    monkeypatch.setattr(server, "settings", server.load_settings())
    with TestClient(server.app) as client:
        yield client
    service.get_face_detector.cache_clear()
    service.get_speech_to_text.cache_clear()


class ScriptedDetector(FaceDetector):
    """Returns the next pre-set list of detections on each call."""

    def __init__(self, frames):
        self._frames = iter(frames)

    def detect(self, image):
        return next(self._frames)


def box(x, mouth_open=0.0):
    return Detection(x=x, y=0.2, width=0.2, height=0.3, mouth_open=mouth_open)


FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


def test_tracker_keeps_ids_as_faces_move():
    tracker = FaceTracker(
        ScriptedDetector([[box(0.1), box(0.6)], [box(0.62), box(0.12)], [box(0.14)]])
    )
    first = tracker.update(FRAME, now=0.0)
    second = tracker.update(FRAME, now=0.1)
    third = tracker.update(FRAME, now=0.2)
    assert [f.id for f in first] == [0, 1]
    assert [f.id for f in second] == [1, 0]
    assert [f.id for f in third] == [0]


def test_tracker_gives_new_id_after_face_is_gone():
    tracker = FaceTracker(ScriptedDetector([[box(0.1)], [], [box(0.1)]]), forget_after_s=1.0)
    assert tracker.update(FRAME, now=0.0)[0].id == 0
    tracker.update(FRAME, now=1.0)
    assert tracker.update(FRAME, now=5.0)[0].id == 1


def test_tracker_flags_the_face_whose_mouth_moves():
    frames = [[box(0.1, mouth), box(0.6, 0.05)] for mouth in (0.0, 0.3, 0.05, 0.25)]
    tracker = FaceTracker(ScriptedDetector(frames))
    for i in range(4):
        talker, listener = tracker.update(FRAME, now=i * 0.1)
    assert talker.speaking and not listener.speaking
    assert talker.mouth_activity == pytest.approx(0.3)


def test_caption_stream_ignores_silence():
    stream = CaptionStream(FakeSpeechToText())
    stream.add_audio(pcm(2.0, loud=False))
    assert stream.poll() is None
    assert stream.poll(flush=True) is None


def test_caption_stream_interim_then_final():
    stream = CaptionStream(FakeSpeechToText(), silence_s=0.5)
    stream.add_audio(pcm(1.0, loud=True))
    interim = stream.poll()
    assert interim.is_final is False and interim.utterance_id == 0
    assert stream.poll() is None  # nothing new since the last poll

    stream.add_audio(pcm(0.6, loud=False))
    final = stream.poll()
    assert final.is_final is True and final.utterance_id == 0
    assert final.audio_ms == 1300  # the speech plus a short tail, not the silence

    stream.add_audio(pcm(1.0, loud=True))
    assert stream.poll().utterance_id == 1


def test_caption_stream_forces_final_on_long_utterance():
    stream = CaptionStream(FakeSpeechToText(), max_utterance_s=2.0)
    stream.add_audio(pcm(2.5, loud=True))
    assert stream.poll().is_final is True


def test_caption_stream_resamples():
    stream = CaptionStream(FakeSpeechToText(), sample_rate=48000)
    stream.add_audio(pcm(1.0, loud=True, rate=48000))
    assert stream.poll().audio_ms == 1000


def test_health(client):
    assert client.get("/health").json() == {
        "status": "ok",
        "face_provider": "fake",
        "stt_provider": "fake",
        "stt_providers": ["fake"],
    }


def test_post_faces(client):
    body = client.post("/v1/faces", content=jpeg()).json()
    assert body["type"] == "faces"
    assert body["faces"][0]["width"] == pytest.approx(0.3)
    assert client.post("/v1/faces", content=b"not an image").status_code == 400
    assert client.post("/v1/faces").status_code == 400


def test_post_transcribe_rejects_non_wav(client):
    assert client.post("/v1/transcribe", content=b"nope").status_code == 400


def test_ws_faces(client):
    with client.websocket_connect("/v1/ws/faces") as ws:
        ws.send_bytes(jpeg())
        first = ws.receive_json()
        ws.send_bytes(b"garbage")
        assert ws.receive_json()["type"] == "error"
        ws.send_bytes(jpeg())
        second = ws.receive_json()
    assert first["faces"][0]["id"] == second["faces"][0]["id"] == 0


def test_ws_captions_stop_returns_final(client):
    with client.websocket_connect("/v1/ws/captions") as ws:
        ws.send_bytes(pcm(1.0, loud=True))
        time.sleep(0.2)
        ws.send_text(json.dumps({"type": "stop"}))
        messages = []
        while not messages or not messages[-1]["is_final"]:
            messages.append(ws.receive_json())
    assert messages[-1]["text"] == "[1.0s of speech]"
    assert all(m["utterance_id"] == 0 for m in messages)


def test_ws_captions_reports_unavailable_provider(client):
    with client.websocket_connect("/v1/ws/captions?provider=deepgram") as ws:
        assert ws.receive_json()["type"] == "error"


def deepgram_session(port: int) -> DeepgramCaptionSession:
    return DeepgramCaptionSession(
        "secret",
        sample_rate=SAMPLE_RATE,
        url=f"ws://127.0.0.1:{port}/v1/listen",
        model="nova-3",
        language="en",
        endpointing_ms=300,
    )


def deepgram_result(text, start, duration, is_final=False, speech_final=False) -> str:
    return json.dumps(
        {
            "type": "Results",
            "start": start,
            "duration": duration,
            "is_final": is_final,
            "speech_final": speech_final,
            "channel": {"alternatives": [{"transcript": text}]},
        }
    )


def test_deepgram_session_builds_utterances_from_results():
    request = {}

    async def fake_deepgram(ws):
        request["auth"] = ws.request.headers["Authorization"]
        request["path"] = ws.request.path
        await ws.recv()  # the first audio
        await ws.send(json.dumps({"type": "Metadata"}))
        await ws.send(deepgram_result("hello", 0.0, 0.5))
        await ws.send(deepgram_result("hello there", 0.0, 0.8, is_final=True))
        await ws.send(deepgram_result("", 0.8, 0.1))
        await ws.send(deepgram_result("how are", 0.8, 0.2))
        await ws.send(deepgram_result("how are you", 0.8, 0.2, is_final=True, speech_final=True))
        async for message in ws:
            if isinstance(message, str) and json.loads(message)["type"] == "CloseStream":
                await ws.send(deepgram_result("goodbye", 1.0, 0.5, is_final=True))
                break

    async def scenario():
        async with websockets.serve(fake_deepgram, "127.0.0.1", 0) as server:
            session = deepgram_session(server.sockets[0].getsockname()[1])
            await session.start()
            await session.add_audio(pcm(1.0, loud=True))
            await asyncio.sleep(0.2)
            await session.stop()
            transcripts = [t async for t in session.transcripts()]
            await session.close()
            return transcripts

    transcripts = asyncio.run(scenario())
    assert request["auth"] == "Token secret"
    assert "sample_rate=16000" in request["path"] and "interim_results=true" in request["path"]
    assert [(t.utterance_id, t.text, t.is_final) for t in transcripts] == [
        (0, "hello", False),
        (0, "hello there", False),
        (0, "hello there", False),
        (0, "hello there how are", False),
        (0, "hello there how are you", True),
        (1, "goodbye", False),
        (1, "goodbye", True),
    ]
    # One second of audio was sent and the final text covered it all
    assert transcripts[4].audio_ms == 1000 and transcripts[4].process_ms == 0


def test_deepgram_session_reports_a_bad_key():
    def reject(connection, request):
        return connection.respond(http.HTTPStatus.UNAUTHORIZED, "no")

    async def scenario():
        async with websockets.serve(None, "127.0.0.1", 0, process_request=reject) as server:
            await deepgram_session(server.sockets[0].getsockname()[1]).start()

    with pytest.raises(ConnectionError, match="rejected the API key"):
        asyncio.run(scenario())
