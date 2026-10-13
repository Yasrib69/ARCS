"""HTTP and WebSocket API for the frontend.

Run from the backend directory:  python -m uvicorn arcs.server:app
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import service
from .audio import SAMPLE_RATE
from .config import BACKEND_DIR, load_settings
from .schemas import ErrorMessage, FacesMessage, Transcript

logger = logging.getLogger("arcs")
DEMO_PAGE = BACKEND_DIR.parent / "frontend" / "index.html"
settings = load_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the models up front so the first request isn't the slow one
    await asyncio.to_thread(service.get_face_detector)
    if settings.stt_provider != "deepgram":
        await asyncio.to_thread(service.get_speech_to_text)
    yield


app = FastAPI(title="ARCS backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _track(tracker, frame: bytes) -> FacesMessage:
    started = time.perf_counter()
    faces = tracker.update(service.decode_image(frame))
    return FacesMessage(faces=faces, process_ms=round((time.perf_counter() - started) * 1000, 1))


async def _receive(websocket: WebSocket) -> tuple[bytes | None, dict | None]:
    """Wait for the next message: (binary payload, None) or (None, parsed JSON)."""
    message = await websocket.receive()
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    if message.get("bytes") is not None:
        return message["bytes"], None
    try:
        control = json.loads(message.get("text") or "")
    except json.JSONDecodeError:
        control = None
    return None, control if isinstance(control, dict) else {}

#=========
# ROUTES
#=========

# Serve the demo page, so it shares an address with the API
@app.get("/", include_in_schema=False)
def demo_page():
    return FileResponse(DEMO_PAGE)


# Verify our app is healthy, and list our current providers
@app.get("/health")
def health():
    return {
        "status": "ok",
        "face_provider": settings.face_provider,
        "stt_provider": settings.stt_provider,
        # Values accepted by the captions endpoint's `provider` parameter
        "stt_providers": service.stt_providers(),
    }


def _binary_body(media_type: str) -> dict:
    """Describe a raw file body, so the /docs page offers a file picker for it."""
    return {
        "requestBody": {
            "required": True,
            "content": {media_type: {"schema": {"type": "string", "format": "binary"}}},
        }
    }


@app.post("/v1/faces", response_model=FacesMessage, openapi_extra=_binary_body("image/jpeg"))
async def faces(request: Request):
    """Detect faces in one image sent as the request body."""
    try:
        return await asyncio.to_thread(_track, service.open_face_tracker(), await request.body())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/v1/transcribe", response_model=Transcript, openapi_extra=_binary_body("audio/wav"))
async def transcribe(request: Request):
    """Transcribe a 16-bit PCM WAV file sent as the request body."""
    try:
        return await asyncio.to_thread(service.transcribe_wav, await request.body())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.websocket("/v1/ws/faces")
async def ws_faces(websocket: WebSocket):
    """Client sends encoded video frames (binary); server replies with a FacesMessage for each
    frame it processes. Frames that arrive while one is being processed replace each other, so
    results always describe the most recent frame."""
    await websocket.accept()
    tracker = service.open_face_tracker()
    latest: list[bytes] = []
    has_frame = asyncio.Event()

    async def receive_frames():
        while True:
            frame, _ = await _receive(websocket)
            if frame is not None:
                latest[:] = [frame]
                has_frame.set()

    async def process_frames():
        while True:
            await has_frame.wait()
            has_frame.clear()
            try:
                reply = await asyncio.to_thread(_track, tracker, latest.pop())
            except ValueError as e:
                reply = ErrorMessage(message=str(e))
            await websocket.send_text(reply.model_dump_json())

    try:
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(receive_frames())
            tasks.create_task(process_frames())
    except* WebSocketDisconnect:
        pass


@app.websocket("/v1/ws/captions")
async def ws_captions(
    websocket: WebSocket, sample_rate: int = SAMPLE_RATE, provider: str | None = None
):
    """Client sends microphone audio as 16-bit little-endian mono PCM (binary) at `sample_rate`;
    server sends Transcript messages. `provider` chooses the speech provider for this
    connection (see /health for the choices). Sending the text message {"type": "stop"}
    returns whatever is left as a final caption and closes the connection."""
    await websocket.accept()
    try:
        # Opening a session may load a model or connect to the cloud provider
        session = await asyncio.to_thread(service.open_caption_session, sample_rate, provider)
        await session.start()
    except (ValueError, ConnectionError) as e:
        await websocket.send_text(ErrorMessage(message=str(e)).model_dump_json())
        await websocket.close(code=1011)
        return

    async def send_captions():
        async for transcript in session.transcripts():
            await websocket.send_text(transcript.model_dump_json())

    sender = asyncio.create_task(send_captions())
    try:
        while not sender.done():
            audio, control = await _receive(websocket)
            if audio is not None:
                await session.add_audio(audio)
            elif control.get("type") == "stop":
                break
        await session.stop()
        await sender
        await websocket.close()
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Captioning failed")
        await websocket.close(code=1011)
    finally:
        sender.cancel()
        await session.close()
