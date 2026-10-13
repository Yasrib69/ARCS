# ARCS backend

Face tracking and live speech-to-text for the ARCS frontend, served over HTTP and WebSockets.

The demo page captions with the browser's built-in speech recognition by default, which needs
no backend. The backend's own speech-to-text (Whisper on this machine, or optionally Deepgram
in the cloud) is the alternative for browsers without it.

## Setup

Use Python 3.11 or 3.12 (MediaPipe does not support every newer version).

```
cd backend
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn arcs.server:app
```

The server listens on `http://localhost:8000`. The first start downloads the face model
(~4 MB) and the Whisper model (~75 MB). Interactive API docs are at `/docs`.

Open `http://localhost:8000` in Chrome or Edge for the demo page (`frontend/index.html`): it
shows the camera with captions under the speaker's face and a running transcript.

### Speech engines

The demo page's **Speech engine** switch chooses where speech is turned into text:

| Engine | Runs | Notes |
| --- | --- | --- |
| Browser (default) | in the browser, via the Web Speech API | Fast, free, no key. Chrome and Edge send the audio to their own speech service, so it needs internet. Not available in Firefox. |
| On device | in this backend, with Whisper | Works offline and keeps audio on the machine, but lags by a second or more on a laptop CPU. |
| Cloud | in this backend, forwarding to Deepgram | Off unless an API key is configured (below). |

### Cloud captions (optional, currently unused)

The Deepgram integration is kept in the code but does nothing without an API key. To turn it
on, create a key at https://console.deepgram.com, copy `.env.example` to `.env`, and put the
key in it:

```
ARCS_DEEPGRAM_API_KEY=your-key-here
```

Restart the server and a **Cloud** option appears in the Speech engine switch. Microphone
audio is then sent to Deepgram. `.env` is ignored by git; never commit the key.

To work on the frontend without the models, start it with fake providers, which return a
fixed face box and placeholder caption text:

```
set ARCS_FACE_PROVIDER=fake
set ARCS_STT_PROVIDER=fake
```

Run the tests with `python -m pytest`.

## API

| Endpoint | Send | Receive |
| --- | --- | --- |
| `GET /health` | | status and active providers |
| `POST /v1/faces` | one image (JPEG/PNG) as the body | faces message |
| `POST /v1/transcribe` | one 16-bit PCM WAV as the body | transcript message |
| `WS /v1/ws/faces` | video frames as binary JPEG messages | a faces message per processed frame |
| `WS /v1/ws/captions?sample_rate=16000&provider=deepgram` | mic audio as binary 16-bit little-endian mono PCM | transcript messages |

### Faces message

```json
{
  "type": "faces",
  "faces": [
    {"id": 0, "x": 0.31, "y": 0.22, "width": 0.28, "height": 0.41,
     "mouth_activity": 0.19, "speaking": true}
  ],
  "process_ms": 14.2
}
```

- `x`, `y`, `width`, `height` are fractions of the frame (0-1) from the top-left of the image
  as sent. Multiply by the displayed video size. If the video is shown mirrored, use
  `1 - x - width` for the left edge.
- `id` stays the same while a face remains in view, so a caption can stay attached to it.
- `speaking` is true when the mouth has been moving. With several faces in view, put the
  caption under the one with the highest `mouth_activity`.
- Send frames as fast as you like; the server skips to the newest frame when it falls behind.
  A frame that cannot be decoded gets `{"type": "error", "message": "..."}`.

### Transcript message

```json
{"type": "transcript", "utterance_id": 3, "text": "nice to meet you",
 "is_final": false, "audio_ms": 1400, "process_ms": 310.5}
```

- While someone is talking, interim messages arrive with the same `utterance_id` and
  `is_final: false`; each one replaces the text of the previous one.
- After a pause the last message for that utterance has `is_final: true`, and the next
  utterance gets a new id.
- `provider` is optional: `whisper` (on-device) or `deepgram` (cloud). `/health` lists the
  ones available in `stt_providers` and the default in `stt_provider`. If the provider can't
  be used (for example a bad API key) the server sends `{"type": "error", "message": "..."}`
  and closes.
- `process_ms` is the model's run time for on-device captions, and for cloud captions how far
  the text trails the audio sent.
- Send the text message `{"type": "stop"}` to get the remaining audio back as a final caption;
  the server then closes the connection.

### Frontend example

```js
// Captions: stream the microphone as 16 kHz PCM
const audioContext = new AudioContext({ sampleRate: 16000 });
const captions = new WebSocket('ws://localhost:8000/v1/ws/captions?sample_rate=16000');
captions.onmessage = (e) => showCaption(JSON.parse(e.data));
// ...in an AudioWorklet, convert each Float32 block to Int16 and captions.send(int16.buffer)

// Faces: send JPEG frames from the <video> element
const faces = new WebSocket('ws://localhost:8000/v1/ws/faces');
faces.onmessage = (e) => positionCaption(JSON.parse(e.data).faces);
canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
canvas.toBlob((blob) => faces.send(blob), 'image/jpeg', 0.7);
```

## Code layout

| File | Purpose |
| --- | --- |
| `arcs/service.py` | The functions to call from Python: `detect_faces`, `open_face_tracker`, `transcribe_wav`, `open_caption_session` |
| `arcs/server.py` | HTTP and WebSocket endpoints over those functions |
| `arcs/schemas.py` | JSON message shapes |
| `arcs/config.py` | Settings, all overridable with `ARCS_*` environment variables |
| `arcs/face/` | `FaceDetector` interface, MediaPipe implementation, `FaceTracker` (ids and speaking) |
| `arcs/stt/` | `CaptionSession` interface for live captions, with on-device (Whisper via `CaptionStream`) and cloud (`deepgram.py`) implementations |

To add another cloud speech service, subclass `CaptionSession` (see `arcs/stt/deepgram.py`)
and add it to `stt_providers` and `open_caption_session` in `arcs/service.py`. To change the
face model, subclass `FaceDetector` and add it to `get_face_detector`.
