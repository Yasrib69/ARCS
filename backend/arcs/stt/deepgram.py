"""Speech-to-text in the cloud, using Deepgram's live streaming API.

Audio is forwarded to Deepgram over a WebSocket as it arrives, and Deepgram
sends back text within a few hundred milliseconds.
"""

import asyncio
import json
import logging
from urllib.parse import urlencode

import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

from ..schemas import Transcript
from .session import CaptionSession

logger = logging.getLogger("arcs")

# A caption this long is made final at the next opportunity, even if the
# speaker hasn't paused, so one utterance doesn't grow without bound
MAX_UTTERANCE_CHARS = 180


class DeepgramCaptionSession(CaptionSession):
    def __init__(
        self,
        api_key: str,
        *,
        sample_rate: int,
        url: str,
        model: str,
        language: str,
        endpointing_ms: int,
    ):
        super().__init__()
        self._api_key = api_key
        self._sample_rate = sample_rate
        query = urlencode(
            {
                "model": model,
                "language": language or "multi",
                "encoding": "linear16",
                "sample_rate": sample_rate,
                "channels": 1,
                "interim_results": "true",
                "smart_format": "true",
                # Milliseconds of silence after which Deepgram ends the utterance
                "endpointing": endpointing_ms,
            }
        )
        self._url = f"{url}?{query}"
        self._socket = None
        self._reader: asyncio.Task | None = None
        self._samples_sent = 0
        # Deepgram finalizes an utterance piece by piece; these are the pieces so far
        self._parts: list[str] = []
        self._utterance_id = 0
        self._utterance_start_s = 0.0

    async def start(self) -> None:
        try:
            self._socket = await websockets.connect(
                self._url, additional_headers={"Authorization": f"Token {self._api_key}"}
            )
        except InvalidStatus as e:
            if e.response.status_code in (401, 403):
                raise ConnectionError("Deepgram rejected the API key") from e
            raise ConnectionError(f"Deepgram refused the connection ({e.response.status_code})") from e
        except OSError as e:
            raise ConnectionError(f"Could not reach Deepgram: {e}") from e
        self._reader = asyncio.create_task(self._read())

    async def add_audio(self, pcm: bytes) -> None:
        self._samples_sent += len(pcm) // 2
        await self._socket.send(pcm)

    async def stop(self) -> None:
        try:
            # Asks Deepgram to return what it has left and then hang up
            await self._socket.send(json.dumps({"type": "CloseStream"}))
        except ConnectionClosed:
            pass
        try:
            await asyncio.wait_for(asyncio.shield(self._reader), timeout=5)
        except TimeoutError:
            await self.close()

    async def close(self) -> None:
        if self._reader:
            self._reader.cancel()
        if self._socket:
            await self._socket.close()

    async def _read(self) -> None:
        try:
            async for raw in self._socket:
                message = json.loads(raw)
                if message.get("type") == "Results":
                    self._handle_result(message)
        except ConnectionClosed as e:
            logger.warning("Deepgram connection dropped: %s", e)
        finally:
            # Anything still pending when the connection ends becomes the last caption
            self._emit(" ".join(self._parts), is_final=True, heard_until_s=None)
            self._results.put_nowait(None)

    def _handle_result(self, message: dict) -> None:
        text = message["channel"]["alternatives"][0]["transcript"].strip()
        heard_until_s = message.get("start", 0.0) + message.get("duration", 0.0)
        so_far = " ".join(self._parts + ([text] if text else []))

        if message.get("is_final") and text:
            self._parts.append(text)
        utterance_over = message.get("speech_final") or (
            message.get("is_final") and len(so_far) >= MAX_UTTERANCE_CHARS
        )
        self._emit(so_far, is_final=bool(utterance_over), heard_until_s=heard_until_s)

    def _emit(self, text: str, *, is_final: bool, heard_until_s: float | None) -> None:
        sent_s = self._samples_sent / self._sample_rate
        if heard_until_s is None:
            heard_until_s = sent_s
        if text:
            self._results.put_nowait(
                Transcript(
                    utterance_id=self._utterance_id,
                    text=text,
                    is_final=is_final,
                    audio_ms=round((heard_until_s - self._utterance_start_s) * 1000),
                    # How far the text trails behind the audio that has been sent
                    process_ms=round(max(0.0, sent_s - heard_until_s) * 1000, 1),
                )
            )
        if is_final:
            if text:
                self._utterance_id += 1
            self._parts = []
            self._utterance_start_s = heard_until_s
