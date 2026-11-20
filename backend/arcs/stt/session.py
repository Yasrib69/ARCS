"""A live captioning session: audio goes in, transcripts come out.

Every speech provider, on this machine or in the cloud, is driven through this
one interface, so the server doesn't care which is in use.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ..schemas import Transcript
from .stream import CaptionStream


class CaptionSession(ABC):
    """Holds the state for one audio stream. Create one per microphone/connection."""

    def __init__(self):
        # None marks the end of the session
        self._results: asyncio.Queue[Transcript | None] = asyncio.Queue()

    @abstractmethod
    async def start(self) -> None:
        """Get ready to receive audio. Raises if the provider can't be reached."""

    @abstractmethod
    async def add_audio(self, pcm: bytes) -> None:
        """Send the next chunk of 16-bit little-endian mono PCM."""

    @abstractmethod
    async def stop(self) -> None:
        """Finish cleanly: whatever was said so far comes out as a final transcript."""

    @abstractmethod
    async def close(self) -> None:
        """Release everything immediately. Safe to call more than once."""

    async def transcripts(self) -> AsyncIterator[Transcript]:
        """Yield transcripts as they are produced, until the session ends."""
        while (transcript := await self._results.get()) is not None:
            yield transcript


class LocalCaptionSession(CaptionSession):
    """Captions from a model running on this machine (see CaptionStream)."""

    def __init__(self, stream: CaptionStream, interval_s: float):
        super().__init__()
        self._stream = stream
        self._interval_s = interval_s
        self._stopping = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def add_audio(self, pcm: bytes) -> None:
        self._stream.add_audio(pcm)

    async def stop(self) -> None:
        self._stopping.set()
        if self._task:
            await self._task

    async def close(self) -> None:
        if self._task:
            self._task.cancel()

    async def _run(self) -> None:
        try:
            while not self._stopping.is_set():
                try:
                    await asyncio.wait_for(self._stopping.wait(), self._interval_s)
                except TimeoutError:
                    pass
                # The model blocks, so it runs in a worker thread
                transcript = await asyncio.to_thread(self._stream.poll, self._stopping.is_set())
                if transcript:
                    self._results.put_nowait(transcript)
        finally:
            self._results.put_nowait(None)
