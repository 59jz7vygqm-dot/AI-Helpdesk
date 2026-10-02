"""OpenAI-compatible TTS backend.

Lets any local server exposing ``/v1/audio/speech`` (openedai-speech, a Kokoro or
XTTS wrapper, LocalAI) be used without touching the session code.  Requests PCM
so there is nothing to decode, and streams the response body so audio starts
flowing before the whole sentence is rendered.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

import numpy as np

from .base import SpeechChunk, Synthesizer
from .text import spoken_form

log = logging.getLogger(__name__)


class OpenAiCompatibleSynthesizer(Synthesizer):
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8880/v1",
        *,
        model: str = "tts-1",
        voice: str = "de_female",
        api_key: str = "none",
        response_format: str = "pcm",
        language: str = "",
        request_stream: bool = True,
        sample_rate: int = 24000,
        speed: float = 1.0,
        timeout: float = 30.0,
        extra_body: Optional[dict] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.voice = voice
        self.api_key = api_key
        self.response_format = response_format
        #: Voxtral TTS needs the language named; servers that do not know the
        #: field would reject it, so it is only sent when set.
        self.language = language
        #: Without this vLLM-Omni renders the whole sentence before answering,
        #: which throws away the entire point of a streaming backend. Not named
        #: `stream`: that is this class's own method, and assigning over it
        #: turns every synthesis into "'bool' object is not callable".
        self.request_stream = request_stream
        self.sample_rate = sample_rate
        self.speed = speed
        self.timeout = timeout
        self.extra_body = extra_body or {}
        self._session = None

    async def _get_session(self):
        import aiohttp  # noqa: PLC0415

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def stream(
        self, text: str, *, cancel: Optional[asyncio.Event] = None
    ) -> AsyncIterator[SpeechChunk]:
        prepared = spoken_form(text)
        if not prepared:
            return
        session = await self._get_session()
        payload = {
            "model": self.model,
            "voice": self.voice,
            "input": prepared,
            "response_format": self.response_format,
        }
        # Only fields the server asked for. A strict server answers 400 to a
        # field it does not know, and a TTS backend that 400s is a silent call.
        if self.request_stream:
            payload["stream"] = True
        if self.language:
            payload["language"] = self.language
        if abs(self.speed - 1.0) > 1e-6:
            payload["speed"] = self.speed
        payload.update(self.extra_body)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with session.post(
            f"{self.base_url}/audio/speech", json=payload, headers=headers
        ) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"TTS server returned {response.status}: {body[:200]}")
            if self.response_format != "pcm":
                raise RuntimeError(
                    f"response_format must be 'pcm' for low latency, got {self.response_format!r}"
                )
            tail = b""
            async for data in response.content.iter_chunked(3200):
                if cancel is not None and cancel.is_set():
                    break
                buffer = tail + data
                # int16 samples must not be split across chunk boundaries.
                usable = len(buffer) - (len(buffer) % 2)
                tail = buffer[usable:]
                if usable:
                    yield SpeechChunk(
                        pcm=np.frombuffer(buffer[:usable], dtype=np.int16),
                        sample_rate=self.sample_rate,
                    )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
