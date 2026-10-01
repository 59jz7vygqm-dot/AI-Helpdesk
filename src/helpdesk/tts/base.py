"""Text-to-speech interface.

Synthesis is streamed: the caller gets audio as soon as the first chunk exists,
and ``cancel_event`` lets a barge-in abandon the rest mid-sentence instead of
paying for audio nobody will hear.
"""

from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import numpy as np


@dataclass
class SpeechChunk:
    pcm: np.ndarray
    sample_rate: int
    #: True for the last chunk of this synthesis call
    final: bool = False


class Synthesizer(abc.ABC):
    #: native output rate; the session resamples to 8 kHz for the call
    sample_rate: int = 22050

    @abc.abstractmethod
    def stream(
        self, text: str, *, cancel: Optional[asyncio.Event] = None
    ) -> AsyncIterator[SpeechChunk]:
        ...

    async def synthesize(self, text: str, *, cancel: Optional[asyncio.Event] = None) -> np.ndarray:
        chunks = []
        async for chunk in self.stream(text, cancel=cancel):
            chunks.append(chunk.pcm)
        if not chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(chunks).astype(np.int16)

    async def warmup(self) -> None:
        return None

    async def close(self) -> None:
        return None
