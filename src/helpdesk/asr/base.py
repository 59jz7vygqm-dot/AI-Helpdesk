"""Speech recognition interface."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Transcript:
    text: str
    language: str = ""
    #: mean log-probability; used to ignore hallucinated output on silence
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    duration_ms: int = 0
    latency_ms: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def is_confident(self) -> bool:
        """Whether this is solid enough to act on irreversibly.

        Short audio with a mediocre log-probability is exactly what produces an
        invented stock phrase, and acting on one by hanging up is the worst thing
        the agent can do to a caller.
        """
        if self.is_empty:
            return False
        if self.duration_ms < 600:
            return False
        return self.avg_logprob > -0.9


class Recognizer(abc.ABC):
    """16 kHz mono int16 in, text out."""

    @abc.abstractmethod
    async def transcribe(self, pcm: np.ndarray, *, language: Optional[str] = None) -> Transcript:
        ...

    async def warmup(self) -> None:
        return None

    async def close(self) -> None:
        return None
