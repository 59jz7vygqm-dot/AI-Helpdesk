"""Piper TTS backend.

Piper is the latency floor: a VITS model running on CPU at roughly 0.05x
realtime, so a short sentence is synthesised in tens of milliseconds and the GPU
stays free for the recogniser and the LLM.  It does not sound like a person --
see the README -- but it never stalls, which is why it is the default.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import AsyncIterator, List, Optional

import numpy as np

from .base import SpeechChunk, Synthesizer
from .text import spoken_form

log = logging.getLogger(__name__)


class PiperSynthesizer(Synthesizer):
    def __init__(
        self,
        model_path: str,
        *,
        config_path: Optional[str] = None,
        speaker_id: Optional[int] = None,
        length_scale: float = 1.0,
        threads: int = 0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
        use_cuda: bool = False,
        normalize_text: bool = True,
    ) -> None:
        self.model_path = model_path
        self.config_path = config_path
        self.speaker_id = speaker_id
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self.use_cuda = use_cuda
        self.normalize_text = normalize_text
        self.threads = threads
        self._voice = None
        self.sample_rate = 22050
        self._lock = asyncio.Lock()

    def _load(self):
        if self._voice is not None:
            return self._voice
        from piper import PiperVoice  # noqa: PLC0415

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Piper voice not found: {self.model_path}. Run scripts/download_models.sh"
            )
        if self.threads > 0:
            # onnxruntime otherwise picks a default that can leave cores idle.
            os.environ.setdefault("OMP_NUM_THREADS", str(self.threads))
        log.info(
            "loading Piper voice %s (cuda=%s, threads=%s)",
            self.model_path, self.use_cuda, self.threads or "auto",
        )
        self._voice = PiperVoice.load(
            self.model_path, config_path=self.config_path, use_cuda=self.use_cuda
        )
        self.sample_rate = int(getattr(self._voice.config, "sample_rate", 22050))
        return self._voice

    def _synthesize_sync(self, text: str) -> np.ndarray:
        voice = self._load()
        # Piper's API changed shape across versions; support both the
        # synthesize()-generator form and the older raw-stream form.
        if hasattr(voice, "synthesize"):
            try:
                pieces: List[np.ndarray] = []
                for chunk in voice.synthesize(text):
                    audio = getattr(chunk, "audio_int16_array", None)
                    if audio is None:
                        raw = getattr(chunk, "audio_int16_bytes", None)
                        if raw is None:
                            continue
                        audio = np.frombuffer(raw, dtype=np.int16)
                    rate = getattr(chunk, "sample_rate", None)
                    if rate:
                        self.sample_rate = int(rate)
                    pieces.append(np.asarray(audio, dtype=np.int16))
                if pieces:
                    return np.concatenate(pieces)
            except TypeError:
                pass

        if hasattr(voice, "synthesize_stream_raw"):
            pieces = [
                np.frombuffer(raw, dtype=np.int16)
                for raw in voice.synthesize_stream_raw(text)
            ]
            if pieces:
                return np.concatenate(pieces)
        return np.zeros(0, dtype=np.int16)

    async def stream(
        self, text: str, *, cancel: Optional[asyncio.Event] = None
    ) -> AsyncIterator[SpeechChunk]:
        prepared = spoken_form(text) if self.normalize_text else text.strip()
        if not prepared:
            return
        if cancel is not None and cancel.is_set():
            return
        loop = asyncio.get_running_loop()
        started = time.monotonic()
        async with self._lock:
            pcm = await loop.run_in_executor(None, self._synthesize_sync, prepared)
        if cancel is not None and cancel.is_set():
            return
        elapsed_ms = int((time.monotonic() - started) * 1000)
        audio_ms = int(pcm.size * 1000 / max(self.sample_rate, 1))
        rtf = elapsed_ms / audio_ms if audio_ms else 0.0
        # A realtime factor above ~0.3 means synthesis is the latency bottleneck;
        # the usual cause is a "high" voice where "medium" would do at 8 kHz.
        log.log(
            logging.INFO if rtf > 0.3 else logging.DEBUG,
            "piper: %d chars -> %d ms audio in %d ms (rtf %.2f)",
            len(prepared), audio_ms, elapsed_ms, rtf,
        )
        if pcm.size:
            yield SpeechChunk(pcm=pcm, sample_rate=self.sample_rate, final=True)

    async def warmup(self) -> None:
        await self.synthesize("Guten Tag.")
        log.info("Piper warmup done (%d Hz)", self.sample_rate)
