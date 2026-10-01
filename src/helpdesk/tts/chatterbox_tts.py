"""Chatterbox Multilingual TTS backend (GPU).

This is the "sounds like a person" option: a 0.5B model with zero-shot voice
cloning, MIT licensed, with German among its languages.  It costs roughly 3 GB of
VRAM and several hundred milliseconds to first audio, so it is worth it only when
the output is streamed sentence by sentence -- which is how the session uses it.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import AsyncIterator, Optional

import numpy as np

from ..audio.codec import float32_to_pcm16
from .base import SpeechChunk, Synthesizer
from .text import spoken_form

log = logging.getLogger(__name__)


class ChatterboxSynthesizer(Synthesizer):
    def __init__(
        self,
        *,
        device: str = "cuda",
        language_id: str = "de",
        reference_audio: Optional[str] = None,
        exaggeration: float = 0.45,
        cfg_weight: float = 0.5,
        temperature: float = 0.6,
        chunk_tokens: int = 25,
        multilingual: bool = True,
    ) -> None:
        self.device = device
        self.language_id = language_id
        self.reference_audio = reference_audio
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.temperature = temperature
        self.chunk_tokens = chunk_tokens
        self.multilingual = multilingual
        self._model = None
        self.sample_rate = 24000
        self._lock = asyncio.Lock()

    def _load(self):
        if self._model is not None:
            return self._model
        started = time.monotonic()
        if self.multilingual:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS as Model  # noqa: PLC0415
        else:
            from chatterbox.tts import ChatterboxTTS as Model  # noqa: PLC0415
        log.info("loading Chatterbox (%s, multilingual=%s)", self.device, self.multilingual)
        self._model = Model.from_pretrained(device=self.device)
        self.sample_rate = int(getattr(self._model, "sr", 24000))
        log.info("Chatterbox ready in %.1fs (%d Hz)", time.monotonic() - started, self.sample_rate)
        return self._model

    def _kwargs(self) -> dict:
        kwargs = {
            "exaggeration": self.exaggeration,
            "cfg_weight": self.cfg_weight,
            "temperature": self.temperature,
        }
        if self.multilingual:
            kwargs["language_id"] = self.language_id
        if self.reference_audio:
            kwargs["audio_prompt_path"] = self.reference_audio
        return kwargs

    def _to_pcm16(self, tensor) -> np.ndarray:
        array = tensor.detach().float().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        return float32_to_pcm16(np.squeeze(array))

    def _produce(self, text: str, sink: "queue.Queue", cancel_flag: threading.Event) -> None:
        """Run synthesis in a worker thread, pushing chunks as they appear."""
        try:
            model = self._load()
            kwargs = self._kwargs()
            if hasattr(model, "generate_stream"):
                for item in model.generate_stream(
                    text, chunk_size=self.chunk_tokens, **kwargs
                ):
                    if cancel_flag.is_set():
                        break
                    audio = item[0] if isinstance(item, tuple) else item
                    sink.put(self._to_pcm16(audio))
            else:
                # No streaming API in this build: one shot, still usable because
                # the session feeds it one sentence at a time.
                sink.put(self._to_pcm16(model.generate(text, **kwargs)))
        except Exception as exc:  # pragma: no cover - depends on the model build
            log.exception("Chatterbox synthesis failed")
            sink.put(exc)
        finally:
            sink.put(None)

    async def stream(
        self, text: str, *, cancel: Optional[asyncio.Event] = None
    ) -> AsyncIterator[SpeechChunk]:
        prepared = spoken_form(text)
        if not prepared:
            return
        if cancel is not None and cancel.is_set():
            return

        loop = asyncio.get_running_loop()
        async with self._lock:
            sink: "queue.Queue" = queue.Queue()
            cancel_flag = threading.Event()
            worker = threading.Thread(
                target=self._produce, args=(prepared, sink, cancel_flag), daemon=True
            )
            started = time.monotonic()
            worker.start()
            first = True
            try:
                while True:
                    if cancel is not None and cancel.is_set():
                        cancel_flag.set()
                        break
                    try:
                        item = await loop.run_in_executor(None, sink.get, True, 0.1)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    if first:
                        log.debug(
                            "chatterbox first audio after %d ms", int((time.monotonic() - started) * 1000)
                        )
                        first = False
                    if item.size:
                        yield SpeechChunk(pcm=item, sample_rate=self.sample_rate)
            finally:
                cancel_flag.set()

    async def warmup(self) -> None:
        await self.synthesize("Guten Tag, wie kann ich helfen?")
        log.info("Chatterbox warmup done")

    async def close(self) -> None:
        self._model = None
