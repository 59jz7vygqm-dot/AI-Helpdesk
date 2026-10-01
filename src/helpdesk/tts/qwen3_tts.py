"""Qwen3-TTS backend -- the quality option.

Qwen3-TTS (Qwen team, January 2026, Apache-2.0) is the strongest open model that
covers German with voice cloning and is licensed for commercial use.  Two sizes
matter here:

* ``0.6B`` -- about 4 GB of VRAM, the sensible choice when the LLM is also large
* ``1.7B`` -- about 8 GB, noticeably better prosody

Three ways to pick a voice:

* ``custom`` -- one of the model's built-in speakers (``speaker``)
* ``clone``  -- a few seconds of reference audio plus its transcript
* ``design`` -- describe the voice in words (``instruct``), needs a VoiceDesign model

Latency note: the published 97 ms figure is for the model's own streaming path.
This wrapper uses it when the installed build exposes one, and otherwise
synthesises per sentence -- which the session already does, so a sentence of
speech is ready while the model is still generating the next one.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import AsyncIterator, Optional, Tuple

import numpy as np

from ..audio.codec import float32_to_pcm16
from .base import SpeechChunk, Synthesizer
from .text import spoken_form

log = logging.getLogger(__name__)

#: generator-style streaming methods seen across builds, tried in order
_STREAM_METHODS = (
    "generate_custom_voice_stream",
    "generate_voice_clone_stream",
    "generate_stream",
    "stream_generate",
)


class Qwen3TtsSynthesizer(Synthesizer):
    def __init__(
        self,
        *,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        language: str = "German",
        mode: str = "custom",
        speaker: str = "",
        instruct: str = "",
        reference_audio: str = "",
        reference_text: str = "",
        attn_implementation: str = "",
        sample_rate: int = 24000,
        streaming: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype_name = dtype
        self.language = language
        self.mode = (mode or "custom").lower()
        self.speaker = speaker
        self.instruct = instruct
        self.reference_audio = reference_audio
        self.reference_text = reference_text
        self.attn_implementation = attn_implementation
        self.sample_rate = sample_rate
        self.streaming = streaming

        self._model = None
        self._clone_prompt = None
        self._stream_method: Optional[str] = None
        self._lock = asyncio.Lock()

    # ---- loading -------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from qwen_tts import Qwen3TTSModel  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Qwen3-TTS is not installed. Install it with:\n"
                "  pip install qwen3-tts\n"
                "or switch tts.backend to 'piper' or 'chatterbox'."
            ) from exc

        import torch  # noqa: PLC0415

        dtype = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
        }.get(self.dtype_name.lower(), torch.bfloat16)

        kwargs = {"device_map": self.device, "dtype": dtype}
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        log.info("loading Qwen3-TTS %s (%s, %s)", self.model_id, self.device, self.dtype_name)
        started = time.monotonic()
        try:
            self._model = Qwen3TTSModel.from_pretrained(self.model_id, **kwargs)
        except TypeError:
            # Older signatures used torch_dtype and no attn_implementation.
            self._model = Qwen3TTSModel.from_pretrained(
                self.model_id, device_map=self.device, torch_dtype=dtype
            )
        log.info("Qwen3-TTS ready in %.1fs", time.monotonic() - started)

        if self.mode == "clone":
            if not self.reference_audio:
                raise RuntimeError("tts.qwen3.mode=clone needs reference_audio")
            if hasattr(self._model, "create_voice_clone_prompt"):
                # Encoding the reference once saves that work on every sentence.
                self._clone_prompt = self._model.create_voice_clone_prompt(
                    self.reference_audio, self.reference_text
                )
                log.info("cached voice-clone prompt from %s", self.reference_audio)

        if self.streaming:
            for name in _STREAM_METHODS:
                if hasattr(self._model, name):
                    self._stream_method = name
                    log.info("Qwen3-TTS streaming via %s()", name)
                    break
            else:
                log.info(
                    "this Qwen3-TTS build exposes no streaming method; "
                    "synthesising per sentence instead"
                )
        return self._model

    # ---- generation ----------------------------------------------------
    def _as_pcm16(self, wavs) -> np.ndarray:
        """Normalise whatever the model returns into one int16 array."""
        array = wavs
        if isinstance(array, (list, tuple)):
            if not array:
                return np.zeros(0, dtype=np.int16)
            array = array[0]
        if hasattr(array, "detach"):
            array = array.detach().float().cpu().numpy()
        array = np.squeeze(np.asarray(array))
        if array.ndim > 1:
            array = array[0]
        if array.dtype == np.int16:
            return array
        return float32_to_pcm16(array.astype(np.float32))

    def _call_kwargs(self) -> dict:
        if self.mode == "design":
            return {"text": None, "language": self.language, "instruct": self.instruct}
        if self.mode == "clone":
            return {
                "text": None,
                "language": self.language,
                "ref_audio": self.reference_audio,
                "ref_text": self.reference_text,
            }
        kwargs = {"text": None, "language": self.language, "speaker": self.speaker}
        if self.instruct:
            kwargs["instruct"] = self.instruct
        return kwargs

    def _generate(self, text: str) -> Tuple[np.ndarray, int]:
        model = self._load()
        kwargs = self._call_kwargs()
        kwargs["text"] = text

        if self.mode == "design":
            method = getattr(model, "generate_voice_design")
        elif self.mode == "clone":
            method = getattr(model, "generate_voice_clone")
            if self._clone_prompt is not None:
                # Reuse the cached reference encoding where the build allows it.
                try:
                    wavs, rate = method(
                        text=text, language=self.language, voice_clone_prompt=self._clone_prompt
                    )
                    return self._as_pcm16(wavs), int(rate or self.sample_rate)
                except TypeError:
                    pass
        else:
            method = getattr(model, "generate_custom_voice")

        result = method(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            wavs, rate = result
            return self._as_pcm16(wavs), int(rate or self.sample_rate)
        return self._as_pcm16(result), self.sample_rate

    def _produce_streaming(self, text: str, sink: "queue.Queue", stop: threading.Event) -> None:
        try:
            model = self._load()
            method = getattr(model, self._stream_method)
            kwargs = self._call_kwargs()
            kwargs["text"] = text
            for item in method(**kwargs):
                if stop.is_set():
                    break
                rate = self.sample_rate
                audio = item
                if isinstance(item, tuple):
                    audio = item[0]
                    if len(item) > 1 and isinstance(item[1], (int, float)) and item[1] > 1000:
                        rate = int(item[1])
                sink.put((self._as_pcm16(audio), rate))
        except Exception as exc:  # pragma: no cover - depends on the build
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
            # Load outside the stream path so a failure is reported once, clearly.
            await loop.run_in_executor(None, self._load)

            if self._stream_method:
                sink: "queue.Queue" = queue.Queue()
                stop = threading.Event()
                worker = threading.Thread(
                    target=self._produce_streaming, args=(prepared, sink, stop), daemon=True
                )
                started = time.monotonic()
                worker.start()
                first = True
                try:
                    while True:
                        if cancel is not None and cancel.is_set():
                            stop.set()
                            return
                        try:
                            item = await loop.run_in_executor(None, sink.get, True, 0.1)
                        except queue.Empty:
                            continue
                        if item is None:
                            return
                        if isinstance(item, Exception):
                            raise item
                        pcm, rate = item
                        if first:
                            log.debug(
                                "qwen3-tts first audio after %d ms",
                                int((time.monotonic() - started) * 1000),
                            )
                            first = False
                        if pcm.size:
                            self.sample_rate = rate
                            yield SpeechChunk(pcm=pcm, sample_rate=rate)
                finally:
                    stop.set()
                return

            started = time.monotonic()
            pcm, rate = await loop.run_in_executor(None, self._generate, prepared)
            if cancel is not None and cancel.is_set():
                return
            self.sample_rate = rate
            log.debug(
                "qwen3-tts synthesized %d chars -> %d ms audio in %d ms",
                len(prepared),
                int(pcm.size * 1000 / max(rate, 1)),
                int((time.monotonic() - started) * 1000),
            )
            if pcm.size:
                yield SpeechChunk(pcm=pcm, sample_rate=rate, final=True)

    async def warmup(self) -> None:
        audio = await self.synthesize("Guten Tag, wie kann ich Ihnen helfen?")
        log.info(
            "Qwen3-TTS warmup done (%d Hz, %d ms of audio)",
            self.sample_rate,
            int(audio.size * 1000 / max(self.sample_rate, 1)),
        )

    async def close(self) -> None:
        self._model = None
        self._clone_prompt = None
