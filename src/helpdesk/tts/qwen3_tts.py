"""Qwen3-TTS backend (package `qwen-tts`, Apache-2.0).

The best German voice available locally, with built-in speakers, zero-shot voice
cloning and voice design from a text description.

Written against the installed package's own source, not its documentation. Two
things that matters for:

* ``generate_custom_voice`` requires a speaker and validates it against
  ``get_supported_speakers()``, so an unset or wrong name is a hard error. When
  none is configured, the first supported speaker is used and logged.
* The language is likewise validated against ``get_supported_languages()``, so the
  value is looked up rather than guessed -- the alternative is a model that loads
  and then fails on the first call.

There is no streaming API: ``non_streaming_mode=False`` only simulates streaming
text input and the return is always ``(List[np.ndarray], sample_rate)``. That is
fine here, because the session synthesises one sentence at a time anyway.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator, Dict, List, Optional, Tuple

import numpy as np

from ..audio.codec import float32_to_pcm16
from .base import SpeechChunk, Synthesizer
from .text import spoken_form

log = logging.getLogger(__name__)

#: language spellings to try when the configured one is not supported
_LANGUAGE_GUESSES = ("German", "german", "de", "de-DE", "Deutsch")


class Qwen3TtsSynthesizer(Synthesizer):
    def __init__(
        self,
        *,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
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
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
        max_new_tokens: int = 0,
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
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens

        self._model = None
        self._clone_prompt = None
        self._lock = asyncio.Lock()

    # ---- loading -------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from qwen_tts import Qwen3TTSModel  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-TTS is not installed. The package is 'qwen-tts' (not "
                "'qwen3-tts', which is an unrelated Apple-Silicon CLI).\n"
                "Rebuild the image with TTS_PROFILE=qwen, or switch tts.backend "
                "to 'piper'."
            ) from exc

        import torch  # noqa: PLC0415

        dtype = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
        }.get(self.dtype_name.lower(), torch.bfloat16)

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"torch sees no GPU, so device={self.device!r} cannot work. "
                "Check that the container gets one, or set tts.qwen3.device to cpu "
                "(far too slow for calls)."
            )

        kwargs: Dict = {"device_map": self.device, "dtype": dtype}
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        log.info("loading Qwen3-TTS %s (%s, %s)", self.model_id, self.device, self.dtype_name)
        started = time.monotonic()
        try:
            self._model = Qwen3TTSModel.from_pretrained(self.model_id, **kwargs)
        except TypeError:
            # Older builds expect torch_dtype.
            self._model = Qwen3TTSModel.from_pretrained(
                self.model_id, device_map=self.device, torch_dtype=dtype
            )
        log.info("Qwen3-TTS loaded in %.1fs", time.monotonic() - started)

        self._resolve_language()
        self._resolve_speaker()
        self._prepare_clone()
        return self._model

    def _resolve_language(self) -> None:
        """Pick a language value the model actually accepts."""
        try:
            supported = self._model.get_supported_languages()
        except Exception:
            supported = None
        if not supported:
            return  # model imposes no constraint

        lowered = {str(s).lower(): str(s) for s in supported}
        for candidate in (self.language, *_LANGUAGE_GUESSES):
            if candidate and candidate.lower() in lowered:
                resolved = lowered[candidate.lower()]
                if resolved != self.language:
                    log.info("language %r -> %r", self.language, resolved)
                self.language = resolved
                return
        raise RuntimeError(
            f"none of {[self.language, *_LANGUAGE_GUESSES]} is supported by "
            f"{self.model_id}. Supported: {sorted(supported)}\n"
            "Set tts.qwen3.language to one of those."
        )

    def _resolve_speaker(self) -> None:
        """generate_custom_voice requires a valid speaker, so settle it now."""
        if self.mode != "custom":
            return
        try:
            supported = self._model.get_supported_speakers()
        except Exception:
            supported = None
        if not supported:
            if not self.speaker:
                log.warning(
                    "no speaker configured and the model lists none; "
                    "synthesis may fail"
                )
            return

        lowered = {str(s).lower(): str(s) for s in supported}
        if self.speaker and self.speaker.lower() in lowered:
            self.speaker = lowered[self.speaker.lower()]
        else:
            if self.speaker:
                log.warning(
                    "speaker %r is not supported, falling back", self.speaker
                )
            self.speaker = sorted(supported)[0]
            log.info(
                "using speaker %r (available: %s)",
                self.speaker,
                ", ".join(sorted(str(s) for s in supported)[:12]),
            )

    def _prepare_clone(self) -> None:
        if self.mode != "clone":
            return
        if not self.reference_audio:
            raise RuntimeError("tts.qwen3.mode=clone needs reference_audio")
        # Encoding the reference once keeps it off the per-sentence path.
        self._clone_prompt = self._model.create_voice_clone_prompt(
            self.reference_audio, self.reference_text or None
        )
        log.info("cached voice-clone prompt from %s", self.reference_audio)

    # ---- generation ----------------------------------------------------
    def _sampling_kwargs(self) -> Dict:
        kwargs: Dict = {
            "do_sample": True,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.max_new_tokens > 0:
            kwargs["max_new_tokens"] = self.max_new_tokens
        return kwargs

    def _as_pcm16(self, wavs) -> np.ndarray:
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
        samples = np.asarray(array, dtype=np.float32)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak > 1.001:
            # Already int16-scaled, just not typed that way.
            return np.clip(samples, -32768, 32767).astype(np.int16)
        return float32_to_pcm16(samples)

    def _generate(self, text: str) -> Tuple[np.ndarray, int]:
        model = self._load()
        kwargs = self._sampling_kwargs()

        if self.mode == "design":
            wavs, rate = model.generate_voice_design(
                text=text, instruct=self.instruct, language=self.language, **kwargs
            )
        elif self.mode == "clone":
            wavs, rate = model.generate_voice_clone(
                text=text,
                language=self.language,
                voice_clone_prompt=self._clone_prompt,
                **kwargs,
            )
        else:
            wavs, rate = model.generate_custom_voice(
                text=text,
                speaker=self.speaker,
                language=self.language,
                instruct=self.instruct or None,
                **kwargs,
            )
        return self._as_pcm16(wavs), int(rate or self.sample_rate)

    async def stream(
        self, text: str, *, cancel: Optional[asyncio.Event] = None
    ) -> AsyncIterator[SpeechChunk]:
        prepared = spoken_form(text)
        if not prepared:
            return
        if cancel is not None and cancel.is_set():
            return

        loop = asyncio.get_running_loop()
        async with self._lock:  # the GPU is shared with ASR and the LLM
            started = time.monotonic()
            pcm, rate = await loop.run_in_executor(None, self._generate, prepared)
            if cancel is not None and cancel.is_set():
                return
            self.sample_rate = rate
            audio_ms = pcm.size * 1000 / max(rate, 1)
            elapsed_ms = (time.monotonic() - started) * 1000
            rtf = elapsed_ms / audio_ms if audio_ms else 0.0
            log.log(
                logging.INFO if rtf > 0.5 else logging.DEBUG,
                "qwen3-tts: %d chars -> %d ms audio in %d ms (rtf %.2f)",
                len(prepared), audio_ms, elapsed_ms, rtf,
            )
            if pcm.size:
                yield SpeechChunk(pcm=pcm, sample_rate=rate, final=True)

    async def warmup(self) -> None:
        audio = await self.synthesize("Guten Tag, wie kann ich Ihnen helfen?")
        log.info(
            "Qwen3-TTS warm: %d Hz, speaker=%r, language=%r, %d ms of audio",
            self.sample_rate, self.speaker or "(default)", self.language,
            int(audio.size * 1000 / max(self.sample_rate, 1)),
        )

    def describe(self) -> Dict:
        """Speakers and languages this model supports, for diagnostics."""
        model = self._load()
        out: Dict = {"model": self.model_id, "speaker": self.speaker, "language": self.language}
        try:
            out["speakers"] = model.get_supported_speakers()
        except Exception:
            out["speakers"] = None
        try:
            out["languages"] = model.get_supported_languages()
        except Exception:
            out["languages"] = None
        return out

    async def close(self) -> None:
        self._model = None
        self._clone_prompt = None
