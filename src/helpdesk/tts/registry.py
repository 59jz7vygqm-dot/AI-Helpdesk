"""TTS backend selection and a cache for fixed phrases.

Anything the agent says verbatim every call -- the greeting, the hold line, the
transfer announcement -- is synthesised once at startup and replayed from memory,
so those turns have zero synthesis latency.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Dict, Optional

import numpy as np

from ..audio.codec import resample
from .base import Synthesizer

log = logging.getLogger(__name__)


def build_synthesizer(config: dict) -> Synthesizer:
    backend = (config.get("backend") or "piper").lower()

    if backend == "piper":
        from .piper_tts import PiperSynthesizer

        piper = config.get("piper", {}) or {}
        return PiperSynthesizer(
            model_path=piper.get("model_path", "/models/piper/de_DE-thorsten-high.onnx"),
            config_path=piper.get("config_path") or None,
            speaker_id=piper.get("speaker_id"),
            length_scale=float(piper.get("length_scale", 1.0)),
            noise_scale=float(piper.get("noise_scale", 0.667)),
            noise_w=float(piper.get("noise_w", 0.8)),
            use_cuda=bool(piper.get("use_cuda", False)),
            threads=int(piper.get("threads", 0)),
        )

    if backend in ("qwen3", "qwen3-tts", "qwen"):
        from .qwen3_tts import Qwen3TtsSynthesizer

        q = config.get("qwen3", {}) or {}
        return Qwen3TtsSynthesizer(
            model_id=q.get("model_id", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
            device=q.get("device", "cuda:0"),
            dtype=q.get("dtype", "bfloat16"),
            language=q.get("language", "German"),
            mode=q.get("mode", "custom"),
            speaker=q.get("speaker", "") or "",
            instruct=q.get("instruct", "") or "",
            reference_audio=q.get("reference_audio", "") or "",
            reference_text=q.get("reference_text", "") or "",
            attn_implementation=q.get("attn_implementation", "") or "",
            sample_rate=int(q.get("sample_rate", 24000)),
            streaming=bool(q.get("streaming", True)),
        )

    if backend == "chatterbox":
        from .chatterbox_tts import ChatterboxSynthesizer

        cb = config.get("chatterbox", {}) or {}
        return ChatterboxSynthesizer(
            device=cb.get("device", "cuda"),
            language_id=cb.get("language_id", "de"),
            reference_audio=cb.get("reference_audio") or None,
            exaggeration=float(cb.get("exaggeration", 0.45)),
            cfg_weight=float(cb.get("cfg_weight", 0.5)),
            temperature=float(cb.get("temperature", 0.6)),
            chunk_tokens=int(cb.get("chunk_tokens", 25)),
            multilingual=bool(cb.get("multilingual", True)),
        )

    if backend in ("openai", "openai_compatible", "http"):
        from .openai_tts import OpenAiCompatibleSynthesizer

        oa = config.get("openai", {}) or {}
        return OpenAiCompatibleSynthesizer(
            base_url=oa.get("base_url", "http://127.0.0.1:8880/v1"),
            model=oa.get("model", "tts-1"),
            voice=oa.get("voice", "de_female"),
            api_key=oa.get("api_key", "none"),
            response_format=oa.get("response_format", "pcm"),
            sample_rate=int(oa.get("sample_rate", 24000)),
            speed=float(oa.get("speed", 1.0)),
            extra_body=oa.get("extra_body") or {},
        )

    raise ValueError(
        f"unknown TTS backend: {backend!r} "
        "(expected piper, qwen3, chatterbox or openai)"
    )


class PhraseCache:
    """Pre-rendered 8 kHz audio for fixed phrases."""

    def __init__(self, synthesizer: Synthesizer, target_rate: int = 8000, cache_dir: Optional[str] = None) -> None:
        self.synthesizer = synthesizer
        self.target_rate = target_rate
        self.cache_dir = cache_dir
        self._audio: Dict[str, np.ndarray] = {}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _disk_path(self, text: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        digest = hashlib.sha1(
            f"{type(self.synthesizer).__name__}|{self.target_rate}|{text}".encode("utf-8")
        ).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{digest}.npy")

    async def prepare(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(0, dtype=np.int16)
        if text in self._audio:
            return self._audio[text]

        path = self._disk_path(text)
        if path and os.path.exists(path):
            try:
                audio = np.load(path).astype(np.int16)
                self._audio[text] = audio
                return audio
            except Exception:  # pragma: no cover - corrupt cache file
                log.warning("ignoring unreadable phrase cache %s", path)

        raw = await self.synthesizer.synthesize(text)
        audio = resample(raw, self.synthesizer.sample_rate, self.target_rate)
        self._audio[text] = audio
        if path:
            try:
                np.save(path, audio)
            except Exception:  # pragma: no cover
                log.debug("could not write phrase cache %s", path, exc_info=True)
        log.info("cached phrase (%d ms): %r", int(audio.size * 1000 / self.target_rate), text[:60])
        return audio

    async def prepare_all(self, texts) -> None:
        for text in texts:
            if text:
                await self.prepare(text)

    def get(self, text: str) -> Optional[np.ndarray]:
        return self._audio.get(text)
