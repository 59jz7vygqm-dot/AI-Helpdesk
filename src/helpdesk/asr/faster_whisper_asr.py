"""faster-whisper (CTranslate2) recogniser.

``large-v3-turbo`` is the right trade for a phone agent: the decoder is 4 layers
instead of 32, so a short utterance transcribes in well under 200 ms on an L4,
while German accuracy stays close to large-v3.  Inference runs in a single-slot
thread pool because the GPU is shared with the LLM and TTS -- letting two
transcriptions overlap would only make both slower.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np

from ..audio.codec import pcm16_to_float32, rms_dbfs
from .base import Recognizer, Transcript

log = logging.getLogger(__name__)

#: Whisper reliably hallucinates these on silence or line noise.
HALLUCINATION_BLOCKLIST = {
    "",
    ".",
    "...",
    "untertitel im auftrag des zdf, 2021",
    "untertitel im auftrag des zdf, 2020",
    "untertitelung im auftrag des zdf, 2021",
    "untertitel der deutschen welle",
    "vielen dank für das zuschauen",
    "vielen dank fürs zuschauen",
    "vielen dank für ihre aufmerksamkeit",
    "das war's für heute",
    "copyright wdr",
    "amara.org",
    "untertitel von stephanie geiges",
    "thank you for watching",
    "thanks for watching",
    "subtitles by the amara.org community",
    "you",
    "bye",
    "mbc 뉴스 이덕영입니다",
}


def _looks_like_hallucination(text: str) -> bool:
    cleaned = text.strip().lower().strip(" .!?,-“”\"'")
    if cleaned in HALLUCINATION_BLOCKLIST:
        return True
    if "amara.org" in cleaned or "untertitel im auftrag" in cleaned:
        return True
    # A single repeated token ("ja ja ja ja ja") is a decode loop, not speech.
    words = cleaned.split()
    if len(words) >= 4 and len(set(words)) == 1:
        return True
    return False


class FasterWhisperRecognizer(Recognizer):
    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        *,
        device: str = "cuda",
        compute_type: str = "int8_float16",
        language: str = "de",
        beam_size: int = 1,
        vad_filter: bool = False,
        download_root: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        temperature: float = 0.0,
        min_avg_logprob: float = -1.1,
        max_no_speech_prob: float = 0.75,
        device_index: int = 0,
        num_workers: int = 1,
        cpu_threads: int = 4,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.download_root = download_root
        self.initial_prompt = initial_prompt
        self.temperature = temperature
        self.min_avg_logprob = min_avg_logprob
        self.max_no_speech_prob = max_no_speech_prob
        self.device_index = device_index
        self.num_workers = num_workers
        self.cpu_threads = cpu_threads

        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
        self._lock = asyncio.Lock()

    def _load(self):
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel  # noqa: PLC0415

        log.info(
            "loading ASR model %s (%s, %s)", self.model_size, self.device, self.compute_type
        )
        started = time.monotonic()
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            device_index=self.device_index,
            compute_type=self.compute_type,
            download_root=self.download_root,
            num_workers=self.num_workers,
            cpu_threads=self.cpu_threads,
        )
        log.info("ASR model ready in %.1fs", time.monotonic() - started)
        return self._model

    def _transcribe_sync(self, pcm: np.ndarray, language: Optional[str]) -> Transcript:
        model = self._load()
        audio = pcm16_to_float32(pcm)
        started = time.monotonic()
        segments, info = model.transcribe(
            audio,
            language=language or self.language or None,
            beam_size=self.beam_size,
            temperature=self.temperature,
            vad_filter=self.vad_filter,
            # The utterance is already endpointed, so conditioning on previous
            # text only invites the model to invent continuations.
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt,
            word_timestamps=False,
            suppress_blank=True,
        )
        parts: List[str] = []
        logprobs: List[float] = []
        no_speech: List[float] = []
        for segment in segments:
            parts.append(segment.text)
            logprobs.append(getattr(segment, "avg_logprob", 0.0))
            no_speech.append(getattr(segment, "no_speech_prob", 0.0))

        text = " ".join(part.strip() for part in parts if part.strip()).strip()
        avg_logprob = float(np.mean(logprobs)) if logprobs else 0.0
        no_speech_prob = float(np.max(no_speech)) if no_speech else 0.0
        level_db = rms_dbfs(pcm)

        reject = None
        if text:
            if _looks_like_hallucination(text):
                reject = "looks like a hallucination"
            elif avg_logprob < self.min_avg_logprob:
                reject = f"logprob {avg_logprob:.2f} < {self.min_avg_logprob}"
            elif no_speech_prob > self.max_no_speech_prob:
                reject = f"no_speech {no_speech_prob:.2f} > {self.max_no_speech_prob}"

        if reject:
            # At INFO: a discarded transcript is the most common reason a caller
            # feels unheard, and the threshold that caused it is the fix.
            log.info(
                "discarded transcript %r (%s; %.0f ms at %.1f dBFS)",
                text, reject, pcm.size * 1000 / 16000, level_db,
            )
            text = ""
        elif not text:
            log.info(
                "recognised nothing in %.0f ms of audio at %.1f dBFS "
                "(no_speech %.2f) -- too quiet, or the caller did not speak",
                pcm.size * 1000 / 16000, level_db, no_speech_prob,
            )

        return Transcript(
            text=text,
            language=getattr(info, "language", "") or (language or self.language),
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
            duration_ms=int(pcm.size * 1000 / 16000),
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    async def transcribe(self, pcm: np.ndarray, *, language: Optional[str] = None) -> Transcript:
        if pcm.size < 16000 * 0.12:  # under ~120 ms is never a real utterance
            return Transcript(text="", duration_ms=int(pcm.size * 1000 / 16000))
        loop = asyncio.get_running_loop()
        async with self._lock:
            return await loop.run_in_executor(self._executor, self._transcribe_sync, pcm, language)

    async def warmup(self) -> None:
        """Pay the model load and the first CUDA kernel launch before any call."""
        silence = np.zeros(16000, dtype=np.int16)
        result = await self.transcribe(silence)
        log.info("ASR warmup done (%d ms on 1s of silence)", result.latency_ms)

    async def close(self) -> None:
        self._executor.shutdown(wait=False)
        self._model = None
