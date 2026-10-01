#!/usr/bin/env python3
"""Qwen3-TTS behind an OpenAI-compatible /v1/audio/speech endpoint.

Exists as a separate service for one reason: the qwen3-tts package requires
Python 3.13, while the agent image is built on a CUDA base with 3.10. Speaking to
it over HTTP keeps both dependency sets intact, and the agent already has an
openai TTS backend that consumes exactly this interface.

Returns raw little-endian int16 PCM, which is what the agent asks for: anything
else would have to be decoded before it could be put on the call.

Configuration is by environment variable:

  QWEN_MODEL_ID     Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
  QWEN_DEVICE       cuda:0
  QWEN_DTYPE        bfloat16
  QWEN_LANGUAGE     German          (some builds expect "de" -- see /health)
  QWEN_MODE         custom | clone | design
  QWEN_SPEAKER      built-in speaker name for mode=custom
  QWEN_INSTRUCT     voice description for mode=design
  QWEN_REF_AUDIO    reference audio path for mode=clone
  QWEN_REF_TEXT     transcript of that reference audio
  QWEN_PORT         8880
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
from aiohttp import web

logging.basicConfig(
    level=os.environ.get("QWEN_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("qwen-tts")

MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
DEVICE = os.environ.get("QWEN_DEVICE", "cuda:0")
DTYPE = os.environ.get("QWEN_DTYPE", "bfloat16")
LANGUAGE = os.environ.get("QWEN_LANGUAGE", "German")
MODE = os.environ.get("QWEN_MODE", "custom").lower()
SPEAKER = os.environ.get("QWEN_SPEAKER", "")
INSTRUCT = os.environ.get("QWEN_INSTRUCT", "")
REF_AUDIO = os.environ.get("QWEN_REF_AUDIO", "")
REF_TEXT = os.environ.get("QWEN_REF_TEXT", "")
PORT = int(os.environ.get("QWEN_PORT", "8880"))
DEFAULT_RATE = int(os.environ.get("QWEN_SAMPLE_RATE", "24000"))

#: language value that actually worked, discovered at warmup
_language_in_use = LANGUAGE
_sample_rate = DEFAULT_RATE
_model = None
_lock = asyncio.Lock()
_clone_prompt = None


def _torch_dtype():
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
    }.get(DTYPE.lower(), torch.bfloat16)


def load_model():
    """Load once, tolerating the signature differences between builds."""
    global _model, _clone_prompt
    if _model is not None:
        return _model

    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise SystemExit(
            f"qwen3-tts is not importable: {exc}\n"
            "The image should have installed it; check the build log."
        ) from exc

    log.info("loading %s on %s (%s)", MODEL_ID, DEVICE, DTYPE)
    started = time.monotonic()
    kwargs = {"device_map": DEVICE, "dtype": _torch_dtype()}
    try:
        _model = Qwen3TTSModel.from_pretrained(MODEL_ID, **kwargs)
    except TypeError:
        # Older signature used torch_dtype.
        _model = Qwen3TTSModel.from_pretrained(
            MODEL_ID, device_map=DEVICE, torch_dtype=_torch_dtype()
        )
    log.info("model ready in %.1fs", time.monotonic() - started)

    if MODE == "clone":
        if not REF_AUDIO:
            raise SystemExit("QWEN_MODE=clone needs QWEN_REF_AUDIO")
        if hasattr(_model, "create_voice_clone_prompt"):
            # Encode the reference once rather than on every sentence.
            _clone_prompt = _model.create_voice_clone_prompt(REF_AUDIO, REF_TEXT)
            log.info("cached voice-clone prompt from %s", REF_AUDIO)
    return _model


def _as_pcm16(wavs) -> np.ndarray:
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
    scaled = np.asarray(array, dtype=np.float32)
    peak = float(np.max(np.abs(scaled))) if scaled.size else 0.0
    if peak > 1.001:
        # Already in int16 range, just not typed as such.
        return np.clip(scaled, -32768, 32767).astype(np.int16)
    return np.clip(scaled * 32767.0, -32768, 32767).astype(np.int16)


def _generate(text: str, language: str) -> Tuple[np.ndarray, int]:
    model = load_model()

    if MODE == "design":
        result = model.generate_voice_design(
            text=text, language=language, instruct=INSTRUCT
        )
    elif MODE == "clone":
        if _clone_prompt is not None:
            try:
                result = model.generate_voice_clone(
                    text=text, language=language, voice_clone_prompt=_clone_prompt
                )
            except TypeError:
                result = model.generate_voice_clone(
                    text=text, language=language, ref_audio=REF_AUDIO, ref_text=REF_TEXT
                )
        else:
            result = model.generate_voice_clone(
                text=text, language=language, ref_audio=REF_AUDIO, ref_text=REF_TEXT
            )
    else:
        kwargs = {"text": text, "language": language}
        if SPEAKER:
            kwargs["speaker"] = SPEAKER
        if INSTRUCT:
            kwargs["instruct"] = INSTRUCT
        try:
            result = model.generate_custom_voice(**kwargs)
        except TypeError:
            kwargs.pop("instruct", None)
            result = model.generate_custom_voice(**kwargs)

    if isinstance(result, tuple) and len(result) == 2:
        wavs, rate = result
        return _as_pcm16(wavs), int(rate or DEFAULT_RATE)
    return _as_pcm16(result), DEFAULT_RATE


def warmup() -> None:
    """Synthesise once at startup, and find a language value that works.

    The accepted spelling differs between builds ("German" vs "de"), and failing
    here with a clear message beats failing on a live call.
    """
    global _language_in_use, _sample_rate

    candidates = [LANGUAGE]
    for alternative in ("German", "de", "German (Germany)", "de-DE"):
        if alternative not in candidates:
            candidates.append(alternative)

    last_error: Optional[Exception] = None
    for language in candidates:
        try:
            started = time.monotonic()
            pcm, rate = _generate("Guten Tag, wie kann ich Ihnen helfen?", language)
            if pcm.size == 0:
                raise RuntimeError("model returned no audio")
            _language_in_use = language
            _sample_rate = rate
            audio_ms = pcm.size * 1000 / max(rate, 1)
            elapsed_ms = (time.monotonic() - started) * 1000
            log.info(
                "warmup ok with language=%r: %.0f ms audio at %d Hz in %.0f ms (rtf %.2f)",
                language, audio_ms, rate, elapsed_ms, elapsed_ms / max(audio_ms, 1),
            )
            return
        except Exception as exc:
            log.warning("language=%r did not work: %s", language, exc)
            last_error = exc

    raise SystemExit(
        f"Qwen3-TTS could not synthesise with any of {candidates}.\n"
        f"Last error: {last_error}\n"
        "Check the model card for the expected language value and speaker names, "
        "then set QWEN_LANGUAGE / QWEN_SPEAKER accordingly."
    )


# ---- HTTP ---------------------------------------------------------------
async def handle_speech(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception:
        raise web.HTTPBadRequest(text="expected a JSON body")

    text = (payload.get("input") or "").strip()
    if not text:
        raise web.HTTPBadRequest(text="'input' is required")

    response_format = (payload.get("response_format") or "pcm").lower()
    if response_format != "pcm":
        raise web.HTTPBadRequest(
            text=f"only response_format=pcm is supported, got {response_format!r}"
        )

    # A per-request voice overrides the configured speaker, as the OpenAI API does.
    voice = payload.get("voice") or ""
    language = payload.get("language") or _language_in_use

    loop = asyncio.get_running_loop()
    started = time.monotonic()
    async with _lock:  # one at a time: the GPU is shared with ASR and the LLM
        previous_speaker = globals()["SPEAKER"]
        if voice and voice != previous_speaker and MODE == "custom":
            globals()["SPEAKER"] = voice
        try:
            pcm, rate = await loop.run_in_executor(None, _generate, text, language)
        except Exception as exc:
            log.exception("synthesis failed for %r", text[:60])
            raise web.HTTPInternalServerError(text=f"synthesis failed: {exc}")
        finally:
            globals()["SPEAKER"] = previous_speaker

    audio_ms = pcm.size * 1000 / max(rate, 1)
    elapsed_ms = (time.monotonic() - started) * 1000
    log.info(
        "%d chars -> %.0f ms audio in %.0f ms (rtf %.2f)",
        len(text), audio_ms, elapsed_ms, elapsed_ms / max(audio_ms, 1),
    )

    return web.Response(
        body=pcm.astype(np.int16).tobytes(),
        content_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(rate),
            "X-Audio-Ms": f"{audio_ms:.0f}",
        },
    )


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "status": "ok" if _model is not None else "loading",
            "model": MODEL_ID,
            "mode": MODE,
            "language": _language_in_use,
            "speaker": SPEAKER or None,
            "sample_rate": _sample_rate,
        }
    )


async def handle_models(request: web.Request) -> web.Response:
    return web.json_response(
        {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}
    )


async def handle_voices(request: web.Request) -> web.Response:
    """Whatever speaker roster this build exposes, for QWEN_SPEAKER."""
    model = load_model()
    found = {}
    for attribute in ("speakers", "speaker_list", "available_speakers", "spk_list", "voices"):
        value = getattr(model, attribute, None)
        if value:
            found[attribute] = list(value.keys() if isinstance(value, dict) else value)
    config = getattr(model, "config", None)
    if config is not None:
        for attribute in ("speakers", "speaker_list", "voices"):
            value = getattr(config, attribute, None)
            if value:
                found[f"config.{attribute}"] = list(
                    value.keys() if isinstance(value, dict) else value
                )
    if not found:
        found["note"] = (
            f"no speaker roster on this build; see https://huggingface.co/{MODEL_ID}"
        )
    return web.json_response(found)


def main() -> int:
    load_model()
    warmup()

    app = web.Application(client_max_size=1024 * 1024)
    app.router.add_post("/v1/audio/speech", handle_speech)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/voices", handle_voices)

    log.info("listening on 0.0.0.0:%d (POST /v1/audio/speech)", PORT)
    web.run_app(app, host="0.0.0.0", port=PORT, print=None, access_log=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
