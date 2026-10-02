"""Configuration: YAML file plus environment overrides.

Secrets belong in the environment (``HELPDESK_SIP_PASSWORD``), everything else in
the YAML file so a tuning change is reviewable.  Any scalar can be overridden
with ``HELPDESK_<SECTION>_<KEY>``, which is what makes the container configurable
without mounting a file.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

DEFAULTS: Dict[str, Any] = {
    "sip": {
        "username": "",
        "password": "",
        "auth_username": "",
        "domain": "",
        "server_host": "",
        "server_port": 5060,
        "display_name": "AI Helpdesk",
        "register_expires": 300,
        "bind_host": "0.0.0.0",
        "bind_port": 5060,
        "advertise_host": "",
        "rtp_port_start": 16000,
        "rtp_port_end": 16200,
        "codec_preference": ["PCMA", "PCMU"],
        "max_concurrent_calls": 1,
        "trace": False,
    },
    "asr": {
        "backend": "faster-whisper",
        "model": "large-v3-turbo",
        "device": "cuda",
        "device_index": 0,
        "compute_type": "int8_float16",
        "language": "de",
        "beam_size": 1,
        "download_root": "/models/whisper",
        # Biases recognition towards your vocabulary. Worth filling in: it is the
        # difference between "Drucker druckt nicht" and "Drucker trug nicht".
        "initial_prompt": "Drucker, Papierstau, Fehlercode, Toner, VPN, Kennung, Passwort, Rechner, Bildschirm, Netzwerk.",
        "min_avg_logprob": -1.1,
        "max_no_speech_prob": 0.75,
        "cpu_threads": 4,
    },
    "llm": {
        # ollama = local Ollama daemon | openai = any OpenAI-compatible server
        # (vLLM, TGI, llama.cpp server, LM Studio)
        "backend": "ollama",
        "base_url": "http://127.0.0.1:11434",
        "api_key": "none",
        "model": "qwen2.5:7b-instruct-q4_K_M",
        # -1 keeps the model resident forever. A string must carry a unit
        # ("30m"); a bare "-1" is rejected by Ollama as a malformed duration.
        "keep_alive": -1,
        "timeout": 60,
        # Reasoning models would spend seconds thinking before the first word.
        "think": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": 110,
            "num_ctx": 4096,
            # Stop as soon as the model starts a second speaker turn.
            "stop": ["\nANRUFER", "\nAnrufer:", "\nUser:"],
        },
    },
    "tts": {
        # piper is the default because it always builds and never stalls.
        # Switch to chatterbox (image built with TTS_PROFILE=quality) or qwen3
        # (own container, see README) for a natural voice.
        "backend": "piper",
        "qwen3": {
            "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "device": "cuda:0",
            "dtype": "bfloat16",
            "language": "German",
            "mode": "custom",
            "speaker": "",
            "instruct": "",
            "reference_audio": "",
            "reference_text": "",
            "attn_implementation": "",
            "sample_rate": 24000,
            "streaming": True,
        },
        "piper": {
            # medium, not high: the call is 8 kHz, so the extra bandwidth is
            # discarded while the synthesis cost is not.
            "model_path": "/models/piper/de_DE-thorsten-medium.onnx",
            "config_path": "",
            "length_scale": 1.0,
            # Synthesis is the dominant cost per answer; give it real cores.
            # 0 lets onnxruntime decide, which is usually too conservative.
            "threads": 8,
            "noise_scale": 0.667,
            "noise_w": 0.8,
            "use_cuda": False,
        },
        "chatterbox": {
            "device": "cuda",
            "language_id": "de",
            "reference_audio": "",
            "exaggeration": 0.45,
            "cfg_weight": 0.5,
            "temperature": 0.6,
            "chunk_tokens": 25,
            "multilingual": True,
        },
        "openai": {
            "base_url": "http://127.0.0.1:8880/v1",
            "model": "tts-1",
            "voice": "de_female",
            "api_key": "none",
            "response_format": "pcm",
            "sample_rate": 24000,
            "speed": 1.0,
        },
        "cache_dir": "/models/phrase-cache",
        # Deliberately small: the first chunk decides when the caller hears
        # anything, and a clause of 15 characters already sounds natural.
        "first_chunk_min_chars": 14,
        "min_chunk_chars": 60,
        "max_chunk_chars": 220,
    },
    "knowledge": {
        "enabled": True,
        "directory": "/app/knowledge",
        "cache_path": "/models/kb-index.npz",
        # Every retrieved chunk is re-read by the model on every single turn, so
        # this trades directly against time-to-first-token. Two good passages
        # answer a helpdesk question; a third mostly adds prefill and distraction.
        "top_k": 2,
        "max_chars": 900,
        "overlap_chars": 120,
        "context_chars": 1100,
        "dense_weight": 0.72,
        # Raised after a live call: at 0.28 a loosely related passage was passed
        # in as context and the model improvised instructions from it. Better to
        # have no context and hand over.
        "min_score": 0.40,
        "embeddings": {
            "backend": "ollama",
            "model": "bge-m3",
            "batch_size": 16,
            "query_prefix": "",
            "document_prefix": "",
            "cache_dir": "/models/fastembed",
            # Synthesis is the dominant cost per answer; give it real cores.
            # 0 lets onnxruntime decide, which is usually too conservative.
            "threads": 8,
        },
    },
    "vad": {
        "backend": "auto",
        # 3 (strictest) on a telephony line: at 2, steady line noise kept the
        # utterance open for seconds after the caller stopped.
        "aggressiveness": 3,
        # The caller waits this out on every single turn, so it is the one number
        # that is felt directly. 320 ms still tolerates a breath mid-sentence.
        "end_silence_ms": 320,
        # Must leave room for recognition to finish before end_silence_ms, or the
        # semantic endpoint can never fire and the hangover is paid in full.
        # Live measurements showed ~70-150 ms for a short utterance, so starting
        # at 140 ms gives it until 320 ms.
        "speculative_silence_ms": 140,
        "start_frames": 3,
        "max_utterance_ms": 20000,
        "pre_roll_ms": 300,
        "barge_in_ms": 200,
        "speculative_asr": True,
        # Answer as soon as the running transcript reads as a finished sentence,
        # instead of waiting out end_silence_ms. The hangover is paid on every
        # turn, so this is the last structural piece of the response time.
        "semantic_endpointing": True,
        "semantic_min_words": 3,
        "echo_guard": True,
        "echo_attenuation_db": 12.0,
        "echo_correlation": 0.72,
        "resume_on_backchannel": True,
    },
    "dialog": {
        "company": "unserem Unternehmen",
        "agent_name": "Alex",
        # helpdesk  = answers only from the knowledge base, hands over otherwise
        # assistant = also chats freely; for demos and general questions
        "mode": "helpdesk",
        "greeting": "Guten Tag, hier ist der automatische Service von unserem Unternehmen. Wie kann ich Ihnen helfen?",
        "transfer_announcement": "Einen Moment bitte, ich verbinde Sie mit einem Kollegen.",
        "transfer_failed": "Ich kann Sie im Moment leider nicht verbinden. Bitte versuchen Sie es später noch einmal. Auf Wiederhören.",
        "goodbye": "Vielen Dank für Ihren Anruf. Auf Wiederhören.",
        "not_understood": "Entschuldigung, das habe ich nicht verstanden. Können Sie das bitte wiederholen?",
        "still_there": "Sind Sie noch da?",
        "thinking": "",
        # Off by default. They were added to cover a multi-second pause; once
        # answers arrive in well under a second they interrupt the flow instead of
        # smoothing it. Fill the list to bring them back.
        "fillers": [],
        "filler_after_ms": 900,
        "transfer_number": "",
        "transfer_method": "auto",
        "transfer_dtmf_feature_code": "##",
        "transfer_dtmf_terminator": "",
        "transfer_dtmf_delay_ms": 700,
        "max_sentences": 2,
        # Enough for a phone call to stay coherent without the prompt growing
        # through a long conversation.
        "history_turns": 6,
        "extra_instructions": "",
        "silence_prompt_after_ms": 7000,
        "silence_hangup_after_ms": 20000,
        "max_call_seconds": 900,
        "max_misunderstood": 2,
        # "Vielen Dank" means goodbye, not "transfer me". Handled in code because
        # the model kept reading it as a request it could not fulfil.
        "farewell_ends_call": True,
        "answer_delay_ms": 0,
        "ring_before_answer": True,
    },
    "logging": {
        "level": "INFO",
        "transcript_dir": "",
        "metrics": True,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _coerce(value: str, reference: Any) -> Any:
    if isinstance(reference, bool):
        return value.strip().lower() in ("1", "true", "yes", "on", "ja")
    if isinstance(reference, int) and not isinstance(reference, bool):
        try:
            return int(value)
        except ValueError:
            return reference
    if isinstance(reference, float):
        try:
            return float(value)
        except ValueError:
            return reference
    if isinstance(reference, list):
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


def apply_env_overrides(config: Dict[str, Any], prefix: str = "HELPDESK_") -> Dict[str, Any]:
    """Override scalars from the environment.

    ``HELPDESK_SIP_SERVER_HOST=10.0.0.1`` sets ``sip.server_host``.  Keys are
    matched longest-first so ``server_host`` wins over a hypothetical ``server``.
    """
    result = copy.deepcopy(config)
    for env_key, raw in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        remainder = env_key[len(prefix) :].lower()
        for section in sorted(result.keys(), key=len, reverse=True):
            if not remainder.startswith(section + "_"):
                continue
            field = remainder[len(section) + 1 :]
            target = result[section]
            if not isinstance(target, dict):
                break
            # Walk into nested dicts, e.g. tts_piper_model_path.
            while True:
                matched_nested = False
                for nested in sorted(
                    (k for k, v in target.items() if isinstance(v, dict)), key=len, reverse=True
                ):
                    if field.startswith(nested + "_"):
                        target = target[nested]
                        field = field[len(nested) + 1 :]
                        matched_nested = True
                        break
                if not matched_nested:
                    break
            if field in target:
                target[field] = _coerce(raw, target[field])
            else:
                target[field] = raw
            break
    return result


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULTS)
    if path and os.path.exists(path):
        import yaml  # noqa: PLC0415

        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        config = _deep_merge(config, loaded)
        log.info("loaded configuration from %s", path)
    elif path:
        log.warning("config file %s not found, using defaults and environment", path)
    return apply_env_overrides(config)


def validate(config: Dict[str, Any]) -> list:
    """Return a list of problems that would stop the agent from working."""
    problems = []
    sip = config["sip"]
    for key in ("username", "password", "server_host"):
        if not sip.get(key):
            problems.append(f"sip.{key} is required (env HELPDESK_SIP_{key.upper()})")
    if not sip.get("domain"):
        sip["domain"] = sip.get("server_host", "")
    if not config["dialog"].get("transfer_number"):
        problems.append(
            "dialog.transfer_number is required so the agent can hand off calls "
            "(env HELPDESK_DIALOG_TRANSFER_NUMBER)"
        )
    start, end = sip.get("rtp_port_start", 0), sip.get("rtp_port_end", 0)
    if end - start < 4:
        problems.append("sip.rtp_port_end must be at least 4 above rtp_port_start")
    return problems
