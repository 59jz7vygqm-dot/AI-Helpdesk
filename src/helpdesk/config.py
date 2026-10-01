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
        "initial_prompt": "",
        "min_avg_logprob": -1.1,
        "max_no_speech_prob": 0.75,
        "cpu_threads": 4,
    },
    "llm": {
        "base_url": "http://127.0.0.1:11434",
        "model": "qwen2.5:7b-instruct-q4_K_M",
        "keep_alive": "-1",
        "timeout": 60,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 160,
            "num_ctx": 4096,
            # Stop as soon as the model starts a second speaker turn.
            "stop": ["\nANRUFER", "\nAnrufer:", "\nUser:"],
        },
    },
    "tts": {
        "backend": "piper",
        "piper": {
            "model_path": "/models/piper/de_DE-thorsten-high.onnx",
            "config_path": "",
            "length_scale": 1.0,
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
        "first_chunk_min_chars": 24,
        "min_chunk_chars": 60,
        "max_chunk_chars": 220,
    },
    "knowledge": {
        "enabled": True,
        "directory": "/app/knowledge",
        "cache_path": "/models/kb-index.npz",
        "top_k": 3,
        "max_chars": 900,
        "overlap_chars": 120,
        "context_chars": 1800,
        "dense_weight": 0.72,
        "min_score": 0.28,
        "embeddings": {
            "backend": "ollama",
            "model": "bge-m3",
            "batch_size": 16,
            "query_prefix": "",
            "document_prefix": "",
            "cache_dir": "/models/fastembed",
            "threads": 4,
        },
    },
    "vad": {
        "backend": "auto",
        "aggressiveness": 2,
        "end_silence_ms": 420,
        "speculative_silence_ms": 220,
        "start_frames": 3,
        "max_utterance_ms": 20000,
        "pre_roll_ms": 300,
        "barge_in_ms": 260,
        "speculative_asr": True,
    },
    "dialog": {
        "company": "unserem Unternehmen",
        "agent_name": "Alex",
        "greeting": "Guten Tag, hier ist der automatische Service von unserem Unternehmen. Wie kann ich Ihnen helfen?",
        "transfer_announcement": "Einen Moment bitte, ich verbinde Sie mit einem Kollegen.",
        "transfer_failed": "Ich kann Sie im Moment leider nicht verbinden. Bitte versuchen Sie es später noch einmal. Auf Wiederhören.",
        "goodbye": "Vielen Dank für Ihren Anruf. Auf Wiederhören.",
        "not_understood": "Entschuldigung, das habe ich nicht verstanden. Können Sie das bitte wiederholen?",
        "still_there": "Sind Sie noch da?",
        "thinking": "",
        "transfer_number": "",
        "max_sentences": 3,
        "history_turns": 10,
        "extra_instructions": "",
        "silence_prompt_after_ms": 7000,
        "silence_hangup_after_ms": 20000,
        "max_call_seconds": 900,
        "max_misunderstood": 2,
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
