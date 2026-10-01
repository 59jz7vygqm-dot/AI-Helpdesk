#!/usr/bin/env python3
"""List the built-in speakers of the configured Qwen3-TTS model.

The speaker names live in the model files, so they are read from the loaded
model rather than hardcoded here. Run it after the first start, then put the
name you want into tts.qwen3.speaker.

    docker compose exec helpdesk python3 /app/scripts/list_qwen_voices.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main() -> int:
    from helpdesk.config import load_config

    config = load_config(os.environ.get("HELPDESK_CONFIG", "/app/config/config.yaml"))
    settings = config["tts"].get("qwen3") or {}
    model_id = settings.get("model_id", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")

    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError:
        print("qwen3-tts is not installed in this container.", file=sys.stderr)
        print("Build with TTS_PROFILE=quality, or: pip install qwen3-tts", file=sys.stderr)
        return 1

    import torch

    print(f"loading {model_id} ...")
    model = Qwen3TTSModel.from_pretrained(
        model_id, device_map=settings.get("device", "cuda:0"), dtype=torch.bfloat16
    )

    # Different builds expose the roster under different names; look for any of
    # them rather than guessing one.
    found = False
    for attribute in ("speakers", "speaker_list", "available_speakers", "spk_list", "voices"):
        value = getattr(model, attribute, None)
        if value:
            print(f"\n{attribute}:")
            for item in (value.keys() if isinstance(value, dict) else value):
                print(f"  {item}")
            found = True

    config_obj = getattr(model, "config", None)
    for attribute in ("speakers", "speaker_list", "voices"):
        value = getattr(config_obj, attribute, None) if config_obj else None
        if value:
            print(f"\nconfig.{attribute}:")
            for item in (value.keys() if isinstance(value, dict) else value):
                print(f"  {item}")
            found = True

    if not found:
        print(
            "\nNo speaker roster found on this build.\n"
            "Check the model card for the speaker names:\n"
            f"  https://huggingface.co/{model_id}\n"
            "Leaving tts.qwen3.speaker empty uses the model's default voice, and\n"
            "mode: clone works without a speaker name at all."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
