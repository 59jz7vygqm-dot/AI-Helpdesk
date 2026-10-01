#!/usr/bin/env python3
"""Synthesise the same sentence with every voice in ./voices and time each one.

Picking a voice by reading model names is guesswork; this writes one WAV per
voice at telephone quality, so the choice is made by ear with the cost visible.

    docker compose exec helpdesk python3 /app/scripts/compare_voices.py
    docker compose exec helpdesk python3 /app/scripts/compare_voices.py "Eigener Satz"
"""

from __future__ import annotations

import asyncio
import glob
import os
import sys
import time
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from helpdesk.audio.codec import resample
from helpdesk.tts.piper_tts import PiperSynthesizer

SENTENCE = "Guten Tag, hier ist der Service. Wie kann ich Ihnen helfen?"
OUT_DIR = "/tmp/voices"


async def main() -> int:
    sentence = " ".join(sys.argv[1:]) or SENTENCE
    voices = sorted(glob.glob("/models/piper/*.onnx")) or sorted(glob.glob("voices/*.onnx"))
    if not voices:
        print("No voices found. Run scripts/download_models.sh first.", file=sys.stderr)
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Satz: {sentence!r}\n")
    print(f"{'Stimme':40} {'Synthese':>10} {'Audio':>8} {'rtf':>6}")
    print("-" * 68)

    results = []
    for path in voices:
        name = os.path.basename(path).replace(".onnx", "")
        synth = PiperSynthesizer(model_path=path, threads=8)
        try:
            await synth.synthesize("Aufwärmen.")  # exclude one-time load cost
            started = time.monotonic()
            pcm = await synth.synthesize(sentence)
            elapsed = time.monotonic() - started
        except Exception as exc:
            print(f"{name:40} FEHLER: {exc}")
            continue

        audio_s = pcm.size / max(synth.sample_rate, 1)
        rtf = elapsed / audio_s if audio_s else 0.0
        print(f"{name:40} {elapsed*1000:>8.0f}ms {audio_s*1000:>6.0f}ms {rtf:>6.2f}")
        results.append((rtf, name))

        # 8 kHz: what the caller actually hears. Judge on this, not the raw file.
        phone = resample(pcm, synth.sample_rate, 8000)
        with wave.open(os.path.join(OUT_DIR, f"{name}-phone.wav"), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            handle.writeframes(phone.astype(np.int16).tobytes())

    print(f"\nWAVs in {OUT_DIR} (8 kHz, wie am Telefon).")
    if results:
        fastest = min(results)
        print(f"Schnellste: {fastest[1]} (rtf {fastest[0]:.2f})")
        print("Ein rtf unter 0,15 ist unkritisch; darüber bremst die Stimme die Antwort.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
