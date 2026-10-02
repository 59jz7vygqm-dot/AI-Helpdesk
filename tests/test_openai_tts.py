"""Verify the openai TTS backend against a server behaving like tts-server.

Qwen3-TTS itself cannot run here (it needs Python 3.13 and a GPU), but the
contract between the agent and the voice container can be checked: the request
shape, the PCM decoding, chunk boundaries, and cancellation.
"""

from __future__ import annotations

import asyncio
import os
import sys

import numpy as np
from aiohttp import web

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from helpdesk.tts.openai_tts import OpenAiCompatibleSynthesizer

RATE = 24000
received: list = []


async def handle_speech(request: web.Request) -> web.Response:
    payload = await request.json()
    received.append(payload)
    if payload.get("response_format") != "pcm":
        return web.Response(status=400, text="only pcm")
    # One second of a tone per 20 characters, as the real server would return it.
    n = max(1, len(payload["input"]) // 20) * RATE
    tone = (np.sin(2 * np.pi * 220 * np.arange(n) / RATE) * 9000).astype(np.int16)
    return web.Response(
        body=tone.tobytes(),
        content_type="audio/pcm",
        headers={"X-Sample-Rate": str(RATE)},
    )


async def handle_models(request: web.Request) -> web.Response:
    return web.json_response({"object": "list", "data": [{"id": "Qwen/Qwen3-TTS"}]})


async def main() -> int:
    app = web.Application()
    app.router.add_post("/v1/audio/speech", handle_speech)
    app.router.add_get("/v1/models", handle_models)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    synth = OpenAiCompatibleSynthesizer(
        base_url=f"http://127.0.0.1:{port}/v1",
        model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        voice="",
        sample_rate=RATE,
    )
    failed = 0

    # 1. A full synthesis must come back as contiguous int16
    pcm = await synth.synthesize("Guten Tag, hier ist der Service von der Beispiel GmbH.")
    expected = max(1, len("Guten Tag, hier ist der Service von der Beispiel GmbH.") // 20) * RATE
    if pcm.dtype == np.int16 and abs(pcm.size - expected) <= 2:
        print(f"PASS decoded {pcm.size} int16 samples ({pcm.size/RATE:.1f}s) intact")
    else:
        print(f"FAIL got {pcm.size} samples of {pcm.dtype}, expected ~{expected} int16")
        failed += 1

    # The waveform must survive chunking: a split int16 would show up as noise
    reference = (np.sin(2 * np.pi * 220 * np.arange(pcm.size) / RATE) * 9000).astype(np.int16)
    corr = float(np.corrcoef(pcm.astype(float), reference.astype(float))[0, 1])
    if corr > 0.99:
        print(f"PASS waveform intact across chunk boundaries (corr={corr:.4f})")
    else:
        print(f"FAIL audio corrupted by chunking (corr={corr:.4f})")
        failed += 1

    # 2. The request must carry what the server needs
    sent = received[-1]
    checks = {
        "model passed through": sent.get("model") == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "response_format is pcm": sent.get("response_format") == "pcm",
        "input is the text": "Beispiel GmbH" in sent.get("input", ""),
    }
    for name, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + name)
        failed += 0 if ok else 1

    # 3. Markup and abbreviations must be spoken form, not read literally
    received.clear()
    await synth.synthesize("Das kostet ca. 50 % **inkl.** MwSt.")
    spoken = received[-1]["input"]
    if "circa" in spoken and "Prozent" in spoken and "*" not in spoken:
        print(f"PASS text normalised before synthesis: {spoken!r}")
    else:
        print(f"FAIL text not normalised: {spoken!r}")
        failed += 1

    # 4. Cancellation must stop consuming mid-stream
    cancel = asyncio.Event()
    cancel.set()
    chunks = [c async for c in synth.stream("Ein langer Satz der abgebrochen wird.", cancel=cancel)]
    if not chunks:
        print("PASS a set cancel event yields nothing")
    else:
        print(f"FAIL cancelled stream still produced {len(chunks)} chunks")
        failed += 1

    await synth.close()
    await runner.cleanup()
    print(f"\n{'all openai TTS checks passed' if not failed else str(failed) + ' failed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
