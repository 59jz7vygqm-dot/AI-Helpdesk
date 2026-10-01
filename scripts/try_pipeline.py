#!/usr/bin/env python3
"""Test the brain without a phone: text in, spoken answer out.

Runs the real knowledge base, the real LLM and the real TTS, prints the
per-stage timings and writes the reply to a WAV file so the voice can be judged
before anyone calls in.

    python3 scripts/try_pipeline.py "Mein Drucker zeigt E-512"
    python3 scripts/try_pipeline.py            # interactive
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from helpdesk.audio.codec import resample
from helpdesk.config import load_config
from helpdesk.kb.embedder import build_embedder
from helpdesk.kb.index import KnowledgeBase
from helpdesk.llm.agent import Action, HelpdeskAgent
from helpdesk.llm.ollama_client import OllamaClient
from helpdesk.tts.registry import build_synthesizer
from helpdesk.tts.text import SentenceStreamer


def write_wav(path: str, pcm: np.ndarray, rate: int) -> None:
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.asarray(pcm, dtype=np.int16).tobytes())


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)-24s %(message)s")
    config = load_config(os.environ.get("HELPDESK_CONFIG", "config/config.yaml"))

    # Paths in the config point into the container; fall back to local ones.
    if not os.path.isdir(config["knowledge"]["directory"]):
        config["knowledge"]["directory"] = "knowledge"
    config["knowledge"]["cache_path"] = "/tmp/kb-index-test.npz"
    piper_path = config["tts"]["piper"]["model_path"]
    if not os.path.exists(piper_path):
        local = os.path.join("voices", os.path.basename(piper_path))
        if os.path.exists(local):
            config["tts"]["piper"]["model_path"] = local
    config["tts"]["cache_dir"] = ""

    llm = OllamaClient(
        base_url=config["llm"]["base_url"],
        model=config["llm"]["model"],
        keep_alive=str(config["llm"].get("keep_alive", "-1")),
        options=config["llm"].get("options") or {},
    )
    await llm.ensure_model()

    kb = None
    if config["knowledge"].get("enabled", True):
        kb = KnowledgeBase(
            config["knowledge"]["directory"],
            build_embedder(config["knowledge"]["embeddings"], llm),
            cache_path=config["knowledge"]["cache_path"],
            dense_weight=float(config["knowledge"]["dense_weight"]),
            min_score=float(config["knowledge"]["min_score"]),
        )
        await kb.build()

    synth = build_synthesizer(config["tts"])
    await synth.warmup()

    dialog = config["dialog"]
    agent = HelpdeskAgent(
        llm, kb,
        company=dialog.get("company", "unserem Unternehmen"),
        agent_name=dialog.get("agent_name", "Alex"),
        max_sentences=int(dialog.get("max_sentences", 3)),
        top_k=int(config["knowledge"].get("top_k", 3)),
        options=config["llm"].get("options") or {},
        extra_instructions=dialog.get("extra_instructions", ""),
    )

    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else None
    turn = 0
    while True:
        if questions is not None:
            if not questions:
                break
            question = questions.pop(0)
        else:
            try:
                question = input("\nAnrufer> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question or question.lower() in ("exit", "quit", "ende"):
                break

        turn += 1
        started = time.monotonic()
        retrieval_started = time.monotonic()
        context, sources = await agent.retrieve(question)
        retrieval_ms = int((time.monotonic() - retrieval_started) * 1000)

        streamer = SentenceStreamer()
        chunks: list = []
        first_audio_ms = 0
        spoken_text = []

        async def speak(text: str) -> None:
            nonlocal first_audio_ms
            async for piece in synth.stream(text):
                if not first_audio_ms:
                    first_audio_ms = int((time.monotonic() - started) * 1000)
                chunks.append(resample(piece.pcm, piece.sample_rate, 8000))

        async for delta in agent.respond_stream(question, context=context):
            spoken_text.append(delta)
            for chunk in streamer.feed(delta):
                await speak(chunk)
        for chunk in streamer.flush():
            await speak(chunk)

        reply = agent.last_reply
        total_ms = int((time.monotonic() - started) * 1000)
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)

        print(f"\nAgent> {reply.text}")
        print(
            f"       [kb {retrieval_ms} ms | first token {reply.first_token_ms} ms | "
            f"first audio {first_audio_ms} ms | total {total_ms} ms | "
            f"{int(audio.size * 1000 / 8000)} ms of speech]"
        )
        if sources:
            print(f"       sources: {', '.join(sources)}")
        if reply.action is Action.TRANSFER:
            print(f"       ACTION: would transfer to {dialog.get('transfer_number') or '(unset)'}")
        elif reply.action is Action.HANGUP:
            print("       ACTION: would hang up")

        if audio.size:
            # 8 kHz is what the caller actually hears; judge the voice on this.
            path = f"/tmp/helpdesk-turn{turn}.wav"
            write_wav(path, audio, 8000)
            write_wav(path.replace(".wav", "-full.wav"), np.concatenate(
                [resample(a, 8000, synth.sample_rate) for a in chunks]
            ), synth.sample_rate)
            print(f"       wrote {path} (as heard on the phone)")

    await synth.close()
    await llm.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
