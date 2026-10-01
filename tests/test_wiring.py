"""Startup wiring test.

Replaces the three heavy model backends with stubs and runs the real
HelpdeskApplication.prepare(), which is where a mistyped config key would
otherwise stay hidden until the first call came in.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from typing import AsyncIterator

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import helpdesk.app as app_module
from helpdesk.asr.base import Recognizer, Transcript
from helpdesk.tts.base import SpeechChunk, Synthesizer


class StubRecognizer(Recognizer):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    async def transcribe(self, pcm, *, language=None):
        return Transcript(text="stub")

    async def warmup(self):
        return None

    async def close(self):
        return None


class StubSynth(Synthesizer):
    sample_rate = 8000

    async def stream(self, text, *, cancel=None) -> AsyncIterator[SpeechChunk]:
        yield SpeechChunk(pcm=np.zeros(800, dtype=np.int16), sample_rate=8000, final=True)

    async def warmup(self):
        return None

    async def close(self):
        return None


class StubLlm:
    def __init__(self, *args, **kwargs):
        self.model = kwargs.get("model", "stub")
        self.calls = []

    async def ensure_model(self, model=None):
        self.calls.append("ensure_model")

    async def warmup(self, model=None):
        return 0.01

    async def chat_stream(self, messages, *, options=None, model=None):
        yield "ok"

    async def embed(self, texts, *, model):
        # deterministic pseudo-embeddings
        return [[float(len(t) % 7), 1.0, 0.5] for t in texts]

    async def close(self):
        return None


async def main() -> int:
    workdir = tempfile.mkdtemp()
    knowledge = os.path.join(workdir, "knowledge")
    os.makedirs(knowledge)
    with open(os.path.join(knowledge, "faq.md"), "w", encoding="utf-8") as fh:
        fh.write("# Drucker\n\n## Papierstau\nFach zwei oeffnen.\n")

    config_path = os.path.join(workdir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as fh:
        fh.write(
            f"""
sip:
  username: "900"
  password: "secret"
  server_host: "127.0.0.1"
  server_port: 15999
  bind_port: 15998
  advertise_host: "127.0.0.1"
  rtp_port_start: 24000
  rtp_port_end: 24050
knowledge:
  directory: "{knowledge}"
  cache_path: "{os.path.join(workdir, 'kb.npz')}"
tts:
  backend: piper
  cache_dir: "{os.path.join(workdir, 'phrases')}"
dialog:
  company: "Testfirma GmbH"
  transfer_number: "200"
logging:
  level: WARNING
  transcript_dir: "{os.path.join(workdir, 'transcripts')}"
"""
        )

    # Swap the heavy parts out
    app_module.FasterWhisperRecognizer = lambda *a, **k: StubRecognizer(*a, **k)
    app_module.OllamaClient = lambda *a, **k: StubLlm(*a, **k)
    app_module.build_synthesizer = lambda cfg: StubSynth()

    from helpdesk.config import load_config, validate

    config = load_config(config_path)
    problems = validate(config)
    assert not problems, problems
    print("PASS config validates")

    app = app_module.HelpdeskApplication(config)
    await app.prepare()
    print("PASS prepare() completed: ASR, TTS, LLM, knowledge base and phrases all wired")

    assert app.kb is not None and app.kb.chunks, "knowledge base did not index anything"
    print(f"PASS knowledge indexed: {len(app.kb.chunks)} chunk(s)")

    # Every fixed phrase must be pre-rendered, otherwise the greeting would be
    # synthesised during the call.
    missing = [t for t in app.texts.fixed_phrases() if t and app.phrases.get(t) is None]
    assert not missing, f"phrases not cached: {missing}"
    cached = [t for t in app.texts.fixed_phrases() if t]
    print(f"PASS {len(cached)} fixed phrase(s) pre-rendered to 8 kHz")

    assert app.ua is not None
    acct = app.ua.account
    assert acct.username == "900" and acct.server_port == 15999, acct
    assert acct.rtp_port_range == (24000, 24050), acct.rtp_port_range
    assert acct.domain == "127.0.0.1", acct.domain
    print(f"PASS SIP account built: {acct.username}@{acct.domain}:{acct.server_port}, "
          f"rtp {acct.rtp_port_range[0]}-{acct.rtp_port_range[1]}")

    # The per-call agent must be constructible with the same config
    from helpdesk.sip.messages import SipMessage
    from helpdesk.sip.ua import Call

    call = Call(
        call_id="w1", local_tag="l", remote_tag="r", from_header="<sip:1@p>;tag=r",
        to_header="<sip:900@p>", remote_target="sip:1@p", route_set=[],
        caller_number="1", caller_name="", dialled_number="900",
        invite=SipMessage(is_request=True, method="INVITE", uri="sip:900@p"),
        source=("127.0.0.1", 5060),
    )
    from helpdesk.llm.agent import HelpdeskAgent

    agent = HelpdeskAgent(app.llm, app.kb, company=config["dialog"]["company"])
    prompt = agent.system_prompt()
    assert "Testfirma GmbH" in prompt, "company not substituted into the prompt"
    assert "[WEITERLEITEN]" in prompt, "transfer marker missing from the prompt"
    print("PASS system prompt renders with company name and control markers")

    hits = await app.kb.search("Papierstau", top_k=2)
    print(f"PASS knowledge search returns {len(hits)} hit(s) for 'Papierstau'")

    await app.llm.close()
    print("\nall wiring checks passed")
    return 0


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.ERROR)
    sys.exit(asyncio.run(main()))
