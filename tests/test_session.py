"""Session-level test with stub models.

Exercises the orchestration that cannot be checked without a call: greeting,
a full turn, transfer on the marker, barge-in cancelling an in-flight reply,
and the silence watchdog.  No GPU, no PBX, no network.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import AsyncIterator, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from helpdesk.asr.base import Recognizer, Transcript
from helpdesk.config import DEFAULTS
from helpdesk.llm.agent import HelpdeskAgent
from helpdesk.session import CallSession, DialogTexts
from helpdesk.sip.ua import Call, CallState
from helpdesk.tts.base import SpeechChunk, Synthesizer
from helpdesk.tts.registry import PhraseCache

RATE = 8000


class StubRecognizer(Recognizer):
    def __init__(self, texts: List[str]) -> None:
        self.texts = list(texts)
        self.calls = 0

    async def transcribe(self, pcm, *, language=None) -> Transcript:
        self.calls += 1
        await asyncio.sleep(0.01)
        text = self.texts.pop(0) if self.texts else ""
        return Transcript(text=text, duration_ms=int(pcm.size * 1000 / 16000), latency_ms=10)


class StubSynthesizer(Synthesizer):
    """Emits 100 ms of tone per 10 characters, in 20 ms chunks."""

    sample_rate = 8000

    def __init__(self, chunk_delay: float = 0.0) -> None:
        self.chunk_delay = chunk_delay
        self.spoken: List[str] = []
        self.cancelled = 0

    async def stream(self, text: str, *, cancel=None) -> AsyncIterator[SpeechChunk]:
        self.spoken.append(text)
        total = max(1, len(text) // 10) * 800
        emitted = 0
        while emitted < total:
            if cancel is not None and cancel.is_set():
                self.cancelled += 1
                return
            if self.chunk_delay:
                await asyncio.sleep(self.chunk_delay)
            n = min(160, total - emitted)
            yield SpeechChunk(
                pcm=(np.sin(np.arange(n) * 0.2) * 5000).astype(np.int16), sample_rate=self.sample_rate
            )
            emitted += n


class StubLlm:
    def __init__(self, replies: List[str], first_token_delay: float = 0.0) -> None:
        self.replies = list(replies)
        self.first_token_delay = first_token_delay
        self.requests: List[list] = []

    async def chat_stream(self, messages, *, options=None, model=None):
        self.requests.append(messages)
        reply = self.replies.pop(0) if self.replies else "Alles klar."
        if self.first_token_delay:
            await asyncio.sleep(self.first_token_delay)
        for word in reply.split(" "):
            await asyncio.sleep(0.002)
            yield word + " "


class FakeRtp:
    """Stands in for RtpSession: records what would go on the wire."""

    def __init__(self) -> None:
        self.on_sent = None
        self.dtmf_sent: List[str] = []
        self.queue: List[np.ndarray] = []
        self.sent_samples = 0
        self.cleared = 0
        self.packets_received = 1
        self.encoding = "PCMA"
        self.dtmf_payload = 101
        self.transport = None
        self.remote = ("127.0.0.1", 1234)
        self._drain_task: Optional[asyncio.Task] = None

    def enqueue(self, pcm) -> None:
        self.queue.append(np.asarray(pcm))
        self.sent_samples += int(np.asarray(pcm).size)

    def is_playing(self) -> bool:
        return bool(self.queue)

    def queued_ms(self) -> int:
        return int(sum(a.size for a in self.queue) * 1000 / RATE)

    def clear_playout(self) -> None:
        self.queue.clear()
        self.cleared += 1

    def send_dtmf(self, digits: str) -> int:
        self.dtmf_sent.append(digits)
        return len(digits) * 200

    def dtmf_pending(self) -> bool:
        return False

    def start_draining(self, speed: float = 1.0) -> None:
        """Mimic the real 20 ms send tick, including the on_sent callback."""

        async def drain():
            while True:
                await asyncio.sleep(0.02 / speed)
                frame = None
                if self.queue:
                    head = self.queue[0]
                    if head.size <= 160:
                        frame = head
                        self.queue.pop(0)
                    else:
                        frame = head[:160]
                        self.queue[0] = head[160:]
                if self.on_sent is not None:
                    self.on_sent(frame)

        self._drain_task = asyncio.ensure_future(drain())

    def stop_draining(self) -> None:
        if self._drain_task:
            self._drain_task.cancel()

    async def close(self) -> None:
        self.stop_draining()


class FakeUa:
    def __init__(self) -> None:
        self.rang = False
        self.transfers: List[str] = []
        self.transfer_methods: List[object] = []
        self.hangups: List[str] = []
        self.transfer_should_fail = False
        self.call: Optional[Call] = None

    def ring(self, call) -> None:
        self.rang = True

    async def answer(self, call, *, on_audio, on_dtmf=None, on_sent=None):
        call.rtp = FakeRtp()
        call.rtp.on_sent = on_sent
        call.state = CallState.ANSWERED
        call.started_at = time.monotonic()
        self.on_audio = on_audio
        self.on_dtmf = on_dtmf
        self.call = call
        return call.rtp

    async def transfer(self, call, number, *, method=None, timeout=6.0,
                       feature_code="##", dtmf_delay_ms=700, dtmf_terminator="",
                       dtmf_settle_s=4.0):
        from helpdesk.sip.ua import TransferError, TransferMethod

        self.transfer_methods.append(method)
        if self.transfer_should_fail:
            raise TransferError("rejected by fake pbx")
        self.transfers.append(number)
        call.state = CallState.TRANSFERRING
        return "dtmf" if method is TransferMethod.DTMF else "refer"

    async def hangup(self, call, reason=""):
        self.hangups.append(reason)
        call.state = CallState.ENDED
        if call.on_end:
            call.on_end(reason)


def make_call() -> Call:
    from helpdesk.sip.messages import SipMessage

    invite = SipMessage(is_request=True, method="INVITE", uri="sip:900@pbx")
    return Call(
        call_id="test-call-1", local_tag="lt", remote_tag="rt",
        from_header='"Max" <sip:49151@pbx>;tag=rt', to_header="<sip:900@pbx>",
        remote_target="sip:49151@pbx", route_set=[], caller_number="49151",
        caller_name="Max", dialled_number="900", invite=invite, source=("127.0.0.1", 5060),
    )


def build_config(**dialog_overrides) -> dict:
    import copy

    config = copy.deepcopy(DEFAULTS)
    config["vad"].update({"backend": "energy", "end_silence_ms": 100,
                          "speculative_silence_ms": 60, "pre_roll_ms": 60,
                          "barge_in_ms": 100, "start_frames": 2})
    config["dialog"].update({
        "greeting": "Guten Tag, hier ist der Service. Wie kann ich helfen?",
        "transfer_number": "200", "silence_prompt_after_ms": 400,
        "silence_hangup_after_ms": 1200, "max_call_seconds": 60,
        "ring_before_answer": True,
    })
    config["dialog"].update(dialog_overrides)
    config["tts"]["cache_dir"] = ""
    config["logging"]["transcript_dir"] = ""
    return config


async def build_session(asr_texts, llm_replies, *, config=None, tts_delay=0.0):
    config = config or build_config()
    call = make_call()
    ua = FakeUa()
    synth = StubSynthesizer(chunk_delay=tts_delay)
    llm = StubLlm(llm_replies)
    agent = HelpdeskAgent(llm, None, company="Testfirma")
    texts = DialogTexts(
        greeting=config["dialog"]["greeting"],
        transfer_announcement="Ich verbinde Sie.",
        transfer_failed="Das geht gerade nicht.",
        goodbye="Auf Wiederhören.",
        not_understood="Das habe ich nicht verstanden.",
        still_there="Sind Sie noch da?",
    )
    session = CallSession(
        call, ua, recognizer=StubRecognizer(asr_texts), synthesizer=synth,
        agent=agent, phrases=PhraseCache(synth, target_rate=RATE), texts=texts, config=config,
    )
    return session, call, ua, synth, llm


def speech(frames: int = 30) -> List[np.ndarray]:
    rng = np.random.default_rng(1)
    return [(rng.normal(0, 7000, 160)).astype(np.int16) for _ in range(frames)]


def silence(frames: int = 15) -> List[np.ndarray]:
    rng = np.random.default_rng(2)
    return [(rng.normal(0, 40, 160)).astype(np.int16) for _ in range(frames)]


async def feed(session, frames, *, drain_first=True):
    """Push frames in as the RTP layer would, letting the loop breathe."""
    if drain_first:
        await wait_until(lambda: not session.call.rtp.is_playing(), 3, "playout drained")
    for frame in frames:
        session._on_audio(frame)
        await asyncio.sleep(0.001)


async def wait_until(predicate, timeout=3.0, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


# ---------------------------------------------------------------- tests
async def test_greeting_and_turn():
    session, call, ua, synth, llm = await build_session(
        ["Ich habe ein Problem mit dem Drucker"], ["Gern, was genau passiert denn?"]
    )
    runner = asyncio.ensure_future(session.run())
    await wait_until(lambda: call.rtp is not None, 2, "answer")
    assert ua.rang, "should send 180 Ringing before answering"
    await wait_until(lambda: call.rtp.sent_samples > 0, 2, "greeting audio")
    greeting_ms = int(call.rtp.sent_samples * 1000 / RATE)
    print(f"PASS greeting played ({greeting_ms} ms of audio, cached)")

    call.rtp.start_draining(speed=20)
    await feed(session, speech(30) + silence(15))
    await wait_until(lambda: session._turn_task is not None and session._turn_task.done(), 4, "turn done")

    assert session.recognizer.calls >= 1
    assert any("Drucker" in t["text"] for t in session.transcript if t["role"] == "caller")
    assert any("was genau" in s for s in synth.spoken), synth.spoken
    assert session.metrics.turns, "no turn metrics recorded"
    m = session.metrics.turns[0]
    print(f"PASS turn: asr={m.asr_ms}ms spec={m.asr_speculative} "
          f"llm_ttft={m.llm_first_token_ms}ms tts={m.tts_first_chunk_ms}ms response={m.response_ms}ms")
    assert m.response_ms > 0, "response latency not measured"

    session._done.set()
    call.rtp.stop_draining()
    await asyncio.wait_for(runner, 3)
    return True


async def test_transfer_on_marker():
    session, call, ua, synth, llm = await build_session(
        ["Ich möchte kündigen"], ["Einen Moment, ich verbinde Sie. [WEITERLEITEN]"]
    )
    runner = asyncio.ensure_future(session.run())
    await wait_until(lambda: call.rtp is not None, 2, "answer")
    call.rtp.start_draining(speed=20)
    await feed(session, speech(30) + silence(15))
    await wait_until(lambda: ua.transfers or ua.hangups, 4, "transfer")

    assert ua.transfers == ["200"], f"transfers={ua.transfers} hangups={ua.hangups}"
    assert any("verbinde" in s for s in synth.spoken), synth.spoken
    # The marker itself must never be spoken
    assert not any("WEITERLEITEN" in s for s in synth.spoken), synth.spoken
    assert session.metrics.transferred
    print(f"PASS transfer to {ua.transfers[0]} on marker, marker not spoken")
    await asyncio.wait_for(runner, 3)
    return True


async def test_transfer_failure_falls_back():
    session, call, ua, synth, llm = await build_session(
        ["Ich will einen Menschen"], ["Ich verbinde Sie. [WEITERLEITEN]"]
    )
    ua.transfer_should_fail = True
    runner = asyncio.ensure_future(session.run())
    await wait_until(lambda: call.rtp is not None, 2, "answer")
    call.rtp.start_draining(speed=20)
    await feed(session, speech(30) + silence(15))
    await wait_until(lambda: ua.hangups, 4, "fallback hangup")
    assert any("geht gerade nicht" in s for s in synth.spoken), synth.spoken
    assert ua.hangups == ["transfer-failed"], ua.hangups
    print("PASS failed transfer: apologises and hangs up instead of dropping silently")
    await asyncio.wait_for(runner, 3)
    return True


async def test_barge_in_cancels_reply():
    # A slow TTS so there is a long reply in flight to interrupt
    session, call, ua, synth, llm = await build_session(
        ["Erzaehl mir alles", "Stop, andere Frage"],
        ["Das ist eine sehr lange Antwort die der Anrufer unterbrechen wird weil sie zu lang ist und weiter geht.",
         "Ja, bitte."],
        tts_delay=0.02,
    )
    runner = asyncio.ensure_future(session.run())
    await wait_until(lambda: call.rtp is not None, 2, "answer")
    call.rtp.start_draining(speed=20)
    await feed(session, speech(30) + silence(15))
    await wait_until(lambda: synth.spoken and call.rtp.sent_samples > 0, 4, "reply started")
    cleared_before = call.rtp.cleared

    # Caller talks over the bot
    for frame in speech(20):
        session._on_audio(frame)
        await asyncio.sleep(0.001)
    await wait_until(lambda: call.rtp.cleared > cleared_before, 3, "playout cleared by barge-in")
    assert synth.cancelled >= 1 or call.rtp.cleared > cleared_before
    print(f"PASS barge-in: playout cleared ({call.rtp.cleared}x), synthesis cancelled {synth.cancelled}x")

    # No audio may keep arriving from the abandoned reply
    await asyncio.sleep(0.15)
    call.rtp.clear_playout()
    samples_before = call.rtp.sent_samples
    await asyncio.sleep(0.15)
    assert call.rtp.sent_samples == samples_before, "cancelled synthesis is still producing audio"
    print("PASS abandoned reply stops producing audio")

    # And the session must accept the next utterance normally
    spoken_before = len(synth.spoken)
    await feed(session, silence(10) + speech(30) + silence(15), drain_first=False)
    await wait_until(lambda: len(synth.spoken) > spoken_before, 5, "reply after barge-in")
    print("PASS session recovers and answers the next question")

    session._done.set()
    call.rtp.stop_draining()
    await asyncio.wait_for(runner, 3)
    return True


async def test_unrecognised_speech_prompts_then_transfers():
    config = build_config(max_misunderstood=1)
    session, call, ua, synth, llm = await build_session(["", ""], [], config=config)
    runner = asyncio.ensure_future(session.run())
    await wait_until(lambda: call.rtp is not None, 2, "answer")
    call.rtp.start_draining(speed=20)

    await feed(session, speech(30) + silence(15))
    await wait_until(lambda: any("nicht verstanden" in s for s in synth.spoken), 4, "reprompt")
    print("PASS unrecognised speech -> asks the caller to repeat")

    await feed(session, speech(30) + silence(15))
    await wait_until(lambda: ua.transfers or ua.hangups, 4, "handoff")
    assert ua.transfers == ["200"], f"transfers={ua.transfers}"
    print("PASS second failure -> hands off to a human instead of looping")
    await asyncio.wait_for(runner, 3)
    return True


async def test_silence_watchdog():
    session, call, ua, synth, llm = await build_session([], [])
    runner = asyncio.ensure_future(session.run())
    await wait_until(lambda: call.rtp is not None, 2, "answer")
    call.rtp.start_draining(speed=20)
    await wait_until(lambda: any("noch da" in s for s in synth.spoken), 4, "still-there prompt")
    print("PASS watchdog asks 'Sind Sie noch da?' after silence")
    await wait_until(lambda: ua.hangups, 5, "silence hangup")
    assert ua.hangups == ["caller-silent"], ua.hangups
    print("PASS watchdog hangs up on a dead line")
    await asyncio.wait_for(runner, 3)
    return True


async def test_dtmf_zero_transfers():
    session, call, ua, synth, llm = await build_session([], [])
    runner = asyncio.ensure_future(session.run())
    await wait_until(lambda: call.rtp is not None, 2, "answer")
    call.rtp.start_draining(speed=20)
    await wait_until(lambda: not call.rtp.is_playing(), 3, "greeting done")
    session._on_dtmf("0")
    await feed(session, speech(20) + silence(15), drain_first=False)
    await wait_until(lambda: ua.transfers, 4, "dtmf transfer")
    assert ua.transfers == ["200"]
    print("PASS pressing 0 reaches a human")
    await asyncio.wait_for(runner, 3)
    return True


async def test_farewell_ends_the_call():
    """Saying goodbye must hang up politely, never transfer."""
    session, call, ua, synth, llm = await build_session(
        ["Vielen Dank."], ["sollte nicht aufgerufen werden [WEITERLEITEN]"]
    )
    runner = asyncio.ensure_future(session.run())
    await wait_until(lambda: call.rtp is not None, 2, "answer")
    call.rtp.start_draining(speed=20)
    requests_before = len(llm.requests)
    await feed(session, speech(30) + silence(15))
    await wait_until(lambda: ua.hangups or ua.transfers, 5, "call ended")

    assert ua.transfers == [], f"a farewell was transferred: {ua.transfers}"
    assert ua.hangups == ["assistant-goodbye"], ua.hangups
    assert len(llm.requests) == requests_before, "the model was asked about a goodbye"
    assert any("Wiederhören" in s for s in synth.spoken), synth.spoken
    print("PASS farewell hangs up with a goodbye, without transferring or asking the model")
    await asyncio.wait_for(runner, 3)
    return True


async def test_backchannel_resumes_instead_of_restarting():
    """A caller saying "mhm" must not restart the answer."""
    session, call, ua, synth, llm = await build_session(
        ["Erzaehl mir von euren Zeiten", "mhm"],
        ["Der Support ist montags bis freitags erreichbar. Von acht bis achtzehn Uhr. "
         "Ausserhalb nehmen wir Stoerungen auf. Die bearbeiten wir am naechsten Werktag.",
         "Sollte nicht aufgerufen werden."],
        tts_delay=0.0,
    )
    runner = asyncio.ensure_future(session.run())
    await wait_until(lambda: call.rtp is not None, 2, "answer")
    # Real-time playout: synthesis outruns transmission, so several sentences sit
    # in the queue unheard -- which is the situation this feature exists for.
    call.rtp.start_draining(speed=1)
    await feed(session, speech(30) + silence(15))
    await wait_until(lambda: len(synth.spoken) >= 2, 12, "reply under way")
    await wait_until(lambda: session._queued_samples > session._sent_samples + 8000,
                     12, "audio buffered ahead of the caller")

    requests_before = len(llm.requests)
    # Interrupt mid-answer, then say only "mhm"
    for frame in speech(20):
        session._on_audio(frame)
        await asyncio.sleep(0.001)
    await wait_until(lambda: session._interrupted_remainder != "", 8, "remainder remembered")
    remainder = session._interrupted_remainder
    print(f"PASS remembered {len(remainder)} chars the caller had not heard yet")

    spoken_before = len(synth.spoken)
    await feed(session, silence(10) + speech(20) + silence(15), drain_first=False)
    await wait_until(lambda: len(synth.spoken) > spoken_before, 12, "resume")

    assert len(llm.requests) == requests_before, "model was asked again for a backchannel"
    resumed = " ".join(synth.spoken[spoken_before:])
    assert resumed.strip(), "nothing was resumed"
    print(f"PASS backchannel resumed the answer without asking the model again")
    print(f"     resumed with: {resumed[:70]!r}")

    session._done.set()
    call.rtp.stop_draining()
    await asyncio.wait_for(runner, 3)
    return True


async def test_real_interruption_does_ask_again():
    """A real question during the answer must reach the model."""
    session, call, ua, synth, llm = await build_session(
        ["Erzaehl mir alles", "Was kostet ein neuer Laptop?"],
        ["Ein sehr langer erster Teil der Antwort. Mit mehreren Saetzen darin. "
         "Und noch einem dritten Satz. Und einem vierten.",
         "Dazu kann ich nichts sagen."],
        tts_delay=0.02,
    )
    runner = asyncio.ensure_future(session.run())
    await wait_until(lambda: call.rtp is not None, 2, "answer")
    call.rtp.start_draining(speed=20)
    await feed(session, speech(30) + silence(15))
    await wait_until(lambda: len(synth.spoken) >= 2, 5, "reply under way")

    requests_before = len(llm.requests)
    for frame in speech(20):
        session._on_audio(frame)
        await asyncio.sleep(0.001)
    await wait_until(lambda: session.call.rtp.cleared > 0, 3, "interrupted")
    await feed(session, silence(10) + speech(30) + silence(15), drain_first=False)
    await wait_until(lambda: len(llm.requests) > requests_before, 6, "model asked again")
    print("PASS a real question after an interruption does reach the model")

    session._done.set()
    call.rtp.stop_draining()
    await asyncio.wait_for(runner, 3)
    return True


async def test_echo_does_not_interrupt():
    """The agent's own voice echoing back must not cut it off."""
    session, call, ua, synth, llm = await build_session(
        ["Eine Frage"], ["Eine recht lange Antwort die weiterlaufen soll. Mit zweitem Satz. Und drittem."],
        tts_delay=0.02,
    )
    runner = asyncio.ensure_future(session.run())
    await wait_until(lambda: call.rtp is not None, 2, "answer")
    call.rtp.start_draining(speed=20)
    await feed(session, speech(30) + silence(15))
    await wait_until(lambda: call.rtp.sent_samples > 0 and synth.spoken, 5, "speaking")

    cleared_before = call.rtp.cleared
    rng = np.random.default_rng(7)
    # Drive transmit and receive on the same tick, as the real pacing loop does:
    # we send a loud frame, an attenuated copy of it comes back ~18 dB down,
    # which is what a speakerphone echo looks like.
    call.rtp.stop_draining()
    call.rtp.on_sent = session._on_sent
    for _ in range(80):
        session._on_sent((rng.normal(0, 6000, 160)).astype(np.int16))
        session._on_audio((rng.normal(0, 700, 160)).astype(np.int16))
        await asyncio.sleep(0.002)
    await asyncio.sleep(0.2)
    assert call.rtp.cleared == cleared_before, "echo was treated as barge-in"
    assert session.barge_in.echo_rejected > 0, "echo guard never fired"
    print(f"PASS echo at -18 dB did not interrupt "
          f"({session.barge_in.echo_rejected} frames rejected as echo)")

    # A caller talking at a comparable level must still get through
    cleared_before = call.rtp.cleared
    for _ in range(40):
        session._on_sent((rng.normal(0, 6000, 160)).astype(np.int16))
        session._on_audio((rng.normal(0, 5500, 160)).astype(np.int16))
        await asyncio.sleep(0.002)
    await asyncio.sleep(0.2)
    assert call.rtp.cleared > cleared_before, "a real caller at speech level was blocked"
    print("PASS a caller at comparable level still interrupts")

    session._done.set()
    await asyncio.wait_for(runner, 3)
    return True


async def test_transfer_method_passed_through():
    """The configured transfer method must reach the user agent."""
    from helpdesk.sip.ua import TransferMethod

    config = build_config(transfer_method="dtmf", transfer_dtmf_feature_code="*2")
    session, call, ua, synth, llm = await build_session(
        ["Ich will einen Menschen"], ["Ich verbinde. [WEITERLEITEN]"], config=config
    )
    runner = asyncio.ensure_future(session.run())
    await wait_until(lambda: call.rtp is not None, 2, "answer")
    call.rtp.start_draining(speed=20)
    await feed(session, speech(30) + silence(15))
    await wait_until(lambda: ua.transfers, 5, "transfer")
    assert ua.transfer_methods == [TransferMethod.DTMF], ua.transfer_methods
    assert session.transfer_feature_code == "*2"
    assert session.metrics.end_reason == "transferred-dtmf", session.metrics.end_reason
    print("PASS transfer_method=dtmf reaches the UA with the configured feature code")
    await asyncio.wait_for(runner, 3)
    return True


async def main() -> int:
    tests = [
        ("greeting + full turn", test_greeting_and_turn),
        ("transfer on marker", test_transfer_on_marker),
        ("transfer failure fallback", test_transfer_failure_falls_back),
        ("barge-in", test_barge_in_cancels_reply),
        ("unrecognised speech", test_unrecognised_speech_prompts_then_transfers),
        ("silence watchdog", test_silence_watchdog),
        ("dtmf 0", test_dtmf_zero_transfers),
        ("backchannel resumes", test_backchannel_resumes_instead_of_restarting),
        ("real interruption asks again", test_real_interruption_does_ask_again),
        ("echo does not interrupt", test_echo_does_not_interrupt),
        ("transfer method passthrough", test_transfer_method_passed_through),
        ("farewell ends call", test_farewell_ends_the_call),
    ]
    failed = 0
    for name, test in tests:
        print(f"\n=== {name} ===")
        try:
            await asyncio.wait_for(test(), 60)
        except Exception as exc:
            failed += 1
            import traceback

            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} session tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    sys.exit(asyncio.run(main()))
