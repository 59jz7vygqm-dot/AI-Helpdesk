"""Per-call orchestration: the loop that turns a phone call into a conversation.

The latency budget this is built around, measured from the moment the caller
stops speaking:

    endpoint hangover   ~420 ms   (vad.end_silence_ms, the dominant term)
    recognition          ~80 ms   (mostly hidden by speculative ASR)
    retrieval            ~15 ms
    model first token   ~150 ms
    synthesis first bit  ~60 ms   (Piper; ~400 ms with Chatterbox)
    -------------------------------
    first audio out     ~600 ms

Three tricks do most of the work: recognition starts at a speculative silence
mark before the hangover expires, the reply is synthesised clause by clause as
tokens arrive, and fixed phrases are pre-rendered so the greeting costs nothing.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .audio.codec import resample
from .audio.vad import BargeInDetector, Endpointer, EndpointerConfig, VadEvent, build_vad
from .llm.agent import Action, HelpdeskAgent
from .metrics import CallMetrics, TurnMetrics
from .sip.ua import Call, SipUserAgent, TransferError, TransferMethod
from .tts.base import Synthesizer
from .tts.registry import PhraseCache
from .tts.text import SentenceStreamer, is_backchannel, is_farewell

log = logging.getLogger(__name__)

TELEPHONY_RATE = 8000
ASR_RATE = 16000


@dataclass
class DialogTexts:
    greeting: str
    transfer_announcement: str
    transfer_failed: str
    goodbye: str
    not_understood: str
    still_there: str
    thinking: str = ""
    #: Short interjections played while the answer is still being produced. A
    #: person says "einen Moment" instead of going silent, and several variants
    #: are needed because the same one every time sounds more mechanical than
    #: the silence it replaces.
    fillers: List[str] = field(default_factory=list)

    def fixed_phrases(self) -> List[str]:
        return [
            self.greeting,
            self.transfer_announcement,
            self.transfer_failed,
            self.goodbye,
            self.not_understood,
            self.still_there,
            self.thinking,
            *self.fillers,
        ]


class CallSession:
    """Drives one call from answer to hang-up."""

    def __init__(
        self,
        call: Call,
        agent_ua: SipUserAgent,
        *,
        recognizer,
        synthesizer: Synthesizer,
        agent: HelpdeskAgent,
        phrases: PhraseCache,
        texts: DialogTexts,
        config: dict,
    ) -> None:
        self.call = call
        self.ua = agent_ua
        self.recognizer = recognizer
        self.synthesizer = synthesizer
        self.agent = agent
        self.phrases = phrases
        self.texts = texts
        self.config = config

        vad_config = config["vad"]
        dialog = config["dialog"]
        self.dialog = dialog
        self.transfer_number = str(dialog.get("transfer_number") or "")
        try:
            self.transfer_method = TransferMethod(str(dialog.get("transfer_method", "auto")).lower())
        except ValueError:
            log.warning(
                "unknown transfer_method %r, using auto", dialog.get("transfer_method")
            )
            self.transfer_method = TransferMethod.AUTO
        self.transfer_feature_code = str(dialog.get("transfer_dtmf_feature_code", "##"))
        self.transfer_dtmf_delay_ms = int(dialog.get("transfer_dtmf_delay_ms", 700))
        self.transfer_dtmf_terminator = str(dialog.get("transfer_dtmf_terminator", "") or "")
        self.max_misunderstood = int(dialog.get("max_misunderstood", 2))
        self.silence_prompt_after_ms = int(dialog.get("silence_prompt_after_ms", 7000))
        self.silence_hangup_after_ms = int(dialog.get("silence_hangup_after_ms", 20000))
        self.max_call_seconds = int(dialog.get("max_call_seconds", 900))
        self.speculative_asr = bool(vad_config.get("speculative_asr", True))
        #: answer as soon as the running transcript looks like a complete
        #: sentence, rather than waiting out end_silence_ms. This is where the
        #: remaining latency is: the hangover is paid on every single turn.
        self.semantic_endpointing = bool(vad_config.get("semantic_endpointing", True))
        self.semantic_min_words = int(vad_config.get("semantic_min_words", 3))
        self.live_asr = bool(vad_config.get("live_asr", True))
        #: new audio needed before another pass is launched. Every pass is a
        #: full encode of the buffer, on the same GPU as synthesis, so this is
        #: a GPU-duty-cycle knob as much as a latency one.
        self.live_interval_ms = int(vad_config.get("live_interval_ms", 500))
        #: no point recognising less than this; it only yields noise words
        self.live_min_audio_ms = int(vad_config.get("live_min_audio_ms", 600))
        self._speculative_result = None

        vad = build_vad(vad_config.get("backend", "auto"), int(vad_config.get("aggressiveness", 2)))
        barge_in_ms = int(vad_config.get("barge_in_ms", 260))
        # The pre-roll has to outlast the barge-in decision, otherwise the
        # frames that proved the caller was talking have already scrolled out of
        # it by the time the turn starts. The margin covers the detector's own
        # hesitation: a non-voiced frame decrements its run, so confirming an
        # interruption takes longer than barge_in_ms of wall clock.
        pre_roll_ms = max(int(vad_config.get("pre_roll_ms", 300)), barge_in_ms + 240)
        self.endpointer = Endpointer(
            EndpointerConfig(
                frame_ms=20,
                sample_rate=TELEPHONY_RATE,
                start_frames=int(vad_config.get("start_frames", 3)),
                end_silence_ms=int(vad_config.get("end_silence_ms", 420)),
                speculative_silence_ms=int(vad_config.get("speculative_silence_ms", 220)),
                max_utterance_ms=int(vad_config.get("max_utterance_ms", 20000)),
                pre_roll_ms=pre_roll_ms,
                barge_in_ms=barge_in_ms,
            ),
            vad,
        )
        self.barge_in = BargeInDetector(
            build_vad(vad_config.get("backend", "auto"), 3),
            min_speech_ms=barge_in_ms,
            frame_ms=20,
            echo_guard=bool(vad_config.get("echo_guard", True)),
            echo_attenuation_db=float(vad_config.get("echo_attenuation_db", 12.0)),
            echo_correlation=float(vad_config.get("echo_correlation", 0.72)),
        )
        self.resume_on_backchannel = bool(vad_config.get("resume_on_backchannel", True))
        self.farewell_ends_call = bool(dialog.get("farewell_ends_call", True))
        self.farewell_min_ms = int(dialog.get("farewell_min_ms", 600))
        # Below this an utterance is not worth a reply of any kind, not even an
        # apology. Tied to the recogniser's own floor so the two cannot disagree.
        self.min_turn_ms = int(config.get("asr", {}).get("min_utterance_ms", 350))
        self.filler_after_ms = int(dialog.get("filler_after_ms", 700))
        self._last_filler = ""
        self._filler_task: Optional[asyncio.Task] = None

        tts_config = config["tts"]
        self._streamer_kwargs = dict(
            first_chunk_min_chars=int(tts_config.get("first_chunk_min_chars", 24)),
            min_chars=int(tts_config.get("min_chunk_chars", 60)),
            max_chars=int(tts_config.get("max_chunk_chars", 220)),
        )

        self._frames: "asyncio.Queue[np.ndarray]" = asyncio.Queue(maxsize=200)
        self._partial = np.zeros(0, dtype=np.int16)
        self._frame_samples = TELEPHONY_RATE * 20 // 1000

        self._speaking = False
        self._cancel_speech = asyncio.Event()
        #: the in-flight reply pipeline for the current turn, cancelled on barge-in
        self._turn_task: Optional[asyncio.Task] = None
        self._speculative: Optional[asyncio.Task] = None
        #: Running recognition while the caller is still talking. The point is
        #: not a lower ASR latency for its own sake -- it is that the semantic
        #: endpoint has a hypothesis ready at the first silent frame. Before
        #: this it could not fire until speculative_silence_ms plus a whole
        #: recognition had elapsed, which is longer than end_silence_ms, so it
        #: lost the race to the plain hangover and that hangover was paid in
        #: full on nearly every turn.
        self._live: Optional[asyncio.Task] = None
        self._live_result = None
        #: the last two hypotheses, for the LocalAgreement-2 prefix
        self._live_texts: List[str] = []
        #: samples the most recently launched pass covered
        self._live_covered = 0
        self._live_passes = 0
        #: silence at which the semantic endpoint fired, 0 when the plain
        #: hangover ended the turn
        self._endpoint_ms = 0
        self._barge_in_flag = False
        self._turn = 0
        self._done = asyncio.Event()
        self._last_caller_audio = time.monotonic()
        self._silence_prompted = False
        self._pending_action = Action.NONE
        #: (sentence, cumulative sample index at which it finishes) for the reply
        #: being spoken, against samples actually transmitted.  Audio sitting in
        #: the playout queue has not been heard yet, so "generated" is the wrong
        #: measure -- only what went on the wire counts.
        self._speech_plan: List[Tuple[str, int]] = []
        self._queued_samples = 0
        self._sent_samples = 0
        self._interrupted_remainder = ""
        self.metrics = CallMetrics(call_id=call.call_id, caller=call.caller_number)
        self.transcript: List[dict] = []

    # ---- audio intake --------------------------------------------------
    def _on_audio(self, pcm: np.ndarray) -> None:
        """Called from the event loop by the RTP protocol; must not block."""
        try:
            self._frames.put_nowait(pcm)
        except asyncio.QueueFull:
            # Dropping the oldest frame is better than growing an unbounded
            # backlog: stale audio would make every later turn feel laggy.
            with contextlib.suppress(asyncio.QueueEmpty):
                self._frames.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                self._frames.put_nowait(pcm)

    def _on_sent(self, frame: Optional[np.ndarray]) -> None:
        """Per-tick notification of what went out.

        Feeds the echo guard and counts transmitted speech, which is what decides
        how much of an interrupted answer the caller actually heard.
        """
        if frame is None:
            self.barge_in.note_silence()
        else:
            self.barge_in.note_sent(frame)
            self._sent_samples += int(frame.size)

    def _on_dtmf(self, digit: str) -> None:
        log.info("DTMF %s on %s", digit, self.call.call_id)
        # A caller pressing 0 is asking for a human in the clearest way there is,
        # so act on it immediately rather than at the end of some later turn.
        if digit == "0" and self.transfer_number and not self._done.is_set():
            self._cancel_turn()
            # Parking it in _turn_task keeps the frame loop from starting a new
            # turn while the hand-off is in progress.
            self._turn_task = asyncio.ensure_future(self._run_action(self._do_transfer()))

    async def _run_action(self, coro) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("action failed on %s", self.call.call_id)

    def _iter_frames(self, pcm: np.ndarray):
        """Re-chunk arriving RTP payloads into exact 20 ms frames."""
        buffer = np.concatenate([self._partial, pcm]) if self._partial.size else pcm
        count = buffer.size // self._frame_samples
        for index in range(count):
            start = index * self._frame_samples
            yield buffer[start : start + self._frame_samples]
        self._partial = buffer[count * self._frame_samples :]

    # ---- speaking ------------------------------------------------------
    async def _play(self, audio: np.ndarray) -> None:
        if self.call.rtp is None or audio.size == 0:
            return
        self._queued_samples += int(audio.size)
        self.call.rtp.enqueue(audio)

    def _reset_speech_plan(self) -> None:
        self._speech_plan = []
        self._queued_samples = 0
        self._sent_samples = 0

    def _unheard_text(self) -> str:
        """The part of the current reply that was never transmitted."""
        unheard = [text for text, end_sample in self._speech_plan if end_sample > self._sent_samples]
        return " ".join(unheard).strip()

    async def _speak_cached(self, text: str) -> int:
        """Play a pre-rendered phrase; returns the audio duration in ms."""
        if not text:
            return 0
        audio = self.phrases.get(text)
        if audio is None:
            audio = await self.phrases.prepare(text)
        self._speaking = True
        self.barge_in.reset()
        await self._play(audio)
        self._log_turn("assistant", text)
        return int(audio.size * 1000 / TELEPHONY_RATE)

    async def _wait_for_playout(self, extra_ms: int = 0) -> bool:
        """Wait until the playout queue drains; False if interrupted."""
        rtp = self.call.rtp
        if rtp is None:
            return False
        while rtp.is_playing():
            if self._cancel_speech.is_set() or self._done.is_set():
                return False
            await asyncio.sleep(0.02)
        if extra_ms:
            try:
                await asyncio.wait_for(self._cancel_speech.wait(), timeout=extra_ms / 1000)
                return False
            except asyncio.TimeoutError:
                pass
        self._speaking = False
        return True

    def _interrupt_speech(self) -> None:
        """Stop producing and discard what is queued, in that order."""
        self._cancel_filler()
        self._cancel_speech.set()
        if self.call.rtp:
            self.call.rtp.clear_playout()
        self._speaking = False

    def _pick_filler(self) -> str:
        """A filler, never the same one twice in a row."""
        options = [f for f in self.texts.fillers if f and f != self._last_filler]
        if not options:
            options = [f for f in self.texts.fillers if f]
        if not options:
            return ""
        chosen = random.choice(options)
        self._last_filler = chosen
        return chosen

    async def _maybe_filler(self) -> None:
        if not self.texts.fillers or self.filler_after_ms <= 0:
            return
        try:
            await asyncio.sleep(self.filler_after_ms / 1000)
        except asyncio.CancelledError:
            return
        if self._cancel_speech.is_set() or self._done.is_set():
            return
        if self._bot_audio_playing():
            return  # the real answer already started
        filler = self._pick_filler()
        if not filler:
            return
        audio = self.phrases.get(filler)
        if audio is None:
            return  # not pre-rendered; never synthesise one on the critical path
        log.info("filler %r while the answer is still coming", filler)
        await self._play(audio)

    def _cancel_filler(self) -> None:
        if self._filler_task is not None and not self._filler_task.done():
            self._filler_task.cancel()
        self._filler_task = None

    async def _resume_speaking(self, text: str) -> None:
        """Speak text that an interruption cut short, without asking the model."""
        self._cancel_speech.clear()
        self._speaking = True
        self.barge_in.reset()
        self._reset_speech_plan()
        streamer = SentenceStreamer(**self._streamer_kwargs)
        sentences = list(streamer.feed(text + " ")) + streamer.flush()
        for sentence in sentences:
            if self._cancel_speech.is_set():
                return
            async for piece in self.synthesizer.stream(sentence, cancel=self._cancel_speech):
                if self._cancel_speech.is_set():
                    return
                await self._play(resample(piece.pcm, piece.sample_rate, TELEPHONY_RATE))
            self._speech_plan.append((sentence, self._queued_samples))
        self._log_turn("assistant", text)

    async def _speak_reply(self, user_text: str, metrics: TurnMetrics) -> None:
        """Stream the model's answer and synthesise it clause by clause."""
        self._cancel_speech.clear()
        self._speaking = True
        self.barge_in.reset()
        streamer = SentenceStreamer(**self._streamer_kwargs)
        first_audio_logged = False
        turn_started = time.monotonic()
        self._reset_speech_plan()
        self._interrupted_remainder = ""

        # If producing the answer takes long enough to be noticeable, say
        # something rather than leaving the line silent.
        self._filler_task = asyncio.ensure_future(self._maybe_filler())

        retrieval_started = time.monotonic()
        context, _ = await self.agent.retrieve(user_text)
        metrics.retrieval_ms = int((time.monotonic() - retrieval_started) * 1000)

        async def synthesize(chunk: str) -> None:
            nonlocal first_audio_logged
            async for piece in self.synthesizer.stream(chunk, cancel=self._cancel_speech):
                if self._cancel_speech.is_set():
                    return
                audio = resample(piece.pcm, piece.sample_rate, TELEPHONY_RATE)
                if not first_audio_logged:
                    self._cancel_filler()
                    metrics.tts_first_chunk_ms = int((time.monotonic() - turn_started) * 1000)
                    if metrics.speech_end_at:
                        metrics.response_ms = int((time.monotonic() - metrics.speech_end_at) * 1000)
                    first_audio_logged = True
                await self._play(audio)
            # Record where this sentence ends in the outgoing stream.
            self._speech_plan.append((chunk, self._queued_samples))

        try:
            async for delta in self.agent.respond_stream(user_text, context=context):
                if self._cancel_speech.is_set():
                    break
                for chunk in streamer.feed(delta):
                    await synthesize(chunk)
                    if self._cancel_speech.is_set():
                        break
            if not self._cancel_speech.is_set():
                for chunk in streamer.flush():
                    await synthesize(chunk)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("generating the reply failed")
            await self._speak_cached(self.texts.not_understood)
            return

        reply = self.agent.last_reply
        if reply.repeat_count >= 2 and self.transfer_number:
            # Three near-identical answers means the agent is stuck, whatever it
            # thinks it is doing. Hand over rather than loop.
            log.warning(
                "answer repeated %d times, handing over instead of looping",
                reply.repeat_count,
            )
            self._pending_action = Action.TRANSFER
        metrics.llm_first_token_ms = reply.first_token_ms
        metrics.llm_total_ms = reply.total_ms
        metrics.reply_chars = len(reply.text)
        self._log_turn("assistant", reply.text, sources=reply.retrieved)

        if reply.action is Action.TRANSFER:
            self._pending_action = Action.TRANSFER
        elif reply.action is Action.HANGUP:
            self._pending_action = Action.HANGUP

    # ---- recognition ---------------------------------------------------
    async def _recognize(self, audio_8k: np.ndarray):
        pcm16k = resample(audio_8k, TELEPHONY_RATE, ASR_RATE)
        return await self.recognizer.transcribe(pcm16k)

    def _start_speculative(self) -> None:
        if not self.speculative_asr or self._speculative is not None:
            return
        audio = self.endpointer.utterance(trim_trailing_silence=True)
        if audio.size < TELEPHONY_RATE * 0.25:
            return
        # Recognise what we have now, betting the caller is finished.  If they
        # resume, the result is discarded; if not, the hangover was free.
        self._speculative_result = None
        self._speculative = asyncio.ensure_future(self._recognize(audio))

    # ---- running recognition -------------------------------------------
    def _maybe_start_live(self) -> None:
        """Recognise the buffer so far, if enough new audio has arrived.

        One pass at a time, always. Queueing passes would put the GPU behind
        the caller instead of ahead of them, and the recogniser serialises on
        its own lock anyway -- a second pass would just wait, holding a stale
        hypothesis while the fresh audio sits unrecognised.
        """
        if not self.live_asr:
            return
        if self._live is not None and not self._live.done():
            return
        buffered = self.endpointer.utterance(trim_trailing_silence=False)
        if buffered.size < TELEPHONY_RATE * self.live_min_audio_ms / 1000:
            return
        if buffered.size - self._live_covered < TELEPHONY_RATE * self.live_interval_ms / 1000:
            return
        self._live_covered = buffered.size
        self._live_passes += 1
        self._live = asyncio.ensure_future(self._recognize(buffered))

    def _collect_live(self) -> None:
        """Take the finished hypothesis, if there is one."""
        task = self._live
        if task is None or not task.done():
            return
        self._live = None
        if task.cancelled():
            return
        try:
            result = task.result()
        except Exception:
            log.debug("live recognition failed", exc_info=True)
            return
        if result is None or result.is_empty:
            return
        self._live_result = result
        self._live_texts.append(result.text)
        del self._live_texts[:-2]

    def _committed_prefix(self) -> str:
        """LocalAgreement-2: the prefix two consecutive hypotheses agree on.

        Whisper happily rewrites the tail of its own output as more audio
        arrives, so only the part that survived one more pass is trustworthy.
        Used for logging and for judging completeness, never spoken.
        """
        if len(self._live_texts) < 2:
            return ""
        a, b = self._live_texts[-2].split(), self._live_texts[-1].split()
        shared = 0
        while shared < min(len(a), len(b)) and a[shared] == b[shared]:
            shared += 1
        return " ".join(b[:shared])

    def _drop_live(self) -> None:
        if self._live is not None:
            self._live.cancel()
            self._live = None
        self._live_result = None
        self._live_texts.clear()
        self._live_covered = 0
        # _live_passes and _endpoint_ms deliberately survive: they are read by
        # the turn that this reset is clearing the way for.

    def _finished_hypothesis(self) -> str:
        """The transcript to answer now, or "" to keep waiting.

        Prefers the speculative result, which was recognised on trimmed audio
        and is what the turn will reuse. Falls back to the running hypothesis,
        which is the whole point of live recognition: at the first silent frame
        the speculative pass has not even started yet.
        """
        if self._speculative_is_complete():
            return self._speculative_result.text
        result = self._live_result
        if result is None or result.is_empty:
            return ""
        # Judge the agreed prefix, not the latest guess: Whisper rewrites its
        # own tail as audio arrives, and a sentence-final dot that vanishes on
        # the next pass would have cut the caller off mid-thought.
        if not self._looks_complete(self._committed_prefix()):
            return ""
        self._speculative_result = result
        return result.text

    def _looks_complete(self, text: str) -> bool:
        """Whether a transcript reads as a finished utterance.

        Conservative: a sentence-final mark plus enough words. Cutting someone
        off mid-thought is worse than the 100 ms this saves.
        """
        stripped = (text or "").strip()
        if len(stripped.split()) < self.semantic_min_words:
            return False
        return stripped.endswith((".", "?", "!", "…"))

    def _speculative_is_complete(self) -> bool:
        """True when the pending speculative transcript is already a full turn."""
        task = self._speculative
        if task is None or not task.done() or task.cancelled():
            return False
        try:
            result = task.result()
        except Exception:
            return False
        if result is None or result.is_empty:
            return False
        self._speculative_result = result
        return self._looks_complete(result.text)

    def _drop_speculative(self) -> None:
        if self._speculative is not None:
            self._speculative.cancel()
            self._speculative = None
        self._speculative_result = None
        self._drop_live()

    async def _finish_recognition(self, metrics: TurnMetrics):
        audio = self.endpointer.utterance(trim_trailing_silence=True)
        metrics.utterance_ms = int(audio.size * 1000 / TELEPHONY_RATE)
        started = time.monotonic()

        if self._speculative_result is not None:
            result = self._speculative_result
            self._speculative_result = None
            self._speculative = None
            metrics.asr_ms = int((time.monotonic() - started) * 1000)
            metrics.asr_speculative = True
            return result

        speculative = self._speculative
        self._speculative = None
        if speculative is not None and not speculative.cancelled():
            try:
                result = await speculative
                metrics.asr_ms = int((time.monotonic() - started) * 1000)
                metrics.asr_speculative = True
                return result
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("speculative recognition failed")

        result = await self._recognize(audio)
        metrics.asr_ms = int((time.monotonic() - started) * 1000)
        return result

    # ---- transcript ----------------------------------------------------
    def _log_turn(self, role: str, text: str, sources: Optional[List[str]] = None) -> None:
        if not text:
            return
        entry = {"t": round(self.call.duration, 2), "role": role, "text": text}
        if sources:
            entry["sources"] = sources
        self.transcript.append(entry)
        log.info("[%s] %s", role, text)

    def _write_transcript(self) -> None:
        directory = (self.config.get("logging") or {}).get("transcript_dir") or ""
        if not directory or not self.transcript:
            return
        try:
            os.makedirs(directory, exist_ok=True)
            safe_id = "".join(c for c in self.call.call_id if c.isalnum() or c in "-_")[:60]
            path = os.path.join(directory, f"{int(time.time())}-{safe_id}.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "call_id": self.call.call_id,
                        "caller": self.call.caller_number,
                        "caller_name": self.call.caller_name,
                        "dialled": self.call.dialled_number,
                        "end_reason": self.metrics.end_reason,
                        "transferred": self.metrics.transferred,
                        "metrics": self.metrics.report(),
                        "turns": self.transcript,
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
        except OSError:
            log.warning("could not write transcript", exc_info=True)

    # ---- actions -------------------------------------------------------
    async def _do_transfer(self) -> None:
        if self._done.is_set() or not self.call.active:
            # The caller hung up while the answer was still being produced; there
            # is nobody left to transfer or apologise to.
            log.info("skipping transfer: call already ended (%s)", self.call.end_reason or "hung up")
            self.metrics.end_reason = self.metrics.end_reason or self.call.end_reason or "ended"
            self._done.set()
            return
        if not self.transfer_number:
            log.warning("transfer requested but no transfer_number configured")
            await self._speak_cached(self.texts.transfer_failed)
            await self._wait_for_playout(200)
            await self.ua.hangup(self.call, reason="no-transfer-target")
            return

        await self._speak_cached(self.texts.transfer_announcement)
        # Let the announcement finish, otherwise the caller hears it cut off
        # the instant the PBX moves the leg.
        await self._wait_for_playout(150)
        try:
            used = await self.ua.transfer(
                self.call,
                self.transfer_number,
                method=self.transfer_method,
                feature_code=self.transfer_feature_code,
                dtmf_delay_ms=self.transfer_dtmf_delay_ms,
                dtmf_terminator=self.transfer_dtmf_terminator,
            )
            self.metrics.transferred = True
            self.metrics.end_reason = f"transferred-{used}"
            log.info(
                "transferred %s to %s via %s", self.call.call_id, self.transfer_number, used
            )
            self._done.set()
        except TransferError as exc:
            log.error("transfer failed: %s", exc)
            self.metrics.end_reason = "transfer-failed"
            await self._speak_cached(self.texts.transfer_failed)
            await self._wait_for_playout(200)
            await self.ua.hangup(self.call, reason="transfer-failed")

    async def _do_hangup(self, reason: str) -> None:
        if self.texts.goodbye and reason == "assistant-goodbye":
            await self._speak_cached(self.texts.goodbye)
            await self._wait_for_playout(200)
        self.metrics.end_reason = reason
        await self.ua.hangup(self.call, reason=reason)
        self._done.set()

    async def _apply_pending_action(self) -> bool:
        action = self._pending_action
        self._pending_action = Action.NONE
        if action is Action.TRANSFER:
            await self._do_transfer()
            return True
        if action is Action.HANGUP:
            await self._do_hangup("assistant-goodbye")
            return True
        return False

    # ---- main loop -----------------------------------------------------
    async def run(self) -> None:
        call = self.call
        call.on_end = self._on_call_end

        if self.dialog.get("ring_before_answer", True):
            self.ua.ring(call)
        delay = int(self.dialog.get("answer_delay_ms", 0))
        if delay:
            await asyncio.sleep(delay / 1000)

        await self.ua.answer(
            call, on_audio=self._on_audio, on_dtmf=self._on_dtmf, on_sent=self._on_sent
        )

        self.agent.reset()
        greeting_ms = await self._speak_cached(self.texts.greeting)
        log.debug("greeting queued (%d ms)", greeting_ms)

        watchdog = asyncio.ensure_future(self._watchdog())
        try:
            await self._loop()
        finally:
            watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watchdog
            self._drop_speculative()
            if self._turn_task is not None and not self._turn_task.done():
                self._turn_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._turn_task
            self.metrics.end_reason = self.metrics.end_reason or call.end_reason or "ended"
            self.metrics.log_report()
            self._write_transcript()

    async def _loop(self) -> None:
        """Read media continuously.

        Nothing in here may await the reply pipeline: the loop has to keep
        consuming frames while the agent thinks and speaks, or barge-in could
        never be detected and the queue would fill with stale audio.
        """
        while not self._done.is_set():
            try:
                pcm = await asyncio.wait_for(self._frames.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if not self.call.active:
                    break
                continue
            except asyncio.CancelledError:
                raise

            self._last_caller_audio = time.monotonic()
            for frame in self._iter_frames(pcm):
                if self._done.is_set():
                    break
                self._dispatch_frame(frame)

    @property
    def _turn_in_flight(self) -> bool:
        return self._turn_task is not None and not self._turn_task.done()

    def _bot_audio_playing(self) -> bool:
        return self.call.rtp is not None and self.call.rtp.is_playing()

    def _dispatch_frame(self, frame: np.ndarray) -> None:
        # While a turn is in flight -- thinking or speaking -- or while a cached
        # phrase such as the greeting is still playing, the only question is
        # whether the caller has started talking over us.
        if self._turn_in_flight or self._bot_audio_playing():
            # Feed the pre-roll even now, so the speech that *earned* the
            # barge-in is part of the utterance rather than thrown away.
            self.endpointer.observe(frame)
            if self.barge_in.push(frame, TELEPHONY_RATE):
                log.info("barge-in on %s", self.call.call_id)
                self._cancel_turn()
                self.endpointer.reset()
                self.endpointer.push(frame)
                self._barge_in_flag = True
            return

        event = self.endpointer.push(frame)
        if event is VadEvent.SPEECH_START:
            self._silence_prompted = False
            self._drop_speculative()
            return
        if event is VadEvent.SPECULATIVE_END:
            self._start_speculative()
            return
        if event in (VadEvent.SPEECH_END, VadEvent.MAX_DURATION):
            self._turn_task = asyncio.ensure_future(self._run_turn())
            return

        if self.endpointer.in_speech:
            # Keep a hypothesis current while the caller talks, so the check
            # below has something to judge at the very first silent frame.
            self._collect_live()
            self._maybe_start_live()

        # Still in the hangover: if what the caller said already reads as a
        # finished sentence, there is nothing to wait for.
        if self.semantic_endpointing and self.endpointer.silence_run > 0:
            self._collect_live()
            text = self._finished_hypothesis()
            if text:
                self._endpoint_ms = self.endpointer.silence_run * 20
                log.debug("semantic endpoint after %d ms of silence: %r",
                          self._endpoint_ms, text[:60])
                self.endpointer.in_speech = False
                self._turn_task = asyncio.ensure_future(self._run_turn())

    def _cancel_turn(self) -> None:
        # Remember what the caller never got to hear, so a backchannel ("mhm")
        # can resume instead of restarting the answer.
        remainder = self._unheard_text()
        if remainder:
            self._interrupted_remainder = remainder
            log.debug("interrupted with %d chars unheard", len(remainder))
        self._interrupt_speech()
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()
        self._drop_speculative()

    async def _run_turn(self) -> None:
        try:
            await self._handle_utterance()
        except asyncio.CancelledError:
            log.debug("turn cancelled on %s", self.call.call_id)
            raise
        except Exception:
            log.exception("turn failed on %s", self.call.call_id)
        finally:
            self._speaking = False

    async def _handle_utterance(self) -> None:
        self._turn += 1
        metrics = TurnMetrics(turn=self._turn, speech_end_at=time.monotonic())
        metrics.barge_in = self._barge_in_flag
        self._barge_in_flag = False
        metrics.live_passes = self._live_passes
        metrics.endpoint_ms = self._endpoint_ms
        self._live_passes = 0
        self._endpoint_ms = 0

        # Decided before recognising, so no backend has to be trusted to have
        # its own floor: too little audio was never a sentence -- a cough, a
        # door, the tail of an interruption. Replying to one is actively
        # harmful, because the agent then talks over a caller who is still
        # speaking, whose next words barge in on another fragment. That loop is
        # what "it does not listen at all" sounds like from the other end.
        pending = self.endpointer.utterance_ms()
        if pending < self.min_turn_ms and self._speculative_result is None:
            metrics.utterance_ms = pending
            log.info(
                "ignoring %d ms of audio: too short to be an utterance (under %d ms)",
                pending,
                self.min_turn_ms,
            )
            self._drop_speculative()
            self.endpointer.reset()
            self.metrics.add(metrics)
            return

        transcript = await self._finish_recognition(metrics)
        self.endpointer.reset()

        if transcript is None or transcript.is_empty:
            self.agent.misunderstood += 1
            self.metrics.add(metrics)
            log.info(
                "nothing recognised in %d ms of audio (attempt %d/%d)",
                metrics.utterance_ms,
                self.agent.misunderstood,
                self.max_misunderstood,
            )
            if self.agent.misunderstood > self.max_misunderstood and self.transfer_number:
                await self._do_transfer()
                return
            await self._speak_cached(self.texts.not_understood)
            await self._wait_for_playout()
            return

        self.agent.misunderstood = 0
        self._log_turn("caller", transcript.text)

        if (
            self.resume_on_backchannel
            and self._interrupted_remainder
            and is_backchannel(transcript.text)
        ):
            # The caller was only acknowledging; carry on where we left off
            # rather than treating "mhm" as a new question.
            remainder = self._interrupted_remainder
            self._interrupted_remainder = ""
            log.info("backchannel %r: resuming the interrupted answer", transcript.text)
            await self._resume_speaking(remainder)
            self.metrics.add(metrics)
            await self._wait_for_playout()
            return
        self._interrupted_remainder = ""

        if self.farewell_ends_call and is_farewell(transcript.text):
            # Hanging up is irreversible, so require more than a short noisy
            # fragment. Whisper reliably invents a goodbye out of half a second of
            # unclear audio, and nobody calls in order to say goodbye first.
            if self._turn <= 1:
                log.info(
                    "ignoring farewell %r on the first turn -- probably a "
                    "misrecognition, asking instead", transcript.text,
                )
            elif metrics.utterance_ms < self.farewell_min_ms or not transcript.is_confident:
                log.info(
                    "ignoring farewell %r: %d ms of audio, logprob %.2f -- not "
                    "confident enough to hang up on",
                    transcript.text, metrics.utterance_ms, transcript.avg_logprob,
                )
            else:
                log.info("farewell %r: saying goodbye", transcript.text)
                self.metrics.add(metrics)
                await self._do_hangup("assistant-goodbye")
                return

        if self.texts.thinking and metrics.utterance_ms > 2500:
            # Only for long questions, where retrieval and generation will take
            # long enough that silence would feel like a dropped call.
            await self._speak_cached(self.texts.thinking)

        await self._speak_reply(transcript.text, metrics)
        self.metrics.add(metrics)

        if await self._apply_pending_action():
            return
        await self._wait_for_playout()

    async def _watchdog(self) -> None:
        """Handle silence and runaway calls."""
        while not self._done.is_set():
            await asyncio.sleep(0.5)
            if not self.call.active:
                self._done.set()
                return
            if self.call.duration > self.max_call_seconds:
                log.info("call %s hit the %ds limit", self.call.call_id, self.max_call_seconds)
                await self._do_hangup("max-duration")
                return
            if self._turn_in_flight or (self.call.rtp and self.call.rtp.is_playing()):
                continue

            idle_ms = (time.monotonic() - self._last_caller_audio) * 1000
            rtp = self.call.rtp
            if rtp is not None and rtp.packets_received == 0:
                continue  # media has not started yet
            if idle_ms > self.silence_hangup_after_ms:
                log.info("no audio from caller for %.0f ms, hanging up", idle_ms)
                await self._do_hangup("caller-silent")
                return
            if idle_ms > self.silence_prompt_after_ms and not self._silence_prompted:
                self._silence_prompted = True
                await self._speak_cached(self.texts.still_there)

    def _on_call_end(self, reason: str) -> None:
        self.metrics.end_reason = self.metrics.end_reason or reason
        self._done.set()
        self._cancel_speech.set()
