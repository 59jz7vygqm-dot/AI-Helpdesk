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
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .audio.codec import resample
from .audio.vad import BargeInDetector, Endpointer, EndpointerConfig, VadEvent, build_vad
from .llm.agent import Action, HelpdeskAgent
from .metrics import CallMetrics, TurnMetrics
from .sip.ua import Call, SipUserAgent, TransferError, TransferMethod
from .tts.base import Synthesizer
from .tts.registry import PhraseCache
from .tts.text import SentenceStreamer, is_backchannel

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

    def fixed_phrases(self) -> List[str]:
        return [
            self.greeting,
            self.transfer_announcement,
            self.transfer_failed,
            self.goodbye,
            self.not_understood,
            self.still_there,
            self.thinking,
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

        vad = build_vad(vad_config.get("backend", "auto"), int(vad_config.get("aggressiveness", 2)))
        self.endpointer = Endpointer(
            EndpointerConfig(
                frame_ms=20,
                sample_rate=TELEPHONY_RATE,
                start_frames=int(vad_config.get("start_frames", 3)),
                end_silence_ms=int(vad_config.get("end_silence_ms", 420)),
                speculative_silence_ms=int(vad_config.get("speculative_silence_ms", 220)),
                max_utterance_ms=int(vad_config.get("max_utterance_ms", 20000)),
                pre_roll_ms=int(vad_config.get("pre_roll_ms", 300)),
                barge_in_ms=int(vad_config.get("barge_in_ms", 260)),
            ),
            vad,
        )
        self.barge_in = BargeInDetector(
            build_vad(vad_config.get("backend", "auto"), 3),
            min_speech_ms=int(vad_config.get("barge_in_ms", 260)),
            frame_ms=20,
            echo_guard=bool(vad_config.get("echo_guard", True)),
            echo_attenuation_db=float(vad_config.get("echo_attenuation_db", 12.0)),
            echo_correlation=float(vad_config.get("echo_correlation", 0.72)),
        )
        self.resume_on_backchannel = bool(vad_config.get("resume_on_backchannel", True))

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
        self._cancel_speech.set()
        if self.call.rtp:
            self.call.rtp.clear_playout()
        self._speaking = False

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
        self._speculative = asyncio.ensure_future(self._recognize(audio))

    def _drop_speculative(self) -> None:
        if self._speculative is not None:
            self._speculative.cancel()
            self._speculative = None

    async def _finish_recognition(self, metrics: TurnMetrics):
        audio = self.endpointer.utterance(trim_trailing_silence=True)
        metrics.utterance_ms = int(audio.size * 1000 / TELEPHONY_RATE)
        started = time.monotonic()

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
        elif event is VadEvent.SPECULATIVE_END:
            self._start_speculative()
        elif event in (VadEvent.SPEECH_END, VadEvent.MAX_DURATION):
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

        transcript = await self._finish_recognition(metrics)
        self.endpointer.reset()

        if transcript is None or transcript.is_empty:
            self.agent.misunderstood += 1
            log.info(
                "nothing recognised (attempt %d/%d)", self.agent.misunderstood, self.max_misunderstood
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
