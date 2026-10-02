"""Application wiring: load config, warm up the models, register, serve calls."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from typing import Optional

from .asr.faster_whisper_asr import FasterWhisperRecognizer
from .config import load_config, validate
from .kb.embedder import build_embedder
from .kb.index import KnowledgeBase
from .llm.agent import HelpdeskAgent
from .llm.ollama_client import OllamaClient, OllamaError
from .llm.openai_client import LlmError, OpenAiCompatibleClient
from .session import CallSession, DialogTexts
from .sip.ua import Call, SipAccount, SipUserAgent
from .tts.registry import PhraseCache, build_synthesizer

log = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # These are chatty at DEBUG and say nothing useful about the call.
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.INFO)
    # Model downloads emit one INFO line per HTTP request, which buries
    # everything else on startup.
    for noisy in ("httpx", "httpcore", "urllib3", "filelock",
                  "huggingface_hub", "hf_transfer", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class HelpdeskApplication:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.recognizer = None
        self.synthesizer = None
        self.llm: Optional[OllamaClient] = None
        self.kb: Optional[KnowledgeBase] = None
        self.phrases: Optional[PhraseCache] = None
        self.ua: Optional[SipUserAgent] = None
        self.texts: Optional[DialogTexts] = None
        self._shutdown = asyncio.Event()
        self._active = 0

    # ---- startup -------------------------------------------------------
    async def prepare(self) -> None:
        config = self.config
        dialog = config["dialog"]

        self.texts = DialogTexts(
            greeting=dialog.get("greeting", ""),
            transfer_announcement=dialog.get("transfer_announcement", ""),
            transfer_failed=dialog.get("transfer_failed", ""),
            goodbye=dialog.get("goodbye", ""),
            not_understood=dialog.get("not_understood", ""),
            still_there=dialog.get("still_there", ""),
            thinking=dialog.get("thinking", "") or "",
            fillers=[f for f in (dialog.get("fillers") or []) if f],
        )

        llm_config = config["llm"]
        backend = str(llm_config.get("backend", "ollama")).lower()
        if backend in ("openai", "openai_compatible", "vllm", "http"):
            self.llm = OpenAiCompatibleClient(
                base_url=llm_config["base_url"],
                model=llm_config["model"],
                api_key=str(llm_config.get("api_key", "none")),
                timeout=float(llm_config.get("timeout", 60)),
                options=llm_config.get("options") or {},
                think=llm_config.get("think", False),
            )
        elif backend == "ollama":
            self.llm = OllamaClient(
                base_url=llm_config["base_url"],
                model=llm_config["model"],
                keep_alive=llm_config.get("keep_alive", -1),
                timeout=float(llm_config.get("timeout", 60)),
                options=llm_config.get("options") or {},
                think=llm_config.get("think", False),
            )
        else:
            raise SystemExit(
                f"unknown llm.backend {backend!r} (expected 'ollama' or 'openai')"
            )
        try:
            await self.llm.ensure_model()
        except (OllamaError, LlmError) as exc:
            raise SystemExit(f"the language model is not usable: {exc}") from exc

        asr_config = config["asr"]
        self.recognizer = FasterWhisperRecognizer(
            model_size=asr_config["model"],
            device=asr_config["device"],
            device_index=int(asr_config.get("device_index", 0)),
            compute_type=asr_config["compute_type"],
            language=asr_config["language"],
            beam_size=int(asr_config.get("beam_size", 1)),
            download_root=asr_config.get("download_root") or None,
            initial_prompt=asr_config.get("initial_prompt") or None,
            min_avg_logprob=float(asr_config.get("min_avg_logprob", -1.1)),
            max_no_speech_prob=float(asr_config.get("max_no_speech_prob", 0.75)),
            cpu_threads=int(asr_config.get("cpu_threads", 4)),
            vad_filter=bool(asr_config.get("vad_filter", True)),
            min_utterance_ms=int(asr_config.get("min_utterance_ms", 350)),
        )

        # onnxruntime reads its thread limits when it is first imported, and the
        # TTS backends import it lazily, so this has to happen before the
        # synthesizer is constructed. Piper on CPU is otherwise capped at whatever
        # OMP_NUM_THREADS happened to be, which on a big host leaves cores idle.
        piper_threads = int((config["tts"].get("piper") or {}).get("threads", 0))
        if piper_threads > 0:
            os.environ["OMP_NUM_THREADS"] = str(piper_threads)
            os.environ["ORT_INTRA_OP_NUM_THREADS"] = str(piper_threads)
            log.info("synthesis thread limit set to %d", piper_threads)

        self.synthesizer = build_synthesizer(config["tts"])
        # Fail fast on a voice that cannot load: this is a pure config error and
        # must not cost a multi-minute model download before it surfaces.
        try:
            await self.synthesizer.warmup()
        except Exception as exc:
            backend = config["tts"].get("backend")
            raise SystemExit(
                f"the TTS backend {backend!r} cannot be used: {exc}\n"
                f"Set tts.backend to 'piper' in config/config.yaml (and "
                f"HELPDESK_TTS_BACKEND / TTS_BACKEND in .env if set), or rebuild "
                f"the image with TTS_PROFILE=quality for chatterbox."
            ) from exc
        await self._check_voice_keeps_up()
        self.phrases = PhraseCache(
            self.synthesizer, target_rate=8000, cache_dir=config["tts"].get("cache_dir") or None
        )

        knowledge_config = config["knowledge"]
        if knowledge_config.get("enabled", True):
            embedder = build_embedder(knowledge_config.get("embeddings") or {}, self.llm)
            self.kb = KnowledgeBase(
                knowledge_config["directory"],
                embedder,
                cache_path=knowledge_config.get("cache_path") or None,
                max_chars=int(knowledge_config.get("max_chars", 900)),
                overlap_chars=int(knowledge_config.get("overlap_chars", 120)),
                dense_weight=float(knowledge_config.get("dense_weight", 0.72)),
                min_score=float(knowledge_config.get("min_score", 0.28)),
            )

        self.ua = SipUserAgent(
            self._build_account(),
            self._handle_call,
            max_concurrent_calls=int(config["sip"].get("max_concurrent_calls", 1)),
        )
        # Claim the SIP port first: it costs nothing and failing here after a
        # minute of model loading wastes the whole warmup.
        try:
            await self.ua.bind()
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

        # Warm everything before the first call: a cold model load during a live
        # call is the difference between 600 ms and 30 seconds.
        log.info("warming up ...")
        started = time.monotonic()
        await self.recognizer.warmup()
        llm_ms = await self.llm.warmup()
        log.info("LLM warm (%d ms for a short completion)", int(llm_ms * 1000))
        if self.kb is not None:
            await self.kb.build()
            # Loading the index from cache skips the embedder, which would then
            # load during the caller's first question and cost seconds there.
            try:
                await self.kb.search("Testfrage zum Aufwaermen", top_k=1)
                log.info("knowledge retrieval warm")
            except Exception:
                log.warning("could not warm up knowledge retrieval", exc_info=True)
        await self.phrases.prepare_all(self.texts.fixed_phrases())
        log.info("warmup complete in %.1fs", time.monotonic() - started)

    #: one sentence of ordinary length, used to time the voice honestly
    _PROBE = "Der Drucker zeigt eine Meldung im Display, bitte lesen Sie sie mir vor."

    async def _check_voice_keeps_up(self) -> None:
        """Measure the voice against the clock and say plainly if it loses.

        Run after warmup, so kernel compilation and model loading are already
        paid for and this is the speed a caller will actually get. A real-time
        factor above 1 means synthesis is slower than playback: the caller waits
        for every sentence and no tuning elsewhere can close that gap. Sentences
        are synthesised one at a time while the previous one plays, so even 0.5
        is felt at the start of a turn.
        """
        started = time.monotonic()
        try:
            audio = await self.synthesizer.synthesize(self._PROBE)
        except Exception:
            log.debug("voice speed probe failed", exc_info=True)
            return
        rate = max(getattr(self.synthesizer, "sample_rate", 0) or 0, 1)
        audio_s = audio.size / rate
        if audio_s < 0.5:
            return
        elapsed = time.monotonic() - started
        rtf = elapsed / audio_s
        if rtf <= 0.5:
            log.info("voice speed: rtf %.2f (%.1fs for %.1fs of speech)", rtf, elapsed, audio_s)
            return
        level = log.warning if rtf > 1.0 else log.info
        level(
            "voice speed: rtf %.2f -- %.1fs to synthesise %.1fs of speech.%s Try piper, "
            "or chatterbox with TTS_PROFILE=quality.",
            rtf, elapsed, audio_s,
            " Above 1.0 the voice cannot keep up with a call: the caller waits"
            " through every sentence." if rtf > 1.0 else
            " Noticeable at the start of each turn.",
        )

    def _build_account(self) -> SipAccount:
        sip_config = self.config["sip"]
        return SipAccount(
            username=str(sip_config["username"]),
            password=str(sip_config["password"]),
            domain=str(sip_config.get("domain") or sip_config["server_host"]),
            server_host=str(sip_config["server_host"]),
            server_port=int(sip_config.get("server_port", 5060)),
            auth_username=str(sip_config.get("auth_username") or "") or None,
            display_name=str(sip_config.get("display_name", "AI Helpdesk")),
            register_expires=int(sip_config.get("register_expires", 300)),
            bind_host=str(sip_config.get("bind_host", "0.0.0.0")),
            bind_port=int(sip_config.get("bind_port", 5060)),
            advertise_host=str(sip_config.get("advertise_host") or ""),
            rtp_port_range=(
                int(sip_config.get("rtp_port_start", 16000)),
                int(sip_config.get("rtp_port_end", 16200)),
            ),
            codec_preference=list(sip_config.get("codec_preference") or ["PCMA", "PCMU"]),
            trace=bool(sip_config.get("trace", False)),
        )

    # ---- per call ------------------------------------------------------
    async def _handle_call(self, call: Call) -> None:
        dialog = self.config["dialog"]
        agent = HelpdeskAgent(
            self.llm,
            self.kb,
            company=dialog.get("company", "unserem Unternehmen"),
            agent_name=dialog.get("agent_name", "Alex"),
            max_sentences=int(dialog.get("max_sentences", 3)),
            history_turns=int(dialog.get("history_turns", 10)),
            top_k=int(self.config["knowledge"].get("top_k", 3)),
            context_chars=int(self.config["knowledge"].get("context_chars", 1800)),
            options=self.config["llm"].get("options") or {},
            extra_instructions=dialog.get("extra_instructions", ""),
            mode=dialog.get("mode", "helpdesk"),
        )
        session = CallSession(
            call,
            self.ua,
            recognizer=self.recognizer,
            synthesizer=self.synthesizer,
            agent=agent,
            phrases=self.phrases,
            texts=self.texts,
            config=self.config,
        )
        self._active += 1
        try:
            await session.run()
        finally:
            self._active -= 1

    # ---- lifecycle -----------------------------------------------------
    async def run(self) -> None:
        assert self.ua is not None
        await self.ua.start()
        log.info(
            "helpdesk ready: extension %s on %s, transfers go to %s",
            self.config["sip"]["username"],
            self.config["sip"]["server_host"],
            self.config["dialog"].get("transfer_number") or "nowhere (not configured)",
        )
        await self._shutdown.wait()
        log.info("shutting down ...")
        await self.ua.stop()
        for closeable in (self.recognizer, self.synthesizer, self.llm):
            if closeable is not None:
                try:
                    await closeable.close()
                except Exception:  # pragma: no cover
                    log.debug("error closing %s", type(closeable).__name__, exc_info=True)

    def request_shutdown(self) -> None:
        self._shutdown.set()

    async def aclose(self) -> None:
        """Release what was opened, tolerating a half-finished startup."""
        for closeable in (self.recognizer, self.synthesizer, self.llm):
            if closeable is None:
                continue
            try:
                await closeable.close()
            except Exception:  # pragma: no cover
                log.debug("error closing %s", type(closeable).__name__, exc_info=True)


async def amain(config_path: Optional[str]) -> int:
    config = load_config(config_path)
    setup_logging((config.get("logging") or {}).get("level", "INFO"))

    if config_path and not os.path.exists(config_path):
        # Not fatal: the environment may carry everything needed. But say so,
        # because running on defaults is rarely what someone intended.
        log.warning(
            "%s does not exist -- running on defaults plus environment overrides. "
            "Copy a profile into place: cp config/profiles/demo-single-gpu.yaml %s",
            config_path,
            config_path,
        )

    problems = validate(config)
    if problems:
        for problem in problems:
            log.error("configuration problem: %s", problem)
        log.error("refusing to start; see config/config.example.yaml")
        return 2

    app = HelpdeskApplication(config)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, app.request_shutdown)
        except NotImplementedError:  # pragma: no cover - not on this platform
            pass

    try:
        await app.prepare()
        await app.run()
    except BaseException:
        # Includes SystemExit from a config problem: release the HTTP session so
        # the real error is not buried under "Unclosed client session".
        await app.aclose()
        raise
    return 0
