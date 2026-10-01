"""Voice activity detection and utterance endpointing.

The endpointer is the single biggest lever on perceived latency: every
millisecond of hangover is a millisecond the caller waits in silence.  It runs on
20 ms frames of 16 kHz PCM and emits events the session loop turns into
recognise / interrupt actions.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .codec import rms_dbfs

log = logging.getLogger(__name__)


class VadEvent(enum.Enum):
    NONE = "none"
    SPEECH_START = "speech_start"
    SPECULATIVE_END = "speculative_end"
    SPEECH_END = "speech_end"
    MAX_DURATION = "max_duration"


class EnergyVad:
    """Adaptive-threshold energy gate.

    Fallback when neither webrtcvad nor silero is installed.  It tracks the noise
    floor of the line and calls anything a configurable margin above it speech,
    which is adequate for G.711 telephony where the noise floor is stable.
    """

    def __init__(self, margin_db: float = 9.0, floor_init_db: float = -45.0) -> None:
        self.margin_db = margin_db
        self.noise_floor = floor_init_db

    def is_speech(self, frame: np.ndarray, sample_rate: int = 16000) -> bool:
        level = rms_dbfs(frame)
        if level < self.noise_floor:
            self.noise_floor += 0.25 * (level - self.noise_floor)
        else:
            self.noise_floor += 0.01 * (level - self.noise_floor)
        self.noise_floor = max(-70.0, min(-20.0, self.noise_floor))
        return level > self.noise_floor + self.margin_db


class WebrtcVad:
    def __init__(self, aggressiveness: int = 2) -> None:
        import webrtcvad  # noqa: PLC0415 - optional dependency

        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame: np.ndarray, sample_rate: int = 16000) -> bool:
        return self._vad.is_speech(np.asarray(frame, dtype=np.int16).tobytes(), sample_rate)


class SileroVad:
    """Silero VAD via torch.  More robust against line noise, ~1 ms per frame."""

    def __init__(self, threshold: float = 0.5) -> None:
        import torch  # noqa: PLC0415 - optional dependency

        self._torch = torch
        self.threshold = threshold
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
        model.eval()
        self._model = model

    def is_speech(self, frame: np.ndarray, sample_rate: int = 16000) -> bool:
        torch = self._torch
        # Silero expects a fixed window: 512 samples at 16 kHz, 256 at 8 kHz.
        window = 256 if sample_rate == 8000 else 512
        buf = np.asarray(frame, dtype=np.float32) / 32768.0
        if buf.size < window:
            buf = np.pad(buf, (0, window - buf.size))
        else:
            buf = buf[:window]
        with torch.no_grad():
            prob = float(self._model(torch.from_numpy(buf), sample_rate).item())
        return prob >= self.threshold


def build_vad(backend: str, aggressiveness: int = 2):
    """Pick a VAD, degrading gracefully so a missing wheel never kills a call."""
    backend = (backend or "auto").lower()
    order = {
        "auto": ("webrtc", "silero", "energy"),
        "webrtc": ("webrtc",),
        "silero": ("silero",),
        "energy": ("energy",),
    }.get(backend, ("webrtc", "energy"))

    for name in order:
        try:
            if name == "webrtc":
                return WebrtcVad(aggressiveness)
            if name == "silero":
                return SileroVad()
            return EnergyVad()
        except Exception as exc:  # pragma: no cover - depends on environment
            log.warning("VAD backend %s unavailable (%s)", name, exc)
    return EnergyVad()


@dataclass
class EndpointerConfig:
    frame_ms: int = 20
    #: 8 kHz: the telephony rate, so detection needs no resampling in the hot
    #: path.  Only the finished utterance is upsampled, once, for the recogniser.
    sample_rate: int = 8000
    #: consecutive voiced frames needed to declare speech
    start_frames: int = 3
    #: trailing silence that ends a turn
    end_silence_ms: int = 420
    #: earlier silence mark that lets us start recognising optimistically
    speculative_silence_ms: int = 220
    #: hard cap so a noisy line cannot hold the turn forever
    max_utterance_ms: int = 20000
    #: audio kept before the trigger so we do not clip the first syllable
    pre_roll_ms: int = 300
    #: while the bot talks, require this much speech before treating it as barge-in
    barge_in_ms: int = 260


@dataclass
class Endpointer:
    """Turn a stream of 20 ms frames into utterance boundaries."""

    config: EndpointerConfig
    vad: object

    in_speech: bool = False
    voiced_run: int = 0
    silence_run: int = 0
    speculative_fired: bool = False
    speech_frames: int = 0
    _pre_roll: list = field(default_factory=list)
    _buffer: list = field(default_factory=list)

    @property
    def _frames_per_ms(self) -> float:
        return 1.0 / self.config.frame_ms

    def _ms_to_frames(self, ms: int) -> int:
        return max(1, int(round(ms / self.config.frame_ms)))

    def reset(self) -> None:
        self.in_speech = False
        self.voiced_run = 0
        self.silence_run = 0
        self.speculative_fired = False
        self.speech_frames = 0
        self._buffer.clear()

    def push(self, frame: np.ndarray) -> VadEvent:
        cfg = self.config
        voiced = bool(self.vad.is_speech(frame, cfg.sample_rate))

        if not self.in_speech:
            max_pre = self._ms_to_frames(cfg.pre_roll_ms)
            self._pre_roll.append(frame)
            if len(self._pre_roll) > max_pre:
                del self._pre_roll[: len(self._pre_roll) - max_pre]

            if voiced:
                self.voiced_run += 1
            else:
                self.voiced_run = 0

            if self.voiced_run >= cfg.start_frames:
                self.in_speech = True
                self.silence_run = 0
                self.speculative_fired = False
                self.speech_frames = self.voiced_run
                self._buffer = list(self._pre_roll)
                self._pre_roll = []
                return VadEvent.SPEECH_START
            return VadEvent.NONE

        self._buffer.append(frame)
        self.speech_frames += 1

        if voiced:
            self.silence_run = 0
            self.speculative_fired = False
        else:
            self.silence_run += 1

        if self.speech_frames * cfg.frame_ms >= cfg.max_utterance_ms:
            self.in_speech = False
            return VadEvent.MAX_DURATION

        silence_ms = self.silence_run * cfg.frame_ms
        if silence_ms >= cfg.end_silence_ms:
            # Latch out of speech so a caller who forgets reset() gets one event,
            # not one per frame.  The buffer survives until the next SPEECH_START.
            self.in_speech = False
            return VadEvent.SPEECH_END
        if not self.speculative_fired and silence_ms >= cfg.speculative_silence_ms:
            self.speculative_fired = True
            return VadEvent.SPECULATIVE_END
        return VadEvent.NONE

    def utterance(self, trim_trailing_silence: bool = True) -> np.ndarray:
        """Collected audio for the current turn, as 16 kHz int16."""
        if not self._buffer:
            return np.zeros(0, dtype=np.int16)
        frames = self._buffer
        if trim_trailing_silence and self.silence_run > 1:
            keep = len(frames) - (self.silence_run - 1)
            frames = frames[: max(1, keep)]
        return np.concatenate(frames).astype(np.int16)

    def utterance_ms(self) -> int:
        return len(self._buffer) * self.config.frame_ms


class BargeInDetector:
    """Separate, deliberately stricter detector used while the bot is speaking.

    A single voiced frame must not cut the bot off -- line echo and the caller's
    own "mhm" would make the agent unusable.  Requiring a sustained run plus a
    level margin keeps false triggers rare.
    """

    def __init__(self, vad, min_speech_ms: int = 260, frame_ms: int = 20, level_margin_db: float = 6.0) -> None:
        self.vad = vad
        self.frames_needed = max(1, int(round(min_speech_ms / frame_ms)))
        self.level_margin_db = level_margin_db
        self.run = 0
        self.reference_db = -45.0

    def reset(self) -> None:
        self.run = 0

    def push(self, frame: np.ndarray, sample_rate: int = 16000) -> bool:
        loud_enough = rms_dbfs(frame) > self.reference_db + self.level_margin_db
        if loud_enough and self.vad.is_speech(frame, sample_rate):
            self.run += 1
        else:
            self.run = max(0, self.run - 1)
        if self.run >= self.frames_needed:
            self.run = 0
            return True
        return False
