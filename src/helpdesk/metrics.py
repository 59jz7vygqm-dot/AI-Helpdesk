"""Per-turn latency accounting.

The numbers printed here are the ones to tune against: they split the pause the
caller experiences into endpoint wait, recognition, model, and synthesis, so it is
obvious which stage to attack.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class TurnMetrics:
    turn: int = 0
    speech_end_at: float = 0.0
    asr_ms: int = 0
    asr_speculative: bool = False
    #: live recognition passes spent on this turn, and the silence at which the
    #: semantic endpoint fired. Together they say whether running recognition
    #: is earning its GPU time: passes without a low endpoint_ms is cost
    #: without benefit.
    live_passes: int = 0
    endpoint_ms: int = 0
    retrieval_ms: int = 0
    llm_first_token_ms: int = 0
    llm_total_ms: int = 0
    tts_first_chunk_ms: int = 0
    #: what the caller actually feels: end of their speech to start of ours
    response_ms: int = 0
    utterance_ms: int = 0
    reply_chars: int = 0
    barge_in: bool = False

    def summary(self) -> str:
        return (
            f"turn {self.turn}: response {self.response_ms} ms "
            f"(asr {self.asr_ms}{'*' if self.asr_speculative else ''}"
            f"{f'/{self.live_passes}live' if self.live_passes else ''}, "
            f"kb {self.retrieval_ms}, llm_ttft {self.llm_first_token_ms}, "
            f"tts {self.tts_first_chunk_ms}) "
            f"utterance {self.utterance_ms} ms, reply {self.reply_chars} chars"
            + (f", endpoint at {self.endpoint_ms} ms silence" if self.endpoint_ms else "")
        )


@dataclass
class CallMetrics:
    call_id: str = ""
    caller: str = ""
    started_at: float = field(default_factory=time.monotonic)
    turns: List[TurnMetrics] = field(default_factory=list)
    end_reason: str = ""
    transferred: bool = False

    def add(self, turn: TurnMetrics) -> None:
        self.turns.append(turn)
        log.info("%s", turn.summary())

    def report(self) -> Dict[str, float]:
        responses = [t.response_ms for t in self.turns if t.response_ms > 0]
        report = {
            "turns": len(self.turns),
            "duration_s": round(time.monotonic() - self.started_at, 1),
            "barge_ins": sum(1 for t in self.turns if t.barge_in),
        }
        if responses:
            report["response_ms_median"] = round(statistics.median(responses))
            report["response_ms_p90"] = round(
                sorted(responses)[min(len(responses) - 1, int(len(responses) * 0.9))]
            )
            report["response_ms_max"] = max(responses)
        return report

    def log_report(self) -> None:
        report = self.report()
        log.info(
            "call %s from %s ended (%s): %s",
            self.call_id,
            self.caller,
            self.end_reason or "unknown",
            ", ".join(f"{k}={v}" for k, v in report.items()),
        )
