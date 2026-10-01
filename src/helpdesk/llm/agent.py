"""Dialogue agent: prompt, retrieval, streaming, and control decisions.

Control actions (transfer, hang up) are signalled by markers the model emits in
its text rather than by tool calls.  That is deliberate: markers work with every
Ollama model regardless of tool support, survive streaming, and cost no extra
round trip -- a tool call would force a second request before the first word
could be spoken.  The marker is stripped before anything reaches the TTS.
"""

from __future__ import annotations

import enum
import logging
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

TRANSFER_MARKER = "[WEITERLEITEN]"
HANGUP_MARKER = "[AUFLEGEN]"

#: tolerate the variations models produce around the exact marker spelling
_MARKER_PATTERNS = [
    (re.compile(r"\[+\s*WEITERLEITEN\s*(?::[^\]]*)?\]+", re.IGNORECASE), "transfer"),
    (re.compile(r"\[+\s*(?:AUFLEGEN|ENDE|BEENDEN)\s*(?::[^\]]*)?\]+", re.IGNORECASE), "hangup"),
]

#: longest marker-ish prefix we might be mid-way through while streaming
_MAX_MARKER_LEN = 24
_PARTIAL_MARKER = re.compile(r"\[[^\]]{0,22}$")


class Action(enum.Enum):
    NONE = "none"
    TRANSFER = "transfer"
    HANGUP = "hangup"


DEFAULT_SYSTEM_PROMPT = """\
Du bist {agent_name}, die telefonische Serviceassistenz von {company}.
Du sprichst mit einem Anrufer am Telefon. Antworte ausschließlich auf Deutsch.

So sprichst du:
- Kurz und natürlich, wie am Telefon. Ein bis zwei Sätze, maximal {max_sentences}.
- Keine Aufzählungen, keine Listen, keine Sonderzeichen, keine Emojis, kein Markdown.
- Keine Links und keine E-Mail-Adressen vorlesen.
- Zahlen und Uhrzeiten ausgeschrieben, wie man sie sagt.
- Stelle immer nur eine Frage auf einmal.
- Wiederhole dich nicht und fasse nicht ständig zusammen.
- Wenn der Anrufer dich unterbricht, gehe sofort auf das Neue ein.

Deine Aufgabe:
- Beantworte Fragen zu {company} ausschließlich mit den Informationen im Abschnitt WISSEN.
- Stelle kurze Rückfragen, wenn dir eine Angabe fehlt, um weiterzuhelfen.
- Erfinde nichts. Keine Preise, Termine, Namen oder Zusagen, die nicht im WISSEN stehen.

Wenn du nicht helfen kannst, leite weiter:
- Schreibe dann {transfer_marker} an das Ende deiner Antwort.
- Sage davor in einem Satz, dass du verbindest, zum Beispiel: "Einen Moment, ich verbinde Sie mit einem Kollegen."
- Leite weiter, wenn: die Information nicht im WISSEN steht, der Anrufer ausdrücklich einen Menschen will,
  es um Kündigung, Reklamation, Rechtliches oder eine Eskalation geht, oder du den Anrufer zweimal nicht verstanden hast.
- Verspreche niemals einen Rückruf und nenne niemals die Zielrufnummer.

Wenn der Anrufer sich verabschiedet oder das Gespräch beenden will:
- Verabschiede dich in einem kurzen Satz und schreibe {hangup_marker} an das Ende.

Die Marker sind Steuerzeichen. Sage sie nicht vor und erkläre sie nicht.
"""

NO_KNOWLEDGE_NOTE = "(Keine passenden Informationen gefunden.)"


@dataclass
class Turn:
    role: str
    content: str


@dataclass
class AgentReply:
    text: str = ""
    action: Action = Action.NONE
    first_token_ms: int = 0
    total_ms: int = 0
    retrieved: List[str] = field(default_factory=list)


class MarkerFilter:
    """Strips control markers from a token stream without leaking fragments.

    Holding back a trailing partial "[WEITERL..." is what keeps the caller from
    hearing the marker read aloud when it lands across two tokens.
    """

    def __init__(self) -> None:
        self._pending = ""
        self.action = Action.NONE

    def feed(self, token: str) -> str:
        self._pending += token
        emit, self._pending = self._scan(self._pending, final=False)
        return emit

    def flush(self) -> str:
        emit, self._pending = self._scan(self._pending, final=True)
        return emit

    def _scan(self, buffer: str, final: bool) -> Tuple[str, str]:
        while True:
            matched = False
            for pattern, kind in _MARKER_PATTERNS:
                match = pattern.search(buffer)
                if match:
                    if kind == "transfer":
                        self.action = Action.TRANSFER
                    elif self.action is Action.NONE:
                        self.action = Action.HANGUP
                    buffer = buffer[: match.start()] + " " + buffer[match.end():]
                    matched = True
                    break
            if not matched:
                break

        if final:
            # Any dangling bracket at the very end was an unterminated marker.
            return re.sub(r"\[[^\]]*$", "", buffer), ""

        partial = _PARTIAL_MARKER.search(buffer)
        if partial:
            return buffer[: partial.start()], buffer[partial.start():]
        # Also hold back a lone trailing '[' in case the marker starts next token.
        if buffer.endswith("["):
            return buffer[:-1], "["
        return buffer, ""


class HelpdeskAgent:
    def __init__(
        self,
        client,
        knowledge_base=None,
        *,
        company: str = "unserem Unternehmen",
        agent_name: str = "Alex",
        system_prompt: Optional[str] = None,
        max_sentences: int = 3,
        history_turns: int = 10,
        top_k: int = 3,
        context_chars: int = 1800,
        options: Optional[Dict] = None,
        extra_instructions: str = "",
    ) -> None:
        self.client = client
        self.kb = knowledge_base
        self.company = company
        self.agent_name = agent_name
        self.max_sentences = max_sentences
        self.history_turns = history_turns
        self.top_k = top_k
        self.context_chars = context_chars
        self.options = options or {}
        self.extra_instructions = extra_instructions
        self.system_prompt_template = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.history: List[Turn] = []
        #: consecutive turns we failed to understand; drives the hand-off rule
        self.misunderstood = 0

    def reset(self) -> None:
        self.history.clear()
        self.misunderstood = 0

    def system_prompt(self) -> str:
        prompt = self.system_prompt_template.format(
            company=self.company,
            agent_name=self.agent_name,
            max_sentences=self.max_sentences,
            transfer_marker=TRANSFER_MARKER,
            hangup_marker=HANGUP_MARKER,
        )
        if self.extra_instructions:
            prompt = f"{prompt}\n{self.extra_instructions.strip()}\n"
        return prompt

    def _messages(self, user_text: str, context: str) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt()}]
        for turn in self.history[-self.history_turns :]:
            messages.append({"role": turn.role, "content": turn.content})
        knowledge = context.strip() or NO_KNOWLEDGE_NOTE
        messages.append(
            {
                "role": "user",
                "content": f"WISSEN:\n{knowledge}\n\nANRUFER SAGT:\n{user_text}",
            }
        )
        return messages

    async def retrieve(self, query: str) -> Tuple[str, List[str]]:
        if self.kb is None:
            return "", []
        try:
            hits = await self.kb.search(query, top_k=self.top_k)
        except Exception:
            log.exception("knowledge lookup failed")
            return "", []
        if not hits:
            return "", []
        context = self.kb.format_context(hits, max_chars=self.context_chars)
        sources = [h.chunk.citation for h in hits]
        log.info(
            "kb hits for %r: %s",
            query[:60],
            ", ".join(f"{h.chunk.citation}({h.score:.2f})" for h in hits),
        )
        return context, sources

    async def respond_stream(
        self, user_text: str, *, context: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Yield speakable text deltas; inspect ``last_reply`` afterwards."""
        started = time.monotonic()
        retrieved: List[str] = []
        if context is None:
            context, retrieved = await self.retrieve(user_text)

        messages = self._messages(user_text, context)
        marker_filter = MarkerFilter()
        spoken: List[str] = []
        first_token_ms = 0

        async for delta in self.client.chat_stream(messages, options=self.options):
            if not first_token_ms:
                first_token_ms = int((time.monotonic() - started) * 1000)
            clean = marker_filter.feed(delta)
            if clean:
                spoken.append(clean)
                yield clean

        tail = marker_filter.flush()
        if tail:
            spoken.append(tail)
            yield tail

        text = re.sub(r"\s{2,}", " ", "".join(spoken)).strip()
        self.last_reply = AgentReply(
            text=text,
            action=marker_filter.action,
            first_token_ms=first_token_ms,
            total_ms=int((time.monotonic() - started) * 1000),
            retrieved=retrieved,
        )
        self.history.append(Turn(role="user", content=user_text))
        self.history.append(Turn(role="assistant", content=text or "(keine Antwort)"))

    last_reply: AgentReply = AgentReply()
