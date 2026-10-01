"""Text shaping for spoken output.

Two jobs: split an LLM token stream into utterances that can be synthesised the
moment they are complete, and rewrite text so a TTS engine pronounces it the way
a German speaker would say it out loud.
"""

from __future__ import annotations

import re
from typing import Iterator, List

#: abbreviations whose trailing dot does not end a sentence
_ABBREVIATIONS = (
    "z.b", "bzw", "bspw", "ca", "evtl", "ggf", "inkl", "exkl", "max", "min",
    "nr", "usw", "u.a", "d.h", "o.ä", "z.t", "vgl", "abs", "art", "bzgl",
    "tel", "str", "mio", "mrd", "dr", "prof", "hr", "fr", "ggfs", "etc",
)

_SENTENCE_END = re.compile(r"([.!?…]+)(\s|$)")
_CLAUSE_END = re.compile(r"([,;:])(\s|$)")

_SPOKEN_REPLACEMENTS = [
    (re.compile(r"\bz\.\s?B\.", re.IGNORECASE), "zum Beispiel"),
    (re.compile(r"\bd\.\s?h\.", re.IGNORECASE), "das heißt"),
    (re.compile(r"\bu\.\s?a\.", re.IGNORECASE), "unter anderem"),
    (re.compile(r"\bbzw\.", re.IGNORECASE), "beziehungsweise"),
    (re.compile(r"\bggf\.", re.IGNORECASE), "gegebenenfalls"),
    (re.compile(r"\bevtl\.", re.IGNORECASE), "eventuell"),
    (re.compile(r"\binkl\.", re.IGNORECASE), "inklusive"),
    (re.compile(r"\busw\.", re.IGNORECASE), "und so weiter"),
    (re.compile(r"\betc\.", re.IGNORECASE), "et cetera"),
    (re.compile(r"\bNr\.", re.IGNORECASE), "Nummer"),
    (re.compile(r"\bTel\.", re.IGNORECASE), "Telefon"),
    (re.compile(r"\bStr\.", re.IGNORECASE), "Straße"),
    (re.compile(r"\bMio\.", re.IGNORECASE), "Millionen"),
    (re.compile(r"\bMrd\.", re.IGNORECASE), "Milliarden"),
    (re.compile(r"\bMwSt\.", re.IGNORECASE), "Mehrwertsteuer"),
    (re.compile(r"\bAbs\.", re.IGNORECASE), "Absatz"),
    (re.compile(r"\bvgl\.", re.IGNORECASE), "vergleiche"),
    (re.compile(r"&"), " und "),
    (re.compile(r"\s*/\s*"), " oder "),
    (re.compile(r"\bca\.", re.IGNORECASE), "circa"),
    (re.compile(r"(\d)\s*%"), r"\1 Prozent"),
    (re.compile(r"(\d)\s*€"), r"\1 Euro"),
    (re.compile(r"€\s*(\d)", re.IGNORECASE), r"\1 Euro"),
]

#: markdown and other artefacts that must never be read aloud
_STRIP_MARKUP = [
    (re.compile(r"```.*?```", re.DOTALL), " "),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"\*\*([^*]*)\*\*"), r"\1"),
    (re.compile(r"\*([^*]*)\*"), r"\1"),
    (re.compile(r"^\s*[#>]+\s*", re.MULTILINE), ""),
    (re.compile(r"^\s*[-*•]\s+", re.MULTILINE), ""),
    (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),
    (re.compile(r"https?://\S+"), "dem Link"),
    # Emoji and symbol ranges a TTS engine would either skip or mangle.
    (re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0000FE0F]"), ""),
]


def strip_markup(text: str) -> str:
    for pattern, replacement in _STRIP_MARKUP:
        text = pattern.sub(replacement, text)
    return re.sub(r"[ \t]+", " ", text).strip()


def spoken_form(text: str) -> str:
    """Rewrite abbreviations and symbols into words.

    A TTS engine reading "z.B." as letters is one of the fastest ways to make an
    agent sound synthetic, and it costs nothing to fix here.
    """
    text = strip_markup(text)
    for pattern, replacement in _SPOKEN_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _ends_with_abbreviation(text: str) -> bool:
    tail = text.rstrip()
    if not tail.endswith("."):
        return False
    word = re.split(r"[\s(]", tail[:-1])[-1].lower()
    if word in _ABBREVIATIONS:
        return True
    # A single letter plus dot is an initial or an abbreviation, not an end.
    return len(word) == 1 and word.isalpha()


def split_sentences(text: str) -> List[str]:
    """Split on real sentence ends, keeping abbreviations intact.

    The test has to be on the text up to the matched dot, not on the end of the
    accumulated buffer, or "z.B. 3 Dinge." breaks into two.
    """
    out: List[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        if _ends_with_abbreviation(text[: match.end(1)]):
            continue
        chunk = text[start : match.end()].strip()
        if chunk:
            out.append(chunk)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


class SentenceStreamer:
    """Accumulate LLM tokens and release speakable chunks as early as possible.

    The first chunk dominates perceived latency, so it is released at the first
    clause boundary once a small minimum is reached; later chunks wait for real
    sentence ends, which gives the engine better prosody.
    """

    def __init__(
        self,
        first_chunk_min_chars: int = 24,
        min_chars: int = 60,
        max_chars: int = 220,
    ) -> None:
        self.first_chunk_min_chars = first_chunk_min_chars
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buffer = ""
        self._emitted = 0

    def feed(self, token: str) -> Iterator[str]:
        self._buffer += token
        while True:
            chunk = self._take()
            if chunk is None:
                return
            yield chunk

    def _threshold(self) -> int:
        return self.first_chunk_min_chars if self._emitted == 0 else self.min_chars

    def _take(self):
        buffer = self._buffer
        if not buffer.strip():
            return None

        threshold = self._threshold()

        match = None
        for candidate in _SENTENCE_END.finditer(buffer):
            head = buffer[: candidate.end()]
            if _ends_with_abbreviation(buffer[: candidate.end(1)]):
                continue
            if len(head.strip()) >= threshold:
                match = candidate
                break

        # Before the first chunk, a comma is a good enough place to start talking.
        if match is None and self._emitted == 0:
            for candidate in _CLAUSE_END.finditer(buffer):
                if len(buffer[: candidate.end()].strip()) >= threshold:
                    match = candidate
                    break

        if match is None:
            if len(buffer) >= self.max_chars:
                cut = buffer.rfind(" ", 0, self.max_chars)
                if cut <= 0:
                    cut = self.max_chars
                chunk = buffer[:cut].strip()
                self._buffer = buffer[cut:]
                self._emitted += 1
                return chunk
            return None

        chunk = buffer[: match.end()].strip()
        self._buffer = buffer[match.end():]
        self._emitted += 1
        return chunk

    def flush(self) -> List[str]:
        remainder = self._buffer.strip()
        self._buffer = ""
        if not remainder:
            return []
        self._emitted += 1
        return [remainder]

    def reset(self) -> None:
        self._buffer = ""
        self._emitted = 0
