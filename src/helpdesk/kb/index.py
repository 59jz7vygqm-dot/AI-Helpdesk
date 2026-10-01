"""Knowledge base: markdown in, retrieved passages out.

For a helpdesk corpus (tens to a few thousand chunks) a numpy dot product over
normalised embeddings is faster than any vector database -- the whole matrix fits
in L2 and a query costs microseconds -- so there is no server to run and nothing
to keep in sync.  Retrieval mixes embeddings with a lexical score, because German
callers say exact product names and error codes that dense vectors blur.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Keep hyphenated identifiers whole: error codes and model numbers ("E-512",
# "HP-4500") are exactly the terms a caller spells out and a dense vector blurs.
_WORD = re.compile(r"[\wäöüßÄÖÜ]+(?:[-/.][\wäöüßÄÖÜ]+)*", re.UNICODE)
_SPLIT_INNER = re.compile(r"[-/.]")

#: German function words carry no retrieval signal
_STOPWORDS = {
    "der", "die", "das", "und", "oder", "ist", "sind", "ein", "eine", "einen",
    "einem", "einer", "ich", "sie", "er", "es", "wir", "ihr", "mein", "meine",
    "mit", "für", "von", "zu", "zum", "zur", "auf", "aus", "bei", "dem", "den",
    "des", "im", "in", "an", "am", "als", "auch", "nicht", "kein", "keine",
    "wie", "was", "wo", "wann", "warum", "wer", "wenn", "dass", "habe", "haben",
    "hat", "kann", "können", "muss", "soll", "will", "würde", "bitte", "danke",
    "ja", "nein", "aber", "noch", "nur", "schon", "sehr", "mehr", "man", "mir",
    "mich", "sich", "dir", "dich", "uns", "euch", "ihnen", "the", "a", "of",
}


def tokenize(text: str) -> List[str]:
    """Lowercase content tokens, with compound identifiers kept and also split.

    Indexing both "e-512" and "512" means the chunk is found whether the caller
    says the code as one word or the recogniser writes it apart.
    """
    out: List[str] = []
    for match in _WORD.finditer(text):
        token = match.group(0).lower().strip("-/.")
        if not token or token in _STOPWORDS:
            continue
        if len(token) > 1:
            out.append(token)
        if _SPLIT_INNER.search(token):
            for part in _SPLIT_INNER.split(token):
                if part and part not in _STOPWORDS and (len(part) > 1 or part.isdigit()):
                    out.append(part)
    return out


@dataclass
class Chunk:
    text: str
    source: str
    title: str = ""
    chunk_id: int = 0

    @property
    def citation(self) -> str:
        return f"{self.title or self.source}"


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0


def split_markdown(text: str, source: str, *, max_chars: int = 900, overlap_chars: int = 120) -> List[Chunk]:
    """Chunk along headings, then along paragraphs when a section is too long.

    Heading-aware splitting keeps a FAQ answer together with its question, which
    is what makes short retrieved passages usable as an answer.
    """
    lines = text.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    heading = ""
    body: List[str] = []
    for line in lines:
        match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if match:
            if body or heading:
                sections.append((heading, body))
            heading = match.group(2).strip()
            body = []
        else:
            body.append(line)
    if body or heading:
        sections.append((heading, body))

    chunks: List[Chunk] = []
    index = 0
    for title, content in sections:
        content_text = "\n".join(content).strip()
        if not content_text:
            continue
        prefix = f"{title}\n" if title else ""
        whole = prefix + content_text
        if len(whole) <= max_chars:
            chunks.append(Chunk(text=whole.strip(), source=source, title=title, chunk_id=index))
            index += 1
            continue

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content_text) if p.strip()]
        buffer = ""
        for paragraph in paragraphs:
            candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
            if len(prefix) + len(candidate) <= max_chars:
                buffer = candidate
                continue
            if buffer:
                chunks.append(Chunk(text=(prefix + buffer).strip(), source=source, title=title, chunk_id=index))
                index += 1
                tail = buffer[-overlap_chars:] if overlap_chars else ""
                buffer = (tail + "\n\n" + paragraph).strip() if tail else paragraph
            else:
                # A single oversized paragraph: hard-wrap it on word boundaries.
                words = paragraph.split()
                piece = ""
                for word in words:
                    if len(prefix) + len(piece) + len(word) + 1 > max_chars:
                        chunks.append(Chunk(text=(prefix + piece).strip(), source=source, title=title, chunk_id=index))
                        index += 1
                        piece = word
                    else:
                        piece = f"{piece} {word}".strip()
                buffer = piece
        if buffer:
            chunks.append(Chunk(text=(prefix + buffer).strip(), source=source, title=title, chunk_id=index))
            index += 1
    return chunks


class KnowledgeBase:
    """Hybrid dense + lexical retrieval over a markdown directory."""

    def __init__(
        self,
        directory: str,
        embedder,
        *,
        cache_path: Optional[str] = None,
        max_chars: int = 900,
        overlap_chars: int = 120,
        dense_weight: float = 0.72,
        min_score: float = 0.28,
    ) -> None:
        self.directory = directory
        self.embedder = embedder
        self.cache_path = cache_path
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.dense_weight = dense_weight
        self.min_score = min_score

        self.chunks: List[Chunk] = []
        self.vectors: Optional[np.ndarray] = None
        self._doc_tokens: List[Dict[str, int]] = []
        self._idf: Dict[str, float] = {}
        self._doc_lengths: np.ndarray = np.zeros(0)

    # ---- building ------------------------------------------------------
    def _load_documents(self) -> List[Chunk]:
        chunks: List[Chunk] = []
        if not os.path.isdir(self.directory):
            log.warning("knowledge directory %s does not exist", self.directory)
            return chunks
        for root, _, files in os.walk(self.directory):
            for name in sorted(files):
                if not name.lower().endswith((".md", ".markdown", ".txt")):
                    continue
                path = os.path.join(root, name)
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        text = handle.read()
                except OSError as exc:
                    log.warning("cannot read %s: %s", path, exc)
                    continue
                relative = os.path.relpath(path, self.directory)
                chunks.extend(
                    split_markdown(
                        text, relative, max_chars=self.max_chars, overlap_chars=self.overlap_chars
                    )
                )
        return chunks

    def _fingerprint(self, chunks: Sequence[Chunk]) -> str:
        digest = hashlib.sha256()
        digest.update(getattr(self.embedder, "identity", "unknown").encode())
        for chunk in chunks:
            digest.update(chunk.text.encode("utf-8"))
        return digest.hexdigest()

    async def build(self, force: bool = False) -> None:
        started = time.monotonic()
        chunks = self._load_documents()
        if not chunks:
            log.warning("knowledge base is empty (%s)", self.directory)
            self.chunks = []
            self.vectors = None
            self._build_lexical()
            return

        fingerprint = self._fingerprint(chunks)
        if not force and self.cache_path and os.path.exists(self.cache_path):
            cached = self._load_cache(fingerprint)
            if cached is not None:
                self.chunks, self.vectors = cached
                self._build_lexical()
                log.info(
                    "knowledge base loaded from cache: %d chunks in %d ms",
                    len(self.chunks),
                    int((time.monotonic() - started) * 1000),
                )
                return

        texts = [c.text for c in chunks]
        vectors = await self.embedder.embed_documents(texts)
        self.chunks = chunks
        self.vectors = _normalize(np.asarray(vectors, dtype=np.float32))
        self._build_lexical()
        if self.cache_path:
            self._save_cache(fingerprint)
        log.info(
            "knowledge base built: %d chunks from %s in %.1fs",
            len(self.chunks),
            self.directory,
            time.monotonic() - started,
        )

    def _build_lexical(self) -> None:
        self._doc_tokens = []
        document_frequency: Dict[str, int] = {}
        for chunk in self.chunks:
            counts: Dict[str, int] = {}
            for token in tokenize(chunk.text):
                counts[token] = counts.get(token, 0) + 1
            self._doc_tokens.append(counts)
            for token in counts:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        total = max(1, len(self.chunks))
        self._idf = {
            token: float(np.log(1.0 + (total - freq + 0.5) / (freq + 0.5)))
            for token, freq in document_frequency.items()
        }
        self._doc_lengths = np.array(
            [max(1, sum(counts.values())) for counts in self._doc_tokens], dtype=np.float32
        )

    # ---- cache ---------------------------------------------------------
    def _save_cache(self, fingerprint: str) -> None:
        try:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            np.savez_compressed(
                self.cache_path,
                fingerprint=np.array([fingerprint]),
                vectors=self.vectors,
                chunks=np.array([json.dumps([asdict(c) for c in self.chunks])]),
            )
        except Exception:  # pragma: no cover
            log.warning("could not write knowledge cache %s", self.cache_path, exc_info=True)

    def _load_cache(self, fingerprint: str):
        try:
            with np.load(self.cache_path, allow_pickle=False) as data:
                if str(data["fingerprint"][0]) != fingerprint:
                    log.info("knowledge cache is stale, rebuilding")
                    return None
                chunks = [Chunk(**item) for item in json.loads(str(data["chunks"][0]))]
                return chunks, data["vectors"].astype(np.float32)
        except Exception:  # pragma: no cover
            log.info("knowledge cache unreadable, rebuilding", exc_info=True)
            return None

    # ---- query ---------------------------------------------------------
    def _lexical_scores(self, query: str) -> np.ndarray:
        """BM25 over the chunk set."""
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        tokens = tokenize(query)
        if not tokens or not self.chunks:
            return scores
        k1, b = 1.5, 0.75
        avg_len = float(np.mean(self._doc_lengths)) if self._doc_lengths.size else 1.0
        for token in set(tokens):
            idf = self._idf.get(token)
            if idf is None:
                continue
            for index, counts in enumerate(self._doc_tokens):
                freq = counts.get(token)
                if not freq:
                    continue
                length = self._doc_lengths[index]
                scores[index] += idf * (freq * (k1 + 1)) / (
                    freq + k1 * (1 - b + b * length / avg_len)
                )
        peak = float(scores.max()) if scores.size else 0.0
        return scores / peak if peak > 0 else scores

    async def search(self, query: str, *, top_k: int = 3) -> List[SearchHit]:
        if not self.chunks or self.vectors is None or not query.strip():
            return []

        query_vector = _normalize(
            np.asarray([await self.embedder.embed_query(query)], dtype=np.float32)
        )[0]
        dense = self.vectors @ query_vector
        lexical = self._lexical_scores(query)
        combined = self.dense_weight * dense + (1.0 - self.dense_weight) * lexical

        order = np.argsort(-combined)[: max(top_k * 3, top_k)]
        hits: List[SearchHit] = []
        for index in order:
            score = float(combined[index])
            if score < self.min_score:
                continue
            hits.append(
                SearchHit(
                    chunk=self.chunks[index],
                    score=score,
                    dense_score=float(dense[index]),
                    lexical_score=float(lexical[index]),
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def format_context(self, hits: Sequence[SearchHit], max_chars: int = 1800) -> str:
        parts: List[str] = []
        used = 0
        for hit in hits:
            block = f"[{hit.chunk.citation}]\n{hit.chunk.text}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n---\n\n".join(parts)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
