"""Embedding backends for the knowledge base.

Ollama is the default because it is already running, which keeps the dependency
list short.  The fastembed path exists for the case where VRAM is tight: an ONNX
multilingual model on CPU embeds a query in a few milliseconds and costs no GPU
memory at all.
"""

from __future__ import annotations

import abc
import logging
from typing import List, Optional, Sequence

log = logging.getLogger(__name__)


class Embedder(abc.ABC):
    #: contributes to the cache fingerprint so a model change forces a rebuild
    identity: str = "unknown"

    @abc.abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        ...

    async def embed_query(self, text: str) -> List[float]:
        result = await self.embed_documents([text])
        return result[0]


class OllamaEmbedder(Embedder):
    def __init__(self, client, model: str = "bge-m3", batch_size: int = 16,
                 query_prefix: str = "", document_prefix: str = "") -> None:
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.identity = f"ollama:{model}:q={query_prefix}:d={document_prefix}"

    async def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        prepared = [f"{self.document_prefix}{t}" for t in texts]
        for start in range(0, len(prepared), self.batch_size):
            batch = prepared[start : start + self.batch_size]
            out.extend(await self.client.embed(batch, model=self.model))
        return out

    async def embed_query(self, text: str) -> List[float]:
        vectors = await self.client.embed([f"{self.query_prefix}{text}"], model=self.model)
        return vectors[0]


class FastEmbedEmbedder(Embedder):
    """CPU ONNX embeddings; keeps the GPU free for ASR/LLM/TTS."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small",
                 query_prefix: str = "query: ", document_prefix: str = "passage: ",
                 cache_dir: Optional[str] = None, threads: int = 4) -> None:
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.cache_dir = cache_dir
        self.threads = threads
        self.identity = f"fastembed:{model_name}"
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding  # noqa: PLC0415

            log.info("loading fastembed model %s", self.model_name)
            self._model = TextEmbedding(
                model_name=self.model_name, cache_dir=self.cache_dir, threads=self.threads
            )
        return self._model

    async def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        import asyncio  # noqa: PLC0415

        model = self._load()
        prepared = [f"{self.document_prefix}{t}" for t in texts]
        loop = asyncio.get_running_loop()
        vectors = await loop.run_in_executor(None, lambda: list(model.embed(prepared)))
        return [v.tolist() for v in vectors]

    async def embed_query(self, text: str) -> List[float]:
        import asyncio  # noqa: PLC0415

        model = self._load()
        loop = asyncio.get_running_loop()
        vectors = await loop.run_in_executor(
            None, lambda: list(model.query_embed([f"{self.query_prefix}{text}"]))
        )
        return vectors[0].tolist()


def build_embedder(config: dict, ollama_client=None) -> Embedder:
    backend = (config.get("backend") or "ollama").lower()
    if backend == "ollama":
        if ollama_client is None:
            raise ValueError("ollama embedder requires a client")
        return OllamaEmbedder(
            ollama_client,
            model=config.get("model", "bge-m3"),
            batch_size=int(config.get("batch_size", 16)),
            query_prefix=config.get("query_prefix", ""),
            document_prefix=config.get("document_prefix", ""),
        )
    if backend in ("fastembed", "cpu", "onnx"):
        return FastEmbedEmbedder(
            model_name=config.get("model", "intfloat/multilingual-e5-small"),
            query_prefix=config.get("query_prefix", "query: "),
            document_prefix=config.get("document_prefix", "passage: "),
            cache_dir=config.get("cache_dir") or None,
            threads=int(config.get("threads", 4)),
        )
    raise ValueError(f"unknown embedding backend: {backend!r}")
