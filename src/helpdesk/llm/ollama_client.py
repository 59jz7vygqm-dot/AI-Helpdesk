"""Streaming Ollama client.

Only two things matter here for latency: keep the model resident
(``keep_alive: -1``) so no call pays a load, and stream tokens so the first
sentence can be spoken while the rest is still being generated.
"""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator, Dict, List, Optional

log = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        *,
        model: str = "qwen2.5:7b-instruct-q4_K_M",
        keep_alive=-1,
        timeout: float = 60.0,
        options: Optional[Dict] = None,
        think: Optional[bool] = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.options = options or {}
        #: Reasoning models (Qwen3, DeepSeek-R1 and friends) emit a thinking block
        #: before answering.  On a phone call that is dead air, so it is off by
        #: default; None leaves the model's own default alone.
        self.think = think
        self._session = None

    @property
    def _keep_alive_value(self):
        """Ollama wants a number of seconds or a duration with a unit.

        A bare "-1" as a string is parsed as a duration and rejected with
        'missing unit in duration', so numeric values are sent as JSON numbers
        (-1 meaning "keep loaded indefinitely") and only real durations such as
        "30m" are passed through as strings.
        """
        value = self.keep_alive
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return value
        text = str(value).strip()
        try:
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return text

    async def _get_session(self):
        import aiohttp  # noqa: PLC0415

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout, sock_connect=5)
            )
        return self._session

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        *,
        options: Optional[Dict] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Yield response text deltas."""
        session = await self._get_session()
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self._keep_alive_value,
            "options": {**self.options, **(options or {})},
        }
        if self.think is not None:
            payload["think"] = self.think
        started = time.monotonic()
        first = True
        async with session.post(f"{self.base_url}/api/chat", json=payload) as response:
            if response.status == 400 and "think" in payload:
                body = await response.text()
                if "think" in body.lower():
                    # This build or model does not accept the flag; drop it and
                    # remember, so we do not pay the round trip again.
                    log.info("ollama rejected think=%s, disabling the flag", payload["think"])
                    self.think = None
                    payload.pop("think")
                    async for delta in self.chat_stream(messages, options=options, model=model):
                        yield delta
                    return
                raise OllamaError(f"ollama 400: {body[:300]}")
            if response.status != 200:
                body = await response.text()
                raise OllamaError(f"ollama {response.status}: {body[:300]}")
            async for raw in response.content:
                line = raw.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in event:
                    raise OllamaError(str(event["error"]))
                message = event.get("message") or {}
                # A thinking model may still return reasoning in its own field;
                # it is never spoken.
                delta = message.get("content") or ""
                if delta:
                    if first:
                        log.debug("ollama first token after %d ms", int((time.monotonic() - started) * 1000))
                        first = False
                    yield delta
                if event.get("done"):
                    break

    async def embed(self, texts: List[str], *, model: str) -> List[List[float]]:
        session = await self._get_session()
        async with session.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": texts, "keep_alive": self._keep_alive_value},
        ) as response:
            if response.status == 404:
                # Older Ollama builds only have the single-input endpoint.
                return [await self._embed_one(text, model) for text in texts]
            if response.status != 200:
                body = await response.text()
                raise OllamaError(f"ollama embed {response.status}: {body[:300]}")
            data = await response.json()
        vectors = data.get("embeddings") or []
        if not vectors:
            raise OllamaError(f"embedding model {model!r} returned nothing")
        return vectors

    async def _embed_one(self, text: str, model: str) -> List[float]:
        session = await self._get_session()
        async with session.post(
            f"{self.base_url}/api/embeddings",
            json={"model": model, "prompt": text, "keep_alive": self._keep_alive_value},
        ) as response:
            if response.status != 200:
                body = await response.text()
                raise OllamaError(f"ollama embeddings {response.status}: {body[:300]}")
            data = await response.json()
        return data.get("embedding") or []

    async def ensure_model(self, model: Optional[str] = None) -> None:
        """Load the model now and fail loudly if it is not installed."""
        name = model or self.model
        session = await self._get_session()
        async with session.get(f"{self.base_url}/api/tags") as response:
            if response.status != 200:
                raise OllamaError(f"cannot reach ollama at {self.base_url}")
            data = await response.json()
        installed = {m.get("name", "") for m in data.get("models", [])}
        installed |= {name.split(":")[0] for name in installed}
        if name not in installed and name.split(":")[0] not in installed:
            raise OllamaError(
                f"model {name!r} is not installed. Run: ollama pull {name}\n"
                f"available: {', '.join(sorted(n for n in installed if ':' in n)) or 'none'}"
            )

    async def warmup(self, model: Optional[str] = None) -> float:
        started = time.monotonic()
        async for _ in self.chat_stream(
            [{"role": "user", "content": "Sag nur: ok"}],
            options={"num_predict": 4},
            model=model,
        ):
            pass
        return time.monotonic() - started

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
