"""OpenAI-compatible chat client, for vLLM and anything that speaks that API.

Preferred over the Ollama client when an inference server is already running:
vLLM's continuous batching gives a lower and far more stable time-to-first-token
than Ollama, and reusing a server that is already resident means the voice agent's
GPU does not have to hold a language model at all.

Also works with TGI, llama.cpp's server, LM Studio and LocalAI.
"""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator, Dict, List, Optional

log = logging.getLogger(__name__)


class LlmError(RuntimeError):
    pass


#: Option names the Ollama config uses, mapped to their OpenAI equivalents, so
#: one config section works for both backends.
_OPTION_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "num_predict": "max_tokens",
    "max_tokens": "max_tokens",
    "stop": "stop",
    "seed": "seed",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "top_k": "top_k",
    "min_p": "min_p",
    "repetition_penalty": "repetition_penalty",
}

#: Options that are not part of the OpenAI schema and must ride in extra_body.
_VLLM_ONLY = {"top_k", "min_p", "repetition_penalty"}

#: Ollama-only options with no meaning here.
_IGNORED = {"num_ctx", "num_gpu", "num_thread", "keep_alive", "mirostat"}


class OpenAiCompatibleClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        *,
        model: str = "Qwen/Qwen3-8B",
        api_key: str = "none",
        timeout: float = 60.0,
        options: Optional[Dict] = None,
        think: Optional[bool] = False,
        extra_body: Optional[Dict] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.options = options or {}
        self.think = think
        self.extra_body = extra_body or {}
        self._session = None

    async def _get_session(self):
        import aiohttp  # noqa: PLC0415

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout, sock_connect=5),
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        return self._session

    def _build_payload(self, messages: List[Dict[str, str]], options: Optional[Dict], model: Optional[str]) -> Dict:
        merged = {**self.options, **(options or {})}
        payload: Dict = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
        }
        extra: Dict = dict(self.extra_body)

        for key, value in merged.items():
            if key in _IGNORED:
                continue
            target = _OPTION_MAP.get(key)
            if target is None:
                extra[key] = value
            elif key in _VLLM_ONLY:
                extra[target] = value
            else:
                payload[target] = value

        if self.think is False:
            # Qwen3 and other hybrid-reasoning models gate thinking through the
            # chat template.  Several seconds of silent reasoning is dead air on a
            # phone call, so it is switched off at the template level.
            template_kwargs = dict(extra.get("chat_template_kwargs") or {})
            template_kwargs.setdefault("enable_thinking", False)
            extra["chat_template_kwargs"] = template_kwargs

        payload.update(extra)
        return payload

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        *,
        options: Optional[Dict] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        session = await self._get_session()
        payload = self._build_payload(messages, options, model)
        started = time.monotonic()
        first = True

        async with session.post(f"{self.base_url}/chat/completions", json=payload) as response:
            if response.status != 200:
                body = await response.text()
                raise LlmError(f"{self.base_url} returned {response.status}: {body[:300]}")
            async for raw in response.content:
                line = raw.strip()
                if not line or not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                # reasoning_content is the thinking stream; never spoken.
                text = delta.get("content") or ""
                if text:
                    if first:
                        log.debug(
                            "llm first token after %d ms", int((time.monotonic() - started) * 1000)
                        )
                        first = False
                    yield text
                if choices[0].get("finish_reason"):
                    break

    async def embed(self, texts: List[str], *, model: str) -> List[List[float]]:
        """Embeddings, if the server hosts an embedding model."""
        session = await self._get_session()
        async with session.post(
            f"{self.base_url}/embeddings", json={"model": model, "input": texts}
        ) as response:
            if response.status != 200:
                body = await response.text()
                raise LlmError(
                    f"embeddings failed ({response.status}): {body[:200]}\n"
                    "vLLM serves one model per endpoint; use knowledge.embeddings."
                    "backend=fastembed to embed on the CPU instead."
                )
            data = await response.json()
        vectors = [item["embedding"] for item in sorted(
            data.get("data", []), key=lambda d: d.get("index", 0)
        )]
        if not vectors:
            raise LlmError(f"embedding model {model!r} returned nothing")
        return vectors

    async def ensure_model(self, model: Optional[str] = None) -> None:
        """Verify the server is up and actually serving the configured model."""
        name = model or self.model
        session = await self._get_session()
        try:
            async with session.get(f"{self.base_url}/models") as response:
                if response.status != 200:
                    raise LlmError(
                        f"cannot list models at {self.base_url} (HTTP {response.status})"
                    )
                data = await response.json()
        except LlmError:
            raise
        except Exception as exc:
            raise LlmError(f"cannot reach the inference server at {self.base_url}: {exc}") from exc

        served = [m.get("id", "") for m in data.get("data", [])]
        if name not in served:
            # vLLM is usually started with one model, so a mismatch here is a
            # config typo; naming what is served saves a round of guessing.
            raise LlmError(
                f"model {name!r} is not served by {self.base_url}.\n"
                f"available: {', '.join(served) or 'none'}\n"
                f"set llm.model to one of those."
            )
        log.info("inference server at %s serves %s", self.base_url, name)

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
