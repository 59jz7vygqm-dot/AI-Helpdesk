# Qwen3-TTS voice service

Qwen3-TTS behind an OpenAI-compatible `/v1/audio/speech` endpoint, so the agent
can use it through its existing `openai` TTS backend.

## Why a separate container

The `qwen3-tts` package requires **Python 3.13**. The agent image is built on an
NVIDIA CUDA base image, which ships Python 3.10, and CTranslate2 (faster-whisper)
needs that base for its cuDNN. Rather than fight both dependency sets into one
image, the voice runs on its own.

This image needs no CUDA base image at all: the torch wheels from the `cu128`
index carry their own CUDA libraries, so a plain `python:3.13-slim` plus `--gpus`
is enough.

## Running

```bash
docker compose -f docker-compose.yml -f docker-compose.qwen.yml up -d --build
docker compose logs -f qwen-tts
```

First start downloads about 5 GB into the shared `/models` volume.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/audio/speech` | Synthesis. Returns raw int16 PCM; `response_format` must be `pcm`. |
| `GET /health` | Readiness, plus the language value and sample rate actually in use. |
| `GET /voices` | Whatever speaker roster this build exposes, for `QWEN_SPEAKER`. |
| `GET /v1/models` | The served model id. |

```bash
curl -s localhost:8880/health
curl -s localhost:8880/voices
curl -s -X POST localhost:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Guten Tag, wie kann ich helfen?","response_format":"pcm"}' \
  --output /tmp/test.pcm
# 24 kHz mono int16:
ffplay -f s16le -ar 24000 -ac 1 /tmp/test.pcm
```

## Configuration

All by environment variable; see `docker-compose.qwen.yml` and `.env.example`.
The two that usually need attention:

- **`QWEN_LANGUAGE`** — builds disagree on whether this is `German` or `de`. The
  server tries several at startup and logs which worked; set it explicitly
  afterwards to skip the search.
- **`QWEN_SPEAKER`** — a built-in speaker name for `QWEN_MODE=custom`. Empty uses
  the model's default. `GET /voices` lists what is available.

## Honest status

This is written against the documented Qwen3-TTS API, not against a running
installation — the model is recent and could not be executed during development.
The code handles the signature variations seen across builds (both
`from_pretrained` forms, tuple and bare return values, several streaming method
names) and the startup warmup fails with a specific message rather than during a
call. Still, expect the first start to need an adjustment to `QWEN_LANGUAGE` or
`QWEN_SPEAKER`.

Piper in the agent image remains the guaranteed fallback: set `tts.backend: piper`
and the call works regardless of what this service is doing.
