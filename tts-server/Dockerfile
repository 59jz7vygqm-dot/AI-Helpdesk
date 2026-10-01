# Python 3.13 because the qwen3-tts package requires it. No CUDA base image: the
# torch wheels from the cu128 index bring their own CUDA libraries, so the only
# host requirement is a recent NVIDIA driver plus --gpus.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch first and in its own layer: it is the big one, and a later failure should
# not re-download it.
RUN pip install --upgrade pip \
    && pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY server.py /app/server.py

VOLUME ["/models"]
EXPOSE 8880

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8880/health || exit 1

ENTRYPOINT ["python3", "/app/server.py"]
