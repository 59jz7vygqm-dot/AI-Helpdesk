# CUDA 12.8 with cuDNN 9: 12.8 is what the Qwen3-TTS torch wheels target, and
# cuDNN 9 is what CTranslate2 >= 4.5 (faster-whisper) links against.
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

# quality = Qwen3-TTS on the GPU (needs torch, adds ~5 GB to the image)
# lite    = Piper only, CPU, much smaller image and faster build
ARG TTS_PROFILE=quality

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/models/hf \
    OMP_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        ca-certificates curl \
        libsndfile1 \
        espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-quality.txt /app/
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -r /app/requirements.txt \
    && if [ "$TTS_PROFILE" = "quality" ]; then \
         python3 -m pip install torch torchaudio \
           --index-url https://download.pytorch.org/whl/cu128 \
         && python3 -m pip install -r /app/requirements-quality.txt ; \
       fi

COPY src /app/src
COPY config /app/config
COPY knowledge /app/knowledge
COPY scripts /app/scripts
RUN chmod +x /app/scripts/*.sh /app/scripts/*.py || true

# Model weights and the phrase/knowledge caches live on a volume so a rebuild
# does not re-download several gigabytes.
VOLUME ["/models"]

# SIP signalling and the RTP range from config/config.yaml.  With
# network_mode: host (recommended) these are informational only.
EXPOSE 5060/udp
EXPOSE 16000-16200/udp

# The agent touches a heartbeat file while its SIP registration is alive.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python3 /app/scripts/healthcheck.py || exit 1

ENTRYPOINT ["python3", "-m", "helpdesk"]
CMD ["/app/config/config.yaml"]
