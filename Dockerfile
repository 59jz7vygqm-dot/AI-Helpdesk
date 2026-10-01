# CUDA 12.4 with cuDNN 9: cuDNN 9 is what CTranslate2 >= 4.5 (faster-whisper)
# links against. Torch, when the quality profile installs it, brings its own CUDA
# libraries as wheels, so this base only has to satisfy CTranslate2.
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

# lite    = Piper only (CPU). Small image, builds in minutes, always works.
# quality = adds Chatterbox and GPU synthesis (~5 GB more image, pulls torch).
ARG TTS_PROFILE=lite

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/models/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        ca-certificates curl \
        libsndfile1 \
        espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Base dependencies in their own layer, so a failure further down does not
# re-download them.
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -r /app/requirements.txt

# The voice stack is a separate layer for the same reason: it is the big one.
COPY requirements-quality.txt /app/requirements-quality.txt
RUN if [ "$TTS_PROFILE" = "quality" ]; then \
      python3 -m pip install -r /app/requirements-quality.txt ; \
    else \
      echo "TTS_PROFILE=$TTS_PROFILE -- skipping the GPU voice stack" ; \
    fi

COPY src /app/src
COPY config /app/config
COPY knowledge /app/knowledge
COPY scripts /app/scripts
RUN chmod +x /app/scripts/*.sh /app/scripts/*.py || true

# Model weights and the phrase/knowledge caches live on a volume so a rebuild
# does not re-download several gigabytes.
VOLUME ["/models"]

EXPOSE 5060/udp
EXPOSE 16000-16200/udp

# The agent touches a heartbeat file while its SIP registration is alive.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python3 /app/scripts/healthcheck.py || exit 1

ENTRYPOINT ["python3", "-m", "helpdesk"]
CMD ["/app/config/config.yaml"]
