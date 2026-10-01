# CUDA 12.4 runtime with cuDNN 9, which is what CTranslate2 >= 4.5 links against.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

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

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -r /app/requirements.txt

COPY src /app/src
COPY config /app/config
COPY knowledge /app/knowledge
COPY scripts /app/scripts
RUN chmod +x /app/scripts/*.sh || true

# Model weights and the phrase/knowledge caches live on a volume so a rebuild
# does not re-download several gigabytes.
VOLUME ["/models"]

# SIP signalling and the RTP range from config/config.yaml.  With
# network_mode: host (recommended) these are informational only.
EXPOSE 5060/udp
EXPOSE 16000-16200/udp

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python3 /app/scripts/healthcheck.py || exit 1

ENTRYPOINT ["python3", "-m", "helpdesk"]
CMD ["/app/config/config.yaml"]
