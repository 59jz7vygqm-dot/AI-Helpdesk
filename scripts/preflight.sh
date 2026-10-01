#!/usr/bin/env bash
# Check everything the agent needs, before it is started for the first time.
# Run on the host (not in the container):  ./scripts/preflight.sh
set -uo pipefail
cd "$(dirname "$0")/.."

PASS=0
WARN=0
FAIL=0

ok()   { printf '  \033[32mOK\033[0m    %s\n' "$1"; PASS=$((PASS+1)); }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$1"; WARN=$((WARN+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }
hint() { printf '        -> %s\n' "$1"; }

echo "=== 1. GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  free=$((total - used))
  ok "GPU: $gpu_name"
  echo "        ${total} MiB total, ${used} MiB in use, ${free} MiB free"
  if   [ "$free" -ge 20000 ]; then ok  "enough VRAM for any profile (22gb-max included)"
  elif [ "$free" -ge 15000 ]; then ok  "enough VRAM for 16gb-quality.yaml (recommended)"
  elif [ "$free" -ge 11000 ]; then warn "only enough for 12gb-shared.yaml"
       hint "free VRAM, or: cp config/profiles/12gb-shared.yaml config/config.yaml"
  else bad "under 11 GiB free -- not enough for any profile"
       hint "find what holds VRAM: nvidia-smi  (often an idle ollama model)"
       hint "unload ollama models: ollama stop <model>"
  fi
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  major=${driver%%.*}
  if [ "${major:-0}" -ge 525 ]; then ok "driver $driver"
  else bad "driver $driver is too old for CUDA 12.x"; fi
else
  bad "nvidia-smi not found -- no NVIDIA driver on this host"
fi

echo
echo "=== 2. Docker ==="
if command -v docker >/dev/null 2>&1; then
  ok "docker $(docker --version | sed 's/Docker version //;s/,.*//')"
  if docker compose version >/dev/null 2>&1; then
    ok "docker compose available"
  else
    bad "docker compose plugin missing"; hint "apt install docker-compose-plugin"
  fi
  # The GPU must be visible inside a container, not just on the host.
  if docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 \
       nvidia-smi -L >/tmp/preflight-gpu.log 2>&1; then
    ok "containers can see the GPU: $(head -1 /tmp/preflight-gpu.log)"
  else
    bad "containers cannot use the GPU (NVIDIA Container Toolkit)"
    hint "install nvidia-container-toolkit, then: systemctl restart docker"
    hint "details: /tmp/preflight-gpu.log"
  fi
else
  bad "docker not found"
fi

echo
echo "=== 3. Ollama ==="
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
if curl -sf --max-time 5 "$OLLAMA_URL/api/tags" -o /tmp/preflight-ollama.json; then
  ok "reachable at $OLLAMA_URL"
  LLM_MODEL="${LLM_MODEL:-qwen2.5:7b-instruct-q4_K_M}"
  if grep -q "\"${LLM_MODEL}\"" /tmp/preflight-ollama.json; then
    ok "model present: $LLM_MODEL"
  else
    bad "model missing: $LLM_MODEL"
    hint "ollama pull $LLM_MODEL"
    echo "        installed: $(sed -n 's/.*"name":"\([^"]*\)".*/\1/p' /tmp/preflight-ollama.json | head -8 | tr '\n' ' ')"
  fi
else
  bad "cannot reach Ollama at $OLLAMA_URL"
  hint "systemctl status ollama"
  hint "with network_mode: host the container uses this same address"
fi

echo
echo "=== 4. Configuration ==="
if [ -f .env ]; then
  ok ".env exists"
  # shellcheck disable=SC1091
  set -a; . ./.env 2>/dev/null; set +a
  for var in SIP_USERNAME SIP_PASSWORD SIP_SERVER_HOST TRANSFER_NUMBER; do
    value="${!var:-}"
    case "$value" in
      ""|change-me|192.168.1.10)
        bad "$var is unset or still the example value"; hint "edit .env" ;;
      *) ok "$var is set" ;;
    esac
  done
else
  bad ".env is missing"; hint "cp .env.example .env && nano .env"
fi
if [ -f config/config.yaml ]; then
  profile=$(grep -m1 'model_id:' config/config.yaml | sed 's/.*12Hz-//;s/-Custom.*//')
  ok "config/config.yaml present (voice model: ${profile:-unknown})"
else
  bad "config/config.yaml is missing"
  hint "cp config/profiles/16gb-quality.yaml config/config.yaml"
fi

echo
echo "=== 5. Network / SIP ==="
BIND_PORT=$(grep -m1 '^  bind_port:' config/config.yaml 2>/dev/null | tr -dc '0-9')
BIND_PORT="${BIND_PORT:-5060}"
if command -v ss >/dev/null 2>&1; then
  holder=$(ss -lunp 2>/dev/null | awk -v p=":$BIND_PORT" '$5 ~ p {print $NF; exit}')
  if [ -n "$holder" ]; then
    bad "UDP port $BIND_PORT is already in use by $holder"
    hint "stop it, or set sip.bind_port to a free port (e.g. 5080)"
  else
    ok "UDP port $BIND_PORT is free"
  fi
else
  warn "ss not available, cannot check whether port $BIND_PORT is free"
fi
if [ -n "${SIP_SERVER_HOST:-}" ] && [ "${SIP_SERVER_HOST}" != "192.168.1.10" ]; then
  # UDP gives no handshake, so this only shows routing, not that SIP answers.
  if ping -c1 -W2 "$SIP_SERVER_HOST" >/dev/null 2>&1; then
    ok "PBX $SIP_SERVER_HOST answers ping"
  else
    warn "PBX $SIP_SERVER_HOST does not answer ping (often blocked -- not conclusive)"
  fi
  route_ip=$(ip route get "$SIP_SERVER_HOST" 2>/dev/null | sed -n 's/.*src \([0-9.]*\).*/\1/p')
  [ -n "$route_ip" ] && ok "traffic to the PBX leaves via $route_ip (goes into SDP/Contact)"
fi

echo
echo "=== 6. Offline test suite ==="
if PYTHONPATH=src timeout 300 ./scripts/run_tests.sh >/tmp/preflight-tests.log 2>&1; then
  ok "all offline tests pass ($(grep -c '  ok ' /tmp/preflight-tests.log) unit checks)"
else
  warn "offline tests did not pass -- see /tmp/preflight-tests.log"
  hint "needs python3 with numpy and scipy on the host; harmless to skip"
fi

echo
printf '=== %d ok, %d warnings, %d problems ===\n' "$PASS" "$WARN" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
  echo "Fix the problems above before starting."
  exit 1
fi
echo "Ready. Next:  docker compose up -d --build && docker compose logs -f"
