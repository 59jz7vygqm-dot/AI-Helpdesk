#!/usr/bin/env bash
# Check everything the agent needs, before it is started for the first time.
# Run on the host (not in the container):  ./scripts/preflight.sh
set -uo pipefail
cd "$(dirname "$0")/.."

PASS=0
WARN=0
FAIL=0
# Not guaranteed to be exported (cron, some sudo configs), and set -u is on.
ME="${USER:-$(id -un 2>/dev/null || echo "$(whoami)")}"

# Logs go to unique paths: a leftover file from a run as another user would make
# the redirection fail, which looked like the command itself failing.
LOGDIR=$(mktemp -d 2>/dev/null || echo "/tmp")
trap 'rm -rf "$LOGDIR" 2>/dev/null' EXIT

# Read one key from a top-level section of a YAML file. Only keys indented by
# exactly two spaces count, so a nested block cannot be mistaken for the section
# itself, and two sections sharing a key name stay distinct.
yaml_get() {
  awk -v want_sec="$2" -v want_key="$3" '
    /^[A-Za-z_][A-Za-z0-9_]*:/ {
      sec = substr($0, 1, index($0, ":") - 1)
      next
    }
    sec != want_sec { next }
    /^  [A-Za-z_]/ {
      line = substr($0, 3)
      if (index(line, want_key ":") == 1) {
        val = substr(line, length(want_key) + 2)
        sub(/#.*/, "", val)
        gsub(/^[ \t]+|[ \t]+$/, "", val)
        gsub(/^"|"$/, "", val)
        print val
        exit
      }
    }
  ' "$1"
}

ok()   { printf '  \033[32mOK\033[0m    %s\n' "$1"; PASS=$((PASS+1)); }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$1"; WARN=$((WARN+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }
hint() { printf '        -> %s\n' "$1"; }

echo "=== 1. GPU ==="
# Load .env early: GPU_ID and the Ollama settings come from there.
if [ -f .env ]; then set -a; . ./.env 2>/dev/null; set +a; fi
GPU_ID="${GPU_ID:-0}"

if command -v nvidia-smi >/dev/null 2>&1; then
  count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  if [ "$count" -gt 1 ]; then
    echo "  note  $count GPUs present; this agent is pinned to GPU_ID=$GPU_ID"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total \
      --format=csv,noheader,nounits |
      awk -F', ' '{printf "        GPU %s  %-12s %6s / %6s MiB used\n", $1, $2, $3, $4}'
  fi
  # Query the configured card specifically, not just the first one.
  if ! stats=$(nvidia-smi --id="$GPU_ID" \
        --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null); then
    bad "GPU_ID=$GPU_ID does not exist on this host"
    hint "pick one from the list above and set GPU_ID in .env"
    stats="unknown, 0, 0"
  fi
  gpu_name=$(echo "$stats" | cut -d',' -f1 | xargs)
  total=$(echo "$stats" | cut -d',' -f2 | xargs)
  used=$(echo "$stats" | cut -d',' -f3 | xargs)
  free=$((total - used))
  ok "GPU $GPU_ID: $gpu_name"
  echo "        ${total} MiB total, ${used} MiB in use, ${free} MiB free"
  # The LLM also lives here when Ollama is pinned to this card, so budget ~5 GB
  # on top of the container's own ASR + TTS.
  if   [ "$free" -ge 16000 ]; then ok  "enough VRAM on GPU $GPU_ID for the agent plus a 7B model"
  elif [ "$free" -ge 11000 ]; then warn "tight: use 12gb-shared.yaml, or a smaller LLM"
       hint "cp config/profiles/12gb-shared.yaml config/config.yaml"
  else bad "only ${free} MiB free on GPU $GPU_ID -- not enough"
       hint "pick a freer card (GPU_ID in .env), or free VRAM there"
       hint "what is holding it: nvidia-smi --id=$GPU_ID"
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
  # Decide once how docker can be reached; everything below uses $DOCKER.
  DOCKER=""
  if docker info >/dev/null 2>&1; then
    DOCKER="docker"
  elif sudo -n docker info >/dev/null 2>&1; then
    DOCKER="sudo docker"
    warn "docker needs sudo for this user"
    hint "every compose command too: sudo docker compose up -d --build"
    hint "or fix it once: sudo usermod -aG docker $ME  (then re-login)"
  else
    warn "cannot reach the docker daemon as $ME"
    hint "fix it once: sudo usermod -aG docker $ME  (then re-login)"
    hint "or re-run this script with sudo to finish the checks"
  fi
  if docker compose version >/dev/null 2>&1; then
    ok "docker compose available"
  else
    bad "docker compose plugin missing"; hint "apt install docker-compose-plugin"
  fi
  # The GPU must be visible inside a container, not just on the host. Docker 29
  # routes --gpus through CDI, and which spelling works depends on how the host
  # was set up, so try each and report the one that does.
  # shellcheck disable=SC2034
  GPU_SPEC=""
  IMAGE=nvidia/cuda:12.8.1-base-ubuntu22.04
  if [ -z "$DOCKER" ]; then
    # Without daemon access the probe says nothing about the GPU; reporting a
    # failure here would point at the wrong thing entirely.
    warn "GPU-in-container check skipped: no docker access"
  else
    for spec in "--gpus device=$GPU_ID" "--device nvidia.com/gpu=$GPU_ID" "--gpus all"; do
      # shellcheck disable=SC2086
      if $DOCKER run --rm $spec "$IMAGE" nvidia-smi -L >"$LOGDIR/gpu.log" 2>&1; then
        GPU_SPEC="$spec"
        break
      fi
    done
  fi
  case "${DOCKER:+$GPU_SPEC}" in
    "")
      if [ -n "$DOCKER" ]; then
        bad "containers cannot use any GPU"
        hint "sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
        hint "sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
        hint "details: $LOGDIR/gpu.log"
      fi
      ;;
    "--gpus all")
      warn "only '--gpus all' works, not per-device selection"
      hint "the container would see every GPU; it uses NVIDIA_VISIBLE_DEVICES=$GPU_ID"
      hint "regenerate the CDI spec to get per-device selection:"
      hint "sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
      ;;
    "--device nvidia.com/gpu=$GPU_ID")
      ok "containers can use GPU $GPU_ID via CDI: $(head -1 "$LOGDIR/gpu.log")"
      warn "this host needs the CDI form in compose"
      hint "start with: docker compose -f docker-compose.yml -f docker-compose.cdi.yml up -d --build"
      ;;
    *)
      ok "containers can use GPU $GPU_ID: $(head -1 "$LOGDIR/gpu.log")"
      ;;
  esac
else
  bad "docker not found"
fi

echo
echo "=== 3. Language model ==="
LLM_BACKEND=$(yaml_get config/config.yaml llm backend 2>/dev/null)
LLM_BACKEND="${LLM_BACKEND:-ollama}"
CONFIG_MODEL=$(yaml_get config/config.yaml llm model 2>/dev/null)
if [ "$LLM_BACKEND" = "ollama" ]; then
  OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
  if curl -sf --max-time 5 "$OLLAMA_URL/api/tags" -o "$LOGDIR/ollama.json"; then
    ok "Ollama reachable at $OLLAMA_URL"
    # .env wins over the file, because compose passes it as an override.
    LLM_MODEL="${LLM_MODEL:-${CONFIG_MODEL:-qwen2.5:7b-instruct-q4_K_M}}"
    if grep -q "\"${LLM_MODEL}\"" "$LOGDIR/ollama.json"; then
      ok "model present: $LLM_MODEL"
    else
      bad "model missing: $LLM_MODEL"
      hint "ollama pull $LLM_MODEL"
      echo "        installed: $(sed -n 's/.*"name":"\([^"]*\)".*/\1/p' "$LOGDIR/ollama.json" | head -8 | tr '\n' ' ')"
    fi
    # On a multi-GPU host Ollama must be pinned to the same free card, or it
    # takes GPU 0 and fails there.
    if [ "${count:-1}" -gt 1 ]; then
      pinned=$(systemctl show ollama -p Environment 2>/dev/null | grep -o 'CUDA_VISIBLE_DEVICES=[^ ]*' | cut -d= -f2)
      if [ -z "$pinned" ]; then
        bad "Ollama is not pinned to a GPU (CUDA_VISIBLE_DEVICES unset)"
        hint "it will use GPU 0 and fail if that card is full -- see the README"
      elif [ "$pinned" = "$GPU_ID" ]; then
        ok "Ollama pinned to GPU $pinned, same as the container"
      else
        warn "Ollama uses GPU $pinned, the container uses GPU $GPU_ID"
        hint "that works, but then both cards need free VRAM"
      fi
    fi
  else
    bad "cannot reach Ollama at $OLLAMA_URL"
    if command -v ollama >/dev/null 2>&1; then
      hint "ollama is installed but not answering: sudo systemctl status ollama"
      hint "start it: sudo systemctl enable --now ollama"
    else
      hint "not installed: curl -fsSL https://ollama.com/install.sh | sh"
    fi
    listening=$(ss -lntp 2>/dev/null | grep -i ollama | awk '{print $4}' | tr '\n' ' ')
    [ -n "$listening" ] && hint "something ollama-ish listens on: $listening (set OLLAMA_URL)"
    hint "or point llm.backend=openai at an existing server (e.g. your vLLM)"
  fi
else
  BASE_URL=$(yaml_get config/config.yaml llm base_url)
  if curl -sf --max-time 5 "${BASE_URL}/models" -o "$LOGDIR/llm.json"; then
    ok "inference server reachable at $BASE_URL"
    served=$(sed -n 's/.*"id":"\([^"]*\)".*/\1/p' "$LOGDIR/llm.json" | head -4 | tr '\n' ' ')
    echo "        serves: $served"
    if [ -n "$CONFIG_MODEL" ] && ! grep -q "\"$CONFIG_MODEL\"" "$LOGDIR/llm.json"; then
      bad "llm.model is $CONFIG_MODEL, but the server serves: $served"
      hint "set llm.model in config/config.yaml to one of those"
    fi
  else
    bad "cannot reach the inference server at $BASE_URL"
    hint "base_url must include /v1 for an OpenAI-compatible server"
  fi
fi

echo
echo "=== 4. Configuration ==="
if [ -f .env ]; then
  ok ".env exists"
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
  TTS_CONF=$(yaml_get config/config.yaml tts backend)
  ok "config/config.yaml present (voice: ${TTS_BACKEND:-${TTS_CONF:-?}})"
  case "${TTS_BACKEND:-$TTS_CONF}" in
    piper)
      # Piper needs the voice file on disk; ./voices is mounted to /models/piper.
      if ls voices/*.onnx >/dev/null 2>&1; then
        ok "Piper voice present: $(ls voices/*.onnx | head -1 | xargs basename)"
        if ! ls voices/*.onnx.json >/dev/null 2>&1; then
          bad "the matching .onnx.json is missing next to the voice"
          hint "./scripts/download_models.sh"
        fi
      else
        bad "no Piper voice in ./voices (the container needs it at /models/piper)"
        hint "./scripts/download_models.sh"
      fi
      ;;
    chatterbox)
      warn "chatterbox needs an image built with TTS_PROFILE=quality"
      hint "TTS_PROFILE=quality docker compose build"
      ;;
    qwen3)
      warn "qwen3-tts is not in the image (its PyPI package needs Python 3.13)"
      hint "see the README section 'Bessere Stimme', or use piper/chatterbox"
      ;;
  esac
else
  bad "config/config.yaml is missing"
  hint "cp config/profiles/demo-single-gpu.yaml config/config.yaml"
fi

echo
echo "=== 5. Network / SIP ==="
BIND_PORT=$(grep -m1 '^  bind_port:' config/config.yaml 2>/dev/null | tr -dc '0-9')
BIND_PORT="${BIND_PORT:-5060}"
if command -v ss >/dev/null 2>&1; then
  # The local address is field 4 in `ss -lun` output (State Recv-Q Send-Q Local
  # Peer); matching field 5 silently passed a port that was taken. Process names
  # need root, so note when they are missing rather than guessing.
  port_line=$(ss -lunp 2>/dev/null | awk -v p="[:.]$BIND_PORT\$" '$4 ~ p {print; exit}')
  if [ -n "$port_line" ]; then
    holder=$(printf '%s' "$port_line" | sed -n 's/.*users:((\"\([^"]*\)\".*/\1/p')
    if [ -n "$holder" ]; then
      bad "UDP port $BIND_PORT is held by '$holder'"
    else
      bad "UDP port $BIND_PORT is already in use"
      hint "re-run with sudo to see which process holds it"
    fi
    hint "set sip.bind_port in config/config.yaml to a free port, e.g. 5080"
    hint "(the PBX learns the port from the Contact header, so nothing else changes)"
    hint "only stop the other service if you know it is not in use"
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
if PYTHONPATH=src timeout 300 ./scripts/run_tests.sh >"$LOGDIR/tests.log" 2>&1; then
  ok "all offline tests pass ($(grep -c '  ok ' "$LOGDIR/tests.log") unit checks)"
else
  warn "offline tests did not pass -- see $LOGDIR/tests.log"
  hint "needs python3 with numpy and scipy on the host; harmless to skip"
fi

echo
printf '=== %d ok, %d warnings, %d problems ===\n' "$PASS" "$WARN" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
  echo "Fix the problems above before starting."
  exit 1
fi
COMPOSE="${DOCKER:-docker} compose"
if [ "$GPU_SPEC" = "--device nvidia.com/gpu=$GPU_ID" ]; then
  COMPOSE="$COMPOSE -f docker-compose.yml -f docker-compose.cdi.yml"
fi
echo "Ready. Next:"
echo "  $COMPOSE up -d --build"
echo "  $COMPOSE logs -f"
