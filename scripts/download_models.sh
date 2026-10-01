#!/usr/bin/env bash
# Fetch the German Piper voice and pull the Ollama models.
# Run this on the host before the first start -- the container will not start
# without the voice file, since Piper is the default backend.
set -euo pipefail

VOICES_DIR="${VOICES_DIR:-./voices}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
LLM_MODEL="${LLM_MODEL:-qwen2.5:7b-instruct-q4_K_M}"
EMBED_MODEL="${EMBED_MODEL:-bge-m3}"

# Piper voices. "medium" is the right default for telephony: the call is 8 kHz
# anyway, so "high" (22.05 kHz) spends several times the synthesis time on detail
# that is discarded on the way to the caller.
VOICE="${VOICE:-de_DE-thorsten-medium}"
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE"

# All German Piper voices, so you can try a few and pick by ear. "medium" is the
# sweet spot for telephony; "low" is faster still and barely worse at 8 kHz.
declare -A VOICE_PATHS=(
  [de_DE-thorsten-medium]="thorsten/medium/de_DE-thorsten-medium"
  [de_DE-thorsten-low]="thorsten/low/de_DE-thorsten-low"
  [de_DE-thorsten-high]="thorsten/high/de_DE-thorsten-high"
  [de_DE-thorsten_emotional-medium]="thorsten_emotional/medium/de_DE-thorsten_emotional-medium"
  [de_DE-eva_k-x_low]="eva_k/x_low/de_DE-eva_k-x_low"
  [de_DE-kerstin-low]="kerstin/low/de_DE-kerstin-low"
  [de_DE-ramona-low]="ramona/low/de_DE-ramona-low"
  [de_DE-karlsson-low]="karlsson/low/de_DE-karlsson-low"
  [de_DE-mls-medium]="mls/medium/de_DE-mls-medium"
  [de_DE-pavoque-low]="pavoque/low/de_DE-pavoque-low"
)

# Fetch several at once to compare:  VOICE="a b c" ./scripts/download_models.sh

mkdir -p "$VOICES_DIR"
for voice in $VOICE; do
  path="${VOICE_PATHS[$voice]:-}"
  if [[ -z "$path" ]]; then
    echo "Unknown voice '$voice'. Available:" >&2
    printf '  %s\n' "${!VOICE_PATHS[@]}" | sort >&2
    exit 1
  fi
  for ext in onnx onnx.json; do
    target="$VOICES_DIR/$voice.$ext"
    if [[ -f "$target" ]]; then
      echo "already present: $target"
      continue
    fi
    echo "downloading $voice.$ext ..."
    curl -fL --retry 3 -o "$target" "$BASE/$path.$ext"
  done
done

echo
echo "Pulling Ollama models (this is the big download) ..."
if ! curl -sf "$OLLAMA_URL/api/tags" >/dev/null; then
  echo "WARNING: cannot reach Ollama at $OLLAMA_URL -- skipping model pulls." >&2
  echo "         Run: ollama pull $LLM_MODEL && ollama pull $EMBED_MODEL" >&2
  exit 0
fi

for model in "$LLM_MODEL" "$EMBED_MODEL"; do
  echo "  -> $model"
  curl -sf -X POST "$OLLAMA_URL/api/pull" -d "{\"model\":\"$model\"}" \
    | while IFS= read -r line; do
        status=$(printf '%s' "$line" | sed -n 's/.*"status":"\([^"]*\)".*/\1/p')
        [[ -n "$status" ]] && printf '\r     %-60s' "$status"
      done
  echo
done

echo
echo "Done. Voice in $VOICES_DIR, models in Ollama."
echo "Next: ./scripts/preflight.sh"
