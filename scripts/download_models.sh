#!/usr/bin/env bash
# Fetch the German Piper voice and pull the Ollama models.
# Run this on the host before the first start.
set -euo pipefail

VOICES_DIR="${VOICES_DIR:-./voices}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
LLM_MODEL="${LLM_MODEL:-qwen2.5:7b-instruct-q4_K_M}"
EMBED_MODEL="${EMBED_MODEL:-bge-m3}"

# Piper voices, best first for a service line.  "high" is 22.05 kHz; the extra
# quality survives the downsample to 8 kHz better than you would expect.
VOICE="${VOICE:-de_DE-thorsten-high}"
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE"

declare -A VOICE_PATHS=(
  [de_DE-thorsten-high]="thorsten/high/de_DE-thorsten-high"
  [de_DE-thorsten-medium]="thorsten/medium/de_DE-thorsten-medium"
  [de_DE-eva_k-x_low]="eva_k/x_low/de_DE-eva_k-x_low"
  [de_DE-kerstin-low]="kerstin/low/de_DE-kerstin-low"
  [de_DE-ramona-low]="ramona/low/de_DE-ramona-low"
  [de_DE-karlsson-low]="karlsson/low/de_DE-karlsson-low"
)

path="${VOICE_PATHS[$VOICE]:-}"
if [[ -z "$path" ]]; then
  echo "Unknown voice '$VOICE'. Available: ${!VOICE_PATHS[*]}" >&2
  exit 1
fi

mkdir -p "$VOICES_DIR"
for ext in onnx onnx.json; do
  target="$VOICES_DIR/$VOICE.$ext"
  if [[ -f "$target" ]]; then
    echo "already present: $target"
    continue
  fi
  echo "downloading $VOICE.$ext ..."
  curl -fL --retry 3 -o "$target" "$BASE/$path.$ext"
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
