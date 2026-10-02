#!/usr/bin/env bash
# Everything that runs without a GPU, a PBX or a network.
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
failed=0

for test in tests/test_sip_flow.py tests/test_session.py tests/test_wiring.py tests/test_units.py tests/test_openai_tts.py; do
  [[ -f "$test" ]] || continue
  echo "=============================================================="
  echo "  $test"
  echo "=============================================================="
  if timeout 300 python3 "$test"; then
    echo "-> OK"
  else
    echo "-> FAILED"
    failed=$((failed + 1))
  fi
  echo
done

if [[ $failed -gt 0 ]]; then
  echo "$failed test file(s) failed"
  exit 1
fi
echo "all test files passed"
