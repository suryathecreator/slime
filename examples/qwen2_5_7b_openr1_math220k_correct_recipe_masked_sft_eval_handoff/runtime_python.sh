#!/usr/bin/env bash
set -euo pipefail

# The runtime venv is pinned to the intact cluster Python 3.11.5 executable;
# the coenv Python 3.11.9 build on this cluster lacks the stdlib _ctypes
# extension.
HANDOFF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HANDOFF/../.." && pwd)"
RUNTIME_PYTHON="${EVAL_RUNTIME_PYTHON:-$REPO_ROOT/checkpoints/runtime/qwen-math500-v1/venv/bin/python}"

[[ -x "$RUNTIME_PYTHON" ]] || { echo "Missing evaluation Python: $RUNTIME_PYTHON" >&2; exit 2; }
[[ -d "$REPO_ROOT/checkpoints/runtime/qwen-math500-v1/venv/lib/python3.11/site-packages/vllm" ]] || {
  echo "Missing pinned evaluation packages" >&2
  exit 2
}
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$RUNTIME_PYTHON" "$@"
