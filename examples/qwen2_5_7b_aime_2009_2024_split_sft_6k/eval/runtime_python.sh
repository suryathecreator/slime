#!/usr/bin/env bash
set -euo pipefail

EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${EVAL_DIR}/../../.." >/dev/null 2>&1 && pwd)"
RUNTIME_PYTHON="${EVAL_RUNTIME_PYTHON:-${REPO_ROOT}/checkpoints/runtime/qwen-math500-v1/venv/bin/python}"

[[ -x "${RUNTIME_PYTHON}" ]] || {
  echo "Missing pinned evaluation Python: ${RUNTIME_PYTHON}" >&2
  exit 2
}
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${RUNTIME_PYTHON}" "$@"
