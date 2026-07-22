#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
exec "${SCRIPT_DIR}/../qwen3_8b_opd_tillicum/container_exec.sh" "$@"
