#!/usr/bin/env bash
set -euo pipefail

HANDOFF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HANDOFF/../.." && pwd)"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-/mmfs1/sw/contrib/chem-src/miniconda3/bin/python3.11}"
RUNTIME_ROOT="${LLAMA_GATE_RUNTIME_ROOT:-$REPO_ROOT/checkpoints/runtime/qwen25-math500-llama-gate-v1}"
VENV="$RUNTIME_ROOT/venv"
MODEL_ROOT="${LLAMA_GATE_MODEL_ROOT:-$REPO_ROOT/checkpoints/models/nvidia--Llama-3.3-70B-Instruct-FP8--68579f675008}"
POLICY="$HANDOFF/llama_gate_policy.json"
LOCK="$RUNTIME_ROOT/setup.lock"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

mkdir -p "$RUNTIME_ROOT" "$MODEL_ROOT"
exec 9>"$LOCK"
flock 9

if [[ ! -x "$VENV/bin/python" ]]; then
  "$BOOTSTRAP_PYTHON" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade "pip==25.1.1" "setuptools==80.9.0" "wheel==0.45.1"
"$VENV/bin/python" -m pip install --requirement "$HANDOFF/requirements_llama_gate.txt"
"$VENV/bin/python" -m pip check
"$VENV/bin/python" "$HANDOFF/llama_math500_eval.py" --repo-root "$REPO_ROOT" check-runtime

if [[ ! -f "$MODEL_ROOT/MODEL_READY.json" ]]; then
  MODEL_ROOT="$MODEL_ROOT" POLICY="$POLICY" "$VENV/bin/python" -c '
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

policy = json.loads(Path(os.environ["POLICY"]).read_text())
model = policy["model"]
root = Path(os.environ["MODEL_ROOT"])
snapshot_download(
    repo_id=model["hf_repo"],
    revision=model["revision"],
    local_dir=root,
    local_dir_use_symlinks=False,
)
'
  MODEL_ROOT="$MODEL_ROOT" POLICY="$POLICY" "$VENV/bin/python" -c '
import hashlib
import json
import os
import tempfile
from pathlib import Path

root = Path(os.environ["MODEL_ROOT"])
policy = json.loads(Path(os.environ["POLICY"]).read_text())
index = json.loads((root / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
if len(shards) != 15 or any(not (root / shard).is_file() for shard in shards):
    raise SystemExit(f"expected 15 complete model shards, found {len(shards)}")
for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
    if not (root / name).is_file():
        raise SystemExit(f"missing required model file: {name}")
value = {
    "artifact_schema_version": 1,
    "files": len([path for path in root.rglob("*") if path.is_file()]),
    "hf_repo": policy["model"]["hf_repo"],
    "revision": policy["model"]["revision"],
    "weight_shards": shards,
}
target = root / "MODEL_READY.json"
with tempfile.NamedTemporaryFile("w", dir=root, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    temp = Path(handle.name)
temp.replace(target)
'
fi

"$VENV/bin/python" "$HANDOFF/llama_math500_eval.py" --repo-root "$REPO_ROOT" check-model
echo "LLAMA_GATE_RUNTIME_READY python=$VENV/bin/python model=$MODEL_ROOT"
