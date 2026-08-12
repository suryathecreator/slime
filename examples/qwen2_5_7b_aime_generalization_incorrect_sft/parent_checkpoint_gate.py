#!/usr/bin/env python3
"""Record/check a fully hashed stage-1 parent checkpoint verification gate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from examples.qwen2_5_7b_aime_generalization_incorrect_sft.data_utils import (
    atomic_json,
    sha256_file,
)


def manifest_value(path: Path, contract_hash: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "contract_hash": contract_hash,
        "family": "aime_generalization",
        "final_iteration": 187,
        "full_sft": True,
        "variant": "aime_correct_3000",
    }
    for key, target in expected.items():
        if value.get(key) != target:
            raise ValueError(f"stage1 parent manifest {key} drift: expected={target!r} " f"actual={value.get(key)!r}")
    identity = value.get("checkpoint_manifest_sha256")
    if not isinstance(identity, str) or len(identity) != 64:
        raise ValueError("stage1 parent identity is not a SHA-256")
    return value


def expected_marker(manifest: Path, checkpoint: Path, contract_hash: str) -> dict[str, object]:
    value = manifest_value(manifest, contract_hash)
    return {
        "artifact_schema_version": 1,
        "checkpoint": str(checkpoint.resolve()),
        "contract_hash": contract_hash,
        "manifest": str(manifest.resolve()),
        "manifest_file_sha256": sha256_file(manifest),
        "parent_checkpoint_manifest_sha256": value["checkpoint_manifest_sha256"],
        "verification": "all_regular_checkpoint_files_sha256_matched_manifest",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("record", "check"))
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contract-hash", required=True)
    parser.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()
    expected = expected_marker(args.manifest, args.checkpoint, args.contract_hash)
    if args.mode == "record":
        if args.marker.exists():
            raise FileExistsError(f"refusing to overwrite parent gate: {args.marker}")
        subprocess.run(
            [
                "python3",
                str(args.tool),
                "verify",
                "--manifest",
                str(args.manifest),
                "--checkpoint",
                str(args.checkpoint),
            ],
            check=True,
        )
        atomic_json(args.marker, expected)
    else:
        actual = json.loads(args.marker.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("stage1 parent verification marker is stale")
    print(
        "AIME_PARENT_CHECKPOINT_GATE_OK " f"mode={args.mode} identity={expected['parent_checkpoint_manifest_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
