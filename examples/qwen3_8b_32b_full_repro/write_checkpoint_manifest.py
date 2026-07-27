#!/usr/bin/env python3
"""Write exact SFT/OPD weights and latest-full-state manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temp = Path(handle.name)
    temp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(root: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        result.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return result


def sft_token_prefixes() -> dict[int, int]:
    targets = set(range(25_000, 200_001, 25_000))
    values: dict[int, int] = {}
    total = 0
    with Path(os.environ["SFT_PARQUET"]).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            metadata = (json.loads(line).get("metadata") or {})
            total += int(metadata.get("rendered_tokens", 0))
            if index in targets:
                values[index] = total
            if index >= max(targets):
                break
    if set(values) != targets:
        raise SystemExit(f"Unable to compute exact SFT token prefixes: {sorted(values)}")
    return values


def opd_token_prefixes() -> dict[int, int]:
    audit = Path(os.environ["OPD_ROLLOUT_AUDIT_JSONL"])
    if not audit.is_file():
        raise SystemExit(f"Missing OPD trajectory audit needed for exact token counts: {audit}")
    by_rollout: dict[int, int] = {}
    with audit.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rollout_id = int(row["rollout_id"])
            by_rollout[rollout_id] = by_rollout.get(rollout_id, 0) + int(row["response_tokens"])
    result: dict[int, int] = {}
    running = 0
    for rollout_id in range(92):
        running += by_rollout.get(rollout_id, 0)
        if rollout_id in {22, 45, 68, 91}:
            result[rollout_id] = running
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("sft", "opd"), required=True)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--split-metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    mappings = {
        "sft": {124: (25_000, None), 249: (50_000, None), 374: (75_000, None), 499: (100_000, None), 624: (125_000, None), 749: (150_000, None), 874: (175_000, None), 999: (200_000, None)},
        "opd": {22: (368, 1472), 45: (736, 2944), 68: (1104, 4416), 91: (1472, 5888)},
    }
    split_sha = sha256_file(args.split_metadata)
    token_prefixes = sft_token_prefixes() if args.stage == "sft" else opd_token_prefixes()
    entries = []
    for iteration, (exposure, trajectories) in mappings[args.stage].items():
        weight_dir = args.weights_root / f"iter_{iteration:07d}"
        if not weight_dir.is_dir():
            raise SystemExit(f"Missing required weights checkpoint: {weight_dir}")
        entries.append(
            {
                "iteration": iteration,
                "examples_or_prompt_rows": exposure,
                "trajectories": trajectories,
                "optimizer_step": iteration + 1,
                "tokens_consumed": token_prefixes[exposure] if args.stage == "sft" else token_prefixes[iteration],
                "token_accounting": "rendered supervised-sequence tokens" if args.stage == "sft" else "generated response tokens",
                "kind": "weights_only",
                "path": str(weight_dir),
                "files": file_manifest(weight_dir),
            }
        )
    tracker = args.state_root / "latest_checkpointed_iteration.txt"
    latest = int(tracker.read_text().strip())
    final_expected = max(mappings[args.stage])
    if latest != final_expected:
        raise SystemExit(f"Latest full-state iteration {latest} != expected final {final_expected}")
    state_dir = args.state_root / f"iter_{latest:07d}"
    if not (state_dir / ".metadata").is_file():
        raise SystemExit(f"Full-state checkpoint is incomplete: {state_dir}")
    final_exposure, final_trajectories = mappings[args.stage][latest]
    state = {
        "iteration": latest,
        "optimizer_step": latest + 1,
        "examples_or_prompt_rows": final_exposure,
        "trajectories": final_trajectories,
        "tokens_consumed": token_prefixes[final_exposure] if args.stage == "sft" else token_prefixes[latest],
        "kind": "full_training_state",
        "path": str(state_dir),
        "files": file_manifest(state_dir),
    }
    manifest = {
        "schema_version": 1,
        "stage": args.stage,
        "repository_commit": os.environ.get("IMPLEMENTATION_COMMIT"),
        "split_metadata": str(args.split_metadata),
        "split_metadata_sha256": split_sha,
        "model_revision": os.environ["STUDENT_HF_REVISION"],
        "tokenizer_revision": os.environ["STUDENT_HF_REVISION"],
        "teacher_revision": os.environ.get("TEACHER_HF_REVISION") if args.stage == "opd" else None,
        "data_revision": os.environ["OT3_REVISION"],
        "weights": entries,
        "latest_full_state": state,
        "load_instructions": "Use the HF snapshot for evaluation or fresh phase initialization; use latest full state only for exact same-geometry resume.",
    }
    atomic_text(args.out, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    atomic_text(args.out.with_suffix(".md"), f"# {args.stage.upper()} checkpoint manifest\n\n- Weights checkpoints: {len(entries)}\n- Latest full state: `{state_dir}`\n- Split hash: `{split_sha}`\n")
    print(json.dumps({"stage": args.stage, "weights": len(entries), "latest_full_state": str(state_dir)}, indent=2))


if __name__ == "__main__":
    main()
