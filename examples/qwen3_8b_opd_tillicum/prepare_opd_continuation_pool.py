#!/usr/bin/env python3
"""Prepare a no-repeat OPD continuation prompt pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    env = os.environ
    effective = int(env.get("OPD_EFFECTIVE_TRAIN_SAMPLES", "24960"))
    already = int(env.get("OPD_ALREADY_TRAINED_SAMPLES", "1024"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=env.get("OT3_DATASET", "open-thoughts/OpenThoughts3-1.2M"))
    parser.add_argument("--split", default=env.get("OT3_SPLIT", "train"))
    parser.add_argument("--seed", type=int, default=int(env.get("DATA_SEED", "1234")))
    parser.add_argument("--math-field", default=env.get("DATA_MATH_FIELD", "domain"))
    parser.add_argument("--math-value", default=env.get("DATA_MATH_VALUE", "math"))
    parser.add_argument("--hf-home", default=env.get("HF_HOME"))
    parser.add_argument("--sft-metadata", default=env.get("SPLIT_METADATA"))
    parser.add_argument("--old-rollout-log-dir", default=env.get("OPD_PREVIOUS_ROLLOUT_LOG_DIR"))
    parser.add_argument("--old-manifest", default=env.get("OPD_PREVIOUS_TRAINED_MANIFEST"))
    parser.add_argument("--out-jsonl", default=env.get("OPD_JSONL"))
    parser.add_argument("--metadata-out", default=env.get("OPD_CONTINUATION_METADATA", env.get("SPLIT_METADATA")))
    parser.add_argument("--target-total-effective-samples", type=int, default=effective)
    parser.add_argument("--already-trained-samples", type=int, default=already)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    missing = [
        name
        for name in ("sft_metadata", "old_rollout_log_dir", "out_jsonl", "metadata_out")
        if getattr(args, name) in (None, "")
    ]
    if missing:
        parser.error(f"missing required path(s): {', '.join(missing)}")
    return args


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def first_present(row: dict[str, Any], keys: list[str]) -> Any | None:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def extract_prompt(row: dict[str, Any]) -> str:
    prompt = first_present(row, ["prompt", "problem", "question", "instruction", "input"])
    if prompt is not None:
        return str(prompt)

    for key in ("messages", "conversations", "conversation"):
        value = row.get(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return value
        if isinstance(value, dict) and "messages" in value:
            value = value["messages"]
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role", item.get("from", item.get("speaker", "")))).lower()
                if role in {"human", "user", "prompt"}:
                    content = item.get("content", item.get("value", item.get("text", "")))
                    return "" if content is None else str(content)

    raise ValueError("Could not infer prompt text from row.")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_sft_source_ids(path: Path) -> list[int]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    return [int(x) for x in metadata["source_row_ids"]["sft"]]


def sample_metadata(sample: Any) -> dict[str, Any]:
    if isinstance(sample, dict):
        return sample.get("metadata") or {}
    return getattr(sample, "metadata", {}) or {}


def load_rollout_source_ids(rollout_log_dir: Path) -> list[int]:
    ids: list[int] = []
    for path in sorted(rollout_log_dir.glob("rollout_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for sample in payload.get("samples", []):
            metadata = sample_metadata(sample)
            if "source_row_id" not in metadata:
                raise RuntimeError(f"Missing source_row_id in sample from {path}")
            ids.append(int(metadata["source_row_id"]))
    return ids


def maybe_load_manifest_ids(path: Path | None) -> list[int]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for row in payload.get("trained_rows", []):
        if row.get("source_row_id") is not None:
            out.append(int(row["source_row_id"]))
    return out


def main() -> None:
    args = parse_args()
    out_jsonl = Path(args.out_jsonl)
    metadata_out = Path(args.metadata_out)
    if (out_jsonl.exists() or metadata_out.exists()) and not args.force:
        raise SystemExit(f"Continuation outputs already exist; use --force to replace: {out_jsonl}, {metadata_out}")

    if args.target_total_effective_samples <= args.already_trained_samples:
        raise SystemExit(
            "target_total_effective_samples must exceed already_trained_samples: "
            f"{args.target_total_effective_samples} <= {args.already_trained_samples}"
        )
    needed_new_rows = args.target_total_effective_samples - args.already_trained_samples

    sft_ids = load_sft_source_ids(Path(args.sft_metadata))
    old_ids = load_rollout_source_ids(Path(args.old_rollout_log_dir))
    manifest_ids = maybe_load_manifest_ids(Path(args.old_manifest) if args.old_manifest else None)
    if len(old_ids) != args.already_trained_samples:
        raise SystemExit(f"Expected {args.already_trained_samples} old OPD trained ids, found {len(old_ids)}")
    if len(set(old_ids)) != len(old_ids):
        raise SystemExit("Duplicate source_row_ids found in old OPD rollout debug files")
    if manifest_ids and set(manifest_ids) != set(old_ids):
        print("WARNING: old manifest source ids differ from rollout-debug source ids; using rollout-debug ids.", flush=True)

    excluded = set(sft_ids).union(old_ids)
    print(f"Loading {args.dataset} split={args.split}")
    ds = load_dataset(args.dataset, split=args.split, cache_dir=args.hf_home)
    if args.math_field not in ds.column_names:
        raise SystemExit(f"Dataset does not contain math filter field {args.math_field!r}")
    ds = ds.map(lambda _, idx: {"source_row_id": idx}, with_indices=True)
    math_value = str(args.math_value).lower()
    math_ds = ds.filter(lambda row: str(row.get(args.math_field, "")).lower() == math_value)

    rng = random.Random(args.seed)
    shuffled = list(range(len(math_ds)))
    rng.shuffle(shuffled)
    selected_math_indices: list[int] = []
    selected_source_ids: list[int] = []
    for math_index in shuffled:
        source_row_id = int(math_ds[int(math_index)]["source_row_id"])
        if source_row_id in excluded:
            continue
        selected_math_indices.append(int(math_index))
        selected_source_ids.append(source_row_id)
        if len(selected_math_indices) == needed_new_rows:
            break
    if len(selected_math_indices) != needed_new_rows:
        raise SystemExit(f"Only found {len(selected_math_indices)} new rows, need {needed_new_rows}")

    materialized_indices = sorted(selected_math_indices)
    selected_set = set(selected_math_indices)
    rows: list[dict[str, Any]] = []
    for row in math_ds.select(materialized_indices):
        source_row_id = int(row["source_row_id"])
        math_index = materialized_indices[len(rows)]
        if math_index not in selected_set:
            raise RuntimeError("Internal selected-index mismatch")
        prompt = extract_prompt(row)
        rows.append(
            {
                "prompt": prompt,
                "metadata": {
                    "source_dataset": args.dataset,
                    "source_split": args.split,
                    "source_row_id": source_row_id,
                    "prompt_sha256": prompt_hash(prompt),
                    "split": "opd_continuation",
                },
            }
        )

    new_ids = [int(row["metadata"]["source_row_id"]) for row in rows]
    if set(new_ids).intersection(sft_ids):
        raise RuntimeError("Continuation OPD rows overlap with SFT rows")
    if set(new_ids).intersection(old_ids):
        raise RuntimeError("Continuation OPD rows overlap with old 1k OPD rows")
    if len(set(new_ids)) != len(new_ids):
        raise RuntimeError("Continuation OPD rows contain duplicate source_row_ids")

    write_jsonl(out_jsonl, rows)
    metadata = {
        "seed": args.seed,
        "dataset": args.dataset,
        "source_split": args.split,
        "filtering_criteria": {
            "field": args.math_field,
            "value": args.math_value,
            "operation": "case-insensitive equality",
        },
        "sampling": "Seeded shuffled math-row scan excluding SFT and prior OPD source_row_ids; output materialized in source order.",
        "counts": {
            "source_rows_seen": len(ds),
            "math_rows_after_filter": len(math_ds),
            "sft_excluded": len(sft_ids),
            "old_opd_excluded": len(old_ids),
            "new_opd_continuation": len(rows),
            "target_total_effective_opd_samples": args.target_total_effective_samples,
            "already_trained_opd_samples": args.already_trained_samples,
        },
        "outputs": {
            "opd_continuation_jsonl": str(out_jsonl),
        },
        "inputs": {
            "sft_metadata": str(args.sft_metadata),
            "old_rollout_log_dir": str(args.old_rollout_log_dir),
            "old_manifest": str(args.old_manifest) if args.old_manifest else None,
        },
        "source_row_ids": {
            "sft": sft_ids,
            "old_opd": old_ids,
            "new_opd_continuation": new_ids,
        },
        "prompt_hashes": {
            "new_opd_continuation": [row["metadata"]["prompt_sha256"] for row in rows],
        },
        "overlap_checks": {
            "sft_intersection_new_opd": 0,
            "old_opd_intersection_new_opd": 0,
            "old_opd_count_from_rollout_debug": len(old_ids),
        },
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        "OPD_CONTINUATION_POOL_OK "
        f"old_opd={len(old_ids)} new_opd={len(rows)} total={args.target_total_effective_samples} jsonl={out_jsonl}"
    )
    print(f"OPD_CONTINUATION_METADATA {metadata_out}")


if __name__ == "__main__":
    main()
