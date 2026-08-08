#!/usr/bin/env python3
"""Freeze matching 64-row custom-token and stock-message canary datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.qwen_correct_only_openr1_math220k_sft_2k.data_utils import atomic_jsonl


def read_first(path: Path, count: int) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) == count:
                    break
    if len(rows) != count:
        raise ValueError(f"expected {count} rows in {path}, found {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custom-source", type=Path, required=True)
    parser.add_argument("--message-source", type=Path, required=True)
    parser.add_argument("--custom-output", type=Path, required=True)
    parser.add_argument("--message-output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=64)
    args = parser.parse_args()
    for path in (args.custom_output, args.message_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    custom = read_first(args.custom_source, args.rows)
    messages = read_first(args.message_source, args.rows)
    custom_ids = [str(row["metadata"]["trace_id"]) for row in custom]
    message_ids = [str(row["metadata"]["trace_id"]) for row in messages]
    if custom_ids != message_ids or len(set(custom_ids)) != args.rows:
        raise ValueError("custom and stock canaries do not contain the same ordered traces")
    if any(
        [message.get("role") for message in row.get("messages", [])]
        != ["user", "assistant"]
        for row in messages
    ):
        raise ValueError("stock canary contains a non-user/assistant conversation")
    atomic_jsonl(args.custom_output, custom)
    atomic_jsonl(args.message_output, messages)
    print(f"CANARY_DATA_READY rows={args.rows} trace_first={custom_ids[0]}", flush=True)


if __name__ == "__main__":
    main()
