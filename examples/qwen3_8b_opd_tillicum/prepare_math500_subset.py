#!/usr/bin/env python3
"""Create a deterministic uniform subset of an existing MATH-500 JSONL file."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    env = os.environ
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", default=env.get("MATH500_JSONL"))
    parser.add_argument("--out-jsonl", default=env.get("MATH500_VAL100_JSONL"))
    parser.add_argument("--out-config", default=env.get("MATH500_VAL100_CONFIG"))
    parser.add_argument("--metadata-out", default=env.get("MATH500_VAL100_METADATA"))
    parser.add_argument("--seed", type=int, default=int(env.get("MATH500_VAL100_SEED", "1234")))
    parser.add_argument("--size", type=int, default=int(env.get("MATH500_VAL100_SIZE", "100")))
    parser.add_argument("--max-response-len", type=int, default=int(env.get("EVAL_MAX_RESPONSE_LEN", "31744")))
    args = parser.parse_args()
    missing = [
        name
        for name in ("source_jsonl", "out_jsonl", "out_config", "metadata_out")
        if getattr(args, name) in (None, "")
    ]
    if missing:
        parser.error(f"missing required path(s): {', '.join(missing)}")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            metadata = row.setdefault("metadata", {})
            metadata.setdefault("source_row_id", line_no - 1)
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_config(path: Path, jsonl_path: Path, max_response_len: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "eval:\n"
        "  defaults:\n"
        f"    max_response_len: {max_response_len}\n"
        "    temperature: 0\n"
        "    top_p: 1\n"
        "    n_samples_per_eval_prompt: 1\n"
        "    input_key: prompt\n"
        "    label_key: label\n"
        "    apply_chat_template: true\n"
        "  datasets:\n"
        "    - name: math500_val100\n"
        f"      path: {jsonl_path}\n"
        "      rm_type: math\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source = Path(args.source_jsonl)
    out_jsonl = Path(args.out_jsonl)
    out_config = Path(args.out_config)
    metadata_out = Path(args.metadata_out)

    rows = read_jsonl(source)
    if args.size > len(rows):
        raise SystemExit(f"Requested {args.size} rows from {source}, but only found {len(rows)}")

    rng = random.Random(args.seed)
    selected_indices = sorted(rng.sample(range(len(rows)), args.size))
    selected = [rows[idx] for idx in selected_indices]
    write_jsonl(out_jsonl, selected)
    write_config(out_config, out_jsonl, args.max_response_len)

    metadata = {
        "source_jsonl": str(source),
        "out_jsonl": str(out_jsonl),
        "out_config": str(out_config),
        "seed": args.seed,
        "size": args.size,
        "source_row_ids": [
            int((row.get("metadata") or {}).get("source_row_id", idx))
            for idx, row in zip(selected_indices, selected)
        ],
        "selected_indices": selected_indices,
        "sampling": "uniform without replacement using random.Random(seed), output sorted by source order",
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"MATH500_VAL100_OK rows={len(selected)} seed={args.seed} jsonl={out_jsonl}")
    print(f"MATH500_VAL100_CONFIG {out_config}")
    print(f"MATH500_VAL100_METADATA {metadata_out}")


if __name__ == "__main__":
    main()
