#!/usr/bin/env python3
"""Repartition the existing 1,024 + 4,096 OPD rows into 2,048 + 3,072."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROMPT_CONTRACT_VERSION = "qwen3_aime2026_boxed_v1"
PROMPT_PREFIX = "Please reason step by step, and put your final answer within \\boxed{}.\n\nProblem:\n"
FIRST_SOURCE_ROWS = 1024
SECOND_SOURCE_ROWS = 4096
FIRST_OUTPUT_ROWS = 2048
SECOND_OUTPUT_ROWS = 3072
TOTAL_ROWS = FIRST_OUTPUT_ROWS + SECOND_OUTPUT_ROWS


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path, expected: int) -> tuple[list[dict[str, Any]], list[str]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != expected:
        raise ValueError(f"Expected {expected} rows in {path}, found {len(lines)}")
    rows = [json.loads(line) for line in lines]
    return rows, lines


def row_identity(row: dict[str, Any]) -> str:
    metadata = dict(row.get("metadata") or {})
    uuid = metadata.get("uuid")
    if uuid:
        return f"uuid:{uuid}"
    return "hash:" + hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def validate_rows(rows: list[dict[str, Any]]) -> None:
    identities = [row_identity(row) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("The original 5,120-row OPD union contains duplicate row identities")
    offsets = [int((row.get("metadata") or {}).get("selected_offset", -1)) for row in rows]
    if offsets != list(range(TOTAL_ROWS)):
        raise ValueError("The original OPD rows do not cover selected_offset 0..5119 in order")
    for index, row in enumerate(rows):
        metadata = dict(row.get("metadata") or {})
        if metadata.get("prompt_contract_version") != PROMPT_CONTRACT_VERSION:
            raise ValueError(f"Row {index} has the wrong prompt contract")
        if not str(row.get("prompt", "")).startswith(PROMPT_PREFIX):
            raise ValueError(f"Row {index} does not use the canonical AIME prompt")


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-1k", required=True)
    parser.add_argument("--source-4k", required=True)
    parser.add_argument("--output-2k", required=True)
    parser.add_argument("--output-3k", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()

    source_1k = Path(args.source_1k)
    source_4k = Path(args.source_4k)
    output_2k = Path(args.output_2k)
    output_3k = Path(args.output_3k)
    metadata_path = Path(args.metadata)

    rows_1k, lines_1k = read_rows(source_1k, FIRST_SOURCE_ROWS)
    rows_4k, lines_4k = read_rows(source_4k, SECOND_SOURCE_ROWS)
    original_rows = rows_1k + rows_4k
    original_lines = lines_1k + lines_4k
    validate_rows(original_rows)

    first_rows = original_rows[:FIRST_OUTPUT_ROWS]
    second_rows = original_rows[FIRST_OUTPUT_ROWS:]
    first_lines = original_lines[:FIRST_OUTPUT_ROWS]
    second_lines = original_lines[FIRST_OUTPUT_ROWS:]
    if len(first_rows) != FIRST_OUTPUT_ROWS or len(second_rows) != SECOND_OUTPUT_ROWS:
        raise AssertionError("Unexpected repartition sizes")
    first_ids = {row_identity(row) for row in first_rows}
    second_ids = {row_identity(row) for row in second_rows}
    if first_ids & second_ids:
        raise ValueError("Repartitioned OPD splits overlap")
    if first_ids | second_ids != {row_identity(row) for row in original_rows}:
        raise ValueError("Repartitioned OPD splits do not preserve the original union")

    first_text = "\n".join(first_lines) + "\n"
    second_text = "\n".join(second_lines) + "\n"
    original_text = "\n".join(original_lines) + "\n"
    if (first_text + second_text).encode("utf-8") != original_text.encode("utf-8"):
        raise AssertionError("Repartition changed the ordered source bytes")

    atomic_write(output_2k, first_text)
    atomic_write(output_3k, second_text)
    payload = {
        "schema_version": 1,
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "source": {
            "opd_001024": {"path": str(source_1k.resolve()), "rows": FIRST_SOURCE_ROWS, "sha256": sha256_file(source_1k)},
            "opd_next_004096": {"path": str(source_4k.resolve()), "rows": SECOND_SOURCE_ROWS, "sha256": sha256_file(source_4k)},
            "ordered_union_sha256": sha256_bytes(original_text.encode("utf-8")),
        },
        "repartition": {
            "opd_002048": {
                "path": str(output_2k.resolve()),
                "rows": FIRST_OUTPUT_ROWS,
                "selected_offset_first": 0,
                "selected_offset_last": FIRST_OUTPUT_ROWS - 1,
                "sha256": sha256_file(output_2k),
            },
            "opd_next_003072": {
                "path": str(output_3k.resolve()),
                "rows": SECOND_OUTPUT_ROWS,
                "selected_offset_first": FIRST_OUTPUT_ROWS,
                "selected_offset_last": TOTAL_ROWS - 1,
                "sha256": sha256_file(output_3k),
            },
            "ordered_union_sha256": sha256_bytes((first_text + second_text).encode("utf-8")),
            "overlap_count": len(first_ids & second_ids),
            "total_unique_rows": len(first_ids | second_ids),
        },
    }
    if payload["source"]["ordered_union_sha256"] != payload["repartition"]["ordered_union_sha256"]:
        raise AssertionError("Repartitioned union hash differs from the original union hash")
    atomic_write(metadata_path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
