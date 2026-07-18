#!/usr/bin/env python3
"""Normalize pinned AIME 2026 data and canonicalize the two OPD slices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from prompt_contract import PROMPT_CONTRACT_VERSION, apply_prompt_contract


AIME_DATASET_REPO = "MathArena/aime_2026"
AIME_DATASET_REVISION = "d2de22f3c656b4f56cf8981212186377d1e23bc3"
AIME_PARQUET_SHA256 = "d91db799651b4cc1f0734f52792a695c9cc60dac342524b3d8e5b2ff31c3e957"
AIME_NORMALIZED_SHA256 = "6e07e802a416526fe52559216e6e3f13f35db1c169443a39c7ae2790c8b5ca2b"
AIME_SIZE = 30


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare_aime(source: Path, output: Path) -> dict[str, Any]:
    import pyarrow.parquet as parquet

    raw_hash = sha256_file(source)
    if raw_hash != AIME_PARQUET_SHA256:
        raise ValueError(f"AIME parquet SHA-256 mismatch: expected {AIME_PARQUET_SHA256}, found {raw_hash}")
    raw_rows = parquet.read_table(source).to_pylist()
    rows: list[dict[str, Any]] = []
    for raw in sorted(raw_rows, key=lambda row: int(row["problem_idx"])):
        index = int(raw["problem_idx"])
        answer = int(raw["answer"])
        problem = str(raw["problem"]).strip()
        if not 1 <= index <= AIME_SIZE or not problem or not 0 <= answer <= 999:
            raise ValueError(f"Invalid AIME row: {raw!r}")
        rows.append(
            {
                "problem_id": f"aime_2026_{index:02d}",
                "problem_idx": index,
                "problem": problem,
                "correct_answer": answer,
                "benchmark": AIME_DATASET_REPO,
                "dataset_revision": AIME_DATASET_REVISION,
            }
        )
    if len(rows) != AIME_SIZE or [row["problem_idx"] for row in rows] != list(range(1, AIME_SIZE + 1)):
        raise ValueError("AIME 2026 must contain exactly problem_idx 1 through 30")
    write_jsonl_atomic(output, rows)
    normalized_hash = sha256_file(output)
    if normalized_hash != AIME_NORMALIZED_SHA256:
        raise ValueError(
            f"AIME normalized SHA-256 mismatch: expected {AIME_NORMALIZED_SHA256}, found {normalized_hash}"
        )
    return {
        "dataset": AIME_DATASET_REPO,
        "revision": AIME_DATASET_REVISION,
        "raw_parquet": str(source),
        "raw_parquet_sha256": raw_hash,
        "normalized_jsonl": str(output),
        "normalized_jsonl_sha256": normalized_hash,
        "rows": len(rows),
        "ordered_problem_ids": [row["problem_id"] for row in rows],
    }


def prepare_opd(source: Path, output: Path, expected_rows: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        rows = [apply_prompt_contract(json.loads(line)) for line in handle if line.strip()]
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in {source}, found {len(rows)}")
    write_jsonl_atomic(output, rows)
    return {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "rows": len(rows),
    }


def row_identity_hashes(path: Path) -> set[str]:
    values: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            value = str(metadata.get("original_problem_sha256") or "")
            if not value or value in values:
                raise ValueError(f"Missing or duplicate OPD problem identity in {path}")
            values.add(value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aime-parquet", required=True)
    parser.add_argument("--aime-output", required=True)
    parser.add_argument("--opd-source-1k", required=True)
    parser.add_argument("--opd-source-4k", required=True)
    parser.add_argument("--opd-output-1k", required=True)
    parser.add_argument("--opd-output-4k", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()
    metadata = {
        "schema_version": 1,
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "aime_2026": prepare_aime(Path(args.aime_parquet), Path(args.aime_output)),
        "opd_001024": prepare_opd(Path(args.opd_source_1k), Path(args.opd_output_1k), 1024),
        "opd_next_004096": prepare_opd(Path(args.opd_source_4k), Path(args.opd_output_4k), 4096),
    }
    first = row_identity_hashes(Path(args.opd_output_1k))
    second = row_identity_hashes(Path(args.opd_output_4k))
    if first & second:
        raise ValueError(f"OPD slices overlap on {len(first & second)} problem identities")
    metadata["opd_overlap_count"] = 0
    destination = Path(args.metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
