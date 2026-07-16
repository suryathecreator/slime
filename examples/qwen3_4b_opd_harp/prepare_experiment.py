#!/usr/bin/env python3
"""Prepare the pinned HARP-500 split and prompt-normalized OPD slices."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from prompt_contract import PROMPT_CONTRACT_VERSION, apply_prompt_contract


HARP_URL = "https://github.com/aadityasingh/HARP/raw/main/HARP.jsonl.zip"
HARP_SOURCE_SHA256 = "40bcbda27876ada4d5de3c26b3f6e4a67d932874f645d4ae7c1fc0aed0e09392"
HARP_SUBSET_SHA256 = "30a43db8e669d7a6cc8fbb8a93a84018d5440b110632372467183d707a30c5ab"
HARP_SEED = 42
HARP_SIZE = 500


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_problem_id(value: Any, index: int) -> str:
    if value is None or str(value).strip() == "":
        value = f"problem_{index:07d}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or f"problem_{index:07d}"


def ensure_harp_source(destination: Path, cached_source: Path | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_name(f".{destination.name}.tmp")
        if cached_source is not None and cached_source.is_file():
            shutil.copyfile(cached_source, temporary)
        else:
            urllib.request.urlretrieve(HARP_URL, temporary)
        temporary.replace(destination)
    found = sha256_file(destination)
    if found != HARP_SOURCE_SHA256:
        raise ValueError(f"HARP source SHA-256 mismatch: expected {HARP_SOURCE_SHA256}, found {found}")


def normalized_harp_rows(source_zip: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(source_zip) as archive:
        names = [name for name in archive.namelist() if name.endswith(".jsonl")]
        if not names:
            raise ValueError("HARP archive contains no JSONL file")
        with archive.open(names[0]) as raw_handle:
            raw_rows = [json.loads(line) for line in io.TextIOWrapper(raw_handle, encoding="utf-8") if line.strip()]
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows):
        problem = row.get("problem")
        answer = row.get("answer")
        if problem is None or answer is None or not str(problem).strip() or not str(answer).strip():
            continue
        pieces = [row.get("contest"), row.get("year"), row.get("number")]
        raw_id = (
            "%s_%s_%s" % tuple(pieces)
            if all(piece is not None and str(piece).strip() for piece in pieces)
            else row.get("id") or row.get("problem_id") or f"harp_{index:05d}"
        )
        rows.append(
            {
                "problem_id": safe_problem_id(raw_id, index),
                "problem": str(problem).strip(),
                "correct_answer": str(answer).strip(),
                "subject": row.get("subject"),
                "level": row.get("level"),
                "source": {"contest": row.get("contest"), "year": row.get("year"), "number": row.get("number")},
                "benchmark": "harp",
            }
        )
    ids = [row["problem_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Normalized HARP pool contains duplicate problem IDs")
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare_harp(source_zip: Path, output: Path) -> dict[str, Any]:
    rows = sorted(normalized_harp_rows(source_zip), key=lambda row: str(row["problem_id"]))
    shuffled = list(rows)
    random.Random(HARP_SEED).shuffle(shuffled)
    selected = sorted(shuffled[:HARP_SIZE], key=lambda row: str(row["problem_id"]))
    if len(selected) != HARP_SIZE:
        raise ValueError(f"Expected {HARP_SIZE} HARP rows, found {len(selected)}")
    write_jsonl_atomic(output, selected)
    found = sha256_file(output)
    if found != HARP_SUBSET_SHA256:
        raise ValueError(f"HARP subset SHA-256 mismatch: expected {HARP_SUBSET_SHA256}, found {found}")
    return {
        "source_sha256": HARP_SOURCE_SHA256,
        "subset_sha256": found,
        "seed": HARP_SEED,
        "size": HARP_SIZE,
        "ordered_problem_ids": [row["problem_id"] for row in selected],
    }


def prepare_opd(source: Path, output: Path, expected_rows: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(apply_prompt_contract(json.loads(line)))
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harp-source", required=True)
    parser.add_argument("--cached-harp-source")
    parser.add_argument("--harp-output", required=True)
    parser.add_argument("--opd-source-1k", required=True)
    parser.add_argument("--opd-source-4k", required=True)
    parser.add_argument("--opd-output-1k", required=True)
    parser.add_argument("--opd-output-4k", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()

    harp_source = Path(args.harp_source)
    cached = Path(args.cached_harp_source) if args.cached_harp_source else None
    ensure_harp_source(harp_source, cached)
    metadata = {
        "schema_version": 1,
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "harp": prepare_harp(harp_source, Path(args.harp_output)),
        "opd_001024": prepare_opd(Path(args.opd_source_1k), Path(args.opd_output_1k), 1024),
        "opd_next_004096": prepare_opd(Path(args.opd_source_4k), Path(args.opd_output_4k), 4096),
    }
    first_ids = set(metadata_row_ids(Path(args.opd_output_1k)))
    next_ids = set(metadata_row_ids(Path(args.opd_output_4k)))
    overlap = first_ids & next_ids
    if overlap:
        raise ValueError(f"OPD slices overlap on {len(overlap)} original prompt hashes")
    metadata["opd_overlap_count"] = 0
    destination = Path(args.metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def metadata_row_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            ids.append(str(metadata.get("original_problem_sha256") or metadata.get("prompt_sha256")))
    return ids


if __name__ == "__main__":
    main()
