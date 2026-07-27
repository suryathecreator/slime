#!/usr/bin/env python3
"""Canonical problem prompt shared by HARP eval, OPD, and future SFT."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


PROMPT_CONTRACT_VERSION = "qwen3_harp_boxed_v1"
PROMPT_TEMPLATE = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
    "Problem:\n{problem}"
)


def canonical_problem_prompt(problem: str) -> str:
    """Wrap raw problem text in the experiment's single prompt contract."""
    text = str(problem).strip()
    prefix = "Please reason step by step, and put your final answer within \\boxed{}.\n\nProblem:\n"
    if text.startswith(prefix):
        return text
    return PROMPT_TEMPLATE.format(problem=text)


def prompt_messages(problem: str) -> list[dict[str, str]]:
    """Return the messages passed to the native Qwen3 chat template."""
    return [{"role": "user", "content": canonical_problem_prompt(problem)}]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def apply_prompt_contract(row: dict[str, Any]) -> dict[str, Any]:
    """Apply the canonical prompt to an OPD row or a future SFT message row."""
    result = copy.deepcopy(row)
    raw_problem: str | None = None
    if result.get("prompt") is not None:
        raw_problem = str(result["prompt"])
        result["prompt"] = canonical_problem_prompt(raw_problem)
    elif isinstance(result.get("messages"), list):
        for message in result["messages"]:
            if isinstance(message, dict) and str(message.get("role", "")).lower() in {"user", "human"}:
                raw_problem = str(message.get("content", ""))
                message["content"] = canonical_problem_prompt(raw_problem)
                break
    if raw_problem is None:
        raise ValueError("Row has neither a prompt nor a user message")
    metadata = dict(result.get("metadata") or {})
    metadata["prompt_contract_version"] = PROMPT_CONTRACT_VERSION
    metadata["original_problem_sha256"] = sha256_text(raw_problem.strip())
    rendered = result.get("prompt")
    if rendered is None:
        rendered = next(
            message["content"]
            for message in result["messages"]
            if str(message.get("role", "")).lower() in {"user", "human"}
        )
    metadata["canonical_prompt_sha256"] = sha256_text(str(rendered))
    result["metadata"] = metadata
    return result


def rewrite_jsonl(source: Path, destination: Path) -> None:
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(apply_prompt_contract(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Cannot apply prompt contract to {source}:{line_number}: {exc}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rewrite_jsonl(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
