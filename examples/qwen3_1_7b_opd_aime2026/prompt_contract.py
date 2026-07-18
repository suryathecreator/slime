#!/usr/bin/env python3
"""Canonical boxed-answer prompt shared by AIME evaluation and OPD."""

from __future__ import annotations

import copy
import hashlib
from typing import Any


PROMPT_CONTRACT_VERSION = "qwen3_aime2026_boxed_v1"
PROMPT_PREFIX = "Please reason step by step, and put your final answer within \\boxed{}.\n\nProblem:\n"


def canonical_problem_prompt(problem: str) -> str:
    text = str(problem).strip()
    return text if text.startswith(PROMPT_PREFIX) else PROMPT_PREFIX + text


def prompt_messages(problem: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": canonical_problem_prompt(problem)}]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def apply_prompt_contract(row: dict[str, Any]) -> dict[str, Any]:
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
