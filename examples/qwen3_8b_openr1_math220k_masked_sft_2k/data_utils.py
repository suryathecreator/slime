"""Shared deterministic helpers for the 2K Slime masking suite."""

from __future__ import annotations

import hashlib
import re
from typing import Any


TRACE_LOCATION = re.compile(r"__row_(\d+)__gen_(\d+)$")


def stable_unit_interval(seed: int, trace_id: str, token_index: int) -> float:
    """Match Axolotl's reproducible per-token Bernoulli draw exactly."""
    payload = "%d\0%s\0%d" % (int(seed), str(trace_id), int(token_index))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def source_location(trace_id: str) -> tuple[int, int]:
    """Extract the original OpenR1 row and generation indices."""
    match = TRACE_LOCATION.search(trace_id)
    if match is None:
        raise ValueError(f"trace ID has no source location: {trace_id}")
    return int(match.group(1)), int(match.group(2))


def find_subsequence_positions(sequence: list[int], needle: list[int]) -> list[int]:
    """Return all token positions covered by exact subsequence matches."""
    if not needle:
        raise ValueError("protected token subsequence is empty")
    covered: list[int] = []
    matches = 0
    for start in range(len(sequence) - len(needle) + 1):
        if sequence[start : start + len(needle)] == needle:
            covered.extend(range(start, start + len(needle)))
            matches += 1
    if matches != 1:
        raise ValueError(f"expected one protected subsequence match, found {matches}")
    return covered


def find_all_subsequence_positions(
    sequence: list[int], needle: list[int]
) -> list[int]:
    """Return positions covered by every non-overlapping exact match."""
    if not needle:
        raise ValueError("protected token subsequence is empty")
    covered: list[int] = []
    start = 0
    while start <= len(sequence) - len(needle):
        if sequence[start : start + len(needle)] == needle:
            covered.extend(range(start, start + len(needle)))
            start += len(needle)
        else:
            start += 1
    return covered


def protected_think_positions(tokenizer: Any, assistant_ids: list[int]) -> list[int]:
    """Return every literal opening/closing think-tag token position."""
    positions: set[int] = set()
    for text in ("<think>", "</think>"):
        tag_ids = [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
        positions.update(find_all_subsequence_positions(assistant_ids, tag_ids))
    if not positions:
        raise ValueError("assistant trace contains no literal think tag tokens")
    return sorted(positions)


def trace_set_sha256(trace_ids: list[str]) -> str:
    """Hash an ordered trace-ID list."""
    payload = "\n".join(trace_ids).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()
